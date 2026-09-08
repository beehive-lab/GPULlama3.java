package org.beehive.gpullama3.model.loader;

import static org.beehive.gpullama3.tokenizer.Vocabulary.fromTokensAndScores;

import java.nio.channels.FileChannel;
import java.util.Map;
import java.util.function.IntFunction;
import org.beehive.gpullama3.auxiliary.Pair;
import org.beehive.gpullama3.format.DataTypeMapping;
import org.beehive.gpullama3.format.GGMLTensorEntry;
import org.beehive.gpullama3.format.GGUF;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.standard.Qwen35StandardWeights;
import org.beehive.gpullama3.model.format.ChatFormat.ChatTokens;
import org.beehive.gpullama3.model.format.Qwen3ChatFormat;
import org.beehive.gpullama3.model.qwen35.Qwen35;
import org.beehive.gpullama3.model.qwen35.Qwen35Configuration;
import org.beehive.gpullama3.runtime.diagnostics.DiagnosticCode;
import org.beehive.gpullama3.tensor.standard.ArrayFloatTensor;
import org.beehive.gpullama3.tensor.standard.FloatTensor;
import org.beehive.gpullama3.tokenizer.Qwen35Tokenizer;
import org.beehive.gpullama3.tokenizer.Tokenizer;
import org.beehive.gpullama3.tokenizer.Vocabulary;

/**
 * Loads a {@code qwen35} GGUF.
 *
 * <p>Two things it does that no other loader here does.
 *
 * <p><b>It reads two disjoint weight sets, chosen per layer.</b> A trunk layer is either an
 * attention layer or a delta-net layer, and the tensors it carries are entirely different. The
 * choice is {@link Qwen35Configuration#isRecurrentLayer(int)} and nothing else, so a layer's kind
 * is decided in one place at load and at execution alike.
 *
 * <p><b>It validates before it reads any weight.</b> The delta-net dimensions are derived from the
 * SSM metadata block rather than stated, and a block that does not divide evenly would produce a
 * model that loads and computes nonsense. That check belongs before a gigabyte is mapped, not
 * after.
 */
public class Qwen35ModelLoader extends AbstractModelLoader<Qwen35, Qwen35Configuration> {

    public Qwen35ModelLoader(
            FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
        super(fileChannel, gguf, contextLength, useTornadovm);
    }

    @Override
    protected Vocabulary loadVocabulary(Map<String, Object> metadata) {
        return fromTokensAndScores(metadata);
    }

    @Override
    protected Tokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        return new Qwen35Tokenizer(metadata, vocabulary);
    }

    // @formatter:off
    @Override
    protected Qwen35Configuration createConfiguration(Map<String, Object> metadata) {
        int modelContextLength = (int) metadata.get("qwen35.context_length");
        int finalContextLength =
                (contextLength < 0 || modelContextLength < contextLength)
                        ? modelContextLength
                        : contextLength;

        // block_count counts the MTP blocks with the trunk; the trunk is what the forward pass
        // runs, so the two are separated here rather than at every use.
        int blockCount = (int) metadata.get("qwen35.block_count");
        int nextnLayers =
                metadata.containsKey("qwen35.nextn_predict_layers")
                        ? (int) metadata.get("qwen35.nextn_predict_layers")
                        : 0;

        Qwen35Configuration config =
                new Qwen35Configuration(
                        getModelQuantization(metadata),
                        (int) metadata.get("qwen35.embedding_length"),
                        (int) metadata.get("qwen35.feed_forward_length"),
                        blockCount - nextnLayers,
                        nextnLayers,
                        (int) metadata.get("qwen35.attention.head_count"),
                        (int) metadata.get("qwen35.attention.head_count_kv"),
                        (int) metadata.get("qwen35.attention.key_length"),
                        (int) metadata.get("qwen35.attention.value_length"),
                        (int) metadata.get("qwen35.full_attention_interval"),
                        (int) metadata.get("qwen35.ssm.conv_kernel"),
                        (int) metadata.get("qwen35.ssm.state_size"),
                        (int) metadata.get("qwen35.ssm.group_count"),
                        (int) metadata.get("qwen35.ssm.time_step_rank"),
                        (int) metadata.get("qwen35.ssm.inner_size"),
                        (int) metadata.get("qwen35.rope.dimension_count"),
                        vocabulary.size(),
                        modelContextLength,
                        finalContextLength,
                        (float) metadata.get("qwen35.attention.layer_norm_rms_epsilon"),
                        (float) metadata.get("qwen35.rope.freq_base"));
        validate(config);
        return config;
    }
    // @formatter:on

    /**
     * Refuses a metadata block that cannot be run, naming which relationship fails.
     *
     * <p>Every check here is a dimension one part of the model derives and another states. They
     * agree in a well-formed file; where they do not, the failure downstream is a silently
     * mis-strided read, which produces fluent text and wrong logits.
     */
    private static void validate(Qwen35Configuration config) {
        String prefix = DiagnosticCode.MODEL_MALFORMED.prefix();
        if (config.ssmInnerSize() % config.ssmTimeStepRank() != 0) {
            throw new ModelLoadException(
                    prefix
                            + "qwen35.ssm.inner_size ("
                            + config.ssmInnerSize()
                            + ") must divide by qwen35.ssm.time_step_rank ("
                            + config.ssmTimeStepRank()
                            + "): the value head width is their quotient");
        }
        if (config.numberOfValueHeads() % config.numberOfKeyHeads() != 0) {
            throw new ModelLoadException(
                    prefix
                            + "qwen35.ssm.time_step_rank ("
                            + config.ssmTimeStepRank()
                            + ") must be a multiple of qwen35.ssm.group_count ("
                            + config.ssmGroupCount()
                            + "): each key head is shared by a whole number of value heads");
        }
        if (config.numberOfHeads() % config.numberOfKeyValueHeads() != 0) {
            throw new ModelLoadException(
                    prefix
                            + "qwen35.attention.head_count ("
                            + config.numberOfHeads()
                            + ") must be a multiple of head_count_kv ("
                            + config.numberOfKeyValueHeads()
                            + ")");
        }
        if (config.numberOfHeadsKey() != config.numberOfHeadsValue()) {
            throw new ModelLoadException(
                    prefix
                            + "qwen35 attention needs equal key and value head widths, got "
                            + config.numberOfHeadsKey()
                            + " and "
                            + config.numberOfHeadsValue());
        }
        if (config.ropeDimensionCount() % 2 != 0
                || config.ropeDimensionCount() > config.numberOfHeadsKey()) {
            throw new ModelLoadException(
                    prefix
                            + "qwen35.rope.dimension_count ("
                            + config.ropeDimensionCount()
                            + ") must be even and at most the head width ("
                            + config.numberOfHeadsKey()
                            + ")");
        }
        if (config.fullAttentionInterval() < 1) {
            throw new ModelLoadException(
                    prefix
                            + "qwen35.full_attention_interval must be at least 1, got "
                            + config.fullAttentionInterval());
        }
        if (config.ssmConvKernel() < 1) {
            throw new ModelLoadException(
                    prefix
                            + "qwen35.ssm.conv_kernel must be at least 1, got "
                            + config.ssmConvKernel());
        }
    }

    /**
     * RoPE tables over {@code rope.dimension_count}, not over the head width.
     *
     * <p>The head is 256 wide and only 64 of it rotates, so tables built for the head would be four
     * times too large and, worse, indexed with the wrong stride.
     *
     * <p>Built for the <i>resolved</i> context length rather than the model's declared maximum:
     * this architecture declares 262144, and no position past what the session can reach is ever
     * looked up.
     */
    @Override
    protected Pair<float[], float[]> precomputeRopeFrequencies(Qwen35Configuration config) {
        return RopeFrequencies.precomputeFreqsCis(
                config.contextLength(), config.ropeDimensionCount(), config.ropeTheta(),
                false, 0, 0, 0, 0);
    }

    @Override
    protected Qwen35 createModel(
            Qwen35Configuration config, Tokenizer tokenizer, Weights weights) {
        ChatTokens chatTokens =
                new ChatTokens("<|im_start|>", "<|im_end|>", "", "<|end_of_text|>", "<|endoftext|>");
        return new Qwen35(
                config,
                tokenizer,
                weights,
                new Qwen3ChatFormat((Qwen35Tokenizer) tokenizer, chatTokens));
    }

    // @formatter:off
    @Override
    protected Weights createStandardWeights(
            Map<String, GGMLTensorEntry> tensorEntries,
            Qwen35Configuration config,
            Pair<float[], float[]> ropeFreqs,
            GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {

        final int blocks = config.numberOfBlocks();
        final int trunk = config.numberOfLayers();

        // Present at every block, of either kind.
        FloatTensor[] attnNorm = perBlock(blocks, l -> tensorEntries.get("blk." + l + ".attn_norm.weight"));
        FloatTensor[] ffnNorm = perBlock(blocks, l -> tensorEntries.get("blk." + l + ".post_attention_norm.weight"));
        FloatTensor[] ffnGate = perBlock(blocks, l -> tensorEntries.get("blk." + l + ".ffn_gate.weight"));
        FloatTensor[] ffnDown = perBlock(blocks, l -> tensorEntries.get("blk." + l + ".ffn_down.weight"));
        FloatTensor[] ffnUp = perBlock(blocks, l -> tensorEntries.get("blk." + l + ".ffn_up.weight"));

        // Attention blocks: every trunk layer that does not recur, plus every MTP block.
        FloatTensor[] wq = new FloatTensor[blocks];
        FloatTensor[] wk = new FloatTensor[blocks];
        FloatTensor[] wv = new FloatTensor[blocks];
        FloatTensor[] wo = new FloatTensor[blocks];
        FloatTensor[] attnQNorm = new FloatTensor[blocks];
        FloatTensor[] attnKNorm = new FloatTensor[blocks];

        // Recurrent blocks.
        FloatTensor[] ssmQkv = new FloatTensor[trunk];
        FloatTensor[] ssmGate = new FloatTensor[trunk];
        FloatTensor[] ssmConv1d = new FloatTensor[trunk];
        FloatTensor[] ssmAlpha = new FloatTensor[trunk];
        FloatTensor[] ssmBeta = new FloatTensor[trunk];
        FloatTensor[] ssmDtBias = new FloatTensor[trunk];
        FloatTensor[] ssmA = new FloatTensor[trunk];
        FloatTensor[] ssmNorm = new FloatTensor[trunk];
        FloatTensor[] ssmOut = new FloatTensor[trunk];

        for (int l = 0; l < blocks; l++) {
            String blk = "blk." + l + ".";
            if (config.isRecurrentLayer(l)) {
                ssmQkv[l] = required(tensorEntries, blk + "attn_qkv.weight");
                ssmGate[l] = required(tensorEntries, blk + "attn_gate.weight");
                ssmConv1d[l] = required(tensorEntries, blk + "ssm_conv1d.weight");
                ssmAlpha[l] = required(tensorEntries, blk + "ssm_alpha.weight");
                ssmBeta[l] = required(tensorEntries, blk + "ssm_beta.weight");
                ssmDtBias[l] = required(tensorEntries, blk + "ssm_dt.bias");
                ssmA[l] = required(tensorEntries, blk + "ssm_a");
                ssmNorm[l] = required(tensorEntries, blk + "ssm_norm.weight");
                ssmOut[l] = required(tensorEntries, blk + "ssm_out.weight");
            } else {
                wq[l] = required(tensorEntries, blk + "attn_q.weight");
                wk[l] = required(tensorEntries, blk + "attn_k.weight");
                wv[l] = required(tensorEntries, blk + "attn_v.weight");
                wo[l] = required(tensorEntries, blk + "attn_output.weight");
                attnQNorm[l] = required(tensorEntries, blk + "attn_q_norm.weight");
                attnKNorm[l] = required(tensorEntries, blk + "attn_k_norm.weight");
            }
        }

        // MTP blocks: an attention block plus the four tensors that make it a draft head.
        FloatTensor[] nextnENorm = new FloatTensor[blocks];
        FloatTensor[] nextnHNorm = new FloatTensor[blocks];
        FloatTensor[] nextnEhProj = new FloatTensor[blocks];
        FloatTensor[] nextnSharedHeadNorm = new FloatTensor[blocks];
        for (int l = trunk; l < blocks; l++) {
            String blk = "blk." + l + ".nextn.";
            nextnENorm[l] = required(tensorEntries, blk + "enorm.weight");
            nextnHNorm[l] = required(tensorEntries, blk + "hnorm.weight");
            nextnEhProj[l] = required(tensorEntries, blk + "eh_proj.weight");
            // Absent means the block shares the trunk's final norm, which llama.cpp allows.
            nextnSharedHeadNorm[l] = optional(tensorEntries, blk + "shared_head_norm.weight");
        }

        return new Qwen35StandardWeights(
                blocks,
                ModelLoader.loadTensor(tokenEmbeddings),
                attnNorm,
                ffnNorm,
                ffnGate,
                ffnDown,
                ffnUp,
                ModelLoader.loadTensor(tensorEntries.get("output_norm.weight")),
                ModelLoader.loadTensor(outputWeight),
                new ArrayFloatTensor(ropeFreqs.first()),
                new ArrayFloatTensor(ropeFreqs.second()),
                wq,
                wk,
                wv,
                wo,
                attnQNorm,
                attnKNorm,
                ssmQkv,
                ssmGate,
                ssmConv1d,
                ssmAlpha,
                ssmBeta,
                ssmDtBias,
                ssmA,
                ssmNorm,
                ssmOut,
                nextnENorm,
                nextnHNorm,
                nextnEhProj,
                nextnSharedHeadNorm,
                DataTypeMapping.sourceType(outputWeight.ggmlType()));
    }
    // @formatter:on

    private static FloatTensor[] perBlock(int blocks, IntFunction<GGMLTensorEntry> entry) {
        FloatTensor[] tensors = new FloatTensor[blocks];
        for (int l = 0; l < blocks; l++) {
            GGMLTensorEntry found = entry.apply(l);
            if (found == null) {
                throw new ModelLoadException(
                        DiagnosticCode.MODEL_MALFORMED.prefix()
                                + "qwen35 block "
                                + l
                                + " is missing a tensor every block must have");
            }
            tensors[l] = ModelLoader.loadTensor(found);
        }
        return tensors;
    }

    private static FloatTensor required(Map<String, GGMLTensorEntry> entries, String name) {
        GGMLTensorEntry entry = entries.get(name);
        if (entry == null) {
            throw new ModelLoadException(
                    DiagnosticCode.MODEL_MALFORMED.prefix()
                            + "qwen35 expects "
                            + name
                            + ", which this file does not carry");
        }
        return ModelLoader.loadTensor(entry);
    }

    private static FloatTensor optional(Map<String, GGMLTensorEntry> entries, String name) {
        GGMLTensorEntry entry = entries.get(name);
        return entry == null ? null : ModelLoader.loadTensor(entry);
    }

    /**
     * Refused, rather than producing device weights no plan consumes.
     *
     * <p>Nothing lowers this architecture, and its delta-net layers have no kernels. Materializing
     * its Q4_0 weights as Q8_0 would also roughly double what the file occupies, which is the other
     * half of why the accelerator path is not merely unimplemented but presently impractical.
     */
    @Override
    protected Weights createTornadoVMWeights(
            Map<String, GGMLTensorEntry> tensorEntries,
            Qwen35Configuration config,
            Pair<float[], float[]> ropeFreqs,
            GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        throw new UnsupportedOperationException(
                "qwen35 has no accelerator path: no TornadoPlanProvider claims it, and its"
                        + " delta-net layers have no kernels. Load it for the CPU.");
    }
}
