package org.beehive.gpullama3.model.loader;

import static org.beehive.gpullama3.tokenizer.Vocabulary.fromTokensAndScores;

import java.nio.channels.FileChannel;
import java.util.Map;
import java.util.function.IntFunction;
import org.beehive.gpullama3.auxiliary.Pair;
import org.beehive.gpullama3.backend.tornado.tensor.TornadoTensor;
import org.beehive.gpullama3.backend.tornado.tensor.TornadoTensorLoader;
import org.beehive.gpullama3.format.DataTypeMapping;
import org.beehive.gpullama3.format.GGMLTensorEntry;
import org.beehive.gpullama3.format.GGMLType;
import org.beehive.gpullama3.format.GGUF;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.standard.Qwen35StandardWeights;
import org.beehive.gpullama3.inference.weights.tornado.Qwen35TornadoWeights;
import org.beehive.gpullama3.model.format.ChatFormat.ChatTokens;
import org.beehive.gpullama3.model.format.Qwen35ChatFormat;
import org.beehive.gpullama3.model.qwen35.Qwen35;
import org.beehive.gpullama3.model.qwen35.Qwen35Configuration;
import org.beehive.gpullama3.runtime.diagnostics.DiagnosticCode;
import org.beehive.gpullama3.runtime.tensor.DataType;
import org.beehive.gpullama3.runtime.tensor.ExecutionTarget;
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

    /**
     * What a caller who asked for no particular context length gets. Chosen to make the key/value
     * arrays about a gigabyte rather than thirty-four; it is the same order as llama.cpp's own
     * default for the same reason.
     */
    private static final int DEFAULT_CONTEXT_LENGTH = 8192;

    private static final System.Logger LOGGER =
            System.getLogger(Qwen35ModelLoader.class.getName());

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
        int finalContextLength = resolveContextLength(modelContextLength);

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
     * The context this family claims, when nobody asked for one, would not fit in memory.
     *
     * <p>Qwen3.5 declares 262144. Sixteen of its layers attend, each holding a 1024-wide key and
     * value per position: at the declared maximum that is 34 GB of host arrays, allocated eagerly
     * at session construction. A caller who passed no context length gets this instead, and one who
     * asked for a specific length gets exactly what they asked for, up to the model's own maximum.
     *
     * <p>Scoped to this family deliberately. Every long-context family has the same shape of
     * problem — it is the facade's default of "the model's own maximum" that is optimistic, not
     * anything about this loader — but this is the first model where the default cannot run at all,
     * and quietly changing the rule for everyone is not this port's decision to make.
     */
    private int resolveContextLength(int modelContextLength) {
        if (contextLength > 0) {
            return Math.min(contextLength, modelContextLength);
        }
        int resolved = Math.min(modelContextLength, DEFAULT_CONTEXT_LENGTH);
        if (resolved < modelContextLength) {
            LOGGER.log(
                    System.Logger.Level.INFO,
                    () ->
                            "qwen35 declares a context of "
                                    + modelContextLength
                                    + ", whose key/value storage does not fit in host memory;"
                                    + " using "
                                    + DEFAULT_CONTEXT_LENGTH
                                    + ". Ask for a length explicitly to override.");
        }
        return resolved;
    }

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
                new Qwen35ChatFormat((Qwen35Tokenizer) tokenizer, chatTokens));
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

    // @formatter:off
    /**
     * Device weights, with the Q4_0 ones kept as they are.
     *
     * <p>Retaining matters more here than anywhere else. Qwen3.8-27B is 16 GB and almost entirely
     * Q4_0; materializing it as Q8_0 costs roughly 28 GB, which does not fit on a 24 GB device,
     * where retaining leaves about 17 GB, which does. The rest of the file — {@code ssm_out} in
     * Q5_K, eight {@code ffn_down} in Q4_1, {@code output} in Q6_K — has no kernel here and is
     * still materialized, so the model is mixed by construction and each weight is read by the
     * kernel matching its own type.
     *
     * <p><b>All or nothing for the layers that share a kernel.</b> The delta-net and attention
     * graphs dispatch per tensor, but the {@code q ‖ k ‖ v} projections of one layer are read by a
     * single fused kernel, so they must agree. The check below is over the tensors a Q4_0 kernel
     * would decode; anything else falls back to Q8_0 for the whole model rather than mixing two
     * block layouts inside one kernel, which would read 18-byte blocks as 34-byte ones and produce
     * fluent, wrong text.
     *
     * <p><b>This does not yet make the model runnable on a device.</b> No {@code
     * TornadoPlanProvider} claims this architecture, so nothing consumes these weights; the layer
     * graphs are the next piece. Building them is what makes it possible to find out whether they
     * are right.
     */
    // @formatter:on
    @Override
    protected Weights createTornadoVMWeights(
            Map<String, GGMLTensorEntry> tensorEntries,
            Qwen35Configuration config,
            Pair<float[], float[]> ropeFreqs,
            GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {

        final int blocks = config.numberOfBlocks();
        final int trunk = config.numberOfLayers();
        boolean retainQ4_0 = allQ4_0Projections(tensorEntries, config);

        TornadoTensor[] attnNorm = perBlockDevice(blocks, l -> tensorEntries.get("blk." + l + ".attn_norm.weight"));
        TornadoTensor[] ffnNorm = perBlockDevice(blocks, l -> tensorEntries.get("blk." + l + ".post_attention_norm.weight"));
        TornadoTensor[] ffnGate = new TornadoTensor[blocks];
        TornadoTensor[] ffnDown = new TornadoTensor[blocks];
        TornadoTensor[] ffnUp = new TornadoTensor[blocks];

        TornadoTensor[] wq = new TornadoTensor[blocks];
        TornadoTensor[] wk = new TornadoTensor[blocks];
        TornadoTensor[] wv = new TornadoTensor[blocks];
        TornadoTensor[] wo = new TornadoTensor[blocks];
        TornadoTensor[] attnQNorm = new TornadoTensor[blocks];
        TornadoTensor[] attnKNorm = new TornadoTensor[blocks];

        TornadoTensor[] ssmQkv = new TornadoTensor[trunk];
        TornadoTensor[] ssmGate = new TornadoTensor[trunk];
        TornadoTensor[] ssmConv1d = new TornadoTensor[trunk];
        TornadoTensor[] ssmAlpha = new TornadoTensor[trunk];
        TornadoTensor[] ssmBeta = new TornadoTensor[trunk];
        TornadoTensor[] ssmDtBias = new TornadoTensor[trunk];
        TornadoTensor[] ssmA = new TornadoTensor[trunk];
        TornadoTensor[] ssmNorm = new TornadoTensor[trunk];
        TornadoTensor[] ssmOut = new TornadoTensor[trunk];

        for (int l = 0; l < blocks; l++) {
            String blk = "blk." + l + ".";
            ffnGate[l] = deviceTensor(tensorEntries, blk + "ffn_gate.weight", retainQ4_0);
            ffnDown[l] = deviceTensor(tensorEntries, blk + "ffn_down.weight", retainQ4_0);
            ffnUp[l] = deviceTensor(tensorEntries, blk + "ffn_up.weight", retainQ4_0);
            if (config.isRecurrentLayer(l)) {
                ssmQkv[l] = deviceTensor(tensorEntries, blk + "attn_qkv.weight", retainQ4_0);
                ssmGate[l] = deviceTensor(tensorEntries, blk + "attn_gate.weight", retainQ4_0);
                // The SSM parameters are F32 in every file that carries them, and are read by
                // dtype-independent kernels; retention does not apply to them.
                ssmConv1d[l] = deviceTensor(tensorEntries, blk + "ssm_conv1d.weight", false);
                ssmAlpha[l] = deviceTensor(tensorEntries, blk + "ssm_alpha.weight", false);
                ssmBeta[l] = deviceTensor(tensorEntries, blk + "ssm_beta.weight", false);
                ssmDtBias[l] = deviceTensor(tensorEntries, blk + "ssm_dt.bias", false);
                ssmA[l] = deviceTensor(tensorEntries, blk + "ssm_a", false);
                ssmNorm[l] = deviceTensor(tensorEntries, blk + "ssm_norm.weight", false);
                ssmOut[l] = deviceTensor(tensorEntries, blk + "ssm_out.weight", retainQ4_0);
            } else {
                wq[l] = deviceTensor(tensorEntries, blk + "attn_q.weight", retainQ4_0);
                wk[l] = deviceTensor(tensorEntries, blk + "attn_k.weight", retainQ4_0);
                wv[l] = deviceTensor(tensorEntries, blk + "attn_v.weight", retainQ4_0);
                wo[l] = deviceTensor(tensorEntries, blk + "attn_output.weight", retainQ4_0);
                attnQNorm[l] = deviceTensor(tensorEntries, blk + "attn_q_norm.weight", false);
                attnKNorm[l] = deviceTensor(tensorEntries, blk + "attn_k_norm.weight", false);
            }
        }

        DataType weightType =
                retainQ4_0
                        ? DataType.Q4_0
                        : DataTypeMapping.materializedType(
                                outputWeight.ggmlType(), ExecutionTarget.GPU);

        return new Qwen35TornadoWeights(
                blocks,
                ModelLoader.loadTornadoTensorRetainingQ4_0(tokenEmbeddings),
                attnNorm,
                ffnNorm,
                ffnGate,
                ffnDown,
                ffnUp,
                ModelLoader.loadTornadoTensor(tensorEntries.get("output_norm.weight")),
                ModelLoader.loadTornadoTensor(outputWeight),
                TornadoTensorLoader.fromFloats(ropeFreqs.first()),
                TornadoTensorLoader.fromFloats(ropeFreqs.second()),
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
                weightType);
    }

    /** A device tensor, keeping Q4_0 as it is when this model retains it. */
    private static TornadoTensor deviceTensor(
            Map<String, GGMLTensorEntry> entries, String name, boolean retainQ4_0) {
        GGMLTensorEntry entry = entries.get(name);
        if (entry == null) {
            throw new ModelLoadException(
                    DiagnosticCode.MODEL_MALFORMED.prefix()
                            + "qwen35 expects "
                            + name
                            + ", which this file does not carry");
        }
        return retainQ4_0
                ? ModelLoader.loadTornadoTensorRetainingQ4_0(entry)
                : ModelLoader.loadTornadoTensor(entry);
    }

    private static TornadoTensor[] perBlockDevice(
            int blocks, IntFunction<GGMLTensorEntry> entry) {
        TornadoTensor[] tensors = new TornadoTensor[blocks];
        for (int l = 0; l < blocks; l++) {
            GGMLTensorEntry found = entry.apply(l);
            if (found == null) {
                throw new ModelLoadException(
                        DiagnosticCode.MODEL_MALFORMED.prefix()
                                + "qwen35 block "
                                + l
                                + " is missing a tensor every block must have");
            }
            tensors[l] = ModelLoader.loadTornadoTensor(found);
        }
        return tensors;
    }

    /**
     * Whether every projection a Q4_0 kernel would decode is in fact Q4_0.
     *
     * <p>The graphs dispatch per tensor, but a layer's {@code q ‖ k ‖ v} projections are read by
     * one fused kernel and must therefore agree. Rather than encode that per-kernel grouping here,
     * the answer is all-or-nothing over the model: mixing two block layouts inside one kernel would
     * read 18-byte blocks as 34-byte ones, and the output would be fluent rather than obviously
     * broken.
     *
     * <p>The SSM parameters, the norms and {@code output} are not consulted. The first two are F32
     * and are read by dtype-independent kernels; {@code output} has its own layer, which reads
     * whatever it actually holds.
     */
    private static boolean allQ4_0Projections(
            Map<String, GGMLTensorEntry> entries, Qwen35Configuration config) {
        for (int l = 0; l < config.numberOfBlocks(); l++) {
            String[] kinds =
                    config.isRecurrentLayer(l)
                            ? new String[] {"attn_qkv", "attn_gate", "ffn_gate", "ffn_up"}
                            : new String[] {
                                "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up"
                            };
            for (String kind : kinds) {
                GGMLTensorEntry entry = entries.get("blk." + l + "." + kind + ".weight");
                if (entry == null || entry.ggmlType() != GGMLType.Q4_0) {
                    return false;
                }
            }
        }
        return config.numberOfBlocks() > 0;
    }
}
