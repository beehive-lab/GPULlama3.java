package org.beehive.gpullama3.tornadovm.plan.components.q8_0;

import org.beehive.gpullama3.inference.state.LlamaState;
import org.beehive.gpullama3.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.llama.LlamaConfiguration;
import org.beehive.gpullama3.tornadovm.layers.AbstractLogitsTaskGraph;
import org.beehive.gpullama3.tornadovm.layers.Activation;
import org.beehive.gpullama3.tornadovm.layers.ActivationTaskGraph;
import org.beehive.gpullama3.tornadovm.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.gpullama3.tornadovm.layers.TransformerLayerTaskGraphs;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.LlamaQ8_0FFNLayers;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.LogitsQ8_0Layer;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.decode.LlamaQ8_0FFNLayersDecode;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.decode.LlamaQ8_0FFNLayersPrefillDecode;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.decode.LogitsQ8_0LayerDecode;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.prefill.LlamaQ8_0LayersBatchPrefill;
import org.beehive.gpullama3.tornadovm.plan.components.BatchPrefillDecodeForwardPlanComponents;
import org.beehive.gpullama3.tornadovm.plan.components.EmbeddingPreparer;
import org.beehive.gpullama3.tornadovm.plan.components.activation.BatchDecodeActivation;
import org.beehive.gpullama3.tornadovm.plan.components.activation.BatchPrefillActivation;
import org.beehive.gpullama3.tornadovm.scheduling.SchedulerDetectionService;
import org.beehive.gpullama3.tornadovm.scheduling.SchedulerType;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;

import java.lang.foreign.MemorySegment;

/**
 * {@link BatchPrefillDecodeForwardPlanComponents} for Llama + Q8_0.
 *
 * <p>Batch embedding prep: CPU dequantizes Q8_0 embeddings into {@code wrapXBatch} (FP32).
 * Decode embedding prep: raw Q8_0 block copy into {@code embeddingX} for on-device conversion.</p>
 */
public class LlamaQ8_0PlanComponents implements BatchPrefillDecodeForwardPlanComponents {

    private static final int BLOCK_SIZE       = 32;
    private static final int Q8_0_BLOCK_BYTES = 34;

    private final LlamaState state;
    private final LlamaTornadoWeights weights;
    private final LlamaConfiguration config;
    private final SchedulerType schedulerType;

    public LlamaQ8_0PlanComponents(LlamaState state, Model model) {
        this.state = state;
        this.config = (LlamaConfiguration) model.configuration();
        this.weights = (LlamaTornadoWeights) model.weights();
        this.schedulerType = SchedulerDetectionService.determineSchedulerType(model);
    }

    // ── Activations ───────────────────────────────────────────────────────────

    @Override public ActivationTaskGraph standardActivation() {
        return new Activation("activationUpdate", state, weights, config);
    }

    @Override public ActivationTaskGraph decodeActivation() {
        return new Activation("decodeActivation", state, weights, config);
    }

    @Override public ActivationTaskGraph batchPrefillActivation(int batchSize) {
        return new BatchPrefillActivation(state, config, batchSize, true);
    }

    @Override public ActivationTaskGraph batchDecodeActivation(String lastBatchLayerId) {
        return new BatchDecodeActivation(state, config, lastBatchLayerId, true);
    }

    // ── Transformer layer task graphs ──────────────────────────────────────────────────────

    @Override public TransformerLayerTaskGraphs standardLayers() {
        return new LlamaQ8_0FFNLayers("llamaFFN", state, weights, config, schedulerType);
    }

    @Override public TransformerLayerTaskGraphs prefillDecodeLayers() {
        return new LlamaQ8_0FFNLayersPrefillDecode("decode", state, weights, config, schedulerType);
    }

    @Override public TransformerLayerTaskGraphs batchDecodeLayers() {
        return new LlamaQ8_0FFNLayersDecode("decode", state, weights, config, schedulerType);
    }

    @Override public BatchPrefillTransformerLayerTaskGraphs batchPrefillLayers(int batchSize) {
        return new LlamaQ8_0LayersBatchPrefill(state, weights, config, batchSize);
    }

    // ── Logits layers ─────────────────────────────────────────────────────────

    @Override public AbstractLogitsTaskGraph standardLogits(String previousGraphId) {
        return new LogitsQ8_0Layer("logits", state, weights, config, previousGraphId, schedulerType);
    }

    @Override public AbstractLogitsTaskGraph decodeLogits(String previousGraphId) {
        return new LogitsQ8_0LayerDecode("logits", state, weights, config, previousGraphId, schedulerType);
    }

    // ── Embedding preparation ─────────────────────────────────────────────────

    @Override public EmbeddingPreparer embeddingPreparer() {
        ByteArray embTable = weights.getTokenEmbeddingTable().asByteArray();
        int dim = config.dim();
        int blocksPerRow = (dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
        long bytesPerToken = (long) blocksPerRow * Q8_0_BLOCK_BYTES;

        return new EmbeddingPreparer() {
            @Override
            public void initBatchState() {
                state.wrapXBatch.clear();
                state.batchStartPosHolder.init(0);
            }
            @Override
            public void copyBatchEmbeddings(int[] tokenIds, int startPos, int chunkSize) {
                state.batchStartPosHolder.set(0, startPos);
                for (int b = 0; b < chunkSize; b++) {
                    int tokenId = tokenIds[b];
                    for (int j = 0; j < dim; j++) {
                        int blockByteOffset = (tokenId * blocksPerRow + j / BLOCK_SIZE) * Q8_0_BLOCK_BYTES;
                        float scale = embTable.getHalfFloat(blockByteOffset).getFloat32();
                        float quant = embTable.get(blockByteOffset + 2 + j % BLOCK_SIZE);
                        state.wrapXBatch.set(b * dim + j, quant * scale);
                    }
                }
            }
            @Override
            public void copyDecodeEmbedding(int token) {
                MemorySegment.copy(embTable.getSegment(), token * bytesPerToken,
                        state.embeddingX.getSegment(), 0L, bytesPerToken);
            }
        };
    }
}
