package org.beehive.gpullama3.backend.tornado.layers;

import org.beehive.gpullama3.backend.tornado.scheduling.SchedulerType;
import org.beehive.gpullama3.inference.state.Qwen35State;
import org.beehive.gpullama3.inference.weights.tornado.Qwen35TornadoWeights;
import org.beehive.gpullama3.model.qwen35.Qwen35Configuration;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

// @formatter:off
/**
 * The decode layers of the batched plan.
 *
 * <p>The same graphs the single-token plan builds, with one difference at layer 0: the key/value
 * store, the block table and the recurrent state were allocated and filled by the batch-prefill
 * graphs, so this layer <b>consumes</b> them from the decode activation rather than uploading its
 * own. Uploading would give decode a second, empty copy of the sequence's history — the model would
 * answer as though the prompt had never been read.
 *
 * <p>The weights come the same way, from the batch-prefill graph for the same block: bound with a
 * transfer in both families, a plan would hold the whole model twice.
 */
// @formatter:on
public class Qwen35FFNLayersBatchDecode extends Qwen35FFNLayers {

    public Qwen35FFNLayersBatchDecode(
            String taskGraphName,
            Qwen35State state,
            Qwen35TornadoWeights weights,
            Qwen35Configuration config,
            SchedulerType schedulerType) {
        super(taskGraphName, state, weights, config, schedulerType, "decodeActivation");
    }

    /** The batch-prefill graph for the same block already uploaded these weights. */
    @Override
    protected String weightSourceGraphName(int layerIndex) {
        return "batchLayer_" + layerIndex;
    }

    @Override
    protected TaskGraph configureLayerDataTransfers(TaskGraph layer, int layerIndex) {
        if (layerIndex != 0) {
            return super.configureLayerDataTransfers(layer, layerIndex);
        }
        Qwen35State state = (Qwen35State) this.state;
        layer.transferToDevice(
                DataTransferMode.EVERY_EXECUTION,
                state.workspace.positionHolder,
                state.workspace.temp,
                state.workspace.tempFFN);
        layer.transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
                context,
                state.workspace.wrapXb,
                state.workspace.wrapQ,
                state.workspace.wrapAttnQ,
                state.workspace.wrapAttnGate,
                state.workspace.wrapK,
                state.workspace.wrapV,
                state.workspace.wrapAtt,
                state.workspace.wrapHb);
        layer.transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
                state.workspace.wrapSsmQkv,
                state.workspace.wrapSsmConvOut,
                state.workspace.wrapSsmZ,
                state.workspace.wrapSsmAlpha,
                state.workspace.wrapSsmBeta,
                state.workspace.wrapSsmQ,
                state.workspace.wrapSsmK,
                state.workspace.wrapSsmV,
                state.workspace.wrapSsmOut);
        // What prefill left behind: the caches, the table that addresses them, and the recurrence.
        layer.consumeFromDevice("decodeActivation", keyStore(), valueStore());
        layer.consumeFromDevice("decodeActivation", state.workspace.wrapBlockTable);
        layer.consumeFromDevice(
                "decodeActivation", state.workspace.wrapConvState, state.workspace.wrapDeltaState);
        return layer;
    }
}
