package org.beehive.gpullama3.backend.tornado.layers;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import static org.junit.Assume.assumeTrue;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import org.beehive.gpullama3.backend.tornado.scheduling.SchedulerType;
import org.beehive.gpullama3.backend.tornado.tensor.FP32TornadoTensor;
import org.beehive.gpullama3.backend.tornado.tensor.Q4_0TornadoTensor;
import org.beehive.gpullama3.backend.tornado.tensor.Q4_1TornadoTensor;
import org.beehive.gpullama3.backend.tornado.tensor.Q5_KTornadoTensor;
import org.beehive.gpullama3.backend.tornado.tensor.Q6_KTornadoTensor;
import org.beehive.gpullama3.backend.tornado.tensor.TornadoTensor;
import org.beehive.gpullama3.inference.state.Qwen35State;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.tornado.Qwen35TornadoWeights;
import org.beehive.gpullama3.model.qwen35.Qwen35Configuration;
import org.beehive.gpullama3.runtime.tensor.DataType;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

// @formatter:off
/**
 * The shape of a {@code qwen35} plan, on the smallest topology that has both layer kinds.
 *
 * <p>The real model is 64 layers of 5120 dimensions and does not fit in a unit test, but every
 * structural question about the plan is answerable at 1/80th of the size: how many graphs there
 * are, how many tasks each holds, which mixer a block gets, which kernel each weight-reading task
 * was bound to, and whether the MTP block leaked into ordinary generation. Getting those wrong on
 * the 27B costs a multi-minute load before the failure appears — and a wrong dispatch does not fail
 * at all, it produces fluent, wrong text.
 *
 * <p>The synthetic model is <b>mixed on purpose</b>, in the same places the real one is: Q4_1 down
 * projections on the early blocks, Q5_K recurrent outputs, a Q6_K vocabulary projection, Q4_0
 * everywhere else, F32 norms and SSM parameters.
 */
// @formatter:on
public class Qwen35GraphTopologyAccelTest {

    private static final int DIM = 256;
    private static final int HIDDEN = 512;
    private static final int TRUNK_LAYERS = 8;
    private static final int NEXTN_LAYERS = 1;
    private static final int BLOCKS = TRUNK_LAYERS + NEXTN_LAYERS;
    private static final int HEADS = 4;
    private static final int KV_HEADS = 2;
    private static final int HEAD_DIM = 32;
    private static final int ATTENTION_INTERVAL = 4;

    private static final int CONV_KERNEL = 4;
    private static final int STATE_SIZE = 64;
    private static final int GROUPS = 4;
    private static final int VALUE_HEADS = 4;
    private static final int INNER = 256;

    private static Qwen35Configuration config() {
        return new Qwen35Configuration(
                "Q8_0",
                DIM,
                HIDDEN,
                TRUNK_LAYERS,
                NEXTN_LAYERS,
                HEADS,
                KV_HEADS,
                HEAD_DIM,
                HEAD_DIM,
                ATTENTION_INTERVAL,
                CONV_KERNEL,
                STATE_SIZE,
                GROUPS,
                VALUE_HEADS,
                INNER,
                16,
                512,
                32,
                32,
                1e-6f,
                1e7f);
    }

    // ---- synthetic tensors --------------------------------------------------

    private static TornadoTensor f32(int elements) {
        return new FP32TornadoTensor(new FloatArray(elements));
    }

    private static TornadoTensor blocked(DataType type, int elements) {
        int blockSize =
                switch (type) {
                    case Q4_0, Q4_1 -> 32;
                    case Q5_K, Q6_K -> 256;
                    default -> throw new IllegalArgumentException(type.toString());
                };
        int blockBytes =
                switch (type) {
                    case Q4_0 -> 18;
                    case Q4_1 -> 20;
                    case Q5_K -> 176;
                    case Q6_K -> 210;
                    default -> throw new IllegalArgumentException(type.toString());
                };
        if (elements % blockSize != 0) {
            fail(type + " needs a whole number of blocks; " + elements + " is not one");
        }
        ByteArray bytes = new ByteArray(elements / blockSize * blockBytes);
        return switch (type) {
            case Q4_0 -> new Q4_0TornadoTensor(bytes);
            case Q4_1 -> new Q4_1TornadoTensor(bytes);
            case Q5_K -> new Q5_KTornadoTensor(bytes);
            case Q6_K -> new Q6_KTornadoTensor(bytes);
            default -> throw new IllegalArgumentException(type.toString());
        };
    }

    /**
     * Weights shaped like the real file: per-layer arrays indexed by absolute block, {@code null}
     * at blocks of the other kind, and the same representation per role that the 27B holds.
     */
    private static Qwen35TornadoWeights weights(Qwen35Configuration config) {
        int queryGate = config.queryGateDim();
        int kvDim = config.kvDim();
        int attnDim = config.attentionOutputInputDim();

        TornadoTensor[] attnNorm = new TornadoTensor[BLOCKS];
        TornadoTensor[] ffnNorm = new TornadoTensor[BLOCKS];
        TornadoTensor[] ffnGate = new TornadoTensor[BLOCKS];
        TornadoTensor[] ffnDown = new TornadoTensor[BLOCKS];
        TornadoTensor[] ffnUp = new TornadoTensor[BLOCKS];
        TornadoTensor[] wq = new TornadoTensor[BLOCKS];
        TornadoTensor[] wk = new TornadoTensor[BLOCKS];
        TornadoTensor[] wv = new TornadoTensor[BLOCKS];
        TornadoTensor[] wo = new TornadoTensor[BLOCKS];
        TornadoTensor[] qNorm = new TornadoTensor[BLOCKS];
        TornadoTensor[] kNorm = new TornadoTensor[BLOCKS];
        TornadoTensor[] ssmQkv = new TornadoTensor[TRUNK_LAYERS];
        TornadoTensor[] ssmGate = new TornadoTensor[TRUNK_LAYERS];
        TornadoTensor[] ssmConv = new TornadoTensor[TRUNK_LAYERS];
        TornadoTensor[] ssmAlpha = new TornadoTensor[TRUNK_LAYERS];
        TornadoTensor[] ssmBeta = new TornadoTensor[TRUNK_LAYERS];
        TornadoTensor[] ssmDtBias = new TornadoTensor[TRUNK_LAYERS];
        TornadoTensor[] ssmA = new TornadoTensor[TRUNK_LAYERS];
        TornadoTensor[] ssmNorm = new TornadoTensor[TRUNK_LAYERS];
        TornadoTensor[] ssmOut = new TornadoTensor[TRUNK_LAYERS];

        for (int l = 0; l < BLOCKS; l++) {
            attnNorm[l] = f32(DIM);
            ffnNorm[l] = f32(DIM);
            ffnGate[l] = blocked(DataType.Q4_0, DIM * HIDDEN);
            ffnUp[l] = blocked(DataType.Q4_0, DIM * HIDDEN);
            // The 27B holds Q4_1 down projections on its first eight blocks only.
            ffnDown[l] = blocked(l < 2 ? DataType.Q4_1 : DataType.Q4_0, HIDDEN * DIM);
            if (l < TRUNK_LAYERS && config.isRecurrentLayer(l)) {
                ssmQkv[l] = blocked(DataType.Q4_0, DIM * config.deltaNetConvDim());
                ssmGate[l] = blocked(DataType.Q4_0, DIM * config.deltaNetValueDim());
                ssmConv[l] = f32(config.deltaNetConvDim() * CONV_KERNEL);
                ssmAlpha[l] = f32(DIM * VALUE_HEADS);
                ssmBeta[l] = f32(DIM * VALUE_HEADS);
                ssmDtBias[l] = f32(VALUE_HEADS);
                ssmA[l] = f32(VALUE_HEADS);
                ssmNorm[l] = f32(config.headValueDim());
                ssmOut[l] = blocked(DataType.Q5_K, config.deltaNetValueDim() * DIM);
            } else {
                wq[l] = blocked(DataType.Q4_0, DIM * queryGate);
                wk[l] = blocked(DataType.Q4_0, DIM * kvDim);
                wv[l] = blocked(DataType.Q4_0, DIM * kvDim);
                wo[l] = blocked(DataType.Q4_0, attnDim * DIM);
                qNorm[l] = f32(HEAD_DIM);
                kNorm[l] = f32(HEAD_DIM);
            }
        }

        return new Qwen35TornadoWeights(
                BLOCKS,
                blocked(DataType.Q4_0, config.vocabularySize() * DIM),
                attnNorm,
                ffnNorm,
                ffnGate,
                ffnDown,
                ffnUp,
                f32(DIM),
                blocked(DataType.Q6_K, config.vocabularySize() * DIM),
                f32(config.contextLength() * HEAD_DIM),
                f32(config.contextLength() * HEAD_DIM),
                wq,
                wk,
                wv,
                wo,
                qNorm,
                kNorm,
                ssmQkv,
                ssmGate,
                ssmConv,
                ssmAlpha,
                ssmBeta,
                ssmDtBias,
                ssmA,
                ssmNorm,
                ssmOut,
                DataType.Q4_0);
    }

    private static Qwen35FFNLayers build(Qwen35Configuration config) {
        String previous = System.getProperty("use.tornadovm");
        System.setProperty("use.tornadovm", "true");
        try {
            Qwen35State state = new Qwen35State(config, -1);
            return new Qwen35FFNLayers(
                    "qwen35FFN", state, weights(config), config, SchedulerType.NVIDIA);
        } finally {
            if (previous == null) {
                System.clearProperty("use.tornadovm");
            } else {
                System.setProperty("use.tornadovm", previous);
            }
        }
    }

    private static List<String> taskNames(GridScheduler scheduler, int layer) {
        List<String> names = new ArrayList<>();
        for (String key : scheduler.keySet()) {
            if (key.startsWith("layer_" + layer + ".")) {
                names.add(key.substring(key.indexOf('.') + 1));
            }
        }
        return names;
    }

    // @formatter:off
    /**
     * Exactly which projections read a quantized activation, and — more to the point — which do
     * not.
     *
     * <p>The packed-integer path is eligible when three things hold at once: the weights are Q4_0,
     * the projection folds no residual, and the buffer it reads still holds the activation the
     * branch quantized. The third is a fact about <b>ordering</b>, and it is the one that can go
     * wrong silently: {@code wrapXb} is written twice more inside a layer, by the attention
     * branch's gated output and by the feed-forward norm, so a projection reading it after either
     * of those would consume the previous activation's quants and produce plausible, wrong numbers.
     *
     * <p>{@code attn_output_proj} is the case that proves the point. It reads {@code wrapXb} — the
     * same array — after the attention branch has overwritten it, and it must <b>not</b> be packed.
     * Today it is also excluded by folding a residual, which is exactly why this asserts the
     * outcome rather than trusting that coincidence.
     */
    // @formatter:on
    @Test
    public void onlyProjectionsReadingTheQuantizedActivationArePacked() {
        assumeTrue(
                "no packed-integer-dot device",
                org.beehive.gpullama3.backend.tornado.device.TornadoDevices.current()
                        .capabilities()
                        .supports(
                                org.beehive.gpullama3.runtime.backend.DeviceCapability
                                        .PACKED_INTEGER_DOT));
        Qwen35Configuration config = config();
        Set<String> packed = new LinkedHashSet<>();
        Set<String> notPacked = new LinkedHashSet<>();
        for (Qwen35FFNLayers.Dispatch dispatch : build(config).dispatchInventory()) {
            (dispatch.quantizedActivation() ? packed : notPacked).add(dispatch.task());
        }

        assertEquals(
                "the projections that read the branch's quantized activation",
                Set.of(
                        "attn_q_proj",
                        "attn_k_proj",
                        "attn_v_proj",
                        "ssm_qkv_proj",
                        "ssm_gate_proj"),
                packed);
        assertTrue(
                "attn_output_proj reads wrapXb after the attention branch overwrote it, so it"
                        + " cannot take the quantized activation",
                notPacked.contains("attn_output_proj"));
        assertTrue(
                "ffn_down_proj reads the feed-forward's own activation",
                notPacked.contains("ffn_down_proj"));
        assertTrue("ssm_out_proj reads the delta-net readout", notPacked.contains("ssm_out_proj"));
    }

    // ---- batched prefill: which projections reach the tensor cores ----------

    /** The batch width the batched-prefill plan is built for here; a whole number of MMA rows. */
    private static final int PREFILL_BATCH = 32;

    private static Qwen35BatchPrefillLayers buildBatched(Qwen35Configuration config) {
        String previousDevice = System.getProperty("use.tornadovm");
        String previousCores = System.getProperty("llama.qwen35.tensorCores");
        System.setProperty("use.tornadovm", "true");
        // Read into a static final when Qwen35BatchPrefillLayers first loads, which is here: no
        // test above this one touches that class.
        System.setProperty("llama.qwen35.tensorCores", "true");
        try {
            Qwen35State state =
                    (Qwen35State)
                            State.withPrefillBatchSize(
                                    PREFILL_BATCH, () -> new Qwen35State(config, -1));
            return new Qwen35BatchPrefillLayers(state, weights(config), config, PREFILL_BATCH);
        } finally {
            restore("use.tornadovm", previousDevice);
            restore("llama.qwen35.tensorCores", previousCores);
        }
    }

    private static void restore(String key, String previous) {
        if (previous == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, previous);
        }
    }

    /** The tasks only the tensor-core branch of a batched layer adds. */
    private static List<String> batchTaskNames(GridScheduler scheduler, int layer) {
        List<String> names = new ArrayList<>();
        for (String key : scheduler.keySet()) {
            if (key.startsWith("batchLayer_" + layer + ".")) {
                names.add(key.substring(key.indexOf('.') + 1));
            }
        }
        return names;
    }

    // @formatter:off
    /**
     * The {@code ffn_down} of <b>every</b> block reaches the tensor cores, in both of the
     * representations this family holds it in.
     *
     * <p>It did not. The condition that opened the branch asked for {@code Q4_0} while the choice
     * of kernel inside it asked whether the tensor was {@code Q4_1}, so the Q4_1 blocks — the first
     * eight of the 27B — took the scalar path and the Q4_1 tensor-core kernel was unreachable. A
     * kernel test could not see it: the kernel was correct, and nothing dispatched to it. This
     * asserts the production dispatch instead, on the same mixed model the rest of this class uses.
     *
     * <p>{@code ffn_down_fp16} is the marker. The scalar path reads the SwiGLU output directly;
     * only the tensor-core branch converts it first, so the task exists exactly when the projection
     * is on the tensor cores.
     */
    // @formatter:on
    @Test
    public void everyFfnDownReachesTheTensorCoresWhateverItsRepresentation() {
        assumeTrue(
                "no tensor-core-capable device",
                org.beehive.gpullama3.backend.tornado.TensorCoreSupport
                        .isTensorCoreCapableBackend());
        Qwen35Configuration config = config();
        GridScheduler scheduler = new GridScheduler();
        buildBatched(config).updateGridScheduler(scheduler);

        for (int layer = 0; layer < TRUNK_LAYERS; layer++) {
            DataType representation = layer < 2 ? DataType.Q4_1 : DataType.Q4_0;
            List<String> tasks = batchTaskNames(scheduler, layer);
            assertTrue(
                    "layer "
                            + layer
                            + " holds its ffn_down as "
                            + representation
                            + " and did not take the tensor-core path; its tasks are "
                            + tasks,
                    tasks.contains("ffn_down_fp16"));
            assertTrue(
                    "layer " + layer + " has a tensor-core ffn_down without its residual pass",
                    tasks.contains("ffn_down_residual"));
        }
    }

    // ---- the assertions -----------------------------------------------------

    /**
     * One graph per trunk layer, and the MTP block is not among them.
     *
     * <p>{@code numberOfLayers()} is the trunk; {@code numberOfBlocks()} counts the draft head with
     * it. Building the draft head into the generation plan would run it every token, for a
     * prediction nothing consumes.
     */
    @Test
    public void thereIsOneGraphPerTrunkLayerAndNoDraftHead() {
        Qwen35Configuration config = config();
        Qwen35FFNLayers layers = build(config);

        assertEquals(
                "one graph per trunk layer",
                TRUNK_LAYERS,
                layers.getFFNLayerImmutableTaskGraphs().size());
        assertEquals(
                "the last graph is the last trunk layer",
                "layer_7",
                layers.getLastFFNLayerTaskGraphID());

        GridScheduler scheduler = layers.updateGridScheduler(new GridScheduler());
        assertTrue(
                "the MTP block must not appear in ordinary generation",
                taskNames(scheduler, TRUNK_LAYERS).isEmpty());
    }

    /** Each layer kind builds its own mixer's tasks, and only those. */
    @Test
    public void eachBlockGetsTheMixerItsKindDeclares() {
        Qwen35Configuration config = config();
        GridScheduler scheduler = build(config).updateGridScheduler(new GridScheduler());

        for (int layer = 0; layer < TRUNK_LAYERS; layer++) {
            List<String> tasks = taskNames(scheduler, layer);
            boolean recurrent = config.isRecurrentLayer(layer);
            assertEquals(
                    "layer " + layer + " kind", (layer + 1) % ATTENTION_INTERVAL != 0, recurrent);

            assertEquals(
                    "layer " + layer + " delta-net tasks",
                    recurrent,
                    tasks.contains("ssm_delta_rule"));
            assertEquals("layer " + layer + " convolution", recurrent, tasks.contains("ssm_conv"));
            assertEquals("layer " + layer + " attention", !recurrent, tasks.contains("attention"));
            assertEquals("layer " + layer + " rotation", !recurrent, tasks.contains("attn_rope"));
            assertEquals(
                    "layer " + layer + " key/value append",
                    !recurrent,
                    tasks.contains("attn_kv_append"));

            // Both kinds run the same dense feed-forward and the same two normalizations.
            assertTrue("layer " + layer + " feed-forward", tasks.contains("ffn_gate_up"));
            assertTrue("layer " + layer + " down projection", tasks.contains("ffn_down_proj"));
            assertTrue("layer " + layer + " input norm", tasks.contains("attn_rms_reduce"));
            assertTrue("layer " + layer + " post-attention norm", tasks.contains("ffn_rms_reduce"));
        }
    }

    /**
     * The task count per layer kind, and the plan's total.
     *
     * <p>An exact number rather than a bound: this is the count that decides whether the real
     * model's plan is 1,300 tasks or 13,000, and a task silently gained or lost is a change in what
     * the layer computes.
     */
    @Test
    public void theTaskCountsAreTheOnesTheTopologyDeclares() {
        Qwen35Configuration config = config();
        GridScheduler scheduler = build(config).updateGridScheduler(new GridScheduler());

        int recurrentTasks = 0;
        int attentionTasks = 0;
        int total = 0;
        for (int layer = 0; layer < TRUNK_LAYERS; layer++) {
            int tasks = taskNames(scheduler, layer).size();
            total += tasks;
            if (config.isRecurrentLayer(layer)) {
                recurrentTasks = tasks;
            } else {
                attentionTasks = tasks;
            }
        }

        // One more per layer where the device takes the packed-integer path: the branch quantizes
        // the normed activation once, for the Q4_0 projections that read it.
        int quantize =
                org.beehive.gpullama3.backend.tornado.device.TornadoDevices.current()
                                .capabilities()
                                .supports(
                                        org.beehive.gpullama3.runtime.backend.DeviceCapability
                                                .PACKED_INTEGER_DOT)
                        ? 1
                        : 0;
        assertEquals("a recurrent layer's tasks", 20 + quantize, recurrentTasks);
        assertEquals("an attention layer's tasks", 16 + quantize, attentionTasks);
        assertEquals("the plan's layer tasks", 6 * recurrentTasks + 2 * attentionTasks, total);
    }

    /** Every task name is unique within its graph, which is what the scheduler keys on. */
    @Test
    public void everyTaskNameIsUniqueWithinItsGraph() {
        GridScheduler scheduler = build(config()).updateGridScheduler(new GridScheduler());
        for (int layer = 0; layer < TRUNK_LAYERS; layer++) {
            List<String> tasks = taskNames(scheduler, layer);
            Set<String> unique = new LinkedHashSet<>(tasks);
            assertEquals(
                    "layer " + layer + " has a duplicate task name", tasks.size(), unique.size());
        }
    }

    // @formatter:off
    /**
     * Every weight-reading task is bound to its own tensor's representation.
     *
     * <p>This is the assertion that a mixed model needs and a uniform one does not. Three roles in
     * this synthetic model are deliberately not the model-wide Q4_0 — the early {@code ffn_down}
     * are Q4_1 and every {@code ssm_out} is Q5_K — and a plan that read them as Q4_0 would decode
     * 20-byte and 176-byte blocks as 18-byte ones, producing weights of plausible magnitude.
     */
    // @formatter:on
    @Test
    public void everyTaskBindsItsOwnTensorsRepresentation() {
        Qwen35Configuration config = config();
        Qwen35FFNLayers layers = build(config);

        for (Qwen35FFNLayers.Dispatch dispatch : layers.dispatchInventory()) {
            DataType expected =
                    switch (dispatch.role()) {
                        case "ffn_down" -> dispatch.layer() < 2 ? DataType.Q4_1 : DataType.Q4_0;
                        case "ssm_out" -> DataType.Q5_K;
                        case "ssm_alpha", "ssm_beta" -> DataType.F32;
                        default -> DataType.Q4_0;
                    };
            assertEquals(
                    "layer "
                            + dispatch.layer()
                            + " task "
                            + dispatch.task()
                            + " reads "
                            + dispatch.role(),
                    expected,
                    dispatch.representation());
        }

        Set<DataType> bound = new LinkedHashSet<>();
        layers.dispatchInventory().forEach(d -> bound.add(d.representation()));
        assertTrue("the Q4_1 down projections must be read as Q4_1", bound.contains(DataType.Q4_1));
        assertTrue(
                "the Q5_K recurrent outputs must be read as Q5_K", bound.contains(DataType.Q5_K));
        assertTrue("the F32 SSM projections must be read as F32", bound.contains(DataType.F32));
        assertFalse(
                "nothing here is materialized as Q8_0 to find a kernel",
                bound.contains(DataType.Q8_0));
    }

    /**
     * Changing one tensor's representation changes what the plan compiles.
     *
     * <p>The legacy path has no compiled-program cache to key, so the identity that matters is the
     * one the graphs themselves carry: which kernel each task was bound to. If that did not change
     * with the tensor, then either the dispatch is not per tensor or a conversion is hiding one.
     */
    @Test
    public void aChangedRepresentationChangesTheDispatch() {
        Qwen35Configuration config = config();
        List<Qwen35FFNLayers.Dispatch> before = build(config).dispatchInventory();

        String previous = System.getProperty("use.tornadovm");
        System.setProperty("use.tornadovm", "true");
        List<Qwen35FFNLayers.Dispatch> after;
        try {
            Qwen35State state = new Qwen35State(config, -1);
            Qwen35TornadoWeights mixed = weights(config);
            // One recurrent block's output projection, re-quantized.
            mixed.ssmOut[0] = blocked(DataType.Q4_0, config.deltaNetValueDim() * DIM);
            after =
                    new Qwen35FFNLayers("qwen35FFN", state, mixed, config, SchedulerType.NVIDIA)
                            .dispatchInventory();
        } finally {
            if (previous == null) {
                System.clearProperty("use.tornadovm");
            } else {
                System.setProperty("use.tornadovm", previous);
            }
        }

        assertEquals("the same tasks either way", before.size(), after.size());
        assertFalse("the dispatch must not be identical", before.equals(after));
        for (int i = 0; i < before.size(); i++) {
            boolean isChangedTensor =
                    before.get(i).layer() == 0 && before.get(i).role().equals("ssm_out");
            if (isChangedTensor) {
                assertEquals(DataType.Q5_K, before.get(i).representation());
                assertEquals(DataType.Q4_0, after.get(i).representation());
            } else {
                assertEquals("only the changed tensor's task changed", before.get(i), after.get(i));
            }
        }
    }
}
