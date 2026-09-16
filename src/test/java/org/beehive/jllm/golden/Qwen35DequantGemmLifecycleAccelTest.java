package org.beehive.jllm.golden;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.beehive.jllm.backend.tornado.PlanDispatchEvidence;
import org.beehive.jllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.beehive.jllm.inference.sampler.Sampler;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.model.format.ChatFormat;
import org.beehive.jllm.model.loader.ModelLoader;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;

// @formatter:off
/**
 * The dequantize-then-GEMM prefill path through a whole session at width 256: two full chunks and a
 * partial tail, several decode steps, a reset, and a different prompt — under CUDA graph replay.
 *
 * <p>The comparison is against the <b>direct quantized path at the same width</b>, selected by a
 * test-only means: the state's dequantization scratch is removed before the plan is built, which is
 * what the dispatch consults, so the direct plan chunks the prompt identically and differs only in
 * the projection kernels. That run happens in a child JVM launched with this one's own JVM
 * arguments and class path, before this JVM builds any plan: device memory a freed plan returns is
 * not given back to the driver until the process exits (see the surefire configuration), so two 27B
 * plans cannot coexist in one process, and the child's exit is what frees its share. Every logits
 * row of the two runs must carry the same raw bits: the pair decodes the halves the direct kernel
 * stages and accumulates each output in the same K order, so the only way the rows can differ is a
 * defect in the pair — a scratch read before it is written, a chunk's inactive rows leaking, a GEMM
 * tile off by a row.
 *
 * <p>Stale state: after the reset, the second prompt's rows must equal those of a <b>fresh</b>
 * state and plan given the same prompt, so nothing the first sequence left in the scratch, the
 * key/value store or the recurrent state can pass. And the two prompts differ, so a plan that
 * ignored its input would fail on the first comparison.
 *
 * <p>The width-128/256 parity tests cover one partial chunk against the host reference with the
 * committed bounds; this test adds the multi-chunk, decode, reset and replay coverage those do not
 * have, at raw-bit strictness against the retained direct path.
 */
// @formatter:on
public class Qwen35DequantGemmLifecycleAccelTest {

    static {
        System.setProperty("jllm.qwen35.tensorCores", "true");
        System.setProperty("jllm.kvcache.fp16", "true");
        // Graph capture on the first execution, replay on every later chunk and decode step.
        System.setProperty("jllm.cudaGraphs", "true");
    }

    private static final int WIDTH = 256;
    private static final int CONTEXT = 1024;
    private static final int DECODE_STEPS = 8;

    /** Long enough for two full 256-token chunks and a partial third. */
    private static final String PROMPT_A =
            repeat(
                            "A matrix multiplication combines two matrices by taking dot products of the"
                                    + " rows of the first with the columns of the second. ",
                            32)
                    + "Explain what a matrix multiplication is in one paragraph.";

    private static final String PROMPT_B =
            repeat(
                            "The river flows past the old mill, turning the wheel that grinds the grain"
                                    + " the farmers bring each autumn. ",
                            20)
                    + "Describe the mill in one paragraph.";

    private static String repeat(String s, int n) {
        StringBuilder b = new StringBuilder();
        for (int i = 0; i < n; i++) {
            b.append(s);
        }
        return b.toString();
    }

    private record Run(List<float[]> rows, GridScheduler scheduler, int promptTokens) {}

    @Test
    public void thePairPathMatchesTheDirectPathAcrossChunksDecodeAndReset() throws Exception {
        Path modelPath = GoldenFixture.locate(Fixture.QWEN3_8_27B_Q4_0);
        if (modelPath == null) {
            System.out.println(
                    "[SKIP] environment absent — "
                            + GoldenFixture.absentMessage(Fixture.QWEN3_8_27B_Q4_0));
            assumeTrue("environment absent", false);
        }
        assumeTrue(
                "no tensor-core-capable device",
                org.beehive.jllm.backend.tornado.TensorCoreSupport.isTensorCoreCapableBackend());
        assertTrue(TornadoVMMasterPlan.CUDA_GRAPHS);

        String previousPrefill = System.getProperty("jllm.withPrefillDecode");
        String previousBatch = System.getProperty("jllm.prefillBatchSize");
        System.setProperty("jllm.withPrefillDecode", "true");
        System.setProperty("jllm.prefillBatchSize", String.valueOf(WIDTH));
        try {
            Model model = ModelLoader.loadModel(modelPath, CONTEXT, true, true);
            List<Integer> promptA = encode(model, PROMPT_A);
            List<Integer> promptB = encode(model, PROMPT_B);
            assertTrue(
                    "prompt A has " + promptA.size() + " tokens; needs > 2 chunks of " + WIDTH,
                    promptA.size() > 2 * WIDTH && promptA.size() % WIDTH != 0);
            assertTrue("prompt B must also span a partial chunk", promptB.size() > WIDTH);

            // Direct path first, in a child JVM (see the class comment), so its device memory is
            // released before this process allocates its own plan.
            List<float[]> directRows = captureDirectInChildJvm(modelPath);

            // Pair path, one state and one plan, three sequences: prompt B on the fresh plan,
            // reset, prompt A (the multi-chunk one, compared with the direct path), reset, prompt
            // B again (compared with its own fresh run, so nothing A left behind can pass).
            State pairState = State.withPrefillBatchSize(WIDTH, model::createNewState);
            assertNotNull(
                    "the state did not allocate the dequantization scratch at width " + WIDTH,
                    pairState.workspace.wrapDequantScratchFP16);
            TornadoVMMasterPlan pairPlan =
                    TornadoVMMasterPlan.initializeTornadoVMPlan(pairState, model);
            Run freshB;
            Run pairA;
            Run pairB;
            try {
                freshB = run(model, pairState, pairPlan, promptB);
                reset(model, pairState, pairPlan);
                pairA = run(model, pairState, pairPlan, promptA);
                reset(model, pairState, pairPlan);
                pairB = run(model, pairState, pairPlan, promptB);
            } finally {
                pairPlan.freeTornadoExecutionPlan();
            }
            PlanDispatchEvidence.assertQwen35AttentionOutputOnDequantGemm(
                    pairA.scheduler(),
                    WIDTH,
                    model.configuration().dim(),
                    ((org.beehive.jllm.model.qwen35.Qwen35Configuration) model.configuration())
                            .attentionOutputInputDim());

            // The direct-path rows for prompt A, captured by the child JVM before this one built
            // its plan.
            List<float[]> directA = directRows;

            assertRowsIdentical("pair vs direct, prompt A", pairA.rows(), directA);
            assertRowsIdentical("after reset vs fresh, prompt B", pairB.rows(), freshB.rows());
            assertTrue(
                    "prompts A and B produced identical first rows, so the input is not reaching"
                            + " the plan",
                    !java.util.Arrays.equals(pairA.rows().get(0), pairB.rows().get(0)));
            System.out.printf(
                    "[LIFECYCLE] width %d: prompt A %d tokens (%d chunks), prompt B %d tokens,"
                            + " %d rows compared each, all raw-bit identical%n",
                    WIDTH,
                    pairA.promptTokens(),
                    (pairA.promptTokens() + WIDTH - 1) / WIDTH,
                    pairB.promptTokens(),
                    pairA.rows().size());
        } finally {
            restore("jllm.withPrefillDecode", previousPrefill);
            restore("jllm.prefillBatchSize", previousBatch);
        }
    }

    /**
     * What a session's reset does (see {@code LegacySessionRuntime.reset}): re-seed the state's
     * latest token, then clear the sequence state on the device through the plan. The plan-level
     * reset alone leaves the previous sequence's last token as the seed the next prompt is fed
     * from, and the next sequence then differs from a fresh one at every row.
     */
    private static void reset(Model model, State state, TornadoVMMasterPlan plan) {
        state.latestToken = model.chatFormat().getBeginOfText();
        plan.resetSequenceState();
    }

    private static List<Integer> encode(Model model, String prompt) {
        ChatFormat chatFormat = model.chatFormat();
        List<Integer> tokens = new ArrayList<>();
        if (model.shouldAddBeginOfText()) {
            tokens.add(chatFormat.getBeginOfText());
        }
        tokens.addAll(
                chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, prompt)));
        tokens.addAll(
                chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
        return tokens;
    }

    /** Prefill the prompt, then decode {@link #DECODE_STEPS} greedy tokens, capturing every row. */
    private static Run run(Model model, State state, TornadoVMMasterPlan plan, List<Integer> prompt)
            throws Exception {
        List<float[]> rows = new ArrayList<>();
        Sampler capturing =
                tensor -> {
                    float[] row = new float[tensor.size()];
                    for (int i = 0; i < row.length; i++) {
                        row[i] = tensor.get(i);
                    }
                    rows.add(row);
                    int token = Sampler.TENSOR_ARGMAX.sampleToken(tensor);
                    return token;
                };
        int skippedSeed =
                org.beehive.jllm.inference.PromptIngestion.of(state, prompt, 0).firstIndex();
        int budget = prompt.size() + DECODE_STEPS - skippedSeed;
        Set<Integer> stopTokens = Set.of();
        model.generateTokensGPU(state, 0, prompt, stopTokens, budget, capturing, false, null, plan);
        GridScheduler scheduler = PlanDispatchEvidence.gridSchedulerIfAvailable(plan);
        // The budget counts every forward, ingestion included; what matters is several decode
        // rows, and the same budget for both runs being compared.
        assertTrue("only " + rows.size() + " rows captured", rows.size() >= DECODE_STEPS - 2);
        return new Run(rows, scheduler, prompt.size());
    }

    /**
     * Every row both runs produced, raw-bit equal. The counts may differ by one: a fresh state is
     * seeded with the token the prompt opens with and ingests it once ({@code
     * PromptIngestion.firstIndex} 1), while a reset state's seed is its last generated token, so
     * the same budget buys one more decode row there. The rows align from the front — both runs
     * start at position zero on the same prompt — so the extra row is the last one.
     */
    private static void assertRowsIdentical(String what, List<float[]> a, List<float[]> b) {
        int common = Math.min(a.size(), b.size());
        assertTrue(
                what + ": counts " + a.size() + " and " + b.size() + " differ by more than one",
                Math.abs(a.size() - b.size()) <= 1);
        assertTrue(what + ": only " + common + " rows to compare", common >= DECODE_STEPS - 2);
        for (int r = 0; r < common; r++) {
            float[] x = a.get(r);
            float[] y = b.get(r);
            assertEquals(what + ": row " + r + " length", x.length, y.length);
            for (int i = 0; i < x.length; i++) {
                assertTrue(
                        what + ": row " + r + " logit " + i + " not finite", Float.isFinite(x[i]));
                if (Float.floatToRawIntBits(x[i]) != Float.floatToRawIntBits(y[i])) {
                    throw new AssertionError(
                            what
                                    + ": row "
                                    + r
                                    + " logit "
                                    + i
                                    + " differs: "
                                    + x[i]
                                    + " vs "
                                    + y[i]);
                }
            }
        }
    }

    /**
     * Runs {@link #main} in a child JVM with this JVM's arguments and class path: the direct path
     * at {@link #WIDTH}, prompt A, rows written raw to a temporary file.
     */
    private static List<float[]> captureDirectInChildJvm(Path modelPath) throws Exception {
        return captureInChildJvm(modelPath, "direct");
    }

    private static List<float[]> captureInChildJvm(Path modelPath, String mode) throws Exception {
        Path out = java.nio.file.Files.createTempFile("qwen35-direct-rows", ".bin");
        List<String> command = new ArrayList<>();
        command.add(Path.of(System.getProperty("java.home"), "bin", "java").toString());
        command.addAll(
                java.lang.management.ManagementFactory.getRuntimeMXBean().getInputArguments());
        command.add("-Djllm.qwen35.tensorCores=true");
        command.add("-Djllm.kvcache.fp16=true");
        command.add("-Djllm.cudaGraphs=true");
        command.add("-cp");
        command.add(System.getProperty("java.class.path"));
        command.add(Qwen35DequantGemmLifecycleAccelTest.class.getName());
        command.add(modelPath.toString());
        command.add(out.toString());
        command.add(mode);
        Process child =
                new ProcessBuilder(command)
                        .redirectErrorStream(true)
                        .redirectOutput(ProcessBuilder.Redirect.INHERIT)
                        .start();
        int status = child.waitFor();
        assertEquals("the direct-path capture in the child JVM failed", 0, status);
        List<float[]> rows = readRows(out);
        java.nio.file.Files.deleteIfExists(out);
        return rows;
    }

    /** Child entry: the direct path at {@link #WIDTH} over prompt A; rows to {@code args[1]}. */
    public static void main(String[] args) throws Exception {
        System.setProperty("jllm.withPrefillDecode", "true");
        System.setProperty("jllm.prefillBatchSize", String.valueOf(WIDTH));
        Model model = ModelLoader.loadModel(Path.of(args[0]), CONTEXT, true, true);
        List<Integer> promptA = encode(model, PROMPT_A);
        State directState = State.withPrefillBatchSize(WIDTH, model::createNewState);
        // The test-only selection: no scratch, so the dispatch keeps the direct kernels.
        if (args[2].equals("direct")) {
            directState.workspace.wrapDequantScratchFP16 = null;
        }
        TornadoVMMasterPlan directPlan =
                TornadoVMMasterPlan.initializeTornadoVMPlan(directState, model);
        Run directA;
        try {
            directA = run(model, directState, directPlan, promptA);
        } finally {
            directPlan.freeTornadoExecutionPlan();
        }
        if (args[2].equals("direct")) {
            PlanDispatchEvidence.assertQwen35AttentionOutputOnTensorCores(
                    directA.scheduler(), WIDTH, model.configuration().dim());
        }
        writeRows(Path.of(args[1]), directA.rows());
        System.out.printf(
                "[LIFECYCLE] child: direct path at width %d, %d rows written%n",
                WIDTH, directA.rows().size());
        System.exit(0);
    }

    private static void writeRows(Path path, List<float[]> rows) throws java.io.IOException {
        try (var out =
                new java.io.DataOutputStream(
                        new java.io.BufferedOutputStream(
                                java.nio.file.Files.newOutputStream(path)))) {
            out.writeInt(rows.size());
            for (float[] row : rows) {
                out.writeInt(row.length);
                for (float v : row) {
                    out.writeInt(Float.floatToRawIntBits(v));
                }
            }
        }
    }

    private static List<float[]> readRows(Path path) throws java.io.IOException {
        try (var in =
                new java.io.DataInputStream(
                        new java.io.BufferedInputStream(
                                java.nio.file.Files.newInputStream(path)))) {
            int count = in.readInt();
            List<float[]> rows = new ArrayList<>(count);
            for (int r = 0; r < count; r++) {
                float[] row = new float[in.readInt()];
                for (int i = 0; i < row.length; i++) {
                    row[i] = Float.intBitsToFloat(in.readInt());
                }
                rows.add(row);
            }
            return rows;
        }
    }

    private static void restore(String key, String previous) {
        if (previous == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, previous);
        }
    }
}
