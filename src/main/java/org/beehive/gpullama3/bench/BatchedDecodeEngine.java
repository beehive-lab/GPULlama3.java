package org.beehive.gpullama3.bench;

import org.beehive.gpullama3.Options;
import org.beehive.gpullama3.inference.state.LlamaState;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.gpullama3.inference.weights.tornado.TornadoWeights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.model.llama.LlamaConfiguration;
import org.beehive.gpullama3.model.qwen3.Qwen3Configuration;
import org.beehive.gpullama3.tornadovm.kernels.TransformerBatchPrefillKernels;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.decode.LlamaFP16LayersBatchDecodeMMA;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.decode.Qwen3FP16LayersBatchDecodeMMA;
import org.beehive.gpullama3.tornadovm.plan.components.activation.BatchPrefillActivation;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

import static org.beehive.gpullama3.model.loader.ModelLoader.loadModel;

/**
 * End-to-end static batched-decode engine for LLaMA FP16: B independent sequences
 * generate together, one token per step, each attending its OWN KV region.
 *
 * <p>Reuses the batched-prefill MMA pipeline (projections as tensor-core GEMMs,
 * weight read once and applied to all B tokens → compute-bound) but swaps the two
 * KV-addressing kernels for the per-slot decode variants
 * ({@link TransformerBatchPrefillKernels#batchedDecodeRopeWithKVCachePacked},
 * {@link TransformerBatchPrefillKernels#batchedDecodeAttentionFP16Out}) over a
 * B-sized KV cache.</p>
 *
 * <p>ONE step routine serves both phases: prompt tokens are fed with logits
 * discarded (fills the B KV regions with identical RoPE to decode → no cache
 * mismatch); then generated tokens are fed back. Greedy sampling with B copies of
 * the same prompt makes all B streams identical AND equal to the single-stream
 * greedy reference — a bit-exact end-to-end correctness check — while the aggregate
 * B×tok/s is the batching win.</p>
 *
 * <p>Run (JVM prop {@code -Dllama.prefillBatchSize=B} MUST equal B):</p>
 * <pre>
 *   java -Dllama.prefillBatchSize=32 -Dbatch.decode.B=32 -Dbatch.decode.ctx=512 \
 *        -Dbatch.decode.n=64 ... BatchedDecodeEngine --model llama-3.2-1b-fp16.gguf --prompt "..."
 * </pre>
 */
public class BatchedDecodeEngine {

    static final int RMS_LOCAL = 256;

    public static void main(String[] args) throws Exception {
        System.setProperty("llama.enableTornadoVM", "true");
        int B = Integer.getInteger("batch.decode.B", 32);
        int decodeCtx = Integer.getInteger("batch.decode.ctx", 512);
        int nDecode = Integer.getInteger("batch.decode.n", 64);
        boolean cudaGraphs = Boolean.parseBoolean(System.getProperty("batch.decode.cudaGraphs", "true"));
        if (Integer.getInteger("llama.prefillBatchSize", 1) != B) {
            throw new IllegalStateException("Set -Dllama.prefillBatchSize=" + B + " to match -Dbatch.decode.B");
        }

        Options options = Options.parseOptions(args);
        Model model = loadModel(options);
        Configuration config = model.configuration();
        TornadoWeights weights = (TornadoWeights) model.weights();
        State state = model.createNewState();
        boolean isQwen3 = config instanceof Qwen3Configuration;

        int dim = config.dim();
        int vocab = config.vocabularySize();
        int numLayers = config.numberOfLayers();
        int kvDim = isQwen3
                ? ((Qwen3Configuration) config).numberOfHeadsValue() * config.numberOfKeyValueHeads()
                : (config.dim() * config.numberOfKeyValueHeads()) / config.numberOfHeads();
        int paddedB = (B + 127) & ~127;

        // ── Prompt tokens ──────────────────────────────────────────────────────
        ChatFormat cf = model.chatFormat();
        List<Integer> prompt = new ArrayList<>();
        if (model.shouldAddBeginOfText()) {
            prompt.add(cf.getBeginOfText());
        }
        prompt.addAll(cf.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, options.prompt())));
        prompt.addAll(cf.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
        int P = prompt.size();
        var stopTokens = cf.getStopTokens();
        if (P + nDecode >= decodeCtx) {
            throw new IllegalStateException("prompt(" + P + ")+n(" + nDecode + ") >= ctx(" + decodeCtx + ")");
        }
        System.out.printf("[engine] B=%d ctx=%d promptLen=%d nDecode=%d dim=%d vocab=%d layers=%d cudaGraphs=%b%n",
                B, decodeCtx, P, nDecode, dim, vocab, numLayers, cudaGraphs);

        // ── Engine-owned buffers ────────────────────────────────────────────────
        long kvElems = (long) B * numLayers * decodeCtx * kvDim;
        FloatArray keyCacheBatch = new FloatArray((int) kvElems);
        FloatArray valueCacheBatch = new FloatArray((int) kvElems);
        keyCacheBatch.init(0.0f);
        valueCacheBatch.init(0.0f);
        IntArray seqPositions = new IntArray(B);
        FloatArray finalScaleBatch = new FloatArray(B);
        HalfFloatArray normedFinalFP16 = new HalfFloatArray(paddedB * dim);
        normedFinalFP16.init(new uk.ac.manchester.tornado.api.types.HalfFloat(0.0f));
        FloatArray logitsBatch = new FloatArray(paddedB * vocab);

        // ── Graphs: [0] activation, [1..N] decode layers, [N+1] batched logits ──
        BatchPrefillActivation activation = new BatchPrefillActivation(state, config, B, false);

        List<ImmutableTaskGraph> layerITGs;
        String lastLayerId;
        java.util.function.Consumer<GridScheduler> updateLayerSched;
        if (isQwen3) {
            Qwen3FP16LayersBatchDecodeMMA q = new Qwen3FP16LayersBatchDecodeMMA(
                    (org.beehive.gpullama3.inference.state.Qwen3State) state,
                    (org.beehive.gpullama3.inference.weights.tornado.Qwen3TornadoWeights) weights,
                    (Qwen3Configuration) config, B, decodeCtx, keyCacheBatch, valueCacheBatch, seqPositions);
            layerITGs = q.getLayerImmutableTaskGraphs();
            lastLayerId = q.getLastLayerTaskGraphID();
            updateLayerSched = q::updateGridScheduler;
        } else {
            LlamaFP16LayersBatchDecodeMMA l = new LlamaFP16LayersBatchDecodeMMA(
                    (LlamaState) state,
                    (LlamaTornadoWeights) weights,
                    (LlamaConfiguration) config, B, decodeCtx, keyCacheBatch, valueCacheBatch, seqPositions);
            layerITGs = l.getLayerImmutableTaskGraphs();
            lastLayerId = l.getLastLayerTaskGraphID();
            updateLayerSched = l::updateGridScheduler;
        }

        KernelContext logitsCtx = new KernelContext();
        HalfFloatArray wclsHalf = weights.wclsByteArray.asHalfFloatArray();
        FloatArray rmsFinal = weights.rms_final_weight_as_floatArray.asFloatArray();
        TaskGraph logitsTg = new TaskGraph("batchLogits")
                .consumeFromDevice(lastLayerId, state.wrapXBatch)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION,
                        logitsCtx, normedFinalFP16, finalScaleBatch, wclsHalf, rmsFinal)
                .task("rms_reduce", TransformerBatchPrefillKernels::batchedRmsReduceParallel,
                        logitsCtx, state.wrapXBatch, finalScaleBatch, dim, config.rmsNormEps(), RMS_LOCAL)
                .task("rms_apply", TransformerBatchPrefillKernels::batchedFFNRmsApplyFP16,
                        logitsCtx, normedFinalFP16, state.wrapXBatch, rmsFinal, finalScaleBatch, dim)
                .task("vocab", TransformerBatchPrefillKernels::gemmMMA,
                        logitsCtx, normedFinalFP16, wclsHalf, logitsBatch, paddedB, vocab, dim)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, logitsBatch);
        ImmutableTaskGraph logitsItg = logitsTg.snapshot();

        // ── Grid scheduler ──────────────────────────────────────────────────────
        GridScheduler gs = new GridScheduler();
        activation.updateGridScheduler(gs);
        updateLayerSched.accept(gs);
        gs.addWorkerGrid("batchLogits.rms_reduce", genericWorker(B * RMS_LOCAL, RMS_LOCAL));
        gs.addWorkerGrid("batchLogits.rms_apply", genericWorker(B * dim, 256));
        gs.addWorkerGrid("batchLogits.vocab", mmaGrid(paddedB, vocab));

        List<ImmutableTaskGraph> all = new ArrayList<>();
        all.add(activation.getImmutableTaskGraph());
        all.addAll(layerITGs);
        all.add(logitsItg);
        int nLayers = layerITGs.size();
        int logitsIdx = 1 + nLayers;

        MemorySegment embTable = weights.getTokenEmbeddingTable().asHalfFloatArray().getSegment();
        long dimBytes = (long) dim * Short.BYTES;

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(all.toArray(new ImmutableTaskGraph[0]))) {

            if (Boolean.parseBoolean(System.getProperty("batch.decode.continuous", "false"))) {
                runContinuous(plan, gs, nLayers, logitsIdx, cudaGraphs, model, B, dim, vocab, nDecode,
                        prompt, stopTokens, embTable, dimBytes, state.embeddingXBatch.getSegment(),
                        seqPositions, logitsBatch);
                return;
            }

            // greedy (bit-exact verification) or per-slot temperature sampling
            // (divergent independent streams — the real concurrent-request case).
            float temperature = Float.parseFloat(System.getProperty("batch.decode.temp", "0.0"));
            boolean sample = temperature > 0.0f;
            java.util.Random[] rng = new java.util.Random[B];
            for (int b = 0; b < B; b++) {
                rng[b] = new java.util.Random(1000 + b);
            }

            // ── Prefill: all slots run the identical prompt, logits only at last pos.
            long prefillStart = System.nanoTime();
            for (int t = 0; t < P; t++) {
                int tok = prompt.get(t);
                for (int b = 0; b < B; b++) {
                    MemorySegment.copy(embTable, (long) tok * dimBytes,
                            state.embeddingXBatch.getSegment(), (long) b * dimBytes, dimBytes);
                    seqPositions.set(b, t);
                }
                boolean needLogits = (t == P - 1);
                runStep(plan, gs, nLayers, logitsIdx, needLogits, cudaGraphs);
            }
            long prefillNs = System.nanoTime() - prefillStart;

            // First generated token per slot (identical → same greedy argmax).
            int[][] streams = new int[B][];
            for (int b = 0; b < B; b++) {
                streams[b] = new int[nDecode];
            }
            int[] cur = new int[B];
            for (int b = 0; b < B; b++) {
                int first = sample ? sampleRow(logitsBatch, b, vocab, temperature, rng[b])
                                   : argmaxRow(logitsBatch, b, vocab);
                cur[b] = first;
                streams[b][0] = first;
            }

            // ── Decode: static batch, greedy. Position advances in lock-step.
            // First few steps capture the CUDA graphs; exclude them from timing so
            // the reported latency is steady-state replay, not capture.
            int warmup = cudaGraphs ? Math.min(4, nDecode - 1) : 0;
            long decodeNs = 0;
            int steps = 0;
            for (int s = 1; s < nDecode; s++) {
                int pos = P + s - 1;
                for (int b = 0; b < B; b++) {
                    MemorySegment.copy(embTable, (long) cur[b] * dimBytes,
                            state.embeddingXBatch.getSegment(), (long) b * dimBytes, dimBytes);
                    seqPositions.set(b, pos);
                }
                long t0 = System.nanoTime();
                runStep(plan, gs, nLayers, logitsIdx, true, cudaGraphs);
                if (s > warmup) {
                    decodeNs += System.nanoTime() - t0;
                    steps++;
                }
                for (int b = 0; b < B; b++) {
                    int nt = sample ? sampleRow(logitsBatch, b, vocab, temperature, rng[b])
                                    : argmaxRow(logitsBatch, b, vocab);
                    cur[b] = nt;
                    streams[b][s] = nt;
                }
            }

            // ── Verify + report ─────────────────────────────────────────────────
            boolean allIdentical = true;
            for (int b = 1; b < B; b++) {
                for (int s = 0; s < nDecode; s++) {
                    if (streams[b][s] != streams[0][s]) {
                        allIdentical = false;
                    }
                }
            }
            java.util.Set<String> distinct = new java.util.HashSet<>();
            for (int b = 0; b < B; b++) {
                distinct.add(java.util.Arrays.toString(streams[b]));
            }
            int showSlots = sample ? Math.min(3, B) : 1;
            for (int b = 0; b < showSlots; b++) {
                System.out.printf("%n──────────── slot %d output ────────────%n", b);
                System.out.println(decodeStream(model, streams[b], nDecode, stopTokens));
                System.out.println("───────────────────────────────────────");
            }
            if (sample) {
                System.out.printf("[verify] mode=sample temp=%.2f: %d/%d streams distinct (independent per-slot KV)%n",
                        temperature, distinct.size(), B);
            } else {
                System.out.printf("[verify] mode=greedy: all %d streams identical (== single-stream greedy ref): %b%n",
                        B, allIdentical);
            }

            double decodeSec = decodeNs / 1e9;
            int genTokens = steps * B;
            System.out.printf("[perf] prefill %d pos in %.1f ms | decode %d steps × B=%d = %d tokens in %.1f ms%n",
                    P, prefillNs / 1e6, steps, B, genTokens, decodeNs / 1e6);
            System.out.printf("[perf] per-step latency %.2f ms | aggregate throughput %.0f tok/s | per-stream %.1f tok/s%n",
                    decodeNs / 1e6 / steps, genTokens / decodeSec, steps / decodeSec);
        }
    }

    /**
     * Continuous (iteration-level) batching: B slots, each independently either
     * PREFILLING its prompt (token-by-token, logits ignored) or DECODING. A slot
     * that hits a stop token / its max-gen is evicted and immediately refilled from
     * the pending queue — no waiting for the slowest slot in a wave. Because the
     * per-step forward already feeds one token per slot at its own {@code seqPositions[b]}
     * with its own KV region, prefill and decode are the same op; new requests join
     * a partly-decoded batch mid-flight (Orca-style scheduling).
     *
     * Workload: R requests of the same prompt (greedy → deterministic) with random
     * max-gen lengths, so slots finish at different steps and exercise evict/admit.
     * Correctness = all completed outputs are mutually prefix-consistent (the shorter
     * is a prefix of the longer), proving each slot decoded correctly regardless of
     * when it joined.
     */
    private static void runContinuous(TornadoExecutionPlan plan, GridScheduler gs, int nLayers, int logitsIdx,
                                      boolean cudaGraphs, Model model, int B, int dim, int vocab, int nDecode,
                                      List<Integer> prompt, java.util.Set<Integer> stopTokens,
                                      MemorySegment embTable, long dimBytes, MemorySegment embBatchSeg,
                                      IntArray seqPositions, FloatArray logitsBatch) {
        int R = Integer.getInteger("batch.decode.requests", 4 * B);
        int minN = Integer.getInteger("batch.decode.minN", Math.max(1, nDecode / 2));
        // refill=true → continuous (admit into a freed slot immediately);
        // refill=false → static-wave baseline (idle until the whole wave finishes, then reload).
        boolean refill = Boolean.parseBoolean(System.getProperty("batch.decode.refill", "true"));
        int promptLen = prompt.size();
        java.util.Random wl = new java.util.Random(7);
        java.util.ArrayDeque<Integer> queue = new java.util.ArrayDeque<>();
        for (int i = 0; i < R; i++) {
            queue.add(minN + wl.nextInt(Math.max(1, nDecode - minN + 1)));
        }
        System.out.printf("[continuous] requests=%d maxGen∈[%d,%d] B=%d%n", R, minN, nDecode, B);

        int[] promptPos = new int[B], pos = new int[B], curTok = new int[B], maxGen = new int[B];
        boolean[] decoding = new boolean[B], active = new boolean[B];
        List<List<Integer>> gen = new ArrayList<>();
        for (int b = 0; b < B; b++) {
            gen.add(new ArrayList<>());
        }
        List<int[]> completed = new ArrayList<>();
        for (int b = 0; b < B; b++) {
            if (!queue.isEmpty()) {
                maxGen[b] = queue.poll();
                active[b] = true;
            }
        }

        int warm = cudaGraphs ? 4 : 0;
        long steps = 0, genTokens = 0, busySlotSteps = 0, timedNs = 0, timedGen = 0, timedBusy = 0, timedSteps = 0;
        long tTimed = 0;
        while (true) {
            boolean anyActive = false;
            for (int b = 0; b < B; b++) {
                if (active[b]) { anyActive = true; break; }
            }
            if (!anyActive) {
                if (queue.isEmpty()) {
                    break;
                }
                // static-wave baseline: whole wave drained → load the next wave.
                for (int b = 0; b < B; b++) {
                    if (!queue.isEmpty()) {
                        maxGen[b] = queue.poll();
                        promptPos[b] = 0;
                        pos[b] = 0;
                        decoding[b] = false;
                        active[b] = true;
                    }
                }
            }
            for (int b = 0; b < B; b++) {
                int tok = active[b] ? (decoding[b] ? curTok[b] : prompt.get(promptPos[b])) : prompt.get(0);
                MemorySegment.copy(embTable, (long) tok * dimBytes, embBatchSeg, (long) b * dimBytes, dimBytes);
                seqPositions.set(b, active[b] ? pos[b] : 0);
            }
            long t0 = System.nanoTime();
            runStep(plan, gs, nLayers, logitsIdx, true, cudaGraphs);
            long dt = System.nanoTime() - t0;
            steps++;

            long stepGen = 0, stepBusy = 0;
            for (int b = 0; b < B; b++) {
                if (!active[b]) {
                    continue;
                }
                stepBusy++;
                pos[b]++;
                boolean produced = false;
                int nt = 0;
                if (!decoding[b]) {
                    if (promptPos[b] == promptLen - 1) {          // just fed the last prompt token
                        nt = argmaxRow(logitsBatch, b, vocab);
                        decoding[b] = true;
                        produced = true;
                    } else {
                        promptPos[b]++;
                    }
                } else {
                    nt = argmaxRow(logitsBatch, b, vocab);
                    produced = true;
                }
                if (produced) {
                    gen.get(b).add(nt);
                    curTok[b] = nt;
                    genTokens++;
                    stepGen++;
                    boolean stop = stopTokens.contains(nt) || gen.get(b).size() >= maxGen[b];
                    if (stop) {
                        completed.add(gen.get(b).stream().mapToInt(Integer::intValue).toArray());
                        gen.get(b).clear();
                        if (refill && !queue.isEmpty()) {          // continuous: admit next request now
                            maxGen[b] = queue.poll();
                            promptPos[b] = 0;
                            pos[b] = 0;
                            decoding[b] = false;
                        } else {                                    // static-wave: idle until wave ends
                            active[b] = false;
                        }
                    }
                }
            }
            busySlotSteps += stepBusy;
            if (steps > warm) {
                if (timedSteps == 0) {
                    tTimed = System.nanoTime();
                }
                timedNs += dt;
                timedGen += stepGen;
                timedBusy += stepBusy;
                timedSteps++;
            }
        }
        long wallNs = System.nanoTime() - tTimed;

        // ── Verify: mutually prefix-consistent (greedy, same prompt) ─────────────
        completed.sort(java.util.Comparator.comparingInt(a -> a.length));
        boolean consistent = true;
        int[] longest = completed.isEmpty() ? new int[0] : completed.get(completed.size() - 1);
        for (int[] r : completed) {
            for (int i = 0; i < r.length; i++) {
                if (r[i] != longest[i]) {
                    consistent = false;
                }
            }
        }
        StringBuilder text = new StringBuilder();
        for (int tk : longest) {
            if (stopTokens.contains(tk)) {
                break;
            }
            text.append(model.tokenizer().decode(List.of(tk)));
        }
        System.out.println("\n──────────── longest completed request ────────────");
        System.out.println(text);
        System.out.println("───────────────────────────────────────────────────");
        System.out.printf("[verify] %d requests completed, all mutually prefix-consistent (greedy): %b%n",
                completed.size(), consistent);

        double sec = wallNs / 1e9;
        double util = timedSteps > 0 ? (double) timedBusy / (timedSteps * B) : 0.0;
        System.out.printf("[perf] continuous: %d steps, %d gen tokens over %.2f s%n", steps, genTokens, sec);
        System.out.printf("[perf] generated %.0f tok/s | %.1f requests/s | slot utilization %.1f%% | per-step %.2f ms%n",
                timedGen / sec, completed.size() / (wallNs / 1e9), util * 100.0,
                timedSteps > 0 ? timedNs / 1e6 / timedSteps : 0.0);
    }

    private static void runStep(TornadoExecutionPlan plan, GridScheduler gs,
                                int nLayers, int logitsIdx, boolean needLogits, boolean cudaGraphs) {
        execGraph(plan, gs, 0, cudaGraphs);
        for (int l = 0; l < nLayers; l++) {
            execGraph(plan, gs, 1 + l, cudaGraphs);
        }
        if (needLogits) {
            execGraph(plan, gs, logitsIdx, cudaGraphs);
        }
    }

    private static void execGraph(TornadoExecutionPlan plan, GridScheduler gs, int idx, boolean cudaGraphs) {
        var g = plan.withGraph(idx).withGridScheduler(gs);
        if (cudaGraphs) {
            g.withCUDAGraph();
        }
        g.execute();
    }

    private static String decodeStream(Model model, int[] stream, int nDecode, java.util.Set<Integer> stopTokens) {
        StringBuilder text = new StringBuilder();
        for (int s = 0; s < nDecode; s++) {
            int tk = stream[s];
            if (stopTokens.contains(tk)) {
                break;
            }
            text.append(model.tokenizer().decode(List.of(tk)));
        }
        return text.toString();
    }

    /** Temperature softmax sampling over one logits row. */
    private static int sampleRow(FloatArray logits, int row, int vocab, float temp, java.util.Random rng) {
        long base = (long) row * vocab;
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < vocab; i++) {
            float v = logits.get((int) (base + i));
            if (v > max) {
                max = v;
            }
        }
        double sum = 0.0;
        double[] probs = new double[vocab];
        for (int i = 0; i < vocab; i++) {
            double e = Math.exp((logits.get((int) (base + i)) - max) / temp);
            probs[i] = e;
            sum += e;
        }
        double r = rng.nextDouble() * sum;
        double acc = 0.0;
        for (int i = 0; i < vocab; i++) {
            acc += probs[i];
            if (acc >= r) {
                return i;
            }
        }
        return vocab - 1;
    }

    private static int argmaxRow(FloatArray logits, int row, int vocab) {
        long base = (long) row * vocab;
        int best = 0;
        float bestV = logits.get((int) base);
        for (int i = 1; i < vocab; i++) {
            float v = logits.get((int) (base + i));
            if (v > bestV) {
                bestV = v;
                best = i;
            }
        }
        return best;
    }

    static WorkerGrid genericWorker(int global, int local) {
        WorkerGrid1D g = new WorkerGrid1D(global);
        g.setLocalWork(local, 1, 1);
        return g;
    }

    static WorkerGrid mmaGrid(int paddedM, int N) {
        int mBlocks = paddedM / 128;
        int nBlocks = N / 128;
        WorkerGrid2D g = new WorkerGrid2D(mBlocks * 256, nBlocks);
        g.setLocalWork(256, 1, 1);
        return g;
    }
}
