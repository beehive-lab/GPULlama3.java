package org.beehive.gpullama3.bench;

import org.beehive.gpullama3.Options;
import org.beehive.gpullama3.inference.InferenceCore;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;

import static org.beehive.gpullama3.model.loader.ModelLoader.loadModel;

/**
 * llama-bench-style performance benchmark for GPULlama3 (GPU forward path).
 *
 * <p>Mirrors llama.cpp's {@code llama-bench}: a cartesian matrix of tests over one or more
 * models — prompt processing ({@code pp N}: N sequential forwards from position 0), token
 * generation ({@code tg N}: N single-token forwards over a growing KV cache) and combined
 * ({@code pg pp+tg}) — each repeated {@code -r} times after an untimed warmup, reported as
 * average tokens/s ± stddev in markdown (default), CSV or JSON. Timings cover the forward
 * pass only: no tokenization, no sampling, no host argmax (llama-bench parity).</p>
 *
 * <pre>
 * gpullama3-bench (via llama-tornado --bench):
 *   -m  model.gguf[,model2.gguf]   models (repeatable / comma-separated)
 *   -p  512[,1024]                 prompt-processing sizes       (default 512)
 *   -n  128[,256]                  generation lengths            (default 128)
 *   -pg 512,128                    combined prompt+gen test      (repeatable)
 *   -d  0[,4096]                   context depths: untimed KV prefill of d positions
 *                                  before each timed test (llama-bench -d)
 *   -r  5                          repetitions                   (default 5)
 *   -o  md|csv|json|jsonl|sql      output format                 (default md)
 *   -oe fmt                        also print results to stderr in this format
 *   --delay N                      sleep N s between tests (GPU thermals)
 *   --no-warmup                    skip the untimed warmup rep
 * </pre>
 */
public class LlamaBench {

    record TestSpec(int nPrompt, int nGen, int depth) {
        String name() {
            String base = (nPrompt > 0 && nGen > 0) ? "pp" + nPrompt + "+tg" + nGen
                    : nPrompt > 0 ? "pp" + nPrompt : "tg" + nGen;
            return depth > 0 ? base + "@d" + depth : base;
        }

        int tokens() {
            return nPrompt + nGen;
        }
    }

    record Result(String model, String quant, double sizeGiB, double paramsB, String backend, String test, double avg, double stddev, double[] samples) {}

    public static void main(String[] args) throws Exception {
        System.setProperty("llama.enableTornadoVM", "true");

        List<String> models = new ArrayList<>();
        List<Integer> pps = new ArrayList<>();
        List<Integer> tgs = new ArrayList<>();
        List<int[]> pgs = new ArrayList<>();
        List<Integer> depths = new ArrayList<>();
        int reps = 5;
        int delay = 0;
        String out = "md";
        String outErr = null;
        boolean warmup = true;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-m", "--model" -> { for (String m : args[++i].split(",")) models.add(m); }
                case "-p", "--n-prompt" -> { for (String v : args[++i].split(",")) pps.add(Integer.parseInt(v)); }
                case "-n", "--n-gen" -> { for (String v : args[++i].split(",")) tgs.add(Integer.parseInt(v)); }
                case "-pg" -> {
                    String[] parts = args[++i].split("[,+]");
                    pgs.add(new int[] { Integer.parseInt(parts[0]), Integer.parseInt(parts[1]) });
                }
                case "-d", "--n-depth" -> { for (String v : args[++i].split(",")) depths.add(Integer.parseInt(v)); }
                case "-r", "--repetitions" -> reps = Integer.parseInt(args[++i]);
                case "-o", "--output" -> out = args[++i];
                case "-oe", "--output-err" -> outErr = args[++i];
                case "--delay" -> delay = Integer.parseInt(args[++i]);
                case "--no-warmup" -> warmup = false;
                default -> {
                    // Ignore the launcher's trailing single-run options (prompt etc.).
                }
            }
        }
        if (models.isEmpty()) {
            System.err.println("usage: LlamaBench -m model.gguf [-m model2.gguf] [-p 512] [-n 128] [-pg 512,128] [-d 0] [-r 5] [-o md|csv|json|jsonl|sql] [-oe fmt] [--delay s] [--no-warmup]");
            System.exit(1);
        }
        if (pps.isEmpty() && tgs.isEmpty() && pgs.isEmpty()) {
            pps.add(512);
            tgs.add(128);
        }

        if (depths.isEmpty()) {
            depths.add(0);
        }
        List<TestSpec> tests = new ArrayList<>();
        for (int d : depths) {
            for (int p : pps) {
                if (p > 0) {
                    tests.add(new TestSpec(p, 0, d));
                }
            }
            for (int n : tgs) {
                if (n > 0) {
                    tests.add(new TestSpec(0, n, d));
                }
            }
            for (int[] pg : pgs) {
                tests.add(new TestSpec(pg[0], pg[1], d));
            }
        }

        List<Result> results = new ArrayList<>();
        for (String modelPath : models) {
            results.addAll(benchModel(modelPath, tests, reps, warmup, delay));
        }

        print(out, results, System.out);
        if (outErr != null) {
            print(outErr, results, System.err);
        }
        // TornadoVM daemon threads keep the JVM alive after plan teardown.
        System.exit(0);
    }

    static void print(String format, List<Result> results, java.io.PrintStream ps) {
        switch (format) {
            case "csv" -> printCsv(results, ps);
            case "json" -> printJson(results, ps);
            case "jsonl" -> printJsonl(results, ps);
            case "sql" -> printSql(results, ps);
            default -> printMarkdown(results, ps);
        }
    }

    static List<Result> benchModel(String modelPath, List<TestSpec> tests, int reps, boolean warmup, int delay) throws Exception {
        Path path = Paths.get(modelPath);
        int maxCtx = tests.stream().mapToInt(t -> t.depth() + t.tokens()).max().orElse(1024) + 8;
        Options options = new Options(path, "bench", null, null, false, 0.0f, 1.0f, 42, maxCtx, false, false, true, false, 1);
        Model model = loadModel(options);
        State state = model.createNewState();
        TornadoVMMasterPlan plan = TornadoVMMasterPlan.initializeTornadoVMPlan(state, model);

        int vocab = model.configuration().vocabularySize();
        String name = path.getFileName().toString().replaceAll("\\.gguf$", "");
        String quant = model.configuration().quantization();
        double sizeGiB = Files.size(path) / (1024.0 * 1024.0 * 1024.0);
        double paramsB = estimateParamsB(Files.size(path), quant);
        String backend = System.getProperty("tornado.backend.name", "TornadoVM " + backendName());

        // Deterministic synthetic token stream (llama-bench uses random ids too).
        Random rng = new Random(42);
        int maxTokens = tests.stream().mapToInt(t -> t.depth() + t.tokens()).max().orElse(0);
        int[] toks = new int[maxTokens];
        for (int i = 0; i < maxTokens; i++) {
            toks[i] = rng.nextInt(vocab);
        }

        List<Result> results = new ArrayList<>();
        for (TestSpec t : tests) {
            if (delay > 0) {
                Thread.sleep(delay * 1000L);
            }
            if (warmup) {
                runTest(model, state, plan, toks, t); // untimed
            }
            double[] samples = new double[reps];
            for (int r = 0; r < reps; r++) {
                samples[r] = runTest(model, state, plan, toks, t);
            }
            double avg = 0;
            for (double s : samples) {
                avg += s;
            }
            avg /= samples.length;
            double var = 0;
            for (double s : samples) {
                var += (s - avg) * (s - avg);
            }
            double stddev = samples.length > 1 ? Math.sqrt(var / (samples.length - 1)) : 0.0;
            results.add(new Result(name, quant, sizeGiB, paramsB, backend, t.name(), avg, stddev, samples));
            System.err.printf(Locale.ROOT, "[bench] %-28s %-12s %8.2f ± %.2f t/s%n", name, t.name(), avg, stddev);
        }
        plan.freeTornadoExecutionPlan();
        return results;
    }

    /**
     * One timed repetition: fresh sequence from position 0 (KV overwritten). With a depth d,
     * d positions are prefilled untimed first and the timed window runs at positions
     * d..d+tokens (llama-bench {@code -d}: measures throughput at context depth). Returns tokens/s.
     */
    static double runTest(Model model, State state, TornadoVMMasterPlan plan, int[] toks, TestSpec t) {
        for (int pos = 0; pos < t.depth(); pos++) {
            InferenceCore.forwardTornadoVM(model, state, toks[pos], pos, plan);
        }
        int total = t.tokens();
        long t0 = System.nanoTime();
        for (int pos = t.depth(); pos < t.depth() + total; pos++) {
            InferenceCore.forwardTornadoVM(model, state, toks[pos], pos, plan);
        }
        long t1 = System.nanoTime();
        return total / ((t1 - t0) / 1e9);
    }

    /** Rough parameter count from file size + quantization (F16 = 2 B/param, Q8_0 = 34/32 B/param). */
    static double estimateParamsB(long bytes, String quant) {
        double bytesPerParam = switch (quant) {
            case "FP16" -> 2.0;
            case "Q8_0" -> 34.0 / 32.0;
            default -> 2.0;
        };
        return bytes / bytesPerParam / 1e9;
    }

    static String backendName() {
        try {
            return uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider.getTornadoRuntime().getBackend(0).getBackendType().name();
        } catch (Throwable e) {
            return "unknown";
        }
    }

    static void printMarkdown(List<Result> rs, java.io.PrintStream ps) {
        ps.println();
        ps.println("| model | quant | size | params | backend | test | t/s |");
        ps.println("| ----- | ----- | ---: | -----: | ------- | ---- | --: |");
        for (Result r : rs) {
            ps.printf(Locale.ROOT, "| %s | %s | %.2f GiB | %.2f B | %s | %s | %.2f ± %.2f |%n",
                    r.model(), r.quant(), r.sizeGiB(), r.paramsB(), r.backend(), r.test(), r.avg(), r.stddev());
        }
    }

    static void printCsv(List<Result> rs, java.io.PrintStream ps) {
        ps.println("model,quant,size_gib,params_b,backend,test,avg_ts,stddev_ts,samples");
        for (Result r : rs) {
            StringBuilder samples = new StringBuilder();
            for (double s : r.samples()) {
                if (samples.length() > 0) {
                    samples.append(';');
                }
                samples.append(String.format(Locale.ROOT, "%.2f", s));
            }
            ps.printf(Locale.ROOT, "%s,%s,%.3f,%.3f,%s,%s,%.2f,%.2f,%s%n",
                    r.model(), r.quant(), r.sizeGiB(), r.paramsB(), r.backend(), r.test(), r.avg(), r.stddev(), samples);
        }
    }

    static String jsonRow(Result r) {
        StringBuilder sb = new StringBuilder(String.format(Locale.ROOT,
                "{\"model\": \"%s\", \"quant\": \"%s\", \"size_gib\": %.3f, \"params_b\": %.3f, \"backend\": \"%s\", \"test\": \"%s\", \"avg_ts\": %.2f, \"stddev_ts\": %.2f, \"samples_ts\": [",
                r.model(), r.quant(), r.sizeGiB(), r.paramsB(), r.backend(), r.test(), r.avg(), r.stddev()));
        for (int j = 0; j < r.samples().length; j++) {
            if (j > 0) {
                sb.append(", ");
            }
            sb.append(String.format(Locale.ROOT, "%.2f", r.samples()[j]));
        }
        sb.append("]}");
        return sb.toString();
    }

    static void printJson(List<Result> rs, java.io.PrintStream ps) {
        StringBuilder sb = new StringBuilder("[\n");
        for (int i = 0; i < rs.size(); i++) {
            sb.append("  ").append(jsonRow(rs.get(i))).append(i < rs.size() - 1 ? "," : "").append('\n');
        }
        sb.append("]");
        ps.println(sb);
    }

    static void printJsonl(List<Result> rs, java.io.PrintStream ps) {
        for (Result r : rs) {
            ps.println(jsonRow(r));
        }
    }

    static void printSql(List<Result> rs, java.io.PrintStream ps) {
        ps.println("CREATE TABLE IF NOT EXISTS llama_bench (model TEXT, quant TEXT, size_gib REAL, params_b REAL, backend TEXT, test TEXT, avg_ts REAL, stddev_ts REAL);");
        for (Result r : rs) {
            ps.printf(Locale.ROOT, "INSERT INTO llama_bench VALUES ('%s', '%s', %.3f, %.3f, '%s', '%s', %.2f, %.2f);%n",
                    r.model(), r.quant(), r.sizeGiB(), r.paramsB(), r.backend(), r.test(), r.avg(), r.stddev());
        }
    }
}
