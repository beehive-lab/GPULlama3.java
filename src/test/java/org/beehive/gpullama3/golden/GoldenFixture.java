package org.beehive.gpullama3.golden;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.util.HexFormat;

/**
 * Locates and verifies the pinned model fixtures for the golden and parity gates.
 *
 * <p>The GGUF files are far too large to commit, so only their SHA-256 is pinned here. The file
 * itself is resolved from {@code $GPULLAMA_TEST_MODELS} or {@code ~/.gpullama3/test-models/}, and
 * an absent fixture produces a fetch instruction rather than a mysterious failure.
 *
 * <p>Per {@code verification-gates.md}, a missing fixture or absent accelerator causes the Class B
 * tests to <b>skip with an explicit marker</b> — never to pass.
 */
public final class GoldenFixture {

    public enum Fixture {
        LLAMA_3_2_1B_F16(
                "Llama-3.2-1B-Instruct-F16.gguf",
                "F16",
                "d4efb14e1eee8d5d9de41211cabd6e81030f79e8070176a3843f6e4e9ecc84da"),
        LLAMA_3_2_1B_Q8_0(
                "Llama-3.2-1B-Instruct-Q8_0.gguf",
                "Q8_0",
                "3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1"),

        /**
         * <b>No recorded golden logits.</b> The checks that use it compare the lowered path against
         * the legacy one <i>in one process on one file</i>, which is a stronger statement than a
         * recorded row and needs no committed data. Recording goldens is a reviewed action; adding
         * a fixture is not the same thing, and conflating them would have made this slice wait on a
         * decision it does not need.
         */
        QWEN2_5_0_5B_F16(
                "Qwen2.5-0.5B-Instruct-f16.gguf",
                "F16",
                "f1ad9d1174ce6ab47b584d522634c47e411b75bffdffd9a4e106e21e882392e5",
                "qwen2.5-0.5b"),
        QWEN2_5_0_5B_Q8_0(
                "Qwen2.5-0.5B-Instruct-Q8_0.gguf",
                "Q8_0",
                "25130a98aa782284a7dabea0c23245b2fd371ed47244e79d78b8ec23245fdf96",
                "qwen2.5-0.5b"),

        MISTRAL_7B_Q8_0(
                "Mistral-7B-Instruct-v0.3.Q8_0.gguf",
                "Q8_0",
                "24df553dc0e725196fe8a3c7be1edfe6ff17a0fe855f508b3f4a0e444e2e4281",
                "mistral-7b"),

        GRANITE_3_2_2B_F16(
                "granite-3.2-2b-instruct-f16.gguf",
                "F16",
                "000535b376c11e1eeb27231d85f19523db42f66d175b0e7cccb704610ae129ce",
                "granite-3.2-2b"),
        GRANITE_3_2_2B_Q8_0(
                "granite-3.2-2b-instruct-Q8_0.gguf",
                "Q8_0",
                "7ffbd0fe17ac37775c3758464aa9a09773a3a162b9459eb9094278a7a809682a",
                "granite-3.2-2b"),

        QWEN3_0_6B_F16(
                "Qwen3-0.6B-f16.gguf",
                "F16",
                "ab9004daf660cd6a6ba1c07556e74fcceb2b756063ccce3f9c69d3a637b361cc",
                "qwen3-0.6b"),
        QWEN3_0_6B_Q8_0(
                "Qwen3-0.6B-Q8_0.gguf",
                "Q8_0",
                "84c0dbe606526d5907251d88ea88b41457f46ce456e9a333d5d2b6245a95cafe",
                "qwen3-0.6b"),

        PHI3_MINI_4K_F16(
                "Phi-3-mini-4k-instruct-fp16.gguf",
                "F16",
                "5d99003e395775659b0dde3f941d88ff378b2837a8dc3a2ea94222ab1420fad3",
                "phi3-mini-4k"),
        PHI3_MINI_4K_Q8_0(
                "Phi-3-mini-4k-instruct-Q8_0.gguf",
                "Q8_0",
                "0ac8ee48aeebf7d1b354691fd1e29e91c32ad88bbad10ad45ac880dcd4372a47",
                "phi3-mini-4k");

        public final String fileName;
        public final String quantization;
        public final String sha256;
        private final String modelDirName;

        Fixture(String fileName, String quantization, String sha256) {
            this(fileName, quantization, sha256, "llama-3.2-1b");
        }

        Fixture(String fileName, String quantization, String sha256, String modelDirName) {
            this.fileName = fileName;
            this.quantization = quantization;
            this.sha256 = sha256;
            this.modelDirName = modelDirName;
        }

        /**
         * Directory name used for this fixture's committed goldens.
         *
         * <p>Derived from the model rather than hardcoded to Llama's, which is what it was until a
         * second model needed a fixture. A fixture with no recorded goldens still answers this —
         * the name is well-defined whether or not the directory exists.
         */
        public String goldenDirName() {
            return modelDirName + "-" + quantization.toLowerCase();
        }
    }

    private GoldenFixture() {}

    /** Root of the local fixture cache. */
    public static Path modelsRoot() {
        String env = System.getenv("GPULLAMA_TEST_MODELS");
        if (env != null && !env.isBlank()) {
            return Paths.get(env);
        }
        return Paths.get(System.getProperty("user.home"), ".gpullama3", "test-models");
    }

    /**
     * @return the fixture path, or {@code null} when it is not present locally.
     */
    public static Path locate(Fixture fixture) {
        Path p = modelsRoot().resolve(fixture.fileName);
        return Files.isRegularFile(p) ? p : null;
    }

    public static String absentMessage(Fixture fixture) {
        return "Model fixture absent: "
                + fixture.fileName
                + "\n  expected under: "
                + modelsRoot()
                + "\n  sha256: "
                + fixture.sha256
                + "\n  Set GPULLAMA_TEST_MODELS to a directory containing it, or place/symlink the"
                + " file there. It is intentionally not committed.";
    }

    /** Full SHA-256 of the fixture; used to prove the golden was produced from this exact file. */
    public static String sha256(Path file) throws IOException {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] buf = new byte[1 << 20];
            try (InputStream in = Files.newInputStream(file)) {
                int n;
                while ((n = in.read(buf)) > 0) {
                    md.update(buf, 0, n);
                }
            }
            return HexFormat.of().formatHex(md.digest());
        } catch (java.security.NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 unavailable", e);
        }
    }
}
