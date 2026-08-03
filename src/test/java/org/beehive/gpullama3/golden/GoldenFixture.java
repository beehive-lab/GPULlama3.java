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
        LLAMA_3_2_1B_F16("Llama-3.2-1B-Instruct-F16.gguf", "F16",
                "d4efb14e1eee8d5d9de41211cabd6e81030f79e8070176a3843f6e4e9ecc84da"),
        LLAMA_3_2_1B_Q8_0("Llama-3.2-1B-Instruct-Q8_0.gguf", "Q8_0",
                "3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1");

        public final String fileName;
        public final String quantization;
        public final String sha256;

        Fixture(String fileName, String quantization, String sha256) {
            this.fileName = fileName;
            this.quantization = quantization;
            this.sha256 = sha256;
        }

        /** Directory name used for this fixture's committed goldens. */
        public String goldenDirName() {
            return "llama-3.2-1b-" + quantization.toLowerCase();
        }
    }

    private GoldenFixture() {
    }

    /** Root of the local fixture cache. */
    public static Path modelsRoot() {
        String env = System.getenv("GPULLAMA_TEST_MODELS");
        if (env != null && !env.isBlank()) {
            return Paths.get(env);
        }
        return Paths.get(System.getProperty("user.home"), ".gpullama3", "test-models");
    }

    /** @return the fixture path, or {@code null} when it is not present locally. */
    public static Path locate(Fixture fixture) {
        Path p = modelsRoot().resolve(fixture.fileName);
        return Files.isRegularFile(p) ? p : null;
    }

    public static String absentMessage(Fixture fixture) {
        return "Model fixture absent: " + fixture.fileName
                + "\n  expected under: " + modelsRoot()
                + "\n  sha256: " + fixture.sha256
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
