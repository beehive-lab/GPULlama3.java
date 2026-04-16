package org.beehive.gpullama3.diagnostic;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import uk.ac.manchester.tornado.api.types.HalfFloat;

/**
 * Diagnostic utility for verifying input embeddings are correctly copied to state.wrapX.
 *
 * <p>Strategy: Keep reference to the entire embedding table, then: 1. When MemorySegment.copy
 * happens in forwardTornadoVM, verify the copied data matches what's at the correct offset in the
 * source embedding table 2. This verifies the copy is working correctly for ANY token
 */
public class EmbeddingDiagnostics {
    private static final int SAMPLE_SIZE = 15;
    private static MemorySegment embeddingTableSegment;
    private static int embeddingTableDim = -1;
    private static boolean hasRecorded = false;
    private static int tokensVerified = 0;
    private static int tokensFailed = 0;
    private static boolean printedHeader = false;
    private static boolean printedSummary = false;

    static {
        // Register shutdown hook to print final summary when program exits
        Runtime.getRuntime()
                .addShutdownHook(
                        new Thread(
                                () -> {
                                    EmbeddingDiagnostics.printFinalSummary();
                                }));
    }

    /**
     * Record the embedding table segment during model loading phase. This allows us to verify the
     * copy operation by reading from the source during forward pass.
     *
     * @param segment MemorySegment containing the loaded embeddings (token_embd.weight)
     * @param tableSize Total number of elements in the embedding table
     * @param dim Model dimension (elements per token)
     */
    public static void recordLoadingStage(MemorySegment segment, int tableSize, int dim) {
        embeddingTableSegment = segment;
        embeddingTableDim = dim;

        System.out.println("\n" + "=".repeat(80));
        System.out.println(
                "[EMBEDDING DIAGNOSTICS] LOADING STAGE - ModelLoader.loadTornadoTensor()");
        System.out.println("=".repeat(80));
        System.out.println("Recording embedding table reference");
        System.out.println("Table Size: " + tableSize + " | Dim per token: " + dim);
        System.out.println("Sample: First 15 values from Token 0 (offset 0):");

        for (int i = 0; i < Math.min(SAMPLE_SIZE, dim); i++) {
            short hfValue = segment.getAtIndex(ValueLayout.JAVA_SHORT, i);
            HalfFloat hf = new HalfFloat(hfValue);
            float value = hf.getFloat32();
            System.out.printf(
                    "  [%2d] Short: 0x%04X -> HalfFloat: %f%n", i, hfValue & 0xFFFF, value);
        }

        hasRecorded = true;
        System.out.println("=".repeat(80) + "\n");
    }

    /**
     * Verify the embedding copy by comparing: - What's at offset (token * dim * 2) in the source
     * embedding table - What got copied to state.wrapX
     *
     * <p>Runs on EVERY token to catch any inconsistencies.
     *
     * @param wrapXSegment MemorySegment of state.wrapX after copy
     * @param token Token being processed
     * @param dim Model dimension
     */
    public static void compareForwardStage(MemorySegment wrapXSegment, int token, int dim) {
        if (!hasRecorded || embeddingTableSegment == null) {
            if (tokensVerified == 0) {
                System.out.println("ERROR: No embedding table was recorded during loading!");
            }
            return;
        }

        // Print header on first token
        if (!printedHeader) {
            System.out.println("\n" + "=".repeat(80));
            System.out.println("[EMBEDDING DIAGNOSTICS] RUNNING ON ALL TOKENS");
            System.out.println("=".repeat(80));
            System.out.println("Format: Token | Result | Mismatches | Max Diff");
            System.out.println("-".repeat(80));
            printedHeader = true;
        }

        // Calculate source offset for this token
        long sourceOffset = (long) token * dim * 2; // 2 bytes per HalfFloat

        boolean matches = true;
        float maxDifference = 0f;
        int differencesCount = 0;

        for (int i = 0; i < Math.min(SAMPLE_SIZE, dim); i++) {
            // Read from source embedding table at token's offset
            // sourceOffset is in bytes, divide by 2 to get logical index for JAVA_SHORT
            short sourceHfValue =
                    embeddingTableSegment.getAtIndex(
                            ValueLayout.JAVA_SHORT, (int) (sourceOffset / 2 + i));
            HalfFloat sourceHf = new HalfFloat(sourceHfValue);
            float sourceValue = sourceHf.getFloat32();

            // Read from destination wrapX at offset i
            short destHfValue = wrapXSegment.getAtIndex(ValueLayout.JAVA_SHORT, i);
            HalfFloat destHf = new HalfFloat(destHfValue);
            float destValue = destHf.getFloat32();

            float diff = Math.abs(destValue - sourceValue);
            boolean match =
                    (Float.isNaN(destValue) && Float.isNaN(sourceValue))
                            || (destValue == sourceValue);

            if (!match) {
                matches = false;
                differencesCount++;
                maxDifference = Math.max(maxDifference, diff);
            }
        }

        tokensVerified++;
        if (!matches) {
            tokensFailed++;
        }

        // Print compact summary for this token
        String result = matches ? "✓ PASS" : "✗ FAIL";
        System.out.printf(
                "%6d | %7s | %10d/%d | %10f%n",
                token, result, differencesCount, SAMPLE_SIZE, maxDifference);

        // If failed, print detailed info
        if (!matches) {
            System.out.println("         ^ MISMATCH DETECTED! Detailed comparison:");
            for (int i = 0; i < Math.min(SAMPLE_SIZE, dim); i++) {
                short sourceHfValue =
                        embeddingTableSegment.getAtIndex(
                                ValueLayout.JAVA_SHORT, (int) (sourceOffset / 2 + i));
                HalfFloat sourceHf = new HalfFloat(sourceHfValue);
                float sourceValue = sourceHf.getFloat32();

                short destHfValue = wrapXSegment.getAtIndex(ValueLayout.JAVA_SHORT, i);
                HalfFloat destHf = new HalfFloat(destHfValue);
                float destValue = destHf.getFloat32();

                float diff = Math.abs(destValue - sourceValue);
                boolean match = (destValue == sourceValue);

                System.out.printf(
                        "           [%2d] Source: %15f | Dest: %15f | %s | Diff: %f%n",
                        i, sourceValue, destValue, match ? "✓" : "✗", diff);
            }
            System.out.println();
        }
    }

    /** Print final summary after all tokens processed */
    public static void printFinalSummary() {
        if (printedSummary || !printedHeader) {
            return; // Already printed or nothing to print
        }
        printedSummary = true;

        System.out.println("-".repeat(80));
        System.out.println("Final Summary:");
        System.out.println("  Total tokens verified: " + tokensVerified);
        System.out.println("  Failed: " + tokensFailed);
        if (tokensVerified > 0) {
            System.out.println(
                    "  Pass rate: "
                            + String.format(
                                    "%.1f",
                                    (100.0 * (tokensVerified - tokensFailed) / tokensVerified))
                            + "%");
        }
        if (tokensFailed == 0) {
            System.out.println("  ✓ All tokens copied correctly!");
        } else {
            System.out.println("  ✗ " + tokensFailed + " token(s) had mismatches");
        }
        System.out.println("=".repeat(80) + "\n");
    }

    /** Clear recorded data (useful for testing multiple models) */
    public static void reset() {
        printFinalSummary();
        embeddingTableSegment = null;
        embeddingTableDim = -1;
        hasRecorded = false;
        tokensVerified = 0;
        tokensFailed = 0;
        printedHeader = false;
        printedSummary = false;
    }

    public static boolean hasRecordedEmbedding() {
        return hasRecorded;
    }

    public static int getRecordedDim() {
        return embeddingTableDim;
    }

    public static int getTokensVerified() {
        return tokensVerified;
    }

    public static int getTokensFailed() {
        return tokensFailed;
    }
}
