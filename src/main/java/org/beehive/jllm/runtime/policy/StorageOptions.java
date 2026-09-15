package org.beehive.gpullama3.runtime.policy;

import java.util.Objects;
import org.beehive.gpullama3.api.Experimental;
import org.beehive.gpullama3.runtime.tensor.DataType;

/**
 * How a model's key/value storage is shaped: the choices that are <b>not</b> execution policy.
 *
 * @param keyValueRepresentation how key/value entries are stored: {@link DataType#F32} or {@link
 *     DataType#F16}
 * @param sharedKeyValuePool whether standalone sessions address one shared pool rather than each
 *     allocating their own cache. Applies to standalone sessions only: engine-batched execution
 *     <i>is</i> the shared pool, so there is no engine setting to make
 */
@Experimental
public record StorageOptions(DataType keyValueRepresentation, boolean sharedKeyValuePool) {

    public StorageOptions {
        Objects.requireNonNull(keyValueRepresentation, "keyValueRepresentation");
        if (keyValueRepresentation != DataType.F32 && keyValueRepresentation != DataType.F16) {
            throw new IllegalArgumentException(
                    "key/value storage is F32 or F16, not " + keyValueRepresentation);
        }
    }

    /**
     * The defaults this build runs with, from the {@code llama.*} system properties.
     *
     * <p>Read per call rather than folded into a constant, for the reason {@link
     * ExecutionPolicy#fromSystemProperties()} gives: a constant is the defect being removed.
     * Nothing calls this in a loop — a model resolves it once, at load.
     */
    public static StorageOptions fromSystemProperties() {
        return new StorageOptions(
                Boolean.getBoolean("llama.kvcache.fp16") ? DataType.F16 : DataType.F32,
                Boolean.getBoolean("llama.kv.sharedPool"));
    }

    /** Whether key/value entries are half precision. */
    public boolean usesFp16KeyValueCache() {
        return keyValueRepresentation == DataType.F16;
    }
}
