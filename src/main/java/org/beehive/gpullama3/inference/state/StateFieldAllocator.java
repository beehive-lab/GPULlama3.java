package org.beehive.gpullama3.inference.state;

import org.beehive.gpullama3.core.model.tensor.ArrayFloatTensor;
import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import org.beehive.gpullama3.model.Configuration;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

import java.util.stream.Stream;

/**
 * Abstract factory for allocating State fields to eliminate duplication across model-specific State implementations.
 *
 * <p>This class provides a template method pattern for state field allocation, where common allocation
 * patterns are implemented in protected helper methods, and model-specific dimension calculations
 * are delegated to subclasses through abstract methods.</p>
 *
 * <p><b>Design Pattern:</b> Abstract Factory + Template Method</p>
 * <ul>
 *   <li>Abstract Factory: Creates families of related state field objects</li>
 *   <li>Template Method: Defines skeleton of allocation algorithm, delegates dimension calculation to subclasses</li>
 * </ul>
 *
 * <p><b>Benefits:</b></p>
 * <ul>
 *   <li>Eliminates ~80% code duplication across State implementations</li>
 *   <li>Centralizes allocation logic for easier maintenance</li>
 *   <li>Enforces consistent initialization patterns</li>
 *   <li>Makes dimension calculations explicit and model-specific</li>
 * </ul>
 */
public abstract class StateFieldAllocator {

    protected final Configuration config;
    protected final int localSize;

    protected StateFieldAllocator(Configuration config, int localSize) {
        this.config = config;
        this.localSize = localSize;
    }

    /**
     * Allocates all state fields using model-specific dimensions.
     * This is the main entry point for state field allocation.
     */
    public State.StateFields allocateFields() {
        State.StateFields fields = new State.StateFields();

        // Allocate tensor fields
        fields.x = ArrayFloatTensor.allocate(getDimX());
        fields.xb = ArrayFloatTensor.allocate(getDimXb());
        fields.xb2 = ArrayFloatTensor.allocate(getDimXb2());
        fields.hb = ArrayFloatTensor.allocate(getDimHb());
        fields.hb2 = ArrayFloatTensor.allocate(getDimHb2());
        fields.q = ArrayFloatTensor.allocate(getDimQ());
        fields.k = ArrayFloatTensor.allocate(getDimK());
        fields.v = ArrayFloatTensor.allocate(getDimV());
        fields.att = ArrayFloatTensor.allocate(config.numberOfHeads(), config.contextLength());
        fields.logits = ArrayFloatTensor.allocate(config.vocabularySize());

        // Allocate KV cache
        int kvCacheDim = getKvCacheDim();
        fields.keyCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength(), kvCacheDim))
                .limit(config.numberOfLayers())
                .toArray(FloatTensor[]::new);
        fields.valueCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength(), kvCacheDim))
                .limit(config.numberOfLayers())
                .toArray(FloatTensor[]::new);

        // Allocate TornadoVM wrappers
        fields.wrapX = new FloatArray(getDimX());
        fields.wrapXb = new FloatArray(getDimXb());
        fields.wrapXb2 = new FloatArray(getDimXb2());
        fields.wrapHb = new FloatArray(getDimHb());
        fields.wrapHb2 = new FloatArray(getDimHb2());
        fields.wrapLogits = new FloatArray(config.vocabularySize());
        fields.wrapQ = new FloatArray(getDimQ());
        fields.wrapK = new FloatArray(getDimK());
        fields.wrapV = new FloatArray(getDimV());

        // Allocate KV cache wrappers
        fields.wrapKeyCache = new FloatArray(config.contextLength() * kvCacheDim * config.numberOfLayers());
        fields.wrapValueCache = new FloatArray(config.contextLength() * kvCacheDim * config.numberOfLayers());
        fields.wrapKeyCache.init(0.f);
        fields.wrapValueCache.init(0.f);

        fields.wrapAtt = new FloatArray(config.numberOfHeads() * config.contextLength());
        fields.positionHolder = new IntArray(1);

        // Allocate temporary arrays with proper dimensions
        fields.temp = new FloatArray(1 + ((getTempReductionDim() + localSize - 1) / localSize));
        fields.tempFFN = new FloatArray(1 + ((getTempFFNReductionDim() + localSize - 1) / localSize));
        fields.tempLogits = new FloatArray(1 + ((getTempLogitsReductionDim() + localSize - 1) / localSize));

        return fields;
    }

    // Abstract methods for model-specific dimensions - subclasses must implement these

    /**
     * @return Dimension for x tensor (current activation)
     */
    protected abstract int getDimX();

    /**
     * @return Dimension for xb tensor (residual branch activation)
     */
    protected abstract int getDimXb();

    /**
     * @return Dimension for xb2 tensor (additional residual buffer)
     */
    protected abstract int getDimXb2();

    /**
     * @return Dimension for hb tensor (hidden dimension buffer for FFN)
     */
    protected abstract int getDimHb();

    /**
     * @return Dimension for hb2 tensor (additional hidden buffer for FFN)
     */
    protected abstract int getDimHb2();

    /**
     * @return Dimension for q tensor (query)
     */
    protected abstract int getDimQ();

    /**
     * @return Dimension for k tensor (key)
     */
    protected abstract int getDimK();

    /**
     * @return Dimension for v tensor (value)
     */
    protected abstract int getDimV();

    /**
     * @return Dimension for KV cache (per layer)
     */
    protected abstract int getKvCacheDim();

    /**
     * @return Dimension for temp reduction buffer (usually config.dim())
     */
    protected int getTempReductionDim() {
        return config.dim();
    }

    /**
     * @return Dimension for tempFFN reduction buffer (usually config.hiddenDim())
     */
    protected int getTempFFNReductionDim() {
        return config.hiddenDim();
    }

    /**
     * @return Dimension for tempLogits reduction buffer (usually config.vocabularySize())
     */
    protected int getTempLogitsReductionDim() {
        return config.vocabularySize();
    }
}
