package org.beehive.gpullama3.tensor.tornado;

import org.beehive.gpullama3.tensor.GGMLType;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

import java.lang.foreign.MemorySegment;

public class FP16TornadoTensor extends TornadoTensor {
    private final HalfFloatArray tornadoNativeArray;

    public FP16TornadoTensor(HalfFloatArray halfFloatArray) {
        this.tornadoNativeArray = halfFloatArray;
    }

    public static FP16TornadoTensor fromTornadoMemorySegment(MemorySegment segment) {
        return new FP16TornadoTensor(HalfFloatArray.fromSegmentShallow(segment));
    }

    @Override
    public HalfFloatArray asHalfFloatArray() {
        return tornadoNativeArray;
    }

    @Override
    public GGMLType type() {
        return GGMLType.F16;
    }
}

