package org.beehive.gpullama3.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Phi3's KV write kernel, addressing the cache through a block table.
 *
 * <p>Addressing is defined once, in {@link KvBlockAddress}.
 */
public class Phi3PagedKvKernels {

    public Phi3PagedKvKernels() {}

    public static void ropeRotationWithCacheCopyPhi3Paged(
            KernelContext context,
            IntArray positionHolder,
            FloatArray sq, // Q vector (in/out)
            FloatArray sk, // K vector (in/out)
            FloatArray sv, // V vector (in only)
            FloatArray keyCache, // Key cache (out)
            FloatArray valueCache, // Value cache (out)
            int nHeadKv,
            int headSize,
            int kvDim,
            int layer,
            IntArray blockTable,
            int blockCfg,
            int blockStride) {

        int idx = context.globalIdx;
        int dimHalf = headSize / 2;

        // Each thread processes one dimension pair
        if (idx >= dimHalf) {
            return;
        }

        int pos = positionHolder.get(0);
        int slot = positionHolder.get(1);
        int cacheOffset =
                KvBlockAddress.offset(
                        blockTable,
                        slot,
                        pos,
                        KvBlockAddress.layerOffset(layer, kvDim, blockCfg),
                        kvDim,
                        blockCfg,
                        blockStride);

        // Calculate frequency for this dimension
        float freq = 1.0f / TornadoMath.pow(10000.0f, (float) (idx * 2) / (float) headSize);
        float val = pos * freq;
        float fcr = TornadoMath.cos(val);
        float fci = TornadoMath.sin(val);

        // Process Q: all heads (dim = nHeads × headSize)
        int totalDimQ = sq.getSize();
        for (int base = 0; base < totalDimQ; base += headSize) {
            if (base + idx >= totalDimQ || base + idx + dimHalf >= totalDimQ) {
                break;
            }

            // Rotate Q with offset pattern
            float v0 = sq.get(base + idx);
            float v1 = sq.get(base + idx + dimHalf);
            sq.set(base + idx, v0 * fcr - v1 * fci);
            sq.set(base + idx + dimHalf, v0 * fci + v1 * fcr);
        }

        // Process K: only kvDim elements, with cache write
        for (int base = 0; base < kvDim; base += headSize) {
            if (base + idx >= kvDim || base + idx + dimHalf >= kvDim) {
                break;
            }

            // Rotate K with offset pattern
            float k0 = sk.get(base + idx);
            float k1 = sk.get(base + idx + dimHalf);
            float rotated0 = k0 * fcr - k1 * fci;
            float rotated1 = k0 * fci + k1 * fcr;

            // Write rotated K back
            sk.set(base + idx, rotated0);
            sk.set(base + idx + dimHalf, rotated1);

            // Fused cache write for K
            keyCache.set(cacheOffset + base + idx, rotated0);
            keyCache.set(cacheOffset + base + idx + dimHalf, rotated1);

            // Fused cache copy for V (no rotation needed)
            valueCache.set(cacheOffset + base + idx, sv.get(base + idx));
            valueCache.set(cacheOffset + base + idx + dimHalf, sv.get(base + idx + dimHalf));
        }
    }
}
