package org.beehive.gpullama3.tornadovm.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
public class Qwen3Kernels {

    /**
     * For explicit copy out useful in debugging.
     * With this kernel we can store the values of an array to a tmp buffer at a timing of interest.
     * In the end of the taskgraph we copy out the tmp buffer to inspect the array values at the timing of interest.
     * @param srcBuffer the array we want to inspect.
     * @param dstBuffer the tmp buffer.
     */
    public static void dbgCopy(FloatArray srcBuffer, FloatArray dstBuffer) {
        for (@Parallel int i = 0; i < srcBuffer.getSize(); i++) {
            dstBuffer.set(i, srcBuffer.get(i));
        }
    }

    /**
     * RmsNorm with parallel offset:
     * The following 3 kernels implement rmsnorm in offset range in parallel for qCur and Kcur rmsnorm calculations.
     *
     * Step 1: Reduction.
     * This kernel implements rmsnorm in offset range in parallel for qCur and Kcur rmsnorm calculations.
     */
    public static void rmsnormReductionWithParallelOffset(KernelContext context, FloatArray output, FloatArray x, int localMemSize) {

        int gid = context.globalIdx;
        int lid = context.localIdx;
        int groupId = context.groupIdx;
        int groupSize = context.localGroupSizeX;

        // Allocate local memory with the provided size
        float[] localX = context.allocateFloatLocalArray(localMemSize);

        // Load input value and compute square
        localX[lid] = x.get(gid);
        localX[lid] = localX[lid] * localX[lid];

        // Perform parallel reduction within the work group
        for (int stride = (groupSize / 2); stride > 0; stride /= 2) {
            context.localBarrier();
            if (lid < stride) {
                localX[lid] += localX[lid + stride];
            }
        }

        // Each workgroup stores its partial sum in a different location
        if (lid == 0) {
            // Store the partial sum from each workgroup
            output.set(groupId, localX[0]);
        }
    }

    /**
     * RmsNorm with parallel offset:
     *
     * Step 2: Combines partial reduction outputs and computes final normalization.
     */
    public static void rmsnormFinalNormalizationWithParallelOffset(
            KernelContext context,
            FloatArray output, // size should be related to offsetIndex
            int offsetIndex,   // = config.numberOfHeads()
            int size,
            float ermsNorm) {

        int gid = context.globalIdx;

        // Only the index threads need to perform this calculation
        if (gid < offsetIndex) {
            // Combine partial sums from all workgroups
            float ss = output.get(gid);

            ss /= size;
            ss += ermsNorm;
            ss = 1.0f / TornadoMath.sqrt(ss);
            // in place
            output.set(gid, ss);  // Store the final scale factor
        }
    }

    /**
     * RmsNorm with parallel offset:
     *
     * Step 3: perform mapIndex operation.
     */
    public static void rmsnormMapIndexInPlaceWithParallelOffset(
            KernelContext context,
            FloatArray out,
            FloatArray weights,
            int size,
            FloatArray ss) {

        int gid = context.globalIdx;
        int groupId = context.groupIdx;

        float finalss = ss.get(groupId);

        if (gid < out.getSize()) { // TODO: check if redundant
            float a = weights.get(gid % size);
            float b = finalss * out.get(gid);
            out.set(gid, a * b);
        }
    }

    /**
     * RmsNorm with parallel offset:
     *
     * Optimized kernel that combines Step 1 (Reduction) and Step 2 (Normalization).
     */
    public static void rmsnormWithParallelOffset(
            KernelContext context,
            FloatArray output,
            FloatArray x,
            int localMemSize,
            int size,
            float ermsNorm) {

        int gid = context.globalIdx;
        int lid = context.localIdx;
        int groupId = context.groupIdx;
        int groupSize = context.localGroupSizeX;

        // Allocate local memory with the provided size
        float[] localX = context.allocateFloatLocalArray(localMemSize);

        // Load input value and compute square
        localX[lid] = x.get(gid);
        localX[lid] = localX[lid] * localX[lid];

        // Perform parallel reduction within the work group
        for (int stride = (groupSize / 2); stride > 0; stride /= 2) {
            context.localBarrier();
            if (lid < stride) {
                localX[lid] += localX[lid + stride];
            }
        }

        // Each workgroup performs the normalization
        if (lid == 0) {
            // Store the partial sum from each workgroup
            localX[0] /= size;
            localX[0] += ermsNorm;
            localX[0] = 1.0f / TornadoMath.sqrt(localX[0]);
            output.set(groupId, localX[0]);
        }
    }

    public static void ropeRotation(
            KernelContext context,
            IntArray position,
            FloatArray q,
            FloatArray k,
            int numberOfKeyValueHeads,
            int nEmbdHead) {

        int h = context.globalIdx;
        int ic = context.globalIdy;

        int rotn = h < numberOfKeyValueHeads ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
        int poffset = h * nEmbdHead;
        int nComplEmbdHead = nEmbdHead / 2;

        // Compute RoPE frequencies for Qwen3
        float theta = 1000000.0f;
        int i = ic * 2; // match i in precompute (see RoPE.precomputeFreqsCis)
        float freq = 1.0f / TornadoMath.pow(theta, (float) i / (float) nEmbdHead);

        float val = position.get(0) * freq;
        float fcr = TornadoMath.cos(val);
        float fci = TornadoMath.sin(val);

        float v0q = q.get(poffset + ic);
        float v1q = q.get(poffset + ic + nComplEmbdHead);
        q.set(poffset + ic, v0q * fcr - v1q * fci);
        q.set(poffset + ic + nComplEmbdHead, v0q * fci + v1q * fcr);

        if (rotn > 1 && (poffset + ic + nComplEmbdHead) < k.getSize()) {
            float v0k = k.get(poffset + ic);
            float v1k = k.get(poffset + ic + nComplEmbdHead);
            k.set(poffset + ic, v0k * fcr - v1k * fci);
            k.set(poffset + ic + nComplEmbdHead, v0k * fci + v1k * fcr);
        }

    }

    public static void processHeadsParallel(
            FloatArray q,
            FloatArray key_cache,
            FloatArray value_cache,
            FloatArray xb,
            int nHeads,
            int nEmbdHead, /* = nEmbdHead, replace headSize in lines: 244, 253,  */
            int nEmbdHeadK, /* = config.numberOfHeadsKey(), replace headSize in line 255 */
            int nEmbdHeadV, /* = config.numberOfHeadsValue(), replace headSize in lines: 266, 268, 273 */
            int nEmbdGqa, /* kvDim */
            int gqa, /* kvMul */
            IntArray positionHolder,
            FloatArray wrapAtt,
            int layer, int contextLength) {

        int pos = positionHolder.get(0);
        int loff = layer * contextLength * nEmbdGqa;

        // Parallelize computation across attention heads
        for (@Parallel int h = 0; h < nHeads; h++) {
            // Process each head in parallel
            //noinspection ExternalInspection
            processHeadTornado(q, key_cache, value_cache, xb, h, nEmbdHead, /* headSize */
                    nEmbdHeadK, /* headSize in line 255 */
                    nEmbdHeadV, /* headSize in lines: 266, 268, 273 */
                    nEmbdGqa, /* kvDim */
                    gqa, /* kvMul */
                    loff, pos, wrapAtt, contextLength);
        }
    }

    private static void processHeadTornado(
            FloatArray allQ,
            FloatArray key_cache,
            FloatArray value_cache,
            FloatArray allXb,
            int h,
            int nEmbdHead, /* = nEmbdHeadV, replace headSize in lines: 244, 253,  */
            int nEmbdHeadK, /* = config.numberOfHeadsKey(), replace headSize in line 255 */
            int nEmbdHeadV, /* = config.numberOfHeadsValue(), replace headSize in lines: 266, 268, 273 */
            int nEmbdGqa, /* kvDim */
            int gqa, /* kvMul */
            long loff,
            int pos,
            FloatArray wrapAtt,
            int contextLength) {

        // Base index for this head's attention weights
        int headOffset = h * (pos + 1);

        // STEP 1: Calculate attention scores for all timesteps
        for (int t = 0; t <= pos; t++) {
            int kvHeadIdx = h / gqa;
            int keyOffset = (int) (loff + t * nEmbdGqa + kvHeadIdx * nEmbdHeadK); // line 255

            float score = 0.0f;
            for (int i = 0; i < nEmbdHeadK; i++) {
                score += allQ.get(h * nEmbdHeadK + i) * key_cache.get(keyOffset + i); // line 255
            }
            score = score / TornadoMath.sqrt(nEmbdHead); // line 257

            // Store in attention buffer
            wrapAtt.set(headOffset + t, score);
        }

        // STEP 2: Find max score for softmax stability
        float maxScore = wrapAtt.get(headOffset);
        for (int t = 1; t <= pos; t++) {
            float val = wrapAtt.get(headOffset + t);
            if (val > maxScore) {
                maxScore = val;
            }
        }

        // STEP 3: Compute exponentials and sum
        float sum = 0.0f;
        for (int t = 0; t <= pos; t++) {
            int idx = headOffset + t;
            float expScore = TornadoMath.exp(wrapAtt.get(idx) - maxScore);
            wrapAtt.set(idx, expScore);
            sum += expScore;
        }

        // STEP 4: Normalize
        float normFactor = (sum > 0.0f) ? (1.0f / sum) : (1.0f / (pos + 1));
        for (int t = 0; t <= pos; t++) {
            int idx = headOffset + t;
            wrapAtt.set(idx, wrapAtt.get(idx) * normFactor);
        }

        // STEP 5: Compute weighted sum of values for each dimension
        for (int i = 0; i < nEmbdHeadV; i++) {
            float weightedSum = 0.0f;
            for (int t = 0; t <= pos; t++) {
                int kvHeadIdx = h / gqa;
                int valueOffset = (int) (loff + t * nEmbdGqa + kvHeadIdx * nEmbdHeadV); //line 273
                weightedSum += wrapAtt.get(headOffset + t) * value_cache.get(valueOffset + i);
            }
            allXb.set(h * nEmbdHeadV + i, weightedSum); // offset from line 266
        }
    }

    /**
     * Fused RoPE rotation with KV cache copy for Qwen3.
     * Combines ropeRotation + copyToCache into a single kernel.
     */
    public static void ropeRotationWithCacheCopy(
            KernelContext context,
            IntArray positionHolder,
            FloatArray q,              // Q vector (in/out)
            FloatArray k,              // K vector (in/out)
            FloatArray v,              // V vector (in only)
            FloatArray keyCache,       // Key cache (out)
            FloatArray valueCache,     // Value cache (out)
            int numberOfKeyValueHeads,
            int nEmbdHead,
            int nEmbdGqa,
            int layer,
            int contextLength) {

        int h = context.globalIdx;
        int ic = context.globalIdy;

        int pos = positionHolder.get(0);
        int rotn = h < numberOfKeyValueHeads ? 2 : 1;
        int poffset = h * nEmbdHead;
        int nComplEmbdHead = nEmbdHead / 2;

        // Compute RoPE frequencies for Qwen3 (theta = 1000000.0f)
        float theta = 1000000.0f;
        int i = ic * 2;
        float freq = 1.0f / TornadoMath.pow(theta, (float) i / (float) nEmbdHead);

        float val = pos * freq;
        float fcr = TornadoMath.cos(val);
        float fci = TornadoMath.sin(val);

        // Rotate Q (all heads)
        float v0q = q.get(poffset + ic);
        float v1q = q.get(poffset + ic + nComplEmbdHead);
        q.set(poffset + ic, v0q * fcr - v1q * fci);
        q.set(poffset + ic + nComplEmbdHead, v0q * fci + v1q * fcr);

        // Rotate K and copy K/V to cache (only for KV heads)
        if (rotn > 1 && (poffset + ic + nComplEmbdHead) < k.getSize()) {
            float v0k = k.get(poffset + ic);
            float v1k = k.get(poffset + ic + nComplEmbdHead);
            float rotatedK0 = v0k * fcr - v1k * fci;
            float rotatedK1 = v0k * fci + v1k * fcr;

            // Write rotated K back
            k.set(poffset + ic, rotatedK0);
            k.set(poffset + ic + nComplEmbdHead, rotatedK1);

            // Direct cache write (fused - no separate copy kernel!)
            int cacheOffset = layer * contextLength * nEmbdGqa + pos * nEmbdGqa;
            int kvIdx = h * nEmbdHead;

            keyCache.set(cacheOffset + kvIdx + ic, rotatedK0);
            keyCache.set(cacheOffset + kvIdx + ic + nComplEmbdHead, rotatedK1);

            // Copy V to cache (V doesn't need rotation)
            valueCache.set(cacheOffset + kvIdx + ic, v.get(poffset + ic));
            valueCache.set(cacheOffset + kvIdx + ic + nComplEmbdHead, v.get(poffset + ic + nComplEmbdHead));
        }
    }

    /**
     * Fused Q/K/V matrix-vector multiplication for Qwen3 GQA.
     * Q has full head dimension, K/V have reduced KV head dimension.
     *
     * Workgroup assignment:
     *   - rowId [0, qDim): Q projection
     *   - rowId [qDim, qDim+kvDim): K projection
     *   - rowId [qDim+kvDim, qDim+2*kvDim): V projection
     */
    public static void fusedQKVMatmul(
            KernelContext context,
            FloatArray x,               // input vector
            FloatArray q,               // output Q
            FloatArray k,               // output K
            FloatArray v,               // output V
            HalfFloatArray wq,          // Q weight matrix
            HalfFloatArray wk,          // K weight matrix
            HalfFloatArray wv,          // V weight matrix
            int inputDim,               // input dimension (config.dim())
            int qDim,                   // Q output dimension
            int kvDim,                  // KV output dimension
            int localWorkGroupSize) {

        int rowId = context.groupIdx;
        int localId = context.localIdx;

        // Allocate local memory for reduction
        float[] localSum = context.allocateFloatLocalArray(localWorkGroupSize);

        if (rowId < qDim) {
            // ========== Q projection ==========
            int rowOffset = rowId * inputDim;

            float partialSum = 0.0f;
            for (int j = localId; j < inputDim; j += localWorkGroupSize) {
                partialSum += wq.get(rowOffset + j).getFloat32() * x.get(j);
            }

            localSum[localId] = partialSum;
            context.localBarrier();

            for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
                if (localId < stride) {
                    localSum[localId] += localSum[localId + stride];
                }
                context.localBarrier();
            }

            if (localId == 0) {
                q.set(rowId, localSum[0]);
            }

        } else if (rowId < qDim + kvDim) {
            // ========== K projection ==========
            int kRow = rowId - qDim;
            int rowOffset = kRow * inputDim;

            float partialSum = 0.0f;
            for (int j = localId; j < inputDim; j += localWorkGroupSize) {
                partialSum += wk.get(rowOffset + j).getFloat32() * x.get(j);
            }

            localSum[localId] = partialSum;
            context.localBarrier();

            for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
                if (localId < stride) {
                    localSum[localId] += localSum[localId + stride];
                }
                context.localBarrier();
            }

            if (localId == 0) {
                k.set(kRow, localSum[0]);
            }

        } else if (rowId < qDim + 2 * kvDim) {
            // ========== V projection ==========
            int vRow = rowId - qDim - kvDim;
            int rowOffset = vRow * inputDim;

            float partialSum = 0.0f;
            for (int j = localId; j < inputDim; j += localWorkGroupSize) {
                partialSum += wv.get(rowOffset + j).getFloat32() * x.get(j);
            }

            localSum[localId] = partialSum;
            context.localBarrier();

            for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
                if (localId < stride) {
                    localSum[localId] += localSum[localId + stride];
                }
                context.localBarrier();
            }

            if (localId == 0) {
                v.set(vRow, localSum[0]);
            }
        }
    }

    /**
     * Fused RMSNorm apply + Q/K/V projection for Qwen3 GQA.
     * Eliminates intermediate wrapXb buffer write/read.
     */
    public static void fusedRmsNormQKVMatmul(
            KernelContext context,
            FloatArray x,               // raw input (FP32)
            FloatArray q,               // output Q
            FloatArray k,               // output K
            FloatArray v,               // output V
            FloatArray rmsWeights,      // RMS norm weights
            FloatArray rmsScale,        // temp[0] = scale factor
            HalfFloatArray wq,          // Q weight matrix
            HalfFloatArray wk,          // K weight matrix
            HalfFloatArray wv,          // V weight matrix
            int inputDim,               // input dimension (config.dim())
            int qDim,                   // Q output dimension
            int kvDim,                  // KV output dimension
            int localWorkGroupSize) {

        int rowId = context.groupIdx;
        int localId = context.localIdx;

        float scale = rmsScale.get(0);

        // Allocate local memory for reduction
        float[] localSum = context.allocateFloatLocalArray(localWorkGroupSize);

        if (rowId < qDim) {
            // ========== Q projection with inline normalization ==========
            int rowOffset = rowId * inputDim;

            float partialSum = 0.0f;
            for (int j = localId; j < inputDim; j += localWorkGroupSize) {
                float normalized = rmsWeights.get(j) * scale * x.get(j);
                partialSum += wq.get(rowOffset + j).getFloat32() * normalized;
            }

            localSum[localId] = partialSum;
            context.localBarrier();

            for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
                if (localId < stride) {
                    localSum[localId] += localSum[localId + stride];
                }
                context.localBarrier();
            }

            if (localId == 0) {
                q.set(rowId, localSum[0]);
            }

        } else if (rowId < qDim + kvDim) {
            // ========== K projection with inline normalization ==========
            int kRow = rowId - qDim;
            int rowOffset = kRow * inputDim;

            float partialSum = 0.0f;
            for (int j = localId; j < inputDim; j += localWorkGroupSize) {
                float normalized = rmsWeights.get(j) * scale * x.get(j);
                partialSum += wk.get(rowOffset + j).getFloat32() * normalized;
            }

            localSum[localId] = partialSum;
            context.localBarrier();

            for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
                if (localId < stride) {
                    localSum[localId] += localSum[localId + stride];
                }
                context.localBarrier();
            }

            if (localId == 0) {
                k.set(kRow, localSum[0]);
            }

        } else if (rowId < qDim + 2 * kvDim) {
            // ========== V projection with inline normalization ==========
            int vRow = rowId - qDim - kvDim;
            int rowOffset = vRow * inputDim;

            float partialSum = 0.0f;
            for (int j = localId; j < inputDim; j += localWorkGroupSize) {
                float normalized = rmsWeights.get(j) * scale * x.get(j);
                partialSum += wv.get(rowOffset + j).getFloat32() * normalized;
            }

            localSum[localId] = partialSum;
            context.localBarrier();

            for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
                if (localId < stride) {
                    localSum[localId] += localSum[localId + stride];
                }
                context.localBarrier();
            }

            if (localId == 0) {
                v.set(vRow, localSum[0]);
            }
        }
    }

    /**
     * Fused RMSNorm apply + Q/K/V projection for Qwen3 GQA with Q8_0 quantized weights.
     * Uses the same Q8_0 block structure as matrixVectorRowMajorOptimizedQ8_0Byte.
     */
    public static void fusedRmsNormQKVMatmulQ8_0(
            KernelContext context,
            FloatArray x,               // raw input (FP32)
            FloatArray q,               // output Q
            FloatArray k,               // output K
            FloatArray v,               // output V
            FloatArray rmsWeights,      // RMS norm weights
            FloatArray rmsScale,        // temp[0] = scale factor
            ByteArray wq,               // Q weight matrix (Q8_0)
            ByteArray wk,               // K weight matrix (Q8_0)
            ByteArray wv,               // V weight matrix (Q8_0)
            int inputDim,               // input dimension (config.dim())
            int qDim,                   // Q output dimension
            int kvDim,                  // KV output dimension
            int localWorkGroupSize) {

        int rowId = context.groupIdx;
        int localId = context.localIdx;

        float scale = rmsScale.get(0);
        final int blockSize = 32;
        final int Q8_0_BLOCK_BYTES = 34; // 2 bytes scale + 32 bytes quants

        // Allocate local memory for reduction
        float[] localSums = context.allocateFloatLocalArray(localWorkGroupSize);

        if (rowId < qDim) {
            // ========== Q projection with inline normalization ==========
            int blocksPerRow = (inputDim + blockSize - 1) / blockSize;
            int rowBlockOffset = rowId * blocksPerRow;

            float partialSum = 0.0f;

            // Main loop with 4-way unrolling
            for (int j = localId * 4; j < inputDim - 3; j += localWorkGroupSize * 4) {
                int blockIdx = j / blockSize;
                int withinBlockIdx = j % blockSize;

                int blockByteOffset = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;

                // Load scale for this block
                HalfFloat blockScale = wq.getHalfFloat(blockByteOffset);
                float scaleFloat = blockScale.getFloat32();

                // Load 4 consecutive quantized values
                int quantsOffset = blockByteOffset + 2 + withinBlockIdx; // Skip 2-byte scale
                byte quant1 = wq.get(quantsOffset);
                byte quant2 = wq.get(quantsOffset + 1);
                byte quant3 = wq.get(quantsOffset + 2);
                byte quant4 = wq.get(quantsOffset + 3);

                // Apply RMS normalization inline and compute dot product
                float norm1 = rmsWeights.get(j) * scale * x.get(j);
                float norm2 = rmsWeights.get(j + 1) * scale * x.get(j + 1);
                float norm3 = rmsWeights.get(j + 2) * scale * x.get(j + 2);
                float norm4 = rmsWeights.get(j + 3) * scale * x.get(j + 3);

                partialSum += ((float) quant1 * scaleFloat) * norm1;
                partialSum += ((float) quant2 * scaleFloat) * norm2;
                partialSum += ((float) quant3 * scaleFloat) * norm3;
                partialSum += ((float) quant4 * scaleFloat) * norm4;
            }

            // Handle remaining elements
            for (int j = ((inputDim / 4) * 4) + localId; j < inputDim; j += localWorkGroupSize) {
                int blockIdx = j / blockSize;
                int withinBlockIdx = j % blockSize;

                int blockByteOffset = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;

                HalfFloat blockScale = wq.getHalfFloat(blockByteOffset);
                float scaleFloat = blockScale.getFloat32();

                byte quant = wq.get(blockByteOffset + 2 + withinBlockIdx);
                float normalized = rmsWeights.get(j) * scale * x.get(j);

                partialSum += ((float) quant * scaleFloat) * normalized;
            }

            localSums[localId] = partialSum;
            context.localBarrier();

            // Parallel reduction
            for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
                if (localId < stride) {
                    localSums[localId] += localSums[localId + stride];
                }
                context.localBarrier();
            }

            if (localId == 0) {
                q.set(rowId, localSums[0]);
            }

        } else if (rowId < qDim + kvDim) {
            // ========== K projection with inline normalization ==========
            int kRow = rowId - qDim;
            int blocksPerRow = (inputDim + blockSize - 1) / blockSize;
            int rowBlockOffset = kRow * blocksPerRow;

            float partialSum = 0.0f;

            // Main loop with 4-way unrolling
            for (int j = localId * 4; j < inputDim - 3; j += localWorkGroupSize * 4) {
                int blockIdx = j / blockSize;
                int withinBlockIdx = j % blockSize;

                int blockByteOffset = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;

                HalfFloat blockScale = wk.getHalfFloat(blockByteOffset);
                float scaleFloat = blockScale.getFloat32();

                int quantsOffset = blockByteOffset + 2 + withinBlockIdx;
                byte quant1 = wk.get(quantsOffset);
                byte quant2 = wk.get(quantsOffset + 1);
                byte quant3 = wk.get(quantsOffset + 2);
                byte quant4 = wk.get(quantsOffset + 3);

                float norm1 = rmsWeights.get(j) * scale * x.get(j);
                float norm2 = rmsWeights.get(j + 1) * scale * x.get(j + 1);
                float norm3 = rmsWeights.get(j + 2) * scale * x.get(j + 2);
                float norm4 = rmsWeights.get(j + 3) * scale * x.get(j + 3);

                partialSum += ((float) quant1 * scaleFloat) * norm1;
                partialSum += ((float) quant2 * scaleFloat) * norm2;
                partialSum += ((float) quant3 * scaleFloat) * norm3;
                partialSum += ((float) quant4 * scaleFloat) * norm4;
            }

            for (int j = ((inputDim / 4) * 4) + localId; j < inputDim; j += localWorkGroupSize) {
                int blockIdx = j / blockSize;
                int withinBlockIdx = j % blockSize;

                int blockByteOffset = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;

                HalfFloat blockScale = wk.getHalfFloat(blockByteOffset);
                float scaleFloat = blockScale.getFloat32();

                byte quant = wk.get(blockByteOffset + 2 + withinBlockIdx);
                float normalized = rmsWeights.get(j) * scale * x.get(j);

                partialSum += ((float) quant * scaleFloat) * normalized;
            }

            localSums[localId] = partialSum;
            context.localBarrier();

            for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
                if (localId < stride) {
                    localSums[localId] += localSums[localId + stride];
                }
                context.localBarrier();
            }

            if (localId == 0) {
                k.set(kRow, localSums[0]);
            }

        } else if (rowId < qDim + 2 * kvDim) {
            // ========== V projection with inline normalization ==========
            int vRow = rowId - qDim - kvDim;
            int blocksPerRow = (inputDim + blockSize - 1) / blockSize;
            int rowBlockOffset = vRow * blocksPerRow;

            float partialSum = 0.0f;

            // Main loop with 4-way unrolling
            for (int j = localId * 4; j < inputDim - 3; j += localWorkGroupSize * 4) {
                int blockIdx = j / blockSize;
                int withinBlockIdx = j % blockSize;

                int blockByteOffset = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;

                HalfFloat blockScale = wv.getHalfFloat(blockByteOffset);
                float scaleFloat = blockScale.getFloat32();

                int quantsOffset = blockByteOffset + 2 + withinBlockIdx;
                byte quant1 = wv.get(quantsOffset);
                byte quant2 = wv.get(quantsOffset + 1);
                byte quant3 = wv.get(quantsOffset + 2);
                byte quant4 = wv.get(quantsOffset + 3);

                float norm1 = rmsWeights.get(j) * scale * x.get(j);
                float norm2 = rmsWeights.get(j + 1) * scale * x.get(j + 1);
                float norm3 = rmsWeights.get(j + 2) * scale * x.get(j + 2);
                float norm4 = rmsWeights.get(j + 3) * scale * x.get(j + 3);

                partialSum += ((float) quant1 * scaleFloat) * norm1;
                partialSum += ((float) quant2 * scaleFloat) * norm2;
                partialSum += ((float) quant3 * scaleFloat) * norm3;
                partialSum += ((float) quant4 * scaleFloat) * norm4;
            }

            for (int j = ((inputDim / 4) * 4) + localId; j < inputDim; j += localWorkGroupSize) {
                int blockIdx = j / blockSize;
                int withinBlockIdx = j % blockSize;

                int blockByteOffset = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;

                HalfFloat blockScale = wv.getHalfFloat(blockByteOffset);
                float scaleFloat = blockScale.getFloat32();

                byte quant = wv.get(blockByteOffset + 2 + withinBlockIdx);
                float normalized = rmsWeights.get(j) * scale * x.get(j);

                partialSum += ((float) quant * scaleFloat) * normalized;
            }

            localSums[localId] = partialSum;
            context.localBarrier();

            for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
                if (localId < stride) {
                    localSums[localId] += localSums[localId + stride];
                }
                context.localBarrier();
            }

            if (localId == 0) {
                v.set(vRow, localSums[0]);
            }
        }
    }


    /**
     * Fused Q and K RMSNorm for Qwen3.
     * Combines rmsnormReduction + rmsnormMapIndexInPlace for both Q and K into one kernel.
     *
     * Workgroup assignment:
     *   - Workgroups [0, nHeads): Process Q heads
     *   - Workgroups [nHeads, nHeads + nHeadKv): Process K heads
     */
    public static void fusedQKRmsNorm(
            KernelContext context,
            FloatArray q,                // Q vector (in/out)
            FloatArray k,                // K vector (in/out)
            FloatArray qWeights,         // Q RMS norm weights
            FloatArray kWeights,         // K RMS norm weights
            int nHeads,                  // number of Q heads
            int nHeadKv,                 // number of K heads
            int nEmbdHead,               // head dimension
            int localMemSize,            // local memory size (must be fixed)
            float rmsNormEps) {

        int groupId = context.groupIdx;
        int localId = context.localIdx;
        int localSize = context.localGroupSizeX;

        // Allocate local memory with FIXED size parameter
        float[] localSum = context.allocateFloatLocalArray(localMemSize);

        if (groupId < nHeads) {
            // === Process Q head ===
            int headOffset = groupId * nEmbdHead;

            // Step 1: Compute sum of squares (reduction)
            float partialSum = 0.0f;
            for (int i = localId; i < nEmbdHead; i += localSize) {
                float val = q.get(headOffset + i);
                partialSum += val * val;
            }

            localSum[localId] = partialSum;
            context.localBarrier();

            // Parallel reduction
            for (int stride = localSize / 2; stride > 0; stride >>= 1) {
                if (localId < stride) {
                    localSum[localId] += localSum[localId + stride];
                }
                context.localBarrier();
            }

            // Compute normalization factor
            float ss = localSum[0];
            ss = ss / nEmbdHead + rmsNormEps;
            ss = 1.0f / TornadoMath.sqrt(ss);

            context.localBarrier();

            // Step 2: Apply normalization with weights (in-place)
            for (int i = localId; i < nEmbdHead; i += localSize) {
                float normalized = ss * q.get(headOffset + i);
                q.set(headOffset + i, qWeights.get(i) * normalized);
            }

        } else if (groupId < nHeads + nHeadKv) {
            // === Process K head ===
            int headIdx = groupId - nHeads;
            int headOffset = headIdx * nEmbdHead;

            // Step 1: Compute sum of squares (reduction)
            float partialSum = 0.0f;
            for (int i = localId; i < nEmbdHead; i += localSize) {
                float val = k.get(headOffset + i);
                partialSum += val * val;
            }

            localSum[localId] = partialSum;
            context.localBarrier();

            // Parallel reduction
            for (int stride = localSize / 2; stride > 0; stride >>= 1) {
                if (localId < stride) {
                    localSum[localId] += localSum[localId + stride];
                }
                context.localBarrier();
            }

            // Compute normalization factor
            float ss = localSum[0];
            ss = ss / nEmbdHead + rmsNormEps;
            ss = 1.0f / TornadoMath.sqrt(ss);

            context.localBarrier();

            // Step 2: Apply normalization with weights (in-place)
            for (int i = localId; i < nEmbdHead; i += localSize) {
                float normalized = ss * k.get(headOffset + i);
                k.set(headOffset + i, kWeights.get(i) * normalized);
            }
        }
    }
}
// @formatter:on
