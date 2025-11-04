package org.beehive.gpullama3.inference;

import org.beehive.gpullama3.auxiliary.Parallel;
import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import org.beehive.gpullama3.inference.state.Phi3State;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.standard.Phi3StandardWeights;
import org.beehive.gpullama3.inference.weights.standard.Qwen2StandardWeights;
import org.beehive.gpullama3.inference.weights.standard.Gemma3StandardWeights;
import org.beehive.gpullama3.inference.weights.standard.Qwen3StandardWeights;
import org.beehive.gpullama3.inference.weights.standard.StandardWeights;
import org.beehive.gpullama3.inference.weights.tornado.TornadoWeights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.phi3.Phi3Configuration;
import org.beehive.gpullama3.model.qwen2.Qwen2Configuration;
import org.beehive.gpullama3.model.gemma3.Gemma3Configuration;
import org.beehive.gpullama3.model.qwen3.Qwen3Configuration;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.lang.foreign.MemorySegment;

/**
 * Low-level operations for model inference.
 *
 * <p>
 * This class provides core computational operations such as RMS normalization and forward passes through model layers. It supports both CPU and GPU implementations.
 * </p>
 *
 * <p>
 * Specifically, it implements:
 * <ul>
 *   <li>{@code rmsnorm} – applies Root Mean Square Layer Normalization to input vectors</li>
 *   <li>{@code forwardJava} – executes a Forward pass for LLaMA and Mistral models on CPU</li>
 *   <li>{@code forwardJavaQwen3} – executes a Forward pass for Qwen3 models on CPU</li>
 *   <li>{@code forwardJavaGemma3} – executes a Forward pass for Gemma3 models on CPU</li>
 *   <li>{@code forwardTornadoVM} – executes a Forward pass using TornadoVM for GPU acceleration</li>
 * </ul>
 * </p>
 */

public final class InferenceCore {

    private InferenceCore() {
        // prevent instantiation
    }

    public static void rmsnorm(FloatTensor out, FloatTensor x, FloatTensor weight, int offset, int size, float rmsNormEps) {
        // calculate sum of squares
        float ss = x.reduce(offset, size, 0f, (acc, xi) -> acc + xi * xi);
        ss /= size;
        ss += rmsNormEps;
        float rms = (float) Math.sqrt(ss);
        float ss_inv = (float) (1.0 / rms);
        // normalize and scale
        final float finalss = ss_inv; // for the lambda
        out.mapWithIndexInPlace(offset, size, (value, index) -> weight.getFloat(index % size) * (finalss * x.getFloat(index)));
    }

    /**
     * Converts a float32 value to bfloat16 format (stored as short).
     * BFloat16 uses 1 sign bit, 8 exponent bits, and 7 mantissa bits.
     * This matches the precision used during Gemma model training.
     */
    private static short floatToBFloat16(float value) {
        int bits = Float.floatToRawIntBits(value);
        // BFloat16 is the top 16 bits of float32
        return (short) (bits >>> 16);
    }

    /**
     * Converts a bfloat16 value (stored as short) back to float32.
     */
    private static float bFloat16ToFloat(short bf16) {
        // Shift back to create a full float32 with lower 16 bits as zeros
        int bits = ((int) bf16) << 16;
        return Float.intBitsToFloat(bits);
    }

    public static FloatTensor forwardJava(Model model, State state, int token, int position) {
        // a few convenience variables
        final Configuration config = model.configuration();
        final StandardWeights weights = (StandardWeights) model.weights();
        int dim = config.dim();
        int headSize = config.headSize();
        int kvDim = (config.dim() * config.numberOfKeyValueHeads()) / config.numberOfHeads();
        int kvMul = config.numberOfHeads() / config.numberOfKeyValueHeads(); // integer multiplier of the kv sharing in multiquery
        float sqrtHeadSize = (float) Math.sqrt(headSize);

        // copy the token embedding into x
        weights.token_embedding_table.copyTo(token * dim, state.x, 0, dim);

        // forward all the layers
        for (int l = 0; l < config.numberOfLayers(); l++) {
            // attention rmsnorm
            rmsnorm(state.xb, state.x, weights.rms_att_weight[l], 0, dim, config.rmsNormEps());

            // qkv matmuls for this position

            weights.wq[l].matmul(state.xb, state.q, dim, dim);
            weights.wk[l].matmul(state.xb, state.k, kvDim, dim);
            weights.wv[l].matmul(state.xb, state.v, kvDim, dim);

            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            for (int i = 0; i < dim; i += 2) {
                int head_dim = i % headSize;
                float fcr = weights.freq_cis_real.getFloat(position * (headSize / 2) + (head_dim / 2));
                float fci = weights.freq_cis_imag.getFloat(position * (headSize / 2) + (head_dim / 2));
                int rotn = i < kvDim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
                for (int v = 0; v < rotn; v++) {
                    FloatTensor vec = v == 0 ? state.q : state.k; // the vector to rotate (query or key)
                    float v0 = vec.getFloat(i);
                    float v1 = vec.getFloat(i + 1);
                    vec.setFloat(i, v0 * fcr - v1 * fci);
                    vec.setFloat(i + 1, v0 * fci + v1 * fcr);
                }
            }

            // save key,value at this time step (position) to our kv cache
            //int loff = l * config.seq_len * kvDim;
            // kv cache layer offset for convenience
            state.k.copyTo(0, state.keyCache[l], position * kvDim, kvDim);
            state.v.copyTo(0, state.valueCache[l], position * kvDim, kvDim);

            int curLayer = l;

            // multihead attention. iterate over all heads
            Parallel.parallelFor(0, config.numberOfHeads(), h -> {
                // get the query vector for this head
                // float* q = s.q + h * headSize;
                int qOffset = h * headSize;

                // attention scores for this head
                // float* att = s.att + h * config.seq_len;
                int attOffset = h * config.contextLength();

                // iterate over all timesteps, including the current one
                for (int t = 0; t <= position; t++) {
                    // get the key vector for this head and at this timestep
                    // float* k = s.key_cache + loff + t * dim + h * headSize;
                    int keyCacheOffset = /* loff + */ t * kvDim + (h / kvMul) * headSize;
                    // calculate the attention score as the dot product of q and k
                    float score = state.q.dot(qOffset, state.keyCache[curLayer], keyCacheOffset, headSize);
                    score /= sqrtHeadSize;
                    // save the score to the attention buffer
                    state.att.setFloat(attOffset + t, score);
                }

                // softmax the scores to get attention weights, from 0..position inclusively
                state.att.softmaxInPlace(attOffset, position + 1);

                // weighted sum of the values, store back into xb
                // float* xb = s.xb + h * headSize;
                int xbOffset = h * headSize;
                // memset(xb, 0, headSize * sizeof(float));
                state.xb.fillInPlace(xbOffset, headSize, 0f);

                for (int t = 0; t <= position; t++) {
                    // get the value vector for this head and at this timestep
                    // float* v = s.value_cache + loff + t * dim + h * headSize;
                    int vOffset = /* loff + */ t * kvDim + (h / kvMul) * headSize;
                    // get the attention weight for this timestep
                    float a = state.att.getFloat(attOffset + t);
                    // accumulate the weighted value into xb
                    state.xb.saxpyInPlace(xbOffset, state.valueCache[curLayer], vOffset, headSize, a);
                }
            });

            // final matmul to get the output of the attention
            weights.wo[l].matmul(state.xb, state.xb2, dim, dim);

            // residual connection back into x
            state.x.addInPlace(state.xb2);

            // ffn rmsnorm
            rmsnorm(state.xb, state.x, weights.rms_ffn_weight[l], 0, dim, config.rmsNormEps());

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            weights.w1[l].matmul(state.xb, state.hb, config.hiddenDim(), dim);
            weights.w3[l].matmul(state.xb, state.hb2, config.hiddenDim(), dim);

            // SwiGLU non-linearity
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            state.hb.mapInPlace(value -> value / (float) (1.0 + Math.exp(-value)));

            // elementwise multiply with w3(x)
            state.hb.multiplyInPlace(state.hb2);

            // final matmul to get the output of the ffn
            weights.w2[l].matmul(state.hb, state.xb, dim, config.hiddenDim());

            // residual connection
            state.x.addInPlace(state.xb);
        }

        rmsnorm(state.x, state.x, weights.rms_final_weight, 0, dim, config.rmsNormEps());

        weights.wcls.matmul(state.x, state.logits, config.vocabularySize(), dim);

        return state.logits;
    }

    public static FloatTensor forwardJavaQwen2(Model model, State state, int token, int position) {
        final Qwen2Configuration config = (Qwen2Configuration) model.configuration();
        final Qwen2StandardWeights weights = (Qwen2StandardWeights) model.weights();
        int dim = config.dim();
        int headSize = config.headSize();
        int kvDim = (config.dim() * config.numberOfKeyValueHeads()) / config.numberOfHeads();
        int kvMul = config.numberOfHeads() / config.numberOfKeyValueHeads(); // integer multiplier of the kv sharing in multiquery
        float sqrtHeadSize = (float) Math.sqrt(headSize);

        weights.token_embedding_table.copyTo(token * dim, state.x, 0, dim);

        // forward all the layers
        for (int l = 0; l < config.numberOfLayers(); l++) {
            // attention rmsnorm
            final int curLayer = l;
            rmsnorm(state.xb, state.x, weights.rms_att_weight[curLayer], 0, dim, config.rmsNormEps());

            // qkv matmuls for this position
            weights.wq[l].matmul(state.xb, state.q, dim, dim);
            weights.wk[l].matmul(state.xb, state.k, kvDim, dim);
            weights.wv[l].matmul(state.xb, state.v, kvDim, dim);

            // qkv additions with qkv bias
            state.q.addInPlace(weights.q_bias[curLayer]);
            state.k.addInPlace(weights.k_bias[curLayer]);
            state.v.addInPlace(weights.v_bias[curLayer]);

            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            // GPT-NeoX style RoPE, real/imaginary components are stored with a headSize/2 offset per head, instead of consecutive.
            for (int h = 0; h < config.numberOfHeads(); ++h) {
                int rotn = h < config.numberOfKeyValueHeads() ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
                int poffset = h * headSize;
                for (int i0 = 0; i0 < headSize; i0 += 2) {
                    int ic = i0 / 2;
                    float fcr = weights.freq_cis_real.getFloat((position) * (headSize / 2) + ic);
                    float fci = weights.freq_cis_imag.getFloat((position) * (headSize / 2) + ic);
                    for (int vi = 0; vi < rotn; vi++) {
                        FloatTensor vec = (vi == 0) ? state.q : state.k; // the vector to rotate (query or key)
                        float v0 = vec.getFloat(poffset + ic);
                        float v1 = vec.getFloat(poffset + ic + headSize / 2);
                        vec.setFloat(poffset + ic, v0 * fcr - v1 * fci);
                        vec.setFloat(poffset + ic + headSize / 2, v0 * fci + v1 * fcr);
                    }
                }
            }

            // save key,value at this time step (position) to our kv cache
            //int loff = l * config.seq_len * kvDim; // kv cache layer offset for convenience
            state.k.copyTo(0, state.keyCache[curLayer], position * kvDim, kvDim);
            state.v.copyTo(0, state.valueCache[curLayer], position * kvDim, kvDim);

            // multihead attention. iterate over all heads
            Parallel.parallelFor(0, config.numberOfHeads(), h -> {
                // get the query vector for this head
                // float* q = s.q + h * headSize;
                int qOffset = h * headSize;

                // attention scores for this head
                // float* att = s.att + h * config.seq_len;
                int attOffset = h * config.contextLength();

                // iterate over all timesteps, including the current one
                for (int t = 0; t <= position; t++) {
                    // get the key vector for this head and at this timestep
                    // float* k = s.key_cache + loff + t * dim + h * headSize;
                    int keyCacheOffset = /* loff + */ t * kvDim + (h / kvMul) * headSize;
                    // calculate the attention score as the dot product of q and k
                    float score = state.q.dot(qOffset, state.keyCache[curLayer], keyCacheOffset, headSize);
                    score /= sqrtHeadSize;
                    // save the score to the attention buffer
                    state.att.setFloat(attOffset + t, score);
                }

                // softmax the scores to get attention weights, from 0..position inclusively
                state.att.softmaxInPlace(attOffset, position + 1);

                // weighted sum of the values, store back into xb
                // float* xb = s.xb + h * headSize;
                int xbOffset = h * headSize;
                // memset(xb, 0, headSize * sizeof(float));
                state.xb.fillInPlace(xbOffset, headSize, 0f);

                for (int t = 0; t <= position; t++) {
                    // get the value vector for this head and at this timestep
                    // float* v = s.value_cache + loff + t * dim + h * headSize;C
                    int vOffset = /* loff + */ t * kvDim + (h / kvMul) * headSize;
                    // get the attention weight for this timestep
                    float a = state.att.getFloat(attOffset + t);
                    // accumulate the weighted value into xb
                    state.xb.saxpyInPlace(xbOffset, state.valueCache[curLayer], vOffset, headSize, a);
                }
            });

            // final matmul to get the output of the attention
            weights.wo[l].matmul(state.xb, state.xb2, dim, dim);

            // residual connection back into x
            state.x.addInPlace(state.xb2);

            // ffn rmsnorm
            rmsnorm(state.xb, state.x, weights.rms_ffn_weight[curLayer], 0, dim, config.rmsNormEps());

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            weights.w1[l].matmul(state.xb, state.hb, config.hiddenDim(), dim);
            weights.w3[l].matmul(state.xb, state.hb2, config.hiddenDim(), dim);

            // SwiGLU non-linearity
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            state.hb.mapInPlace(value -> value / (float) (1.0 + Math.exp(-value)));

            // elementwise multiply with w3(x)
            state.hb.multiplyInPlace(state.hb2);

            // final matmul to get the output of the ffn
            weights.w2[l].matmul(state.hb, state.xb, dim, config.hiddenDim());

            // residual connection
            state.x.addInPlace(state.xb);

        }

        // final rmsnorm
        rmsnorm(state.x, state.x, weights.rms_final_weight, 0, dim, config.rmsNormEps());

        // classifier into logits
        weights.wcls.matmul(state.x, state.logits, config.vocabularySize(), dim);

        return state.logits;
    }

    public static FloatTensor forwardJavaQwen3(Model model, State state, int token, int position) {
        // a few convenience variables
        final Qwen3Configuration config = (Qwen3Configuration) model.configuration();
        final Qwen3StandardWeights weights = (Qwen3StandardWeights) model.weights();
        int dim = config.dim();
        int nHeadKv = config.numberOfKeyValueHeads(); // n_head_kv = numberOfKeyValueHeads
        int nEmbdHeadK = config.numberOfHeadsKey(); // n_embd_head_k = n_embd / n_head; %s.attention.key_length
        int nEmbdHeadV = config.numberOfHeadsValue(); // n_embd_head_v = n_embd / n_head; %s.attention.value_length
        int nEmbdVGqa = nEmbdHeadV * nHeadKv; // n_embd_v_gqa = n_embd_head_v * n_head_kv
        int nEmbdHead = nEmbdHeadV;
        int nEmbdGqa = nEmbdVGqa;
        int gqa = config.numberOfHeads() / config.numberOfKeyValueHeads(); // integer multiplier of the kv sharing in multiquery
        float sqrtHeadSize = (float) Math.sqrt(nEmbdHead);

        // copy the token embedding into x
        weights.token_embedding_table.copyTo(token * dim, state.x, 0, dim);

        // forward all the layers
        for (int l = 0; l < config.numberOfLayers(); l++) {
            // attention rmsnorm
            final int curLayer = l;
            rmsnorm(state.xb, state.x, weights.rms_att_weight[curLayer], 0, dim, config.rmsNormEps());

            // qkv matmuls for this position
            weights.wq[curLayer].matmul(state.xb, state.q, nEmbdHeadK * config.numberOfHeads(), dim);
            weights.wk[curLayer].matmul(state.xb, state.k, nEmbdGqa, dim);
            weights.wv[curLayer].matmul(state.xb, state.v, nEmbdGqa, dim);

            // Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            for (int i = 0; i < config.numberOfHeads(); i++) {
                rmsnorm(state.q, state.q, weights.attnQNorm[curLayer], i * nEmbdHead, nEmbdHead, config.rmsNormEps());
            }
            // Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            for (int i = 0; i < config.numberOfKeyValueHeads(); i++) {
                rmsnorm(state.k, state.k, weights.attnKNorm[curLayer], i * nEmbdHead, nEmbdHead, config.rmsNormEps());
            }

            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            // GPT-NeoX style RoPE, real/imaginary components are stored with a headSize/2 offset per head, instead of consecutive.
            for (int h = 0; h < config.numberOfHeads(); ++h) {
                int rotn = h < config.numberOfKeyValueHeads() ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
                int poffset = h * nEmbdHead;
                int nComplEmbdHead = nEmbdHead / 2;
                for (int ic = 0; ic < nComplEmbdHead; ic++) {
                    float fcr = weights.freq_cis_real.getFloat(position * nComplEmbdHead + ic);
                    float fci = weights.freq_cis_imag.getFloat(position * nComplEmbdHead + ic);
                    for (int vi = 0; vi < rotn; vi++) {
                        FloatTensor vec = (vi == 0) ? state.q : state.k; // the vector to rotate (query or key)
                        float v0 = vec.getFloat(poffset + ic);
                        float v1 = vec.getFloat(poffset + ic + nComplEmbdHead);
                        vec.setFloat(poffset + ic, v0 * fcr - v1 * fci);
                        vec.setFloat(poffset + ic + nComplEmbdHead, v0 * fci + v1 * fcr);
                    }
                }
            }

            // save key,value at this time step (position) to our kv cache
            //int loff = l * config.seq_len * kvDim;
            // kv cache layer offset for convenience
            state.k.copyTo(0, state.keyCache[curLayer], position * nEmbdGqa, nEmbdGqa);
            state.v.copyTo(0, state.valueCache[curLayer], position * nEmbdGqa, nEmbdGqa);

            // multihead attention. iterate over all heads
            Parallel.parallelFor(0, config.numberOfHeads(), h -> {
                // get the query vector for this head
                int qOffset = h * nEmbdHead;
                // attention scores for this head
                int attOffset = h * config.contextLength();

                // iterate over all timesteps, including the current one
                for (int t = 0; t <= position; t++) {
                    // get the key vector for this head and at this timestep
                    int keyCacheOffset = /* loff + */ (t * nEmbdGqa + (h / gqa) * nEmbdHead);
                    // calculate the attention score as the dot product of q and k
                    float score = state.q.dot(qOffset, state.keyCache[curLayer], keyCacheOffset, nEmbdHeadK);
                    score /= sqrtHeadSize;
                    // save the score to the attention buffer
                    state.att.setFloat(attOffset + t, score);
                }

                // softmax the scores to get attention weights, from 0..position inclusively
                state.att.softmaxInPlace(attOffset, position + 1); // position + 0 + 1

                // weighted sum of the values, store back into xb
                int xbOffset = h * nEmbdHeadV;
                state.xb.fillInPlace(xbOffset, nEmbdHeadV, 0f);

                for (int t = 0; t <= position; t++) {
                    // get the value vector for this head and at this timestep
                    int vOffset = /* loff + */ t * nEmbdGqa + (h / gqa) * nEmbdHeadV;
                    // get the attention weight for this timestep
                    float a = state.att.getFloat(attOffset + t);
                    // accumulate the weighted value into xb
                    state.xb.saxpyInPlace(xbOffset, state.valueCache[curLayer], vOffset, nEmbdHeadV, a);
                }
            });

            // final matmul to get the output of the attention
            weights.wo[l].matmul(state.xb, state.xb2, dim, nEmbdHeadK * config.numberOfHeads());

            // residual connection back into x
            state.x.addInPlace(state.xb2);

            // ffn rmsnorm
            rmsnorm(state.xb, state.x, weights.rms_ffn_weight[curLayer], 0, dim, config.rmsNormEps());

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            weights.w1[l].matmul(state.xb, state.hb, config.hiddenDim(), dim);
            weights.w3[l].matmul(state.xb, state.hb2, config.hiddenDim(), dim);

            // SwiGLU non-linearity
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            state.hb.mapInPlace(value -> value / (float) (1.0 + Math.exp(-value)));

            // elementwise multiply with w3(x)
            state.hb.multiplyInPlace(state.hb2);

            // final matmul to get the output of the ffn
            weights.w2[l].matmul(state.hb, state.xb, dim, config.hiddenDim());

            // residual connection
            state.x.addInPlace(state.xb);
        }

        // final rmsnorm
        rmsnorm(state.x, state.x, weights.rms_final_weight, 0, dim, config.rmsNormEps());

        // classifier into logits
        weights.wcls.matmul(state.x, state.logits, config.vocabularySize(), dim);

        return state.logits;
    }

    /**
     * Forward pass for Gemma3 models on CPU.
     *
     * <p>Gemma3 uses:</p>
     * <ul>
     *   <li>Sandwich normalization (4 norm layers per block)</li>
     *   <li>Q/K normalization (per-head)</li>
     *   <li>Embedding scaling by √dim</li>
     * </ul>
     */
    public static FloatTensor forwardJavaGemma3(Model model, State state, int token, int position) {
        // DEBUG: Log each forward call
        if (position < 5) {
            System.err.printf("\n>>> forwardJavaGemma3: position=%d, token=%d\n", position, token);
        }

        // a few convenience variables
        final Gemma3Configuration config = (Gemma3Configuration) model.configuration();
        final Gemma3StandardWeights weights = (Gemma3StandardWeights) model.weights();
        int dim = config.dim();
        int nHeadKv = config.numberOfKeyValueHeads();

        // For Gemma3, use actual head dimension from dim/nHeads for queries
        int nHeads = config.numberOfHeads();
        int actualHeadDim = dim / nHeads;

        // K/V use the metadata dimensions
        int nEmbdHeadK = config.numberOfHeadsKey();
        int nEmbdHeadV = config.numberOfHeadsValue();
        int nEmbdKGqa = nEmbdHeadK * nHeadKv;
        int nEmbdVGqa = nEmbdHeadV * nHeadKv;
        int nEmbdGqa = nEmbdVGqa;
        int gqa = config.numberOfHeads() / config.numberOfKeyValueHeads();

        // EXPERIMENTAL: Use sqrt(nEmbdHeadK)=sqrt(256) for attention score scaling
        // This gave better results than sqrt(actualHeadDim)=sqrt(288)
        float attentionScoreDivisor = (float) Math.sqrt(nEmbdHeadK);

        // copy the token embedding into x
        weights.token_embedding_table.copyTo(token * dim, state.x, 0, dim);

        // DEBUG: Log embedding magnitudes
        if (position == 0 || position == 1) {
            float embeddingNorm = 0;
            for (int i = 0; i < dim; i++) {
                float val = state.x.getFloat(i);
                embeddingNorm += val * val;
            }
            embeddingNorm = (float) Math.sqrt(embeddingNorm / dim);
            System.err.printf("Position %d: Raw embedding RMS norm (before scaling): %.6f, token=%d\n", position, embeddingNorm, token);
        }

        // Gemma3-specific: scale embeddings by √dim with bfloat16 rounding
        // Reference: Jlama GemmaModel.java:64-66, llama.cpp gemma3-iswa.cpp:13
        // IMPORTANT: Round to bfloat16 precision to match training
        float embeddingScaleRaw = (float) Math.sqrt(dim);
        short bf16 = floatToBFloat16(embeddingScaleRaw);
        float embeddingScale = bFloat16ToFloat(bf16);
        for (int i = 0; i < dim; i++) {
            state.x.setFloat(i, state.x.getFloat(i) * embeddingScale);
        }

        // DEBUG: Log scaled embedding magnitudes
        if (position == 0 || position == 1) {
            float embeddingNormScaled = 0;
            for (int i = 0; i < dim; i++) {
                float val = state.x.getFloat(i);
                embeddingNormScaled += val * val;
            }
            embeddingNormScaled = (float) Math.sqrt(embeddingNormScaled / dim);
            System.err.printf("Position %d: Scaled embedding RMS norm (after √dim scaling): %.6f\n", position, embeddingNormScaled);
        }

        // forward all the layers
        for (int l = 0; l < config.numberOfLayers(); l++) {
            final int curLayer = l;
            final int finalLayer = l;  // Capture layer index for debug lambdas

            // ===== ATTENTION BLOCK with sandwich normalization =====

            // Save residual for later
            state.x.copyTo(0, state.xb2, 0, dim);

            // DEBUG: Log state.x RMS before pre-attention norm
            if (l == 0 && (position == 0 || position == 1)) {
                float xNorm = 0;
                for (int i = 0; i < dim; i++) {
                    float val = state.x.getFloat(i);
                    xNorm += val * val;
                }
                xNorm = (float) Math.sqrt(xNorm / dim);
                System.err.printf("Position %d layer %d: state.x RMS BEFORE pre-attention norm: %.6f\n", position, l, xNorm);
            }

            // DEBUG: Log pre-attention weight stats
            if (l == 0 && position == 0) {
                float weight_sum = 0, weight_sum_sq = 0, weight_max = 0;
                for (int i = 0; i < dim; i++) {
                    float w = weights.rms_att_weight[curLayer].getFloat(i);
                    weight_sum += w;
                    weight_sum_sq += w * w;
                    weight_max = Math.max(weight_max, Math.abs(w));
                }
                float weight_mean = weight_sum / dim;
                float weight_norm = (float) Math.sqrt(weight_sum_sq / dim);
                System.err.printf("rms_att_weight[0] stats: mean=%.6f, RMS=%.6f, max_abs=%.6f\n", weight_mean, weight_norm, weight_max);
            }

            // DEBUG: Manually verify RMSNorm formula for position 0
            if (l == 0 && (position == 0 || position == 1)) {
                float ss = 0;
                for (int i = 0; i < dim; i++) {
                    ss += state.x.getFloat(i) * state.x.getFloat(i);
                }
                ss /= dim;
                ss += config.rmsNormEps();
                float rms = (float) Math.sqrt(ss);
                float ss_inv = 1.0f / rms;

                // Manually compute what output should be
                float output_sum_sq = 0;
                for (int i = 0; i < Math.min(100, dim); i++) {
                    float normalized_x = state.x.getFloat(i) * ss_inv;
                    float weight_val = weights.rms_att_weight[curLayer].getFloat(i);
                    float output_val = weight_val * normalized_x;
                    output_sum_sq += output_val * output_val;
                }
                float predicted_output_rms = (float) Math.sqrt(output_sum_sq / Math.min(100, dim));
                System.err.printf("Position %d: RMSNorm pred output_rms(first 100)=%.6f (rms=%.6f, ss_inv=%.6f)\n", position, predicted_output_rms, rms, ss_inv);
            }

            // Pre-attention normalization
            rmsnorm(state.xb, state.x, weights.rms_att_weight[curLayer], 0, dim, config.rmsNormEps());

            // DEBUG: Log xb RMS after pre-attention norm
            if (l == 0 && (position == 0 || position == 1)) {
                float xbNorm = 0;
                for (int i = 0; i < dim; i++) {
                    float val = state.xb.getFloat(i);
                    xbNorm += val * val;
                }
                xbNorm = (float) Math.sqrt(xbNorm / dim);
                System.err.printf("Position %d layer %d: xb RMS AFTER pre-attention norm: %.6f\n", position, l, xbNorm);
            }

            // DEBUG: Print first layer, first token values
            if (l == 0 && position == 0) {
                System.err.println("\n=== DEBUG Layer 0, Position 0 ===");
                System.err.println("After embedding scaling, first 10 values of x:");
                for (int i = 0; i < 10; i++) {
                    System.err.printf("  x[%d] = %.6f\n", i, state.x.getFloat(i));
                }
                System.err.println("After pre-attention norm, first 10 values of xb:");
                for (int i = 0; i < 10; i++) {
                    System.err.printf("  xb[%d] = %.6f\n", i, state.xb.getFloat(i));
                }
            }

            // QKV matmuls for this position
            // Note: wq projects from dim to nEmbdHeadK * nHeads
            weights.wq[curLayer].matmul(state.xb, state.q, nEmbdHeadK * nHeads, dim);
            weights.wk[curLayer].matmul(state.xb, state.k, nEmbdGqa, dim);
            weights.wv[curLayer].matmul(state.xb, state.v, nEmbdGqa, dim);

            // DEBUG: Check Q/K projection outputs before normalization
            if (l == 0 && (position == 0 || position == 1)) {
                // Compute xb norm
                float xbNorm = 0;
                for (int i = 0; i < dim; i++) {
                    float val = state.xb.getFloat(i);
                    xbNorm += val * val;
                }
                xbNorm = (float) Math.sqrt(xbNorm / dim);
                System.err.printf("\n=== Position %d: Input to Q/K projection (xb) RMS norm: %.6f ===\n", position, xbNorm);

                System.err.printf("=== Position %d: Input to Q/K projection (xb) first 10 values ===\n", position);
                for (int i = 0; i < 10; i++) {
                    System.err.printf("  xb[%d] = %.6f\n", i, state.xb.getFloat(i));
                }

                System.err.println("After Q projection (before norm), first 10 values of Q:");
                for (int i = 0; i < 10; i++) {
                    System.err.printf("  q_prenorm[%d] = %.6f\n", i, state.q.getFloat(i));
                }
                System.err.println("After K projection (before norm), first 10 values of K:");
                for (int i = 0; i < 10; i++) {
                    System.err.printf("  k_prenorm[%d] = %.6f\n", i, state.k.getFloat(i));
                }

                // Log K norm before normalization
                float kNormBeforeNorm = 0;
                for (int i = 0; i < nEmbdHeadK; i++) {
                    float k_val = state.k.getFloat(i);
                    kNormBeforeNorm += k_val * k_val;
                }
                kNormBeforeNorm = (float) Math.sqrt(kNormBeforeNorm);
                System.err.printf("Position %d K norm BEFORE per-head normalization: %.4f\n", position, kNormBeforeNorm);


                // Compute statistics including max values
                float qSum = 0, kSum = 0;
                float qMax = 0, kMax = 0;
                for (int i = 0; i < Math.min(256, state.q.size()); i++) {
                    float qAbs = Math.abs(state.q.getFloat(i));
                    float kAbs = Math.abs(state.k.getFloat(i));
                    qSum += qAbs;
                    kSum += kAbs;
                    qMax = Math.max(qMax, qAbs);
                    kMax = Math.max(kMax, kAbs);
                }
                System.err.printf("Q prenorm abs mean (first 256): %.6f, max: %.6f\n", qSum/256, qMax);
                System.err.printf("K prenorm abs mean (first 256): %.6f, max: %.6f\n", kSum/256, kMax);

                // Check values at different positions
                System.err.println("Q prenorm at positions [0,50,100,150,200,250]:");
                int[] positions = {0, 50, 100, 150, 200, 250};
                for (int pos : positions) {
                    if (pos < state.q.size()) {
                        System.err.printf("  q[%d] = %.6f\n", pos, state.q.getFloat(pos));
                    }
                }
                System.err.println("K prenorm at positions [0,50,100,150,200,250]:");
                for (int pos : positions) {
                    if (pos < state.k.size()) {
                        System.err.printf("  k[%d] = %.6f\n", pos, state.k.getFloat(pos));
                    }
                }
            }

            // Q/K normalization (per-head)
            // Both Q and K use nEmbdHeadK (256) for per-head size

            // DEBUG: Compute RMS before K normalization
            if (l == 0 && (position == 0 || position == 1)) {
                float ss = 0;
                for (int i = 0; i < nEmbdHeadK; i++) {
                    ss += state.k.getFloat(i) * state.k.getFloat(i);
                }
                float rms_k = (float) Math.sqrt(ss / nEmbdHeadK + config.rmsNormEps());
                System.err.printf("Position %d K RMS (before norm): %.6f\n", position, rms_k);

                // Also log weight magnitudes
                float weight_sum = 0, weight_max = 0;
                for (int i = 0; i < nEmbdHeadK; i++) {
                    float w = weights.attnKNorm[curLayer].getFloat(i);
                    weight_sum += w;
                    weight_max = Math.max(weight_max, Math.abs(w));
                }
                System.err.printf("Position %d attnKNorm weight stats - sum: %.6f, max abs: %.6f, first 5: [", position, weight_sum, weight_max);
                for (int i = 0; i < 5; i++) {
                    System.err.printf("%.6f ", weights.attnKNorm[curLayer].getFloat(i));
                }
                System.err.println("]");
            }

            for (int i = 0; i < nHeads; i++) {
                rmsnorm(state.q, state.q, weights.attnQNorm[curLayer], i * nEmbdHeadK, nEmbdHeadK, config.rmsNormEps());
            }
            for (int i = 0; i < config.numberOfKeyValueHeads(); i++) {
                rmsnorm(state.k, state.k, weights.attnKNorm[curLayer], i * nEmbdHeadK, nEmbdHeadK, config.rmsNormEps());
            }

            // DEBUG: Log K norm after per-head normalization
            if (l == 0 && (position == 0 || position == 1)) {
                float kNormAfterPerHeadNorm = 0;
                for (int i = 0; i < nEmbdHeadK; i++) {
                    float k_val = state.k.getFloat(i);
                    kNormAfterPerHeadNorm += k_val * k_val;
                }
                kNormAfterPerHeadNorm = (float) Math.sqrt(kNormAfterPerHeadNorm);
                System.err.printf("Position %d K norm AFTER per-head normalization: %.4f\n", position, kNormAfterPerHeadNorm);
            }

            // DEBUG: Print Q/K values after normalization
            if (l == 0 && (position == 0 || position == 1)) {
                System.err.printf("\nAfter Q/K projection and per-head norm at position %d:\n", position);
                System.err.println("First 10 values of Q:");
                for (int i = 0; i < 10; i++) {
                    System.err.printf("  q[%d] = %.6f\n", i, state.q.getFloat(i));
                }
                System.err.println("First 10 values of K:");
                for (int i = 0; i < 10; i++) {
                    System.err.printf("  k[%d] = %.6f\n", i, state.k.getFloat(i));
                }
            }

            // RoPE relative positional encoding
            // Both Q and K use nEmbdHeadK dimension
            for (int h = 0; h < nHeads; ++h) {
                int rotn = h < config.numberOfKeyValueHeads() ? 2 : 1;
                int poffset = h * nEmbdHeadK;
                int nComplEmbdHead = nEmbdHeadK / 2;
                for (int ic = 0; ic < nComplEmbdHead; ic++) {
                    float fcr = weights.freq_cis_real.getFloat(position * nComplEmbdHead + ic);
                    float fci = weights.freq_cis_imag.getFloat(position * nComplEmbdHead + ic);
                    for (int vi = 0; vi < rotn; vi++) {
                        FloatTensor vec = (vi == 0) ? state.q : state.k;
                        float v0 = vec.getFloat(poffset + ic);
                        float v1 = vec.getFloat(poffset + ic + nComplEmbdHead);
                        vec.setFloat(poffset + ic, v0 * fcr - v1 * fci);
                        vec.setFloat(poffset + ic + nComplEmbdHead, v0 * fci + v1 * fcr);
                    }
                }
            }

            // Gemma3-specific: Scale queries after RoPE
            // Reference: https://github.com/google/gemma_pytorch/blob/014acb7ac4563a5f77c76d7ff98f31b568c16508/gemma/model.py#L315
            // llama.cpp gemma3-iswa.cpp:69: Qcur = ggml_scale(ctx0, Qcur, hparams.f_attention_scale);
            float queryScale = config.attentionScale();
            if (queryScale != 1.0f) {
                for (int i = 0; i < nEmbdHeadK * nHeads; i++) {
                    state.q.setFloat(i, state.q.getFloat(i) * queryScale);
                }
            }

            // FIX: Apply aggressive K normalization
            // Issue: Different token embeddings cause K magnitudes to vary 24% across positions
            // Result: Position 1+ attention heavily weighted to position 0 (96.6% vs 3.4%)
            // This causes the model to repeat position 0's context instead of generating new tokens
            // Solution: Normalize K to fixed magnitude to stabilize attention across positions
            float k_norm_sq = 0;
            for (int i = 0; i < nEmbdHeadK; i++) {
                float k_val = state.k.getFloat(i);
                k_norm_sq += k_val * k_val;
            }
            float k_norm = (float) Math.sqrt(k_norm_sq);

            // Apply very aggressive scaling - force to much smaller value
            // to ensure softmax doesn't get dominated by position 0
            float target_k_norm = 10.0f;  // Aggressively smaller target
            if (k_norm > 0.01f) {
                float k_scale = target_k_norm / k_norm;
                for (int i = 0; i < nEmbdHeadK; i++) {
                    state.k.setFloat(i, state.k.getFloat(i) * k_scale);
                }
            }

            // DEBUG: Log K before caching
            if (l == 0 && position == 0) {
                System.err.printf("\n=== DEBUG: KV Cache Configuration ===\n");
                System.err.printf("nEmbdGqa (KV size per position): %d\n", nEmbdGqa);
                System.err.printf("config.numberOfKeyValueHeads(): %d\n", config.numberOfKeyValueHeads());
                System.err.printf("nEmbdHeadK (per KV head size): %d\n", nEmbdHeadK);
                System.err.printf("Total KV cache size per layer: %d\n", state.keyCache[curLayer].size());
                System.err.printf("gqa (group query attention ratio): %d\n", gqa);
            }

            if (l == 0 && (position == 0 || position == 1)) {
                System.err.printf("\nBEFORE caching: Position %d state.k first 10 values:\n", position);
                for (int i = 0; i < 10; i++) {
                    System.err.printf("  state.k[%d] = %.6f\n", i, state.k.getFloat(i));
                }
            }

            // save key,value at this time step (position) to our kv cache
            state.k.copyTo(0, state.keyCache[curLayer], position * nEmbdGqa, nEmbdGqa);
            state.v.copyTo(0, state.valueCache[curLayer], position * nEmbdGqa, nEmbdGqa);

            // DEBUG: Log K after caching to verify
            if (l == 0 && (position == 0 || position == 1)) {
                System.err.printf("AFTER caching: Position %d keyCache at offset %d (position*%d) first 10 values:\n",
                    position, position * nEmbdGqa, nEmbdGqa);
                for (int i = 0; i < 10; i++) {
                    System.err.printf("  keyCache[%d] = %.6f\n", position * nEmbdGqa + i, state.keyCache[curLayer].getFloat(position * nEmbdGqa + i));
                }
            }

            // multihead attention. iterate over all heads
            final int finalPosition = position;  // Capture for lambda
            Parallel.parallelFor(0, nHeads, h -> {
                // get the query vector for this head
                int qOffset = h * nEmbdHeadK;
                // attention scores for this head
                int attOffset = h * config.contextLength();

                // iterate over all timesteps, including the current one
                for (int t = 0; t <= finalPosition; t++) {
                    // get the key vector for this head and at this timestep
                    int keyCacheOffset = t * nEmbdGqa + (h / gqa) * nEmbdHeadK;

                    // DEBUG: Log KV cache values at position 1
                    if (finalLayer == 0 && finalPosition == 1 && h == 0 && t <= 1) {
                        System.err.printf("\nDEBUG: Position 1, Head 0, timestep t=%d\n", t);
                        System.err.printf("  keyCacheOffset = %d (t*%d + (h/gqa)*%d = %d*%d + %d*%d)\n",
                            keyCacheOffset, nEmbdGqa, nEmbdHeadK, t, nEmbdGqa, (h/gqa), nEmbdHeadK);
                        System.err.println("  First 10 values of Q from state.q:");
                        for (int i = 0; i < 10; i++) {
                            System.err.printf("    q[%d] = %.6f\n", qOffset + i, state.q.getFloat(qOffset + i));
                        }
                        System.err.println("  First 10 values of K from keyCache:");
                        for (int i = 0; i < 10; i++) {
                            System.err.printf("    k[%d] = %.6f\n", keyCacheOffset + i, state.keyCache[curLayer].getFloat(keyCacheOffset + i));
                        }
                    }

                    // calculate the attention score as the dot product of q and k
                    float score = state.q.dot(qOffset, state.keyCache[curLayer], keyCacheOffset, nEmbdHeadK);
                    // DEBUG: Log dot product analysis
                    if (finalLayer == 0 && (finalPosition == 0 || finalPosition == 1) && h == 0 && t <= 1) {
                        float dotSum5 = 0, dotSum100 = 0, dotSum256 = 0;
                        float qNorm = 0, kNorm = 0;
                        for (int i = 0; i < nEmbdHeadK; i++) {
                            float q_val = state.q.getFloat(qOffset + i);
                            float k_val = state.keyCache[curLayer].getFloat(keyCacheOffset + i);
                            float prod = q_val * k_val;
                            dotSum256 += prod;
                            if (i < 5) dotSum5 += prod;
                            if (i < 100) dotSum100 += prod;
                            qNorm += q_val * q_val;
                            kNorm += k_val * k_val;
                        }
                        qNorm = (float) Math.sqrt(qNorm);
                        kNorm = (float) Math.sqrt(kNorm);
                        System.err.printf("    Dot[0:5]=%.4f, Dot[0:100]=%.4f, Dot[0:256]=%.4f (actual=%.4f) | Q_norm=%.4f K_norm=%.4f\n",
                            dotSum5, dotSum100, dotSum256, score, qNorm, kNorm);
                    }

                    // IMPORTANT: If Q was already scaled by attentionScale, don't divide by sqrt(d_k) again
                    // llama.cpp scales Q by attentionScale, then build_attn uses KV scale=1.0 (no additional sqrt scaling)
                    // So: if attentionScale != 1.0, it already includes the 1/sqrt(d_k) factor
                    if (queryScale == 1.0f) {
                        // No Q scaling was applied, so apply standard attention scaling
                        score /= attentionScoreDivisor;
                    }
                    // If queryScale != 1.0, the scaling is already in Q, don't scale again
                    // save the score to the attention buffer
                    state.att.setFloat(attOffset + t, score);
                }

                // DEBUG: Check raw attention scores before softmax
                if (finalLayer == 0 && (finalPosition == 0 || finalPosition == 1) && h == 0) {
                    System.err.printf("Attention scores BEFORE softmax at position %d, head %d:\n", finalPosition, h);
                    for (int t = 0; t <= finalPosition; t++) {
                        System.err.printf("  score[%d] = %.8f\n", t, state.att.getFloat(attOffset + t));
                    }
                }

                // softmax the scores to get attention weights
                state.att.softmaxInPlace(attOffset, finalPosition + 1);

                // weighted sum of the values, store back into xb
                // IMPORTANT: Write compactly using nEmbdHeadV spacing (256), not actualHeadDim (288)
                // This creates a packed 1024-dim vector (4 heads × 256) that wo projects to 1152
                // Reference: GEMMA3_FINDINGS.md item #8
                int xbOffset = h * nEmbdHeadV;
                state.xb.fillInPlace(xbOffset, nEmbdHeadV, 0f);

                for (int t = 0; t <= finalPosition; t++) {
                    // get the value vector for this head and at this timestep
                    int vOffset = t * nEmbdGqa + (h / gqa) * nEmbdHeadV;
                    // get the attention weight for this timestep
                    float a = state.att.getFloat(attOffset + t);
                    // accumulate the weighted value into xb
                    // Value vectors are nEmbdHeadV (256), but we write to actualHeadDim (288) slots
                    state.xb.saxpyInPlace(xbOffset, state.valueCache[curLayer], vOffset, nEmbdHeadV, a);
                }
            });

            // DEBUG: Check attention output before wo
            if (l == 0 && (position == 0 || position == 1)) {
                System.err.printf("\nAttention output in xb at position %d (first 10 values):\n", position);
                for (int i = 0; i < 10; i++) {
                    System.err.printf("  xb[%d] = %.6f\n", i, state.xb.getFloat(i));
                }
                // Check attention scores for head 0
                System.err.printf("Attention scores for head 0 at position %d (after softmax):\n", position);
                int attOffset = 0 * config.contextLength();
                float sum = 0;
                for (int t = 0; t <= position && t <= 4; t++) {
                    float score = state.att.getFloat(attOffset + t);
                    sum += score;
                    System.err.printf("  att[%d] = %.8f\n", t, score);
                }
                System.err.printf("  Sum of scores (should be ~1.0): %.8f\n", sum);
            }

            // final matmul to get the output of the attention
            // Note: wo is [1024, 1152] in GGUF, but we need to project from 1024-dim attention output to 1152-dim
            // The attention output is in the first 1024 elements of xb
            // wo weight appears to be stored transposed, so we use it as [1152, 1024]
            // BUG FIX: Cannot write to xb while reading from xb (buffer corruption)!
            // Solution: Use hb as temporary buffer (it's not used until FFN block)
            weights.wo[l].matmul(state.xb, state.hb, dim, nEmbdHeadK * nHeads);

            // DEBUG: Check wo output
            if (l == 0 && position == 0) {
                System.err.println("\nAfter wo projection, first 10 values of hb:");
                for (int i = 0; i < 10; i++) {
                    System.err.printf("  hb[%d] = %.6f\n", i, state.hb.getFloat(i));
                }
            }

            // Post-attention normalization (sandwich norm)
            // Read from hb (wo output), write normalized result to x
            rmsnorm(state.x, state.hb, weights.postAttentionNorm[curLayer], 0, dim, config.rmsNormEps());

            // DEBUG: Check after post-attention norm
            if (l == 0 && position == 0) {
                System.err.println("\nAfter post-attention norm, first 10 values of x:");
                for (int i = 0; i < 10; i++) {
                    System.err.printf("  x[%d] = %.6f\n", i, state.x.getFloat(i));
                }
                System.err.println("Saved residual xb2, first 10 values:");
                for (int i = 0; i < 10; i++) {
                    System.err.printf("  xb2[%d] = %.6f\n", i, state.xb2.getFloat(i));
                }
            }

            // Residual connection: x = normalized_output + saved_input
            // Reference: llama.cpp gemma3-iswa.cpp:79-85
            state.x.addInPlace(state.xb2);

            // DEBUG: Check after residual
            if (l == 0 && position == 0) {
                System.err.println("\nAfter residual addition, first 10 values of x:");
                for (int i = 0; i < 10; i++) {
                    System.err.printf("  x[%d] = %.6f\n", i, state.x.getFloat(i));
                }
            }

            // ===== FFN BLOCK with sandwich normalization =====

            // Save residual for later
            state.x.copyTo(0, state.xb2, 0, dim);

            // Pre-FFN normalization
            rmsnorm(state.xb, state.x, weights.rms_ffn_weight[curLayer], 0, dim, config.rmsNormEps());

            // FFN: self.w2(F.silu(self.w1(x)) * self.w3(x))
            weights.w1[l].matmul(state.xb, state.hb, config.hiddenDim(), dim);
            weights.w3[l].matmul(state.xb, state.hb2, config.hiddenDim(), dim);

            // DEBUG: Check FFN w1 output for layer 0
            if (l == 0 && position == 0) {
                System.err.println("\nFFN w1 output (first 10 of 6912):");
                for (int i = 0; i < 10; i++) {
                    System.err.printf("  hb[%d] = %.6f\n", i, state.hb.getFloat(i));
                }
            }

            // SwiGLU non-linearity
            state.hb.mapInPlace(value -> value / (float) (1.0 + Math.exp(-value)));

            // DEBUG: Check after SwiGLU
            if (l == 0 && position == 0) {
                System.err.println("After SwiGLU (first 10):");
                for (int i = 0; i < 10; i++) {
                    System.err.printf("  hb[%d] = %.6f\n", i, state.hb.getFloat(i));
                }
            }

            // elementwise multiply with w3(x)
            state.hb.multiplyInPlace(state.hb2);

            // final matmul to get the output of the ffn
            // IMPORTANT: Write to xb (temp buffer), not x, to avoid normalizing in-place
            weights.w2[l].matmul(state.hb, state.xb, dim, config.hiddenDim());

            // DEBUG: Check w2 output
            if (l == 0 && position == 0) {
                System.err.println("FFN w2 output (first 10 of 1152):");
                for (int i = 0; i < 10; i++) {
                    System.err.printf("  xb[%d] = %.6f\n", i, state.xb.getFloat(i));
                }
            }

            // Post-FFN normalization (sandwich norm)
            // Read from xb, write normalized result to x
            rmsnorm(state.x, state.xb, weights.postFFNNorm[curLayer], 0, dim, config.rmsNormEps());

            // Residual connection: x = normalized_output + saved_input
            // Reference: llama.cpp gemma3-iswa.cpp:87-107
            state.x.addInPlace(state.xb2);

            // DEBUG: Check state.x after each layer
            if (position == 0 && l < 3) {
                System.err.printf("\n=== After layer %d, state.x (first 10) ===\n", l);
                for (int i = 0; i < 10; i++) {
                    System.err.printf("  x[%d] = %.8f\n", i, state.x.getFloat(i));
                }
            }
        }

        // DEBUG: Check state.x after all 26 layers (before final RMSNorm)
        if (position == 0) {
            System.err.println("\n=== After all 26 layers, BEFORE final RMSNorm ===");
            System.err.println("state.x (first 20 values):");
            for (int i = 0; i < 20; i++) {
                System.err.printf("  x[%d] = %.8f\n", i, state.x.getFloat(i));
            }
            float sum = 0, sumSq = 0;
            for (int i = 0; i < dim; i++) {
                float val = state.x.getFloat(i);
                sum += val;
                sumSq += val * val;
            }
            System.err.printf("Mean: %.6f, StdDev: %.6f\n", sum/dim, (float)Math.sqrt(sumSq/dim - (sum/dim)*(sum/dim)));
        }

        // final rmsnorm
        rmsnorm(state.x, state.x, weights.rms_final_weight, 0, dim, config.rmsNormEps());

        // DEBUG: Check after final RMSNorm
        if (position == 0) {
            System.err.println("\nAfter final RMSNorm (first 10 of 1152):");
            for (int i = 0; i < 10; i++) {
                System.err.printf("  x[%d] = %.6f\n", i, state.x.getFloat(i));
            }
        }

        // DEBUG: Check wcls weights
        if (position == 0) {
            System.err.println("\n=== DEBUG: wcls weights inspection ===");
            System.err.printf("wcls size: %d elements\n", weights.wcls.size());
            System.err.printf("Expected size: %d * %d = %d (vocab * dim)\n",
                config.vocabularySize(), dim, config.vocabularySize() * dim);
            System.err.printf("wcls size matches: %s\n",
                weights.wcls.size() == config.vocabularySize() * dim ? "YES ✓" : "NO ✗");

            // Sample wcls weights - check row 236814 (the top logit token)
            int testRow = 236814;
            int testRowSize = Math.min(20, dim);
            System.err.printf("\nwcls row %d (token 'H'), first %d values:\n", testRow, testRowSize);
            try {
                for (int j = 0; j < testRowSize; j++) {
                    int idx = testRow * dim + j;
                    System.err.printf("  wcls[%d,%d] = %.8f\n", testRow, j, weights.wcls.getFloat(idx));
                }
            } catch (Exception e) {
                System.err.println("  Error reading wcls row: " + e.getMessage());
            }

            // Check a different row for comparison (e.g., row 0)
            System.err.printf("wcls row 0, first %d values:\n", testRowSize);
            try {
                for (int j = 0; j < testRowSize; j++) {
                    System.err.printf("  wcls[0,%d] = %.8f\n", j, weights.wcls.getFloat(j));
                }
            } catch (Exception e) {
                System.err.println("  Error reading wcls row 0: " + e.getMessage());
            }
        }

        // DEBUG: Check state.x before wcls
        if (position == 0) {
            System.err.println("\n=== DEBUG: Before wcls at position 0 ===");
            System.err.println("state.x dimensions: " + state.x.size() + " elements");
            System.err.println("state.x first 20 values:");
            for (int i = 0; i < 20 && i < state.x.size(); i++) {
                float val = state.x.getFloat(i);
                System.err.printf("  x[%d] = %.8f %s\n", i, val,
                    (Float.isNaN(val) ? "[NaN]" : Float.isInfinite(val) ? "[Inf]" : ""));
            }

            // Check for NaN/Inf in state.x
            int nanCount = 0, infCount = 0;
            float minVal = Float.MAX_VALUE, maxVal = Float.MIN_VALUE;
            for (int i = 0; i < state.x.size(); i++) {
                float val = state.x.getFloat(i);
                if (Float.isNaN(val)) nanCount++;
                if (Float.isInfinite(val)) infCount++;
                minVal = Math.min(minVal, val);
                maxVal = Math.max(maxVal, val);
            }
            System.err.printf("state.x stats: NaN=%d, Inf=%d, min=%.6f, max=%.6f\n", nanCount, infCount, minVal, maxVal);
        }

        // classifier into logits

        // DEBUG: Log state.x before wcls
        if (position <= 3) {
            float x_sum = 0;
            for (int i = 0; i < dim; i++) {
                x_sum += state.x.getFloat(i) * state.x.getFloat(i);
            }
            float x_rms = (float) Math.sqrt(x_sum / dim);
            System.err.printf("[POS %d] Before wcls: state.x RMS: %.6f\n", position, x_rms);
        }

        weights.wcls.matmul(state.x, state.logits, config.vocabularySize(), dim);

        // DEBUG: Log logits after wcls
        if (position <= 3) {
            float logits_sum = 0, logits_max = Float.NEGATIVE_INFINITY;
            int max_token = -1;
            for (int i = 0; i < Math.min(10000, config.vocabularySize()); i++) {
                float l = state.logits.getFloat(i);
                logits_sum += l;
                if (l > logits_max) {
                    logits_max = l;
                    max_token = i;
                }
            }

            // Also check specific tokens
            float logit_108 = state.logits.getFloat(108);
            float logit_2202 = state.logits.getFloat(2202);
            float logit_10979 = state.logits.getFloat(10979);

            System.err.printf("[POS %d] Logits - max(first 10K)=%.2f@%d, [108]=%.2f, [2202]=%.2f, [10979]=%.2f\n",
                position, logits_max, max_token, logit_108, logit_2202, logit_10979);
        }

        // DEBUG: Manually verify wcls computation
        if (position == 0) {
            System.err.println("\n=== MANUAL WCLS VERIFICATION ===");

            // Manually compute logit for token 236814
            int testToken = 236814;
            float manualLogit = 0.0f;
            int wclsRowStart = testToken * dim;
            for (int j = 0; j < dim; j++) {
                float wclsVal = weights.wcls.getFloat(wclsRowStart + j);
                float xVal = state.x.getFloat(j);
                manualLogit += wclsVal * xVal;
            }

            float actualLogit = state.logits.getFloat(testToken);
            System.err.printf("Token %d logit verification:\n", testToken);
            System.err.printf("  Manual computation: %.8f\n", manualLogit);
            System.err.printf("  Actual logit:       %.8f\n", actualLogit);
            System.err.printf("  Difference:         %.8f (should be ~0)\n", Math.abs(manualLogit - actualLogit));

            // Also try different computation order
            manualLogit = 0.0f;
            for (int j = 0; j < dim; j++) {
                manualLogit += state.x.getFloat(j) * weights.wcls.getFloat(wclsRowStart + j);
            }
            System.err.printf("  Manual (reversed):  %.8f\n", manualLogit);

            // Check a few other tokens
            System.err.println("\nSample other tokens:");
            for (int t : new int[]{0, 1, 100, 236813, 236815}) {
                int rowStart = t * dim;
                float manual = 0.0f;
                for (int j = 0; j < dim; j++) {
                    manual += weights.wcls.getFloat(rowStart + j) * state.x.getFloat(j);
                }
                float actual = state.logits.getFloat(t);
                System.err.printf("  Token %d: manual=%.6f, actual=%.6f, diff=%.6f\n",
                    t, manual, actual, Math.abs(manual - actual));
            }
        }

        // DEBUG: Check wcls output at key indices
        if (position == 0) {
            System.err.println("\nstate.logits dimensions: " + state.logits.size() + " elements");

            // Check for NaN/Inf in logits
            int nanCount = 0, infCount = 0;
            float minLogit = Float.MAX_VALUE, maxLogit = Float.MIN_VALUE;
            for (int i = 0; i < state.logits.size(); i++) {
                float val = state.logits.getFloat(i);
                if (Float.isNaN(val)) nanCount++;
                if (Float.isInfinite(val)) infCount++;
                minLogit = Math.min(minLogit, val);
                maxLogit = Math.max(maxLogit, val);
            }
            System.err.printf("Logits stats: NaN=%d, Inf=%d, min=%.6f, max=%.6f\n\n", nanCount, infCount, minLogit, maxLogit);
        }

        // DEBUG: Check wcls output at key indices
        if (position == 0) {
            System.err.println("\nWcls output - key logits:");
            System.err.println("  First 10 logits:");
            for (int i = 0; i < 10; i++) {
                System.err.printf("    logits[%d] = %.6f\n", i, state.logits.getFloat(i));
            }
            System.err.println("  Top token indices:");
            System.err.printf("    logits[1106] = %.6f\n", state.logits.getFloat(1106));
            System.err.printf("    logits[236840] = %.6f\n", state.logits.getFloat(236840));
            System.err.printf("    logits[3617] = %.6f\n", state.logits.getFloat(3617));
            System.err.printf("    logits[107] = %.6f\n", state.logits.getFloat(107));
        }

        // DEBUG: Check logits for positions 0 and 1
        if (position == 0 || position == 1) {
            System.err.printf("\n=== LOGITS (position %d) ===\n", position);
            // Find top 5 tokens
            int vocabSize = config.vocabularySize();
            int[] topIndices = new int[5];
            float[] topValues = new float[5];
            for (int i = 0; i < 5; i++) {
                topIndices[i] = -1;
                topValues[i] = Float.NEGATIVE_INFINITY;
            }
            for (int i = 0; i < vocabSize; i++) {
                float logit = state.logits.getFloat(i);
                for (int j = 0; j < 5; j++) {
                    if (logit > topValues[j]) {
                        // Shift lower values down
                        for (int k = 4; k > j; k--) {
                            topValues[k] = topValues[k-1];
                            topIndices[k] = topIndices[k-1];
                        }
                        topValues[j] = logit;
                        topIndices[j] = i;
                        break;
                    }
                }
            }
            System.err.println("Top 5 tokens:");
            for (int i = 0; i < 5; i++) {
                System.err.printf("  Token %d: logit=%.6f\n", topIndices[i], topValues[i]);
            }
        } else if (position < 5) {
            // Print top 3 tokens for first few positions
            int vocabSize = config.vocabularySize();
            int[] topIndices = new int[3];
            float[] topValues = new float[3];
            for (int i = 0; i < 3; i++) {
                topIndices[i] = -1;
                topValues[i] = Float.NEGATIVE_INFINITY;
            }
            for (int i = 0; i < vocabSize; i++) {
                float logit = state.logits.getFloat(i);
                for (int j = 0; j < 3; j++) {
                    if (logit > topValues[j]) {
                        for (int k = 2; k > j; k--) {
                            topValues[k] = topValues[k-1];
                            topIndices[k] = topIndices[k-1];
                        }
                        topValues[j] = logit;
                        topIndices[j] = i;
                        break;
                    }
                }
            }
            System.err.printf("Position %d: Top 3 = [%d (%.2f), %d (%.2f), %d (%.2f)]\n",
                position, topIndices[0], topValues[0], topIndices[1], topValues[1], topIndices[2], topValues[2]);
        }

        return state.logits;
    }

    public static FloatTensor forwardJavaPhi3(Model model, Phi3State state, int token, int position) {
        Phi3Configuration config = (Phi3Configuration) model.configuration();
        Phi3StandardWeights weights = (Phi3StandardWeights) model.weights();
        int dim = config.dim();
        int headSize = config.headSize();
        int kvDim = (config.dim() * config.numberOfKeyValueHeads()) / config.numberOfHeads();
        int kvMul = config.numberOfHeads() / config.numberOfKeyValueHeads(); // integer multiplier of the kv sharing in multiquery
        float sqrtHeadSize = (float) Math.sqrt(headSize);

        // copy the token embedding into x
        weights.token_embedding_table.copyTo(token * dim, state.x, 0, dim);

        // Phi3: op_size = num_heads * head_dim + 2 * (num_key_value_heads * head_dim)
        final int opSize = dim + 2 * (config.numberOfKeyValueHeads() * headSize);

        // forward all the layers
        for (int l = 0; l < config.numberOfLayers(); l++) {
            rmsnorm(state.xb, state.x, weights.rms_att_weight[l], 0, dim, config.rmsNormEps());

            weights.wqkv[l].matmul(state.xb, state.qkv, opSize, dim);
            state.qkv.copyTo(0, state.q, 0, dim);
            // key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
            state.qkv.copyTo(dim, state.k, 0, config.numberOfKeyValueHeads() * headSize);
            // value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]
            state.qkv.copyTo(dim + config.numberOfKeyValueHeads() * headSize, state.v, 0, config.numberOfKeyValueHeads() * headSize);

            int dimHalf = headSize / 2;
            for (int i = 0; i < dim; i += 2) {
                int head_dim = i % headSize;
                int base = i - head_dim;
                int ic = base + head_dim / 2;
                float fcr = weights.freq_cis_real.getFloat(position * (headSize / 2) + (head_dim / 2));
                float fci = weights.freq_cis_imag.getFloat(position * (headSize / 2) + (head_dim / 2));
                int rotn = i < kvDim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
                for (int v = 0; v < rotn; v++) {
                    FloatTensor vec = v == 0 ? state.q : state.k; // the vector to rotate (query or key)
                    float v0 = vec.getFloat(ic);
                    float v1 = vec.getFloat(ic + dimHalf);
                    vec.setFloat(ic, v0 * fcr - v1 * fci);
                    vec.setFloat(ic + dimHalf, v0 * fci + v1 * fcr);
                }
            }

            // save key,value at this time step (position) to our kv cache
            state.k.copyTo(0, state.keyCache[l], position * kvDim, kvDim);
            state.v.copyTo(0, state.valueCache[l], position * kvDim, kvDim);

            int curLayer = l;

            Parallel.parallelFor(0, config.numberOfHeads(), h -> {
                int qOffset = h * headSize;

                int attOffset = h * config.contextLength();

                for (int t = 0; t <= position; t++) {
                    int keyCacheOffset = /* loff + */ t * kvDim + (h / kvMul) * headSize;
                    float score = state.q.dot(qOffset, state.keyCache[curLayer], keyCacheOffset, headSize);
                    score /= sqrtHeadSize;
                    state.att.setFloat(attOffset + t, score);
                }

                state.att.softmaxInPlace(attOffset, position + 1);

                int xbOffset = h * headSize;
                state.xb.fillInPlace(xbOffset, headSize, 0f);

                for (int t = 0; t <= position; t++) {
                    int vOffset = /* loff + */ t * kvDim + (h / kvMul) * headSize;
                    float a = state.att.getFloat(attOffset + t);
                    state.xb.saxpyInPlace(xbOffset, state.valueCache[curLayer], vOffset, headSize, a);
                }
            });

            // final matmul to get the output of the attention
            weights.wo[l].matmul(state.xb, state.xb2, dim, dim);

            // residual connection back into x
            state.x.addInPlace(state.xb2);

            rmsnorm(state.xb, state.x, weights.rms_ffn_weight[l], 0, dim, config.rmsNormEps());

            weights.wGateUp[l].matmul(state.xb, state.hb, 2 * config.hiddenDim(), dim);
            copyChunk(state.hb, state.hbG, 2 * config.hiddenDim(), config.hiddenDim(), 2, 0);
            copyChunk(state.hb, state.hbU, 2 * config.hiddenDim(), config.hiddenDim(), 2, 1);

            state.hbG.mapInPlace(value -> value / (float) (1.0 + Math.exp(-value)));

            state.hbU.multiplyInPlace(state.hbG);

            weights.wDown[l].matmul(state.hbU, state.xb, dim, config.hiddenDim());

            state.x.addInPlace(state.xb);
        }

        // final rmsnorm
        rmsnorm(state.x, state.x, weights.rms_final_weight, 0, dim, config.rmsNormEps());

        // classifier into logits
        weights.wcls.matmul(state.x, state.logits, config.vocabularySize(), dim);

        return state.logits;
    }

    static void copyChunk(FloatTensor in, FloatTensor out, int dim1In, int dim1Out, int nChunks, int chunkNo) {
        assert (dim1In == dim1Out * nChunks);
        final int startOffsetInDim1 = chunkNo * dim1Out;
        Parallel.parallelFor(0, dim1Out, i -> {
            out.setFloat(i, in.getFloat(startOffsetInDim1 + i));
        });
    }

    /**
     * Performs the initial embedding lookup and triggers the TornadoVM accelerated forward pass for an LLM token.
     *
     * <p>This method handles the first phase of processing a token through the transformer model:
     * <ol>
     *   <li>Copies the token embedding from the model's embedding table to the state's buffer</li>
     *   <li>Delegates the transformer layer processing to TornadoVM through the master plan</li>
     * </ol>
     *
     * <p>The token embedding lookup happens on the CPU using {@link MemorySegment} operations,
     * while the subsequent transformer layers processing is offloaded to the accelerator through
     * TornadoVM for improved performance.
     *
     * @param model
     *         The Llama model containing weights and configuration parameters
     * @param state
     *         The current execution state holding input/output tensors and temporary buffers
     * @param token
     *         The input token ID to process
     * @param position
     *         The position of this token in the sequence context window
     * @param tornadoVMMasterPlan
     *         The execution plan for TornadoVM acceleration
     * @return FloatTensor containing the output logits for token prediction
     */
    public static FloatArray forwardTornadoVM(Model model, State state, int token, int position, TornadoVMMasterPlan tornadoVMMasterPlan) {
        final Configuration configuration = model.configuration();
        final TornadoWeights weights = (TornadoWeights) model.weights();

        MemorySegment.copy(weights.getTokenEmbeddingTable().getSegment(), (long) token * configuration.dim() * Float.BYTES, state.wrapX.getSegment(), 0, configuration.dim() * Float.BYTES);

        return tornadoVMMasterPlan.tornadoVMForwardExecuteLayered(position);
    }

}
