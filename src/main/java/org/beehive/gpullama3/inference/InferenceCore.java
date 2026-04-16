package org.beehive.gpullama3.inference;

import org.beehive.gpullama3.auxiliary.Parallel;
import org.beehive.gpullama3.tensor.GGMLType;
import org.beehive.gpullama3.tensor.standard.FloatTensor;
import org.beehive.gpullama3.inference.state.Gemma4State;
import org.beehive.gpullama3.inference.state.Phi3State;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.standard.Gemma4StandardWeights;
import org.beehive.gpullama3.inference.weights.standard.Phi3StandardWeights;
import org.beehive.gpullama3.inference.weights.standard.Qwen2StandardWeights;
import org.beehive.gpullama3.inference.weights.standard.Qwen3StandardWeights;
import org.beehive.gpullama3.inference.weights.standard.StandardWeights;
import org.beehive.gpullama3.inference.weights.tornado.TornadoWeights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.gemma4.Gemma4Configuration;
import org.beehive.gpullama3.model.granite.GraniteConfiguration;
import org.beehive.gpullama3.model.devstral.DevstralConfiguration;
import org.beehive.gpullama3.model.phi3.Phi3Configuration;
import org.beehive.gpullama3.model.qwen2.Qwen2Configuration;
import org.beehive.gpullama3.model.qwen3.Qwen3Configuration;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.lang.foreign.MemorySegment;
import java.nio.FloatBuffer;

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
        ss = (float) (1.0 / Math.sqrt(ss));
        // normalize and scale
        final float finalss = ss; // for the lambda
        out.mapWithIndexInPlace(offset, size, (value, index) -> weight.getFloat(index % size) * (finalss * x.getFloat(index)));
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

    /**
     * Forward pass for Devstral 2 models where head_dim != dim/num_heads.
     * Q projection outputs qDim (num_heads * head_dim) instead of dim.
     */
    public static FloatTensor forwardJavaDevstral(Model model, State state, int token, int position) {
        final DevstralConfiguration config = (DevstralConfiguration) model.configuration();
        final StandardWeights weights = (StandardWeights) model.weights();
        int dim = config.dim();
        int headSize = config.headSize();       // 128 (independent head_dim)
        int qDim = config.qDim();               // 4096 = 32 * 128
        int kvDim = config.kvDim();              // 1024 = 8 * 128
        int kvMul = config.kvMul();
        float sqrtHeadSize = (float) Math.sqrt(headSize);

        weights.token_embedding_table.copyTo(token * dim, state.x, 0, dim);

        for (int l = 0; l < config.numberOfLayers(); l++) {
            rmsnorm(state.xb, state.x, weights.rms_att_weight[l], 0, dim, config.rmsNormEps());

            weights.wq[l].matmul(state.xb, state.q, qDim, dim);
            weights.wk[l].matmul(state.xb, state.k, kvDim, dim);
            weights.wv[l].matmul(state.xb, state.v, kvDim, dim);

            // RoPE over qDim (not dim)
            for (int i = 0; i < qDim; i += 2) {
                int head_dim = i % headSize;
                float fcr = weights.freq_cis_real.getFloat(position * (headSize / 2) + (head_dim / 2));
                float fci = weights.freq_cis_imag.getFloat(position * (headSize / 2) + (head_dim / 2));
                int rotn = i < kvDim ? 2 : 1;
                for (int v = 0; v < rotn; v++) {
                    FloatTensor vec = v == 0 ? state.q : state.k;
                    float v0 = vec.getFloat(i);
                    float v1 = vec.getFloat(i + 1);
                    vec.setFloat(i, v0 * fcr - v1 * fci);
                    vec.setFloat(i + 1, v0 * fci + v1 * fcr);
                }
            }

            state.k.copyTo(0, state.keyCache[l], position * kvDim, kvDim);
            state.v.copyTo(0, state.valueCache[l], position * kvDim, kvDim);

            int curLayer = l;

            Parallel.parallelFor(0, config.numberOfHeads(), h -> {
                int qOffset = h * headSize;
                int attOffset = h * config.contextLength();

                for (int t = 0; t <= position; t++) {
                    int keyCacheOffset = t * kvDim + (h / kvMul) * headSize;
                    float score = state.q.dot(qOffset, state.keyCache[curLayer], keyCacheOffset, headSize);
                    score /= sqrtHeadSize;
                    state.att.setFloat(attOffset + t, score);
                }

                state.att.softmaxInPlace(attOffset, position + 1);

                int xbOffset = h * headSize;
                state.xb.fillInPlace(xbOffset, headSize, 0f);

                for (int t = 0; t <= position; t++) {
                    int vOffset = t * kvDim + (h / kvMul) * headSize;
                    float a = state.att.getFloat(attOffset + t);
                    state.xb.saxpyInPlace(xbOffset, state.valueCache[curLayer], vOffset, headSize, a);
                }
            });

            // O projection: input qDim, output dim
            weights.wo[l].matmul(state.xb, state.xb2, dim, qDim);

            state.x.addInPlace(state.xb2);

            rmsnorm(state.xb, state.x, weights.rms_ffn_weight[l], 0, dim, config.rmsNormEps());

            weights.w1[l].matmul(state.xb, state.hb, config.hiddenDim(), dim);
            weights.w3[l].matmul(state.xb, state.hb2, config.hiddenDim(), dim);

            state.hb.mapInPlace(value -> value / (float) (1.0 + Math.exp(-value)));
            state.hb.multiplyInPlace(state.hb2);

            weights.w2[l].matmul(state.hb, state.xb, dim, config.hiddenDim());
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

    /**
     * Forward pass for Granite models with µP scaling factors applied.
     * <p>
     * Granite uses the same transformer architecture as Llama but with maximal update parameterization (µP)
     * scaling factors applied at specific points:
     * <ul>
     *   <li>Embedding scaling: multiply embeddings after lookup</li>
     *   <li>Attention scaling: use custom multiplier instead of 1/sqrt(headDim)</li>
     *   <li>Residual scaling: multiply residual connections</li>
     *   <li>Logit scaling: divide logits by the scaling factor</li>
     * </ul>
     */
    public static FloatTensor forwardGranite(Model model, State state, int token, int position) {
        final GraniteConfiguration config = (GraniteConfiguration) model.configuration();
        final StandardWeights weights = (StandardWeights) model.weights();
        int dim = config.dim();
        int headSize = config.headSize();
        int kvDim = (config.dim() * config.numberOfKeyValueHeads()) / config.numberOfHeads();
        int kvMul = config.numberOfHeads() / config.numberOfKeyValueHeads();
        float attentionScale = config.attentionScale();
        float residualScale = config.residualScale();
        float embeddingScale = config.embeddingScale();
        float logitScale = config.logitScale();

        // copy the token embedding into x
        weights.token_embedding_table.copyTo(token * dim, state.x, 0, dim);
        // Apply Granite embedding scaling
        state.x.mapInPlace(v -> v * embeddingScale);

        // forward all the layers
        for (int l = 0; l < config.numberOfLayers(); l++) {
            // attention rmsnorm
            rmsnorm(state.xb, state.x, weights.rms_att_weight[l], 0, dim, config.rmsNormEps());

            // qkv matmuls for this position
            weights.wq[l].matmul(state.xb, state.q, dim, dim);
            weights.wk[l].matmul(state.xb, state.k, kvDim, dim);
            weights.wv[l].matmul(state.xb, state.v, kvDim, dim);

            // RoPE relative positional encoding
            for (int i = 0; i < dim; i += 2) {
                int head_dim = i % headSize;
                float fcr = weights.freq_cis_real.getFloat(position * (headSize / 2) + (head_dim / 2));
                float fci = weights.freq_cis_imag.getFloat(position * (headSize / 2) + (head_dim / 2));
                int rotn = i < kvDim ? 2 : 1;
                for (int v = 0; v < rotn; v++) {
                    FloatTensor vec = v == 0 ? state.q : state.k;
                    float v0 = vec.getFloat(i);
                    float v1 = vec.getFloat(i + 1);
                    vec.setFloat(i, v0 * fcr - v1 * fci);
                    vec.setFloat(i + 1, v0 * fci + v1 * fcr);
                }
            }

            // save key,value at this time step to kv cache
            state.k.copyTo(0, state.keyCache[l], position * kvDim, kvDim);
            state.v.copyTo(0, state.valueCache[l], position * kvDim, kvDim);

            int curLayer = l;

            // multihead attention with Granite attention scaling
            Parallel.parallelFor(0, config.numberOfHeads(), h -> {
                int qOffset = h * headSize;
                int attOffset = h * config.contextLength();

                for (int t = 0; t <= position; t++) {
                    int keyCacheOffset = t * kvDim + (h / kvMul) * headSize;
                    float score = state.q.dot(qOffset, state.keyCache[curLayer], keyCacheOffset, headSize);
                    // Granite uses custom attention multiplier instead of 1/sqrt(headSize)
                    score *= attentionScale;
                    state.att.setFloat(attOffset + t, score);
                }

                state.att.softmaxInPlace(attOffset, position + 1);

                int xbOffset = h * headSize;
                state.xb.fillInPlace(xbOffset, headSize, 0f);

                for (int t = 0; t <= position; t++) {
                    int vOffset = t * kvDim + (h / kvMul) * headSize;
                    float a = state.att.getFloat(attOffset + t);
                    state.xb.saxpyInPlace(xbOffset, state.valueCache[curLayer], vOffset, headSize, a);
                }
            });

            // final matmul to get the output of the attention
            weights.wo[l].matmul(state.xb, state.xb2, dim, dim);

            // residual connection with Granite scaling
            state.xb2.mapInPlace(v -> v * residualScale);
            state.x.addInPlace(state.xb2);

            // ffn rmsnorm
            rmsnorm(state.xb, state.x, weights.rms_ffn_weight[l], 0, dim, config.rmsNormEps());

            // FFN: self.w2(F.silu(self.w1(x)) * self.w3(x))
            weights.w1[l].matmul(state.xb, state.hb, config.hiddenDim(), dim);
            weights.w3[l].matmul(state.xb, state.hb2, config.hiddenDim(), dim);

            // SwiGLU non-linearity
            state.hb.mapInPlace(value -> value / (float) (1.0 + Math.exp(-value)));
            state.hb.multiplyInPlace(state.hb2);

            // final matmul to get the output of the ffn
            weights.w2[l].matmul(state.hb, state.xb, dim, config.hiddenDim());

            // residual connection with Granite scaling
            state.xb.mapInPlace(v -> v * residualScale);
            state.x.addInPlace(state.xb);
        }

        rmsnorm(state.x, state.x, weights.rms_final_weight, 0, dim, config.rmsNormEps());

        weights.wcls.matmul(state.x, state.logits, config.vocabularySize(), dim);

        // Apply Granite logit scaling (divide by the scaling factor)
        state.logits.mapInPlace(v -> v * logitScale);

        return state.logits;
    }

    static void copyChunk(FloatTensor in, FloatTensor out, int dim1In, int dim1Out, int nChunks, int chunkNo) {
        assert (dim1In == dim1Out * nChunks);
        final int startOffsetInDim1 = chunkNo * dim1Out;
        Parallel.parallelFor(0, dim1Out, i -> {
            out.setFloat(i, in.getFloat(startOffsetInDim1 + i));
        });
    }

    // === Gemma 4 helper methods ===

    static float gelu(float x) {
        return (float) (0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3)))));
    }

    /** Read a float from a quantized MemorySegment at a long element index. Supports Q8_0, F32, F16. */
    static float getQuantizedFloat(MemorySegment segment, GGMLType type, long index) {
        return switch (type) {
            case Q8_0 -> {
                long blockIndex = index / GGMLType.Q8_0.getBlockSize();
                long withinBlock = index % GGMLType.Q8_0.getBlockSize();
                long blockOffset = blockIndex * GGMLType.Q8_0.getTypeSize();
                // Q8_0 layout: 2 bytes fp16 scale + 32 bytes int8 quants
                float scale = Float.float16ToFloat(FloatTensor.readShort(segment, blockOffset));
                byte quant = FloatTensor.readByte(segment, blockOffset + 2 + withinBlock);
                yield quant * scale;
            }
            case F32 -> {
                // F32: 4 bytes per element, read as int bits then convert
                long byteOffset = index * Float.BYTES;
                int bits = (FloatTensor.readByte(segment, byteOffset) & 0xFF)
                        | ((FloatTensor.readByte(segment, byteOffset + 1) & 0xFF) << 8)
                        | ((FloatTensor.readByte(segment, byteOffset + 2) & 0xFF) << 16)
                        | ((FloatTensor.readByte(segment, byteOffset + 3) & 0xFF) << 24);
                yield Float.intBitsToFloat(bits);
            }
            case F16 -> Float.float16ToFloat(FloatTensor.readShort(segment, index * 2));
            default -> throw new UnsupportedOperationException("Unsupported type for long-indexed access: " + type);
        };
    }

    /** RMS norm using FloatBuffer weights (for Gemma 4 norm tensors stored as F32 buffers). */
    static void rmsnormBuf(FloatTensor out, FloatTensor x, FloatBuffer weight, int size, float rmsNormEps) {
        float ss = x.reduce(0, size, 0f, (acc, xi) -> acc + xi * xi);
        ss /= size;
        ss += rmsNormEps;
        ss = (float) (1.0 / Math.sqrt(ss));
        final float finalss = ss;
        out.mapWithIndexInPlace(0, size, (value, index) -> weight.get(index) * (finalss * x.getFloat(index)));
    }

    /** RMS norm with offset, using FloatBuffer weights. */
    static void rmsnormBuf(FloatTensor out, int outOffset, FloatTensor x, int xOffset,
                           FloatBuffer weight, int size, float rmsNormEps) {
        float ss = 0f;
        for (int i = 0; i < size; i++) {
            float xi = x.getFloat(xOffset + i);
            ss += xi * xi;
        }
        ss /= size;
        ss += rmsNormEps;
        ss = (float) (1.0 / Math.sqrt(ss));
        for (int i = 0; i < size; i++) {
            out.setFloat(outOffset + i, weight.get(i) * ss * x.getFloat(xOffset + i));
        }
    }

    /** Bare RMS norm without learned weights (just normalize to unit RMS). */
    static void rmsnormNoWeight(FloatTensor out, int outOffset, FloatTensor x, int xOffset,
                                int size, float rmsNormEps) {
        float ss = 0f;
        for (int i = 0; i < size; i++) {
            float xi = x.getFloat(xOffset + i);
            ss += xi * xi;
        }
        ss /= size;
        ss += rmsNormEps;
        ss = (float) (1.0 / Math.sqrt(ss));
        for (int i = 0; i < size; i++) {
            out.setFloat(outOffset + i, ss * x.getFloat(xOffset + i));
        }
    }

    /**
     * Forward pass for Gemma 4 models.
     * Key differences: dual head sizes (SWA vs full), per-head Q/K norm, GELU activation,
     * post-attention/FFN norms, no attention scaling, embedding scaling, per-layer output scaling,
     * sliding window attention, shared KV cache, optional per-layer embeddings, optional MoE.
     */
    public static FloatTensor forwardJavaGemma4(Model model, Gemma4State state, int token, int position) {
        final Gemma4Configuration config = (Gemma4Configuration) model.configuration();
        final Gemma4StandardWeights weights = (Gemma4StandardWeights) model.weights();
        int dim = config.dim();
        float sqrtDim = (float) Math.sqrt(dim);

        // Embedding: x = embeddings[token] * sqrt(dim)
        weights.token_embedding_table.copyTo(token * dim, state.x, 0, dim);
        state.x.mapInPlace(v -> v * sqrtDim);

        // Compute per-layer inputs (if model has per-layer embeddings)
        int plDim = config.embeddingLengthPerLayer();
        int plTotal = plDim * config.numberOfLayers();
        if (plDim > 0 && weights.perLayerTokenEmbdSegment != null && state.perLayerInputs != null) {
            float sqrtPlDim = (float) Math.sqrt(plDim);
            float projScale = (float) (1.0 / Math.sqrt(dim));
            float inputScale = (float) (1.0 / Math.sqrt(2.0));

            weights.perLayerModelProj.matmul(state.x, state.perLayerInputs, plTotal, dim);
            state.perLayerInputs.mapWithIndexInPlace(0, plTotal, (v, i) -> v * projScale);
            for (int l = 0; l < config.numberOfLayers(); l++) {
                rmsnormBuf(state.perLayerInputs, l * plDim, state.perLayerInputs, l * plDim,
                        weights.perLayerProjNorm, plDim, config.rmsNormEps());
            }

            long tokEmbOffset = (long) token * plTotal;
            for (int i = 0; i < plTotal; i++) {
                float tokEmb = getQuantizedFloat(weights.perLayerTokenEmbdSegment,
                        weights.perLayerTokenEmbdType, tokEmbOffset + i) * sqrtPlDim;
                state.perLayerInputs.setFloat(i, state.perLayerInputs.getFloat(i) + tokEmb);
            }

            state.perLayerInputs.mapWithIndexInPlace(0, plTotal, (v, i) -> v * inputScale);
        }

        // Forward all layers
        for (int l = 0; l < config.numberOfLayers(); l++) {
            boolean layerIsSWA = config.isSWA()[l];
            int headSize = config.headSize(l);
            int halfHead = headSize / 2;
            int queryDim = config.queryDim(l);
            int kvDim = config.kvDim(l);
            int hiddenDim = config.feedForwardLength()[l];

            // Attention RMSNorm (parent's FloatTensor field)
            rmsnorm(state.xb, state.x, weights.rms_att_weight[l], 0, dim, config.rmsNormEps());

            // Q projection + per-head RMS norm
            weights.wq[l].matmul(state.xb, state.q, queryDim, dim);
            for (int h = 0; h < config.numberOfHeads(); h++) {
                rmsnormBuf(state.q, h * headSize, state.q, h * headSize,
                        weights.attnQNorm[l], headSize, config.rmsNormEps());
            }

            // RoPE for Q (NeoX style)
            FloatTensor freqsReal = layerIsSWA ? weights.freq_cis_real_swa : weights.freq_cis_real_full;
            FloatTensor freqsImag = layerIsSWA ? weights.freq_cis_imag_swa : weights.freq_cis_imag_full;
            for (int h = 0; h < config.numberOfHeads(); ++h) {
                int poffset = h * headSize;
                for (int i0 = 0; i0 < headSize; i0 += 2) {
                    int ic = i0 / 2;
                    float fcr = freqsReal.getFloat(position * halfHead + ic);
                    float fci = freqsImag.getFloat(position * halfHead + ic);
                    float v0 = state.q.getFloat(poffset + ic);
                    float v1 = state.q.getFloat(poffset + ic + halfHead);
                    state.q.setFloat(poffset + ic, v0 * fcr - v1 * fci);
                    state.q.setFloat(poffset + ic + halfHead, v0 * fci + v1 * fcr);
                }
            }

            // KV projection
            int kvLayer = config.kvSourceLayer(l);
            int nKvHeads = config.numberOfKeyValueHeads(l);
            int kvMul = config.numberOfHeads() / nKvHeads;
            if (config.hasKv(l)) {
                weights.wk[l].matmul(state.xb, state.k, kvDim, dim);
                // V = wv @ xb if V weight exists, otherwise V = K
                if (weights.wv[l] != null) {
                    weights.wv[l].matmul(state.xb, state.v, kvDim, dim);
                } else {
                    state.k.copyTo(0, state.v, 0, kvDim);
                }

                // Per-head K norm (learned) and V norm (bare RMS)
                for (int h = 0; h < nKvHeads; h++) {
                    rmsnormBuf(state.k, h * headSize, state.k, h * headSize,
                            weights.attnKNorm[l], headSize, config.rmsNormEps());
                    rmsnormNoWeight(state.v, h * headSize, state.v, h * headSize,
                            headSize, config.rmsNormEps());
                }

                // RoPE for K
                for (int h = 0; h < nKvHeads; ++h) {
                    int poffset = h * headSize;
                    for (int i0 = 0; i0 < headSize; i0 += 2) {
                        int ic = i0 / 2;
                        float fcr = freqsReal.getFloat(position * halfHead + ic);
                        float fci = freqsImag.getFloat(position * halfHead + ic);
                        float v0 = state.k.getFloat(poffset + ic);
                        float v1 = state.k.getFloat(poffset + ic + halfHead);
                        state.k.setFloat(poffset + ic, v0 * fcr - v1 * fci);
                        state.k.setFloat(poffset + ic + halfHead, v0 * fci + v1 * fcr);
                    }
                }

                // Store K,V to cache
                int kvPos = config.kvCacheIndex(l, position);
                state.k.copyTo(0, state.keyCache[kvLayer], kvPos * kvDim, kvDim);
                state.v.copyTo(0, state.valueCache[kvLayer], kvPos * kvDim, kvDim);
            }

            // Attention (scale=1.0, NO 1/sqrt(headSize))
            int attStart = layerIsSWA ? Math.max(0, position - config.slidingWindow() + 1) : 0;
            int finalKvLayer = kvLayer;
            int finalKvDim = kvDim;
            int finalAttStart = attStart;
            int finalLayer = l;
            int finalHeadSize = headSize;

            Parallel.parallelFor(0, config.numberOfHeads(), h -> {
                int qOffset = h * finalHeadSize;
                int attOffset = h * config.contextLength();
                int kvHeadOffset = (h / kvMul) * finalHeadSize;
                for (int t = finalAttStart; t <= position; t++) {
                    int keyCacheOffset = config.kvCacheIndex(finalLayer, t) * finalKvDim + kvHeadOffset;
                    float score = state.q.dot(qOffset, state.keyCache[finalKvLayer], keyCacheOffset, finalHeadSize);
                    state.att.setFloat(attOffset + t, score);
                }

                state.att.softmaxInPlace(attOffset + finalAttStart, position - finalAttStart + 1);
                int xbOffset = h * finalHeadSize;
                state.xb_k.fillInPlace(xbOffset, finalHeadSize, 0f);
                for (int t = finalAttStart; t <= position; t++) {
                    int vOffset = config.kvCacheIndex(finalLayer, t) * finalKvDim + kvHeadOffset;
                    float a = state.att.getFloat(attOffset + t);
                    state.xb_k.saxpyInPlace(xbOffset, state.valueCache[finalKvLayer], vOffset, finalHeadSize, a);
                }
            });

            // Output projection + post-attention norm + residual
            weights.wo[l].matmul(state.xb_k, state.xb2, dim, queryDim);
            rmsnormBuf(state.xb2, state.xb2, weights.postAttentionNorm[l], dim, config.rmsNormEps());
            state.x.addInPlace(state.xb2);

            // FFN (MoE vs dense)
            boolean isMoELayer = config.isMoE() && weights.ffnGateInp != null && weights.ffnGateInp[l] != null;
            if (isMoELayer) {
                // === MoE FFN ===
                // Shared MLP path
                rmsnorm(state.xb, state.x, weights.rms_ffn_weight[l], 0, dim, config.rmsNormEps());
                weights.w1[l].matmul(state.xb, state.hb, hiddenDim, dim);
                weights.w3[l].matmul(state.xb, state.hb2, hiddenDim, dim);
                state.hb.mapWithIndexInPlace(0, hiddenDim, (v, i) -> gelu(v));
                state.hb.mapWithIndexInPlace(0, hiddenDim, (v, i) -> v * state.hb2.getFloat(i));
                weights.w2[l].matmul(state.hb, state.xb, dim, hiddenDim);
                rmsnormBuf(state.xb, state.xb, weights.ffnPostNorm1[l], dim, config.rmsNormEps());

                // Expert routing
                rmsnormBuf(state.moeInput, state.x, weights.preFfwNorm2[l], dim, config.rmsNormEps());

                // Router computation
                float ss = state.x.reduce(0, dim, 0f, (acc, xi) -> acc + xi * xi);
                ss /= dim;
                ss += config.rmsNormEps();
                float rmsScale = (float) (1.0 / Math.sqrt(ss)) / (float) Math.sqrt(dim);
                for (int i = 0; i < dim; i++) {
                    state.xb2.setFloat(i, state.x.getFloat(i) * rmsScale * weights.ffnGateInpScale[l].get(i));
                }
                weights.ffnGateInp[l].matmul(state.xb2, state.routerLogits, config.expertCount(), dim);
                state.routerLogits.softmaxInPlace(0, config.expertCount());

                // Top-k expert selection
                int topK = config.expertUsedCount();
                int[] topExperts = new int[topK];
                float[] topProbs = new float[topK];
                for (int ki = 0; ki < topK; ki++) {
                    int bestIdx = 0;
                    float bestVal = Float.NEGATIVE_INFINITY;
                    for (int ei = 0; ei < config.expertCount(); ei++) {
                        float val = state.routerLogits.getFloat(ei);
                        if (val > bestVal) {
                            bestVal = val;
                            bestIdx = ei;
                        }
                    }
                    topExperts[ki] = bestIdx;
                    topProbs[ki] = bestVal;
                    state.routerLogits.setFloat(bestIdx, Float.NEGATIVE_INFINITY);
                }

                // Run experts and accumulate
                int expertFF = config.expertFeedForwardLength();
                int gateUpDim = 2 * expertFF;
                state.moeOutput.fillInPlace(0, dim, 0f);

                for (int ki = 0; ki < topK; ki++) {
                    int expertIdx = topExperts[ki];
                    float prob = topProbs[ki];
                    float downScale = weights.ffnDownExpsScale[l].get(expertIdx);

                    // gate_up computation via offset matmul
                    int gateUpOffset = expertIdx * gateUpDim * dim;
                    // Manual offset matmul for expert weights
                    for (int i = 0; i < gateUpDim; i++) {
                        float dot = 0f;
                        for (int j = 0; j < dim; j++) {
                            dot += weights.ffnGateUpExps[l].getFloat(gateUpOffset + i * dim + j) * state.moeInput.getFloat(j);
                        }
                        state.expertGateUp.setFloat(i, dot);
                    }

                    // GELU on gate part, multiply by up part
                    state.expertGateUp.mapWithIndexInPlace(0, expertFF, (v, i) -> gelu(v));
                    for (int i = 0; i < expertFF; i++) {
                        state.expertGateUp.setFloat(i,
                                state.expertGateUp.getFloat(i) * state.expertGateUp.getFloat(expertFF + i));
                    }

                    // down projection
                    int downOffset = expertIdx * dim * expertFF;
                    for (int i = 0; i < dim; i++) {
                        float dot = 0f;
                        for (int j = 0; j < expertFF; j++) {
                            dot += weights.ffnDownExps[l].getFloat(downOffset + i * expertFF + j) * state.expertGateUp.getFloat(j);
                        }
                        state.expertDown.setFloat(i, dot);
                    }

                    float finalWeight = prob * downScale;
                    state.moeOutput.saxpyInPlace(0, state.expertDown, 0, dim, finalWeight);
                }

                // Post-norm for MoE + combine
                rmsnormBuf(state.moeOutput, state.moeOutput, weights.ffnPostNorm2[l], dim, config.rmsNormEps());
                state.xb.mapWithIndexInPlace(0, dim, (v, i) -> v + state.moeOutput.getFloat(i));

                // Overall post-FFW norm + residual
                rmsnormBuf(state.xb, state.xb, weights.postFfwNorm[l], dim, config.rmsNormEps());
                state.x.addInPlace(state.xb);
            } else {
                // Standard dense FFN: w2(GELU(w1(x)) * w3(x))
                rmsnorm(state.xb, state.x, weights.rms_ffn_weight[l], 0, dim, config.rmsNormEps());
                weights.w1[l].matmul(state.xb, state.hb, hiddenDim, dim);
                weights.w3[l].matmul(state.xb, state.hb2, hiddenDim, dim);
                state.hb.mapWithIndexInPlace(0, hiddenDim, (v, i) -> gelu(v));
                state.hb.mapWithIndexInPlace(0, hiddenDim, (v, i) -> v * state.hb2.getFloat(i));
                weights.w2[l].matmul(state.hb, state.xb, dim, hiddenDim);
                rmsnormBuf(state.xb, state.xb, weights.postFfwNorm[l], dim, config.rmsNormEps());
                state.x.addInPlace(state.xb);
            }

            // Per-layer embedding: GELU-gated projection
            if (plDim > 0 && weights.perLayerInpGate != null && state.perLayerInputs != null) {
                weights.perLayerInpGate[l].matmul(state.x, state.plGate, plDim, dim);
                state.plGate.mapWithIndexInPlace(0, plDim, (v, i) -> gelu(v));
                int plOffset = l * plDim;
                for (int i = 0; i < plDim; i++) {
                    state.plGate.setFloat(i,
                            state.plGate.getFloat(i) * state.perLayerInputs.getFloat(plOffset + i));
                }
                weights.perLayerProj[l].matmul(state.plGate, state.plProj, dim, plDim);
                rmsnormBuf(state.plProj, state.plProj, weights.perLayerPostNorm[l], dim, config.rmsNormEps());
                state.x.addInPlace(state.plProj);
            }

            // Layer output scale
            float scale = weights.layerOutputScale[l];
            if (scale != 1.0f) {
                state.x.mapWithIndexInPlace(0, dim, (v, i) -> v * scale);
            }
        }

        // Final norm + logits (parent's FloatTensor field)
        rmsnorm(state.x, state.x, weights.rms_final_weight, 0, dim, config.rmsNormEps());
        weights.wcls.matmul(state.x, state.logits, config.vocabularySize(), dim);

        // Optional logit softcapping
        if (config.logitSoftcapping() > 0) {
            float cap = config.logitSoftcapping();
            state.logits.mapInPlace(v -> cap * (float) Math.tanh(v / cap));
        }

        return state.logits;
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

        switch (weights.getWeightType()) {
            case F16 -> {
                MemorySegment tokenEmbeddings = weights.getTokenEmbeddingTable().asHalfFloatArray().getSegment();
                int bytes = Short.BYTES;
                MemorySegment.copy(tokenEmbeddings, (long) token * configuration.dim() * bytes, state.embeddingX.getSegment(), 0, (long) configuration.dim() * bytes);
            }
            case Q8_0 -> {
                MemorySegment tokenEmbeddings = weights.getTokenEmbeddingTable().asByteArray().getSegment();
                int blockSize = 32;
                int Q8_0_BLOCK_BYTES = 34; // 2 bytes scale + 32 bytes quants
                int blocksPerToken = (configuration.dim() + blockSize - 1) / blockSize; // Ceiling division
                long bytesPerToken = (long) blocksPerToken * Q8_0_BLOCK_BYTES;

                MemorySegment.copy(tokenEmbeddings, (long) token * bytesPerToken, state.embeddingX.getSegment(), 0, bytesPerToken);

            }
            default -> throw new IllegalArgumentException("Unsupported weight type: " + weights.getWeightType());
        }

        return tornadoVMMasterPlan.tornadoVMForwardExecuteLayered(position);
    }

}
