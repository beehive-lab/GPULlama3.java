package org.beehive.gpullama3.inference;

import org.beehive.gpullama3.auxiliary.Parallel;
import org.beehive.gpullama3.tensor.standard.FloatTensor;
import org.beehive.gpullama3.inference.state.Phi3State;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.standard.Phi3StandardWeights;
import org.beehive.gpullama3.inference.weights.standard.Qwen2StandardWeights;
import org.beehive.gpullama3.inference.weights.standard.Qwen3StandardWeights;
import org.beehive.gpullama3.inference.weights.standard.StandardWeights;
import org.beehive.gpullama3.inference.weights.tornado.TornadoWeights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.phi3.Phi3Configuration;
import org.beehive.gpullama3.model.qwen2.Qwen2Configuration;
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

        MemorySegment.copy(weights.getTokenEmbeddingTable().asHalfFloatArray().getSegment(), (long) token * configuration.dim() * Short.BYTES, state.embeddingX.getSegment(), 0, configuration.dim() * Short.BYTES);

        return tornadoVMMasterPlan.tornadoVMForwardExecuteLayered(position);
    }

}
