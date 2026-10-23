#include "nexora_lm5m_trainer.h"
#include "nexora_vulkan.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <android/log.h>

#define TAG "NexoraTrainerNative"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)

namespace nexora {

static float s_invFreq[32];
static bool s_ropeInitialized = false;

static void ensureRopeInit() {
    if (s_ropeInitialized) return;
    for (int i = 0; i < 32; ++i) {
        s_invFreq[i] = 1.0f / std::pow(10000.0f, (2.0f * i / 64.0f));
    }
    s_ropeInitialized = true;
}

static inline void applyRope(float* vec, uint32_t offset, uint32_t pos) {
    for (uint32_t i = 0; i < 32; ++i) {
        float ang = pos * s_invFreq[i];
        float c = std::cos(ang);
        float s = std::sin(ang);
        float v1 = vec[offset + i];
        float v2 = vec[offset + i + 32];
        vec[offset + i] = v1 * c - v2 * s;
        vec[offset + i + 32] = v2 * c + v1 * s;
    }
}

TrainStepResult executeLM5MTrainStep(
    const int32_t* tokens,
    uint32_t seqLen,
    float* allWeights,
    float* allGrads,
    float* allM,
    float* allV,
    float learningRate,
    float weightDecay,
    float maxGradNorm,
    uint32_t step,
    bool preferVulkan
) {
    TrainStepResult result;
    if (seqLen < 2 || !tokens || !allWeights || !allGrads || !allM || !allV) {
        return result;
    }
    for (uint32_t t = 0; t < seqLen; ++t) {
        if (tokens[t] < 0 || tokens[t] >= static_cast<int32_t>(LM5MOffsets::VOCAB_SIZE)) {
            LOGD("Invalid token id at position %u; aborting training step before modifying weights.", t);
            return result;
        }
    }

    ensureRopeInit();
    auto& vkEngine = VulkanComputeEngine::instance();
    const uint64_t gpuDispatchesAtStart = vkEngine.gpuDispatchCount();
    const bool useVulkan = preferVulkan && vkEngine.isAvailable();

    const uint32_t D = LM5MOffsets::D_MODEL; // 256
    const uint32_t DFF = LM5MOffsets::D_FF;   // 704
    const uint32_t H = LM5MOffsets::NUM_HEADS; // 4
    const uint32_t HD = LM5MOffsets::HEAD_DIM; // 64
    const float scale = 1.0f / std::sqrt(static_cast<float>(HD)); // 0.125f

    float* tokEmb = allWeights;
    float* gradTokEmb = allGrads;
    float* finalNorm = allWeights + 5065216;
    float* gradFinalNorm = allGrads + 5065216;

    // Layer weight pointers
    struct LayerPtrs {
        float* attnNorm;
        float* wq;
        float* wk;
        float* wv;
        float* wo;
        float* ffnNorm;
        float* w1;
        float* w3;
        float* w2;

        float* gAttnNorm;
        float* gWq;
        float* gWk;
        float* gWv;
        float* gWo;
        float* gFfnNorm;
        float* gW1;
        float* gW3;
        float* gW2;
    } layers[5];

    for (uint32_t l = 0; l < 5; ++l) {
        uint32_t off = LM5MOffsets::TOK_EMB_SIZE + l * LM5MOffsets::LAYER_TOTAL;
        layers[l].attnNorm = allWeights + off;
        layers[l].wq       = allWeights + off + 256;
        layers[l].wk       = allWeights + off + 256 + 65536;
        layers[l].wv       = allWeights + off + 256 + 131072;
        layers[l].wo       = allWeights + off + 256 + 196608;
        layers[l].ffnNorm  = allWeights + off + 256 + 262144;
        layers[l].w1       = allWeights + off + 262656;
        layers[l].w3       = allWeights + off + 442880;
        layers[l].w2       = allWeights + off + 623104;

        layers[l].gAttnNorm = allGrads + off;
        layers[l].gWq       = allGrads + off + 256;
        layers[l].gWk       = allGrads + off + 256 + 65536;
        layers[l].gWv       = allGrads + off + 256 + 131072;
        layers[l].gWo       = allGrads + off + 256 + 196608;
        layers[l].gFfnNorm  = allGrads + off + 256 + 262144;
        layers[l].gW1       = allGrads + off + 262656;
        layers[l].gW3       = allGrads + off + 442880;
        layers[l].gW2       = allGrads + off + 623104;
    }

    // Reset gradients
    std::memset(allGrads, 0, LM5MOffsets::TOTAL_PARAMS * sizeof(float));

    // 1. FORWARD PASS
    // Activation storage for backward pass
    std::vector<float> xIn(seqLen * D);
    for (uint32_t t = 0; t < seqLen; ++t) {
        int32_t tok = tokens[t];
        if (tok >= 0 && tok < static_cast<int32_t>(LM5MOffsets::VOCAB_SIZE)) {
            std::memcpy(&xIn[t * D], tokEmb + (tok * D), D * sizeof(float));
        }
    }

    struct LayerCache {
        std::vector<float> hAttn;
        std::vector<float> invStdAttn;
        std::vector<float> q;
        std::vector<float> k;
        std::vector<float> v;
        std::vector<float> attnWeights; // [H, seqLen, seqLen]
        std::vector<float> outHeads;
        std::vector<float> xMid;
        std::vector<float> hFfn;
        std::vector<float> invStdFfn;
        std::vector<float> gate;
        std::vector<float> up;
        std::vector<float> swiglu;
        std::vector<float> xOut;

        LayerCache(uint32_t T, uint32_t D, uint32_t DFF, uint32_t H)
            : hAttn(T * D), invStdAttn(T), q(T * D), k(T * D), v(T * D),
              attnWeights(H * T * T), outHeads(T * D), xMid(T * D),
              hFfn(T * D), invStdFfn(T), gate(T * DFF), up(T * DFF),
              swiglu(T * DFF), xOut(T * D) {}
    };

    std::vector<LayerCache> caches;
    caches.reserve(5);
    for (uint32_t l = 0; l < 5; ++l) {
        caches.emplace_back(seqLen, D, DFF, H);
    }

    const float* currentX = xIn.data();

    for (uint32_t l = 0; l < 5; ++l) {
        auto& lc = caches[l];
        const auto& lw = layers[l];

        // Attention Pre-RMSNorm. Weight matrices are stored output-major [out, in],
        // so forward projections use A * W^T (matmul mode 1). Batch tokens into
        // Batch RMSNorm over all token rows in a single compute dispatch where supported.
        vkEngine.rmsNorm(currentX, lw.attnNorm, lc.hAttn.data(), lc.invStdAttn.data(),
                         D, 1e-5f, seqLen, useVulkan);
        vkEngine.matmul(lc.hAttn.data(), lw.wq, lc.q.data(), seqLen, D, D, 1, useVulkan);
        vkEngine.matmul(lc.hAttn.data(), lw.wk, lc.k.data(), seqLen, D, D, 1, useVulkan);
        vkEngine.matmul(lc.hAttn.data(), lw.wv, lc.v.data(), seqLen, D, D, 1, useVulkan);
        for (uint32_t t = 0; t < seqLen; ++t) {
            for (uint32_t h = 0; h < H; ++h) {
                applyRope(&lc.q[t * D], h * HD, t);
                applyRope(&lc.k[t * D], h * HD, t);
            }
        }

        // Multi-head Attention
        for (uint32_t h = 0; h < H; ++h) {
            uint32_t hOffset = h * HD;
            for (uint32_t t = 0; t < seqLen; ++t) {
                float maxScore = -1e30f;
                std::vector<float> scores(t + 1);
                for (uint32_t s = 0; s <= t; ++s) {
                    float dot = 0.0f;
                    #pragma clang loop vectorize(enable)
                    for (uint32_t i = 0; i < HD; ++i) {
                        dot += lc.q[t * D + hOffset + i] * lc.k[s * D + hOffset + i];
                    }
                    float sc = dot * scale;
                    scores[s] = sc;
                    if (sc > maxScore) maxScore = sc;
                }

                float sumExp = 0.0f;
                for (uint32_t s = 0; s <= t; ++s) {
                    float p = std::exp(scores[s] - maxScore);
                    lc.attnWeights[(h * seqLen + t) * seqLen + s] = p;
                    sumExp += p;
                }
                float invSum = 1.0f / sumExp;
                for (uint32_t s = 0; s <= t; ++s) {
                    lc.attnWeights[(h * seqLen + t) * seqLen + s] *= invSum;
                }

                for (uint32_t i = 0; i < HD; ++i) {
                    float vVal = 0.0f;
                    for (uint32_t s = 0; s <= t; ++s) {
                        vVal += lc.attnWeights[(h * seqLen + t) * seqLen + s] * lc.v[s * D + hOffset + i];
                    }
                    lc.outHeads[t * D + hOffset + i] = vVal;
                }
            }
        }

        // Wo projection + Residual -> xMid. All projection weights are [out, in].
        vkEngine.matmul(lc.outHeads.data(), lw.wo, lc.xMid.data(), seqLen, D, D, 1, useVulkan);
        #pragma clang loop vectorize(enable)
        for (uint32_t i = 0; i < seqLen * D; ++i) {
            lc.xMid[i] += currentX[i];
        }

        // SwiGLU FFN: batched normalization, dense projections, and activation kernel.
        vkEngine.rmsNorm(lc.xMid.data(), lw.ffnNorm, lc.hFfn.data(), lc.invStdFfn.data(),
                         D, 1e-5f, seqLen, useVulkan);
        vkEngine.matmul(lc.hFfn.data(), lw.w1, lc.gate.data(), seqLen, D, DFF, 1, useVulkan);
        vkEngine.matmul(lc.hFfn.data(), lw.w3, lc.up.data(), seqLen, D, DFF, 1, useVulkan);
        vkEngine.swiglu(lc.gate.data(), lc.up.data(), lc.swiglu.data(),
                        static_cast<uint32_t>(seqLen * DFF), useVulkan);
        // W2 is also stored [out, in], therefore transpose it for forward projection.
        vkEngine.matmul(lc.swiglu.data(), lw.w2, lc.xOut.data(), seqLen, DFF, D, 1, useVulkan);
        #pragma clang loop vectorize(enable)
        for (uint32_t i = 0; i < seqLen * D; ++i) {
            lc.xOut[i] += lc.xMid[i];
        }

        currentX = lc.xOut.data();
    }

    // Final RMSNorm across all positions.
    std::vector<float> xFinal(seqLen * D);
    std::vector<float> invStdFinal(seqLen);
    vkEngine.rmsNorm(currentX, finalNorm, xFinal.data(), invStdFinal.data(),
                     D, 1e-5f, seqLen, useVulkan);

    // 2. LOSS & OUTPUT LOGITS GRADIENT
    uint32_t numTargets = seqLen - 1;
    float batchLoss = 0.0f;
    std::vector<float> dXFinal(seqLen * D, 0.0f);
    // Compute all next-token logits in one matrix operation: [T-1,D] * [V,D]^T.
    // This is the largest repeated per-token dispatch in the original implementation.
    std::vector<float> logits(static_cast<size_t>(numTargets) * LM5MOffsets::VOCAB_SIZE);
    float invNormFactor = 1.0f / static_cast<float>(numTargets);
    vkEngine.matmul(xFinal.data(), tokEmb, logits.data(), numTargets, D,
                    LM5MOffsets::VOCAB_SIZE, 1, useVulkan);

    for (uint32_t t = 0; t < numTargets; ++t) {
        float* rowLogits = logits.data() + static_cast<size_t>(t) * LM5MOffsets::VOCAB_SIZE;
        int32_t target = tokens[t + 1];
        float maxL = -1e30f;
        for (uint32_t v = 0; v < LM5MOffsets::VOCAB_SIZE; ++v) {
            if (rowLogits[v] > maxL) maxL = rowLogits[v];
        }

        float sumExp = 0.0f;
        for (uint32_t v = 0; v < LM5MOffsets::VOCAB_SIZE; ++v) {
            float e = std::exp(rowLogits[v] - maxL);
            rowLogits[v] = e;
            sumExp += e;
        }
        float invSum = 1.0f / sumExp;
        for (uint32_t v = 0; v < LM5MOffsets::VOCAB_SIZE; ++v) {
            rowLogits[v] *= invSum;
        }

        float targetProb = rowLogits[target];
        batchLoss += -std::log(std::max(1e-12f, targetProb));
        rowLogits[target] -= 1.0f; // dSoftmax = p - oneHot(target)
        for (uint32_t v = 0; v < LM5MOffsets::VOCAB_SIZE; ++v) {
            rowLogits[v] *= invNormFactor;
        }
    }

    // Exact tied-embedding gradients are matrix products too:
    // dX = dLogits * tokenEmbedding and dEmbedding = dLogits^T * X.
    // This removes the huge per-target x vocabulary x hidden CPU loop.
    vkEngine.matmul(logits.data(), tokEmb, dXFinal.data(), numTargets,
                    LM5MOffsets::VOCAB_SIZE, D, 0, useVulkan);
    vkEngine.matmul(logits.data(), xFinal.data(), gradTokEmb,
                    LM5MOffsets::VOCAB_SIZE, numTargets, D, 2, useVulkan);

    result.loss = batchLoss / static_cast<float>(numTargets);

    // 3. BACKPROPAGATION THROUGH LAYERS
    std::vector<float> dXCurrent(seqLen * D);

    // Backprop final_norm
    for (uint32_t t = 0; t < seqLen; ++t) {
        float invStd = invStdFinal[t];
        const float* xRow = &currentX[t * D];
        float sumDxW = 0.0f;
        for (uint32_t i = 0; i < D; ++i) {
            gradFinalNorm[i] += dXFinal[t * D + i] * xRow[i] * invStd;
            sumDxW += dXFinal[t * D + i] * finalNorm[i] * xRow[i];
        }
        float factor = sumDxW / (static_cast<float>(D) * (1.0f / (invStd * invStd)));
        #pragma clang loop vectorize(enable)
        for (uint32_t i = 0; i < D; ++i) {
            dXCurrent[t * D + i] = (dXFinal[t * D + i] * finalNorm[i] - xRow[i] * factor) * invStd;
        }
    }

    std::vector<float> dOutHeads(seqLen * D);
    std::vector<float> dXMid(seqLen * D);
    std::vector<float> dXPrev(seqLen * D);
    std::vector<float> dQ(seqLen * D);
    std::vector<float> dK(seqLen * D);
    std::vector<float> dV(seqLen * D);

    for (int l = 4; l >= 0; --l) {
        const auto& lw = layers[l];
        const auto& lc = caches[l];

        // 1. SwiGLU FFN backward. Batch all dense gradients using the same
        // [output,input] layout as the model's stored parameter matrices.
        std::vector<float> dSwigluAll(static_cast<size_t>(seqLen) * DFF);
        std::vector<float> dGateAll(static_cast<size_t>(seqLen) * DFF);
        std::vector<float> dUpAll(static_cast<size_t>(seqLen) * DFF);
        vkEngine.matmul(dXCurrent.data(), lw.w2, dSwigluAll.data(), seqLen, D, DFF, 0, useVulkan);
        vkEngine.swigluBackward(lc.gate.data(), lc.up.data(), dSwigluAll.data(),
                                dGateAll.data(), dUpAll.data(),
                                static_cast<uint32_t>(seqLen * DFF), useVulkan);
        for (uint32_t t = 0; t < seqLen; ++t) {
            std::memcpy(&dXMid[t * D], &dXCurrent[t * D], D * sizeof(float));
        }

        // dW2 = dX^T * swiglu; dW1 = dGate^T * hFfn; dW3 = dUp^T * hFfn.
        vkEngine.matmul(dXCurrent.data(), lc.swiglu.data(), lw.gW2, D, seqLen, DFF, 2, useVulkan);
        vkEngine.matmul(dGateAll.data(), lc.hFfn.data(), lw.gW1, DFF, seqLen, D, 2, useVulkan);
        vkEngine.matmul(dUpAll.data(), lc.hFfn.data(), lw.gW3, DFF, seqLen, D, 2, useVulkan);

        // dHffn = dGate * W1 + dUp * W3.
        std::vector<float> dHGateAll(static_cast<size_t>(seqLen) * D);
        std::vector<float> dHUpAll(static_cast<size_t>(seqLen) * D);
        vkEngine.matmul(dGateAll.data(), lw.w1, dHGateAll.data(), seqLen, DFF, D, 0, useVulkan);
        vkEngine.matmul(dUpAll.data(), lw.w3, dHUpAll.data(), seqLen, DFF, D, 0, useVulkan);

        for (uint32_t t = 0; t < seqLen; ++t) {
            const size_t row = static_cast<size_t>(t) * D;
            const float* xRow = lc.xMid.data() + row;
            const float invStd = lc.invStdFfn[t];
            float sumDxW = 0.0f;
            for (uint32_t i = 0; i < D; ++i) {
                const float dH = dHGateAll[row + i] + dHUpAll[row + i];
                lw.gFfnNorm[i] += dH * xRow[i] * invStd;
                sumDxW += dH * lw.ffnNorm[i] * xRow[i];
            }
            const float factor = sumDxW / (static_cast<float>(D) * (1.0f / (invStd * invStd)));
            #pragma clang loop vectorize(enable)
            for (uint32_t i = 0; i < D; ++i) {
                const float dH = dHGateAll[row + i] + dHUpAll[row + i];
                dXMid[row + i] += (dH * lw.ffnNorm[i] - xRow[i] * factor) * invStd;
            }
        }

        // 2. Attention backward. dOutHeads = dXMid * W_o and
        // dW_o = dXMid^T * outHeads; both are batched dense operations.
        vkEngine.matmul(dXMid.data(), lw.wo, dOutHeads.data(), seqLen, D, D, 0, useVulkan);
        vkEngine.matmul(dXMid.data(), lc.outHeads.data(), lw.gWo, D, seqLen, D, 2, useVulkan);
        for (uint32_t t = 0; t < seqLen; ++t) {
            std::memcpy(&dXPrev[t * D], &dXMid[t * D], D * sizeof(float));
        }

        std::fill(dQ.begin(), dQ.end(), 0.0f);
        std::fill(dK.begin(), dK.end(), 0.0f);
        std::fill(dV.begin(), dV.end(), 0.0f);

        for (uint32_t h = 0; h < H; ++h) {
            uint32_t hOffset = h * HD;
            for (uint32_t t = 0; t < seqLen; ++t) {
                std::vector<float> dScore(t + 1);
                for (uint32_t s = 0; s <= t; ++s) {
                    float dValSum = 0.0f;
                    float attn = lc.attnWeights[(h * seqLen + t) * seqLen + s];
                    #pragma clang loop vectorize(enable)
                    for (uint32_t i = 0; i < HD; ++i) {
                        float dOut = dOutHeads[t * D + hOffset + i];
                        dV[s * D + hOffset + i] += attn * dOut;
                        dValSum += dOut * lc.v[s * D + hOffset + i];
                    }
                    dScore[s] = dValSum;
                }

                float sumDScoreP = 0.0f;
                for (uint32_t s = 0; s <= t; ++s) {
                    sumDScoreP += dScore[s] * lc.attnWeights[(h * seqLen + t) * seqLen + s];
                }
                for (uint32_t s = 0; s <= t; ++s) {
                    float dAttn = lc.attnWeights[(h * seqLen + t) * seqLen + s] * (dScore[s] - sumDScoreP) * scale;
                    #pragma clang loop vectorize(enable)
                    for (uint32_t i = 0; i < HD; ++i) {
                        dQ[t * D + hOffset + i] += dAttn * lc.k[s * D + hOffset + i];
                        dK[s * D + hOffset + i] += dAttn * lc.q[t * D + hOffset + i];
                    }
                }
            }
        }

        // Inverse RoPE on dQ and dK before projecting gradients back to activations.
        for (uint32_t t = 0; t < seqLen; ++t) {
            for (uint32_t h = 0; h < H; ++h) {
                const uint32_t hOffset = h * HD;
                for (uint32_t i = 0; i < 32; ++i) {
                    const float ang = t * s_invFreq[i];
                    const float c = std::cos(ang);
                    const float sn = std::sin(ang);

                    const float dq1 = dQ[t * D + hOffset + i];
                    const float dq2 = dQ[t * D + hOffset + i + 32];
                    dQ[t * D + hOffset + i] = dq1 * c + dq2 * sn;
                    dQ[t * D + hOffset + i + 32] = dq2 * c - dq1 * sn;

                    const float dk1 = dK[t * D + hOffset + i];
                    const float dk2 = dK[t * D + hOffset + i + 32];
                    dK[t * D + hOffset + i] = dk1 * c + dk2 * sn;
                    dK[t * D + hOffset + i + 32] = dk2 * c - dk1 * sn;
                }
            }
        }

        // dHAttn = dQ*Wq + dK*Wk + dV*Wv, and dW = dQ^T*hAttn.
        // Weight gradients are independent dense products; batch all sequence rows.
        std::vector<float> dHq(static_cast<size_t>(seqLen) * D);
        std::vector<float> dHk(static_cast<size_t>(seqLen) * D);
        std::vector<float> dHv(static_cast<size_t>(seqLen) * D);
        vkEngine.matmul(dQ.data(), lw.wq, dHq.data(), seqLen, D, D, 0, useVulkan);
        vkEngine.matmul(dK.data(), lw.wk, dHk.data(), seqLen, D, D, 0, useVulkan);
        vkEngine.matmul(dV.data(), lw.wv, dHv.data(), seqLen, D, D, 0, useVulkan);
        vkEngine.matmul(dQ.data(), lc.hAttn.data(), lw.gWq, D, seqLen, D, 2, useVulkan);
        vkEngine.matmul(dK.data(), lc.hAttn.data(), lw.gWk, D, seqLen, D, 2, useVulkan);
        vkEngine.matmul(dV.data(), lc.hAttn.data(), lw.gWv, D, seqLen, D, 2, useVulkan);

        // Attention RMSNorm backward.
        for (uint32_t t = 0; t < seqLen; ++t) {
            const size_t row = static_cast<size_t>(t) * D;
            const float invStd = lc.invStdAttn[t];
            const float* prevX = (l == 0) ? &xIn[row] : &caches[l - 1].xOut[row];
            float sumDxW = 0.0f;
            for (uint32_t i = 0; i < D; ++i) {
                const float dH = dHq[row + i] + dHk[row + i] + dHv[row + i];
                lw.gAttnNorm[i] += dH * prevX[i] * invStd;
                sumDxW += dH * lw.attnNorm[i] * prevX[i];
            }
            const float factor = sumDxW / (static_cast<float>(D) * (1.0f / (invStd * invStd)));
            #pragma clang loop vectorize(enable)
            for (uint32_t i = 0; i < D; ++i) {
                const float dH = dHq[row + i] + dHk[row + i] + dHv[row + i];
                dXPrev[row + i] += (dH * lw.attnNorm[i] - prevX[i] * factor) * invStd;
            }
        }

        dXCurrent = dXPrev;
    }

    // Backprop to Token Embeddings
    for (uint32_t t = 0; t < seqLen; ++t) {
        int32_t tok = tokens[t];
        if (tok >= 0 && tok < static_cast<int32_t>(LM5MOffsets::VOCAB_SIZE)) {
            float* gRow = gradTokEmb + (tok * D);
            const float* dRow = &dXCurrent[t * D];
            #pragma clang loop vectorize(enable)
            for (uint32_t i = 0; i < D; ++i) {
                gRow[i] += dRow[i];
            }
        }
    }

    // 4. GLOBAL GRADIENT L2 NORM & CLIPPING
    double sumSqGrad = 0.0;
    #pragma clang loop vectorize(enable)
    for (uint32_t i = 0; i < LM5MOffsets::TOTAL_PARAMS; ++i) {
        float g = allGrads[i];
        sumSqGrad += static_cast<double>(g * g);
    }
    float globalNorm = std::sqrt(sumSqGrad);
    result.gradNorm = globalNorm;

    float clipScale = (globalNorm > maxGradNorm) ? (maxGradNorm / globalNorm) : 1.0f;

    // 5. DECOUPLED ADAMW OPTIMIZER STEP
    float b1 = 0.9f;
    float b2 = 0.95f;
    float eps = 1e-8f;
    float b1Pow = std::pow(b1, static_cast<float>(step));
    float b2Pow = std::pow(b2, static_cast<float>(step));
    float alphaT = learningRate * std::sqrt(1.0f - b2Pow) / (1.0f - b1Pow);
    // With bias-corrected alphaT and uncorrected second moment, scale epsilon
    // by sqrt(1-beta2^step) to match standard AdamW denominator semantics.
    float epsCorrected = eps * std::sqrt(1.0f - b2Pow);

    auto updateAdam = [&](float* param, float* grad, float* m, float* v, uint32_t count, bool is2D) {
        vkEngine.adamWUpdate(param, grad, m, v, count, b1, b2, alphaT,
                             epsCorrected, learningRate, weightDecay, clipScale, is2D, useVulkan);
    };

    // tokEmb [4096, 256]: 2D weight matrix -> weight decay active
    updateAdam(allWeights, allGrads, allM, allV, LM5MOffsets::TOK_EMB_SIZE, true);

    // 5 Transformer Layers
    for (uint32_t l = 0; l < 5; ++l) {
        uint32_t off = LM5MOffsets::TOK_EMB_SIZE + l * LM5MOffsets::LAYER_TOTAL;
        // attnNorm (1D)
        updateAdam(allWeights + off, allGrads + off, allM + off, allV + off, 256, false);
        // wq, wk, wv, wo (2D)
        updateAdam(allWeights + off + 256, allGrads + off + 256, allM + off + 256, allV + off + 256, 4 * 65536, true);
        // ffnNorm (1D)
        updateAdam(allWeights + off + 256 + 262144, allGrads + off + 256 + 262144, allM + off + 256 + 262144, allV + off + 256 + 262144, 256, false);
        // w1, w3, w2 (2D)
        updateAdam(allWeights + off + 262656, allGrads + off + 262656, allM + off + 262656, allV + off + 262656, 3 * 180224, true);
    }

    // finalNorm (1D)
    updateAdam(finalNorm, gradFinalNorm, allM + 5065216, allV + 5065216, 256, false);

    // Report GPU only if at least one Vulkan compute dispatch really completed.
    result.vulkanUsed = useVulkan && (vkEngine.gpuDispatchCount() > gpuDispatchesAtStart);
    return result;
}

} // namespace nexora
