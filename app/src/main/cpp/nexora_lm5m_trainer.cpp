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

static inline float applyRmsNorm(const float* x, const float* w, float* out, uint32_t D, float eps = 1e-5f) {
    float sumSq = 0.0f;
    #pragma clang loop vectorize(enable)
    for (uint32_t i = 0; i < D; ++i) {
        sumSq += x[i] * x[i];
    }
    float invStd = 1.0f / std::sqrt((sumSq / static_cast<float>(D)) + eps);
    #pragma clang loop vectorize(enable)
    for (uint32_t i = 0; i < D; ++i) {
        out[i] = x[i] * invStd * w[i];
    }
    return invStd;
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
        std::vector<float> silu;
        std::vector<float> swiglu;
        std::vector<float> xOut;

        LayerCache(uint32_t T, uint32_t D, uint32_t DFF, uint32_t H)
            : hAttn(T * D), invStdAttn(T), q(T * D), k(T * D), v(T * D),
              attnWeights(H * T * T), outHeads(T * D), xMid(T * D),
              hFfn(T * D), invStdFfn(T), gate(T * DFF), up(T * DFF),
              silu(T * DFF), swiglu(T * DFF), xOut(T * D) {}
    };

    std::vector<LayerCache> caches;
    caches.reserve(5);
    for (uint32_t l = 0; l < 5; ++l) {
        caches.emplace_back(seqLen, D, DFF, H);
    }

    const float* currentX = xIn.data();

    std::vector<float> hTemp(D);
    std::vector<float> downTemp(D);

    for (uint32_t l = 0; l < 5; ++l) {
        auto& lc = caches[l];
        const auto& lw = layers[l];

        // Attention Pre-RMSNorm & Projections
        for (uint32_t t = 0; t < seqLen; ++t) {
            lc.invStdAttn[t] = applyRmsNorm(&currentX[t * D], lw.attnNorm, &lc.hAttn[t * D], D);

            // Q = h * wq, K = h * wk, V = h * wv
            vkEngine.matmul(&lc.hAttn[t * D], lw.wq, &lc.q[t * D], 1, D, D, 0, useVulkan);
            vkEngine.matmul(&lc.hAttn[t * D], lw.wk, &lc.k[t * D], 1, D, D, 0, useVulkan);
            vkEngine.matmul(&lc.hAttn[t * D], lw.wv, &lc.v[t * D], 1, D, D, 0, useVulkan);

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

        // Wo projection + Residual -> xMid
        for (uint32_t t = 0; t < seqLen; ++t) {
            vkEngine.matmul(&lc.outHeads[t * D], lw.wo, hTemp.data(), 1, D, D, 0, useVulkan);
            #pragma clang loop vectorize(enable)
            for (uint32_t i = 0; i < D; ++i) {
                lc.xMid[t * D + i] = currentX[t * D + i] + hTemp[i];
            }
        }

        // SwiGLU FFN
        for (uint32_t t = 0; t < seqLen; ++t) {
            lc.invStdFfn[t] = applyRmsNorm(&lc.xMid[t * D], lw.ffnNorm, &lc.hFfn[t * D], D);

            vkEngine.matmul(&lc.hFfn[t * D], lw.w1, &lc.gate[t * DFF], 1, D, DFF, 0, useVulkan);
            vkEngine.matmul(&lc.hFfn[t * D], lw.w3, &lc.up[t * DFF], 1, D, DFF, 0, useVulkan);

            #pragma clang loop vectorize(enable)
            for (uint32_t j = 0; j < DFF; ++j) {
                float z = lc.gate[t * DFF + j];
                float sVal = z / (1.0f + std::exp(-z));
                lc.silu[t * DFF + j] = sVal;
                lc.swiglu[t * DFF + j] = sVal * lc.up[t * DFF + j];
            }

            vkEngine.matmul(&lc.swiglu[t * DFF], lw.w2, downTemp.data(), 1, DFF, D, 0, useVulkan);
            #pragma clang loop vectorize(enable)
            for (uint32_t i = 0; i < D; ++i) {
                lc.xOut[t * D + i] = lc.xMid[t * D + i] + downTemp[i];
            }
        }

        currentX = lc.xOut.data();
    }

    // Final RMSNorm
    std::vector<float> xFinal(seqLen * D);
    std::vector<float> invStdFinal(seqLen);
    for (uint32_t t = 0; t < seqLen; ++t) {
        invStdFinal[t] = applyRmsNorm(&currentX[t * D], finalNorm, &xFinal[t * D], D);
    }

    // 2. LOSS & OUTPUT LOGITS GRADIENT
    uint32_t numTargets = seqLen - 1;
    float batchLoss = 0.0f;
    std::vector<float> dXFinal(seqLen * D, 0.0f);
    std::vector<float> logits(LM5MOffsets::VOCAB_SIZE);
    float invNormFactor = 1.0f / static_cast<float>(numTargets);

    for (uint32_t t = 0; t < numTargets; ++t) {
        // Logits = xFinal[t] * tokEmb^T
        vkEngine.matmul(&xFinal[t * D], tokEmb, logits.data(), 1, D, LM5MOffsets::VOCAB_SIZE, 1, useVulkan);
        int32_t target = tokens[t + 1];

        float maxL = -1e30f;
        for (uint32_t v = 0; v < LM5MOffsets::VOCAB_SIZE; ++v) {
            if (logits[v] > maxL) maxL = logits[v];
        }

        float sumExp = 0.0f;
        for (uint32_t v = 0; v < LM5MOffsets::VOCAB_SIZE; ++v) {
            float e = std::exp(logits[v] - maxL);
            logits[v] = e;
            sumExp += e;
        }
        float invSum = 1.0f / sumExp;
        for (uint32_t v = 0; v < LM5MOffsets::VOCAB_SIZE; ++v) {
            logits[v] *= invSum;
        }

        float targetProb = logits[target];
        batchLoss += -std::log(std::max(1e-12f, targetProb));

        logits[target] -= 1.0f; // dLogits

        for (uint32_t v = 0; v < LM5MOffsets::VOCAB_SIZE; ++v) {
            float dL = logits[v] * invNormFactor;
            const float* embRow = tokEmb + (v * D);
            float* gEmbRow = gradTokEmb + (v * D);
            #pragma clang loop vectorize(enable)
            for (uint32_t i = 0; i < D; ++i) {
                dXFinal[t * D + i] += dL * embRow[i];
                gEmbRow[i] += dL * xFinal[t * D + i];
            }
        }
    }

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

    std::vector<float> dSwiglu(DFF);
    std::vector<float> dGate(DFF);
    std::vector<float> dUp(DFF);
    std::vector<float> dHFfn(D);
    std::vector<float> dHAttn(D);
    std::vector<float> dOutHeads(seqLen * D);
    std::vector<float> dXMid(seqLen * D);
    std::vector<float> dXPrev(seqLen * D);
    std::vector<float> dQ(seqLen * D);
    std::vector<float> dK(seqLen * D);
    std::vector<float> dV(seqLen * D);

    for (int l = 4; l >= 0; --l) {
        const auto& lw = layers[l];
        const auto& lc = caches[l];

        // 1. SwiGLU FFN backward
        for (uint32_t t = 0; t < seqLen; ++t) {
            std::memcpy(&dXMid[t * D], &dXCurrent[t * D], D * sizeof(float));

            // Down projection backward:
            // dSwiglu = dXCurrent * w2^T
            vkEngine.matmul(&dXCurrent[t * D], lw.w2, dSwiglu.data(), 1, D, DFF, 1, useVulkan);

            // gradW2 += dXCurrent^T * swiglu
            for (uint32_t i = 0; i < D; ++i) {
                float dxi = dXCurrent[t * D + i];
                float* gw2Row = lw.gW2 + (i * DFF);
                const float* swRow = &lc.swiglu[t * DFF];
                #pragma clang loop vectorize(enable)
                for (uint32_t j = 0; j < DFF; ++j) {
                    gw2Row[j] += dxi * swRow[j];
                }
            }

            // SwiGLU activation backward
            #pragma clang loop vectorize(enable)
            for (uint32_t j = 0; j < DFF; ++j) {
                float z = lc.gate[t * DFF + j];
                float sig = 1.0f / (1.0f + std::exp(-z));
                float siluVal = z * sig;
                float ds = dSwiglu[j];

                dUp[j] = ds * siluVal;
                float dsiluVal = ds * lc.up[t * DFF + j];
                dGate[j] = dsiluVal * (sig + siluVal * (1.0f - sig));
            }

            // Up & Gate projections backward:
            std::fill(dHFfn.begin(), dHFfn.end(), 0.0f);
            for (uint32_t j = 0; j < DFF; ++j) {
                float dg = dGate[j];
                float du = dUp[j];
                float* gw1Row = lw.gW1 + (j * D);
                float* gw3Row = lw.gW3 + (j * D);
                const float* w1Row = lw.w1 + (j * D);
                const float* w3Row = lw.w3 + (j * D);
                const float* hRow = &lc.hFfn[t * D];
                #pragma clang loop vectorize(enable)
                for (uint32_t i = 0; i < D; ++i) {
                    gw1Row[i] += dg * hRow[i];
                    gw3Row[i] += du * hRow[i];
                    dHFfn[i] += dg * w1Row[i] + du * w3Row[i];
                }
            }

            // FFN RMSNorm backward
            float invStd = lc.invStdFfn[t];
            const float* xRow = &lc.xMid[t * D];
            float sumDxW = 0.0f;
            for (uint32_t i = 0; i < D; ++i) {
                lw.gFfnNorm[i] += dHFfn[i] * xRow[i] * invStd;
                sumDxW += dHFfn[i] * lw.ffnNorm[i] * xRow[i];
            }
            float factor = sumDxW / (static_cast<float>(D) * (1.0f / (invStd * invStd)));
            #pragma clang loop vectorize(enable)
            for (uint32_t i = 0; i < D; ++i) {
                dXMid[t * D + i] += (dHFfn[i] * lw.ffnNorm[i] - xRow[i] * factor) * invStd;
            }
        }

        // 2. Attention backward
        for (uint32_t t = 0; t < seqLen; ++t) {
            std::memcpy(&dXPrev[t * D], &dXMid[t * D], D * sizeof(float));

            // wo projection backward:
            vkEngine.matmul(&dXMid[t * D], lw.wo, &dOutHeads[t * D], 1, D, D, 1, useVulkan);

            // gradWo += dXMid^T * outHeads
            for (uint32_t i = 0; i < D; ++i) {
                float dxi = dXMid[t * D + i];
                float* gwoRow = lw.gWo + (i * D);
                const float* ohRow = &lc.outHeads[t * D];
                #pragma clang loop vectorize(enable)
                for (uint32_t j = 0; j < D; ++j) {
                    gwoRow[j] += dxi * ohRow[j];
                }
            }
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

        // Inverse RoPE on dQ and dK
        for (uint32_t t = 0; t < seqLen; ++t) {
            for (uint32_t h = 0; h < H; ++h) {
                uint32_t hOffset = h * HD;
                for (uint32_t i = 0; i < 32; ++i) {
                    float ang = t * s_invFreq[i];
                    float c = std::cos(ang);
                    float s = std::sin(ang);

                    float dq1 = dQ[t * D + hOffset + i];
                    float dq2 = dQ[t * D + hOffset + i + 32];
                    dQ[t * D + hOffset + i] = dq1 * c + dq2 * s;
                    dQ[t * D + hOffset + i + 32] = dq2 * c - dq1 * s;

                    float dk1 = dK[t * D + hOffset + i];
                    float dk2 = dK[t * D + hOffset + i + 32];
                    dK[t * D + hOffset + i] = dk1 * c + dk2 * s;
                    dK[t * D + hOffset + i + 32] = dk2 * c - dk1 * s;
                }
            }

            // Projections backward
            std::fill(dHAttn.begin(), dHAttn.end(), 0.0f);
            for (uint32_t j = 0; j < D; ++j) {
                float dq = dQ[t * D + j];
                float dk = dK[t * D + j];
                float dv = dV[t * D + j];
                float* gwqRow = lw.gWq + (j * D);
                float* gwkRow = lw.gWk + (j * D);
                float* gwvRow = lw.gWv + (j * D);
                const float* wqRow = lw.wq + (j * D);
                const float* wkRow = lw.wk + (j * D);
                const float* wvRow = lw.wv + (j * D);
                const float* hRow = &lc.hAttn[t * D];
                #pragma clang loop vectorize(enable)
                for (uint32_t i = 0; i < D; ++i) {
                    gwqRow[i] += dq * hRow[i];
                    gwkRow[i] += dk * hRow[i];
                    gwvRow[i] += dv * hRow[i];
                    dHAttn[i] += dq * wqRow[i] + dk * wkRow[i] + dv * wvRow[i];
                }
            }

            // Attention RMSNorm backward
            float invStd = lc.invStdAttn[t];
            const float* prevX = (l == 0) ? &xIn[t * D] : &caches[l - 1].xOut[t * D];
            float sumDxW = 0.0f;
            for (uint32_t i = 0; i < D; ++i) {
                lw.gAttnNorm[i] += dHAttn[i] * prevX[i] * invStd;
                sumDxW += dHAttn[i] * lw.attnNorm[i] * prevX[i];
            }
            float factor = sumDxW / (static_cast<float>(D) * (1.0f / (invStd * invStd)));
            #pragma clang loop vectorize(enable)
            for (uint32_t i = 0; i < D; ++i) {
                dXPrev[t * D + i] += (dHAttn[i] * lw.attnNorm[i] - prevX[i] * factor) * invStd;
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

    auto updateAdam = [&](float* param, float* grad, float* m, float* v, uint32_t count, bool is2D) {
        vkEngine.adamWUpdate(param, grad, m, v, count, b1, b2, alphaT, eps, weightDecay, clipScale, is2D, useVulkan);
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
