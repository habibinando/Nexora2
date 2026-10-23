#pragma once
#include <cstdint>
#include <vector>

namespace nexora {

struct LM5MOffsets {
    static constexpr uint32_t TOTAL_PARAMS = 5065472;
    static constexpr uint32_t VOCAB_SIZE = 4096;
    static constexpr uint32_t D_MODEL = 256;
    static constexpr uint32_t D_FF = 704;
    static constexpr uint32_t NUM_LAYERS = 5;
    static constexpr uint32_t NUM_HEADS = 4;
    static constexpr uint32_t HEAD_DIM = 64;

    static constexpr uint32_t TOK_EMB_SIZE = VOCAB_SIZE * D_MODEL; // 1048576
    static constexpr uint32_t ATTN_NORM_SIZE = D_MODEL; // 256
    static constexpr uint32_t PROJ_SIZE = D_MODEL * D_MODEL; // 65536
    static constexpr uint32_t FFN_NORM_SIZE = D_MODEL; // 256
    static constexpr uint32_t W1_SIZE = D_FF * D_MODEL; // 180224
    static constexpr uint32_t W3_SIZE = D_FF * D_MODEL; // 180224
    static constexpr uint32_t W2_SIZE = D_MODEL * D_FF; // 180224
    static constexpr uint32_t FINAL_NORM_SIZE = D_MODEL; // 256

    // Single layer total: 256 + 4*65536 + 256 + 3*180224 = 803328
    static constexpr uint32_t LAYER_TOTAL = 803328;
};

struct TrainStepResult {
    float loss = 0.0f;
    float gradNorm = 0.0f;
    bool vulkanUsed = false;
};

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
);

} // namespace nexora
