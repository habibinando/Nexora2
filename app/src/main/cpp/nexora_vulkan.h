#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace nexora {

struct VulkanDeviceInfo {
    bool available = false;
    std::string deviceName = "Vulkan Inactive";
    std::string driverVersion = "";
    uint32_t apiVersion = 0;
    uint32_t computeQueueIndex = 0;
    bool hasUnifiedMemory = false;
};

class VulkanComputeEngine {
public:
    static VulkanComputeEngine& instance();

    bool init();
    void cleanup();
    bool isAvailable() const { return m_info.available; }
    const VulkanDeviceInfo& getDeviceInfo() const { return m_info; }
    uint64_t gpuDispatchCount() const { return m_gpuDispatchCount; }
    std::string getStatusString() const;

    // Matmul: mode 0: C = A * B, mode 1: C = A * B^T, mode 2: C = A^T * B
    bool matmul(const float* A, const float* B, float* C, uint32_t M, uint32_t K, uint32_t N, uint32_t mode = 0, bool useGpu = true);

    // SwiGLU activation: swiglu = (gate * sigmoid(gate)) * up
    bool swiglu(const float* gate, const float* up, float* swigluOut, uint32_t count);

    // SwiGLU backward pass: produces dGate and dUp from dSwiglu
    bool swigluBackward(const float* gate, const float* up, const float* dSwiglu, float* dGate, float* dUp, uint32_t count);

    // RMSNorm: outNorm = x / sqrt(mean(x^2) + eps) * w
    bool rmsNorm(const float* x, const float* w, float* outNorm, float* outInvStd, uint32_t D, float eps, uint32_t totalTokens);

    // AdamW Decoupled parameter update
    bool adamWUpdate(float* param, float* grad, float* m, float* v, uint32_t count,
                     float beta1, float beta2, float alphaT, float eps, float learningRate,
                     float weightDecay, float clipScale, bool is2D, bool useGpu = true);

private:
    VulkanComputeEngine();
    ~VulkanComputeEngine();

    VulkanDeviceInfo m_info;
    uint64_t m_gpuDispatchCount = 0;
    void* m_vulkanLib = nullptr;

    // Internal state pointer (VkInstance, VkDevice, VkQueue, Pipelines, Buffers, etc.)
    struct Impl;
    Impl* m_impl = nullptr;
};

} // namespace nexora
