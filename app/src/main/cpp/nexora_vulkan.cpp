#include "nexora_vulkan.h"
#include <nexora_spv.h>
#include <vulkan/vulkan.h>
#include <android/log.h>
#include <dlfcn.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <sstream>

#define TAG "NexoraVulkan"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

namespace nexora {

struct VulkanComputeEngine::Impl {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;
    uint32_t queueFamilyIndex = 0;
    VkPhysicalDeviceMemoryProperties memProperties{};

    VkCommandPool cmdPool = VK_NULL_HANDLE;
    VkCommandBuffer cmdBuffer = VK_NULL_HANDLE;
    VkDescriptorPool descPool = VK_NULL_HANDLE;

    // Shader modules
    VkShaderModule shMatmul = VK_NULL_HANDLE;
    VkShaderModule shAdamW = VK_NULL_HANDLE;
    VkShaderModule shSwiglu = VK_NULL_HANDLE;
    VkShaderModule shSwigluBack = VK_NULL_HANDLE;
    VkShaderModule shRmsNorm = VK_NULL_HANDLE;

    // Descriptor set layouts
    VkDescriptorSetLayout dslMatmul = VK_NULL_HANDLE;
    VkDescriptorSetLayout dslAdamW = VK_NULL_HANDLE;
    VkDescriptorSetLayout dslSwiglu = VK_NULL_HANDLE;
    VkDescriptorSetLayout dslSwigluBack = VK_NULL_HANDLE;
    VkDescriptorSetLayout dslRmsNorm = VK_NULL_HANDLE;

    // Pipeline layouts
    VkPipelineLayout plMatmul = VK_NULL_HANDLE;
    VkPipelineLayout plAdamW = VK_NULL_HANDLE;
    VkPipelineLayout plSwiglu = VK_NULL_HANDLE;
    VkPipelineLayout plSwigluBack = VK_NULL_HANDLE;
    VkPipelineLayout plRmsNorm = VK_NULL_HANDLE;

    // Pipelines
    VkPipeline pipeMatmul = VK_NULL_HANDLE;
    VkPipeline pipeAdamW = VK_NULL_HANDLE;
    VkPipeline pipeSwiglu = VK_NULL_HANDLE;
    VkPipeline pipeSwigluBack = VK_NULL_HANDLE;
    VkPipeline pipeRmsNorm = VK_NULL_HANDLE;

    // Shared reusable GPU buffers for matmul / activations (up to 4MB each)
    static constexpr VkDeviceSize MAX_BUF_SIZE = 16 * 1024 * 1024; // 16MB
    VkBuffer bufA = VK_NULL_HANDLE;
    VkDeviceMemory memA = VK_NULL_HANDLE;
    float* mapA = nullptr;

    VkBuffer bufB = VK_NULL_HANDLE;
    VkDeviceMemory memB = VK_NULL_HANDLE;
    float* mapB = nullptr;

    VkBuffer bufC = VK_NULL_HANDLE;
    VkDeviceMemory memC = VK_NULL_HANDLE;
    float* mapC = nullptr;

    VkBuffer bufD = VK_NULL_HANDLE;
    VkDeviceMemory memD = VK_NULL_HANDLE;
    float* mapD = nullptr;

    VkBuffer bufE = VK_NULL_HANDLE;
    VkDeviceMemory memE = VK_NULL_HANDLE;
    float* mapE = nullptr;

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) &&
                (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        return 0xFFFFFFFF;
    }

    bool createGpuBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer& buffer, VkDeviceMemory& memory, float** mappedPtr) {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            return false;
        }

        VkMemoryRequirements memReq;
        vkGetBufferMemoryRequirements(device, buffer, &memReq);

        uint32_t memType = findMemoryType(memReq.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        if (memType == 0xFFFFFFFF) {
            return false;
        }

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReq.size;
        allocInfo.memoryTypeIndex = memType;

        if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
            return false;
        }

        vkBindBufferMemory(device, buffer, memory, 0);

        if (mappedPtr) {
            void* data = nullptr;
            if (vkMapMemory(device, memory, 0, size, 0, &data) == VK_SUCCESS) {
                *mappedPtr = static_cast<float*>(data);
            }
        }
        return true;
    }

    VkShaderModule createShaderModule(const uint32_t* code, size_t size) {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = size;
        createInfo.pCode = code;
        VkShaderModule module = VK_NULL_HANDLE;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &module) != VK_SUCCESS) {
            return VK_NULL_HANDLE;
        }
        return module;
    }
};

VulkanComputeEngine& VulkanComputeEngine::instance() {
    static VulkanComputeEngine engine;
    return engine;
}

VulkanComputeEngine::VulkanComputeEngine() {
    m_impl = new Impl();
    init();
}

VulkanComputeEngine::~VulkanComputeEngine() {
    cleanup();
    delete m_impl;
}

bool VulkanComputeEngine::init() {
    if (m_info.available) return true;

    // 1. Try to load libvulkan.so
    m_vulkanLib = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    if (!m_vulkanLib) {
        LOGD("Vulkan library libvulkan.so not available in this environment. Falling back to multi-threaded SIMD CPU engine.");
        m_info.available = false;
        m_info.deviceName = "CPU Fallback (Vulkan library not loaded)";
        return false;
    }

    // 2. Create Vulkan Instance
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Nexora LM5M Engine";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "NexoraVulkanCompute";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    // The engine uses Vulkan 1.0 core compute features; query the physical device's actual version below.
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    VkResult res = vkCreateInstance(&createInfo, nullptr, &m_impl->instance);
    if (res != VK_SUCCESS) {
        LOGD("vkCreateInstance failed with code %d. Using high-performance SIMD CPU engine.", res);
        m_info.available = false;
        m_info.deviceName = "CPU Fallback (vkCreateInstance failed)";
        return false;
    }

    // 3. Enumerate Physical Devices
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(m_impl->instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        LOGD("No Vulkan physical devices found.");
        m_info.available = false;
        m_info.deviceName = "CPU Fallback (No Vulkan GPU)";
        return false;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(m_impl->instance, &deviceCount, devices.data());

    // Choose best compute device
    VkPhysicalDevice chosenDevice = VK_NULL_HANDLE;
    uint32_t chosenQueueIndex = 0;
    VkPhysicalDeviceProperties chosenProps{};

    for (const auto& dev : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(dev, &props);

        uint32_t qfCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qfCount, nullptr);
        std::vector<VkQueueFamilyProperties> qfProps(qfCount);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qfCount, qfProps.data());

        for (uint32_t q = 0; q < qfCount; ++q) {
            if (qfProps[q].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                chosenDevice = dev;
                chosenQueueIndex = q;
                chosenProps = props;
                break;
            }
        }
        if (chosenDevice != VK_NULL_HANDLE) break;
    }

    if (chosenDevice == VK_NULL_HANDLE) {
        LOGD("No device with compute queue found.");
        m_info.available = false;
        m_info.deviceName = "CPU Fallback (No compute queue)";
        return false;
    }

    m_impl->physicalDevice = chosenDevice;
    m_impl->queueFamilyIndex = chosenQueueIndex;
    vkGetPhysicalDeviceMemoryProperties(chosenDevice, &m_impl->memProperties);

    m_info.deviceName = chosenProps.deviceName;
    m_info.apiVersion = chosenProps.apiVersion;
    m_info.driverVersion = std::to_string(chosenProps.driverVersion);

    // 4. Create Logical Device
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = chosenQueueIndex;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo devCreateInfo{};
    devCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    devCreateInfo.queueCreateInfoCount = 1;
    devCreateInfo.pQueueCreateInfos = &queueCreateInfo;

    if (vkCreateDevice(chosenDevice, &devCreateInfo, nullptr, &m_impl->device) != VK_SUCCESS) {
        LOGD("vkCreateDevice failed.");
        m_info.available = false;
        m_info.deviceName = "CPU Fallback (Device create failed)";
        return false;
    }

    vkGetDeviceQueue(m_impl->device, chosenQueueIndex, 0, &m_impl->computeQueue);

    // 5. Command Pool & Buffer
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = chosenQueueIndex;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(m_impl->device, &poolInfo, nullptr, &m_impl->cmdPool) != VK_SUCCESS) {
        LOGE("Failed to create Vulkan command pool.");
        return false;
    }

    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = m_impl->cmdPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    vkAllocateCommandBuffers(m_impl->device, &cmdAllocInfo, &m_impl->cmdBuffer);

    // 6. Descriptor Pool
    VkDescriptorPoolSize poolSizes[1] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64 }
    };
    VkDescriptorPoolCreateInfo descPoolInfo{};
    descPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descPoolInfo.maxSets = 32;
    descPoolInfo.poolSizeCount = 1;
    descPoolInfo.pPoolSizes = poolSizes;

    vkCreateDescriptorPool(m_impl->device, &descPoolInfo, nullptr, &m_impl->descPool);

    // 7. Compile Shaders & Pipelines
    // (a) MatMul Tiled (3 storage buffers: A, B, C; PushConstants 16 bytes: M, K, N, mode)
    m_impl->shMatmul = m_impl->createShaderModule(spv_matmul_tiled, spv_matmul_tiled_size);
    if (m_impl->shMatmul != VK_NULL_HANDLE) {
        VkDescriptorSetLayoutBinding bindings[3] = {
            { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
        };
        VkDescriptorSetLayoutCreateInfo dslInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 3, bindings };
        vkCreateDescriptorSetLayout(m_impl->device, &dslInfo, nullptr, &m_impl->dslMatmul);

        VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, 4 * sizeof(uint32_t) };
        VkPipelineLayoutCreateInfo plInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &m_impl->dslMatmul, 1, &pcr };
        vkCreatePipelineLayout(m_impl->device, &plInfo, nullptr, &m_impl->plMatmul);

        VkComputePipelineCreateInfo pipeInfo{};
        pipeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeInfo.layout = m_impl->plMatmul;
        pipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeInfo.stage.module = m_impl->shMatmul;
        pipeInfo.stage.pName = "main";
        vkCreateComputePipelines(m_impl->device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &m_impl->pipeMatmul);
    }

    // (b) AdamW (4 storage buffers: param, grad, m, v; PushConstants 36 bytes)
    m_impl->shAdamW = m_impl->createShaderModule(spv_adamw, spv_adamw_size);
    if (m_impl->shAdamW != VK_NULL_HANDLE) {
        VkDescriptorSetLayoutBinding bindings[4] = {
            { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
        };
        VkDescriptorSetLayoutCreateInfo dslInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 4, bindings };
        vkCreateDescriptorSetLayout(m_impl->device, &dslInfo, nullptr, &m_impl->dslAdamW);

        VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, 9 * sizeof(uint32_t) };
        VkPipelineLayoutCreateInfo plInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &m_impl->dslAdamW, 1, &pcr };
        vkCreatePipelineLayout(m_impl->device, &plInfo, nullptr, &m_impl->plAdamW);

        VkComputePipelineCreateInfo pipeInfo{};
        pipeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeInfo.layout = m_impl->plAdamW;
        pipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeInfo.stage.module = m_impl->shAdamW;
        pipeInfo.stage.pName = "main";
        vkCreateComputePipelines(m_impl->device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &m_impl->pipeAdamW);
    }

    // (c) SwiGLU (3 storage buffers: gate, up, swiglu; PushConstants count)
    m_impl->shSwiglu = m_impl->createShaderModule(spv_swiglu, spv_swiglu_size);
    if (m_impl->shSwiglu != VK_NULL_HANDLE) {
        VkDescriptorSetLayoutBinding bindings[3] = {
            { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
        };
        VkDescriptorSetLayoutCreateInfo dslInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 3, bindings };
        vkCreateDescriptorSetLayout(m_impl->device, &dslInfo, nullptr, &m_impl->dslSwiglu);

        VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, 1 * sizeof(uint32_t) };
        VkPipelineLayoutCreateInfo plInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &m_impl->dslSwiglu, 1, &pcr };
        vkCreatePipelineLayout(m_impl->device, &plInfo, nullptr, &m_impl->plSwiglu);

        VkComputePipelineCreateInfo pipeInfo{};
        pipeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeInfo.layout = m_impl->plSwiglu;
        pipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeInfo.stage.module = m_impl->shSwiglu;
        pipeInfo.stage.pName = "main";
        vkCreateComputePipelines(m_impl->device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &m_impl->pipeSwiglu);
    }

    // (d) SwiGLU backward (5 storage buffers; count push constant)
    m_impl->shSwigluBack = m_impl->createShaderModule(spv_swiglu_backward, spv_swiglu_backward_size);
    if (m_impl->shSwigluBack != VK_NULL_HANDLE) {
        VkDescriptorSetLayoutBinding bindings[5] = {
            { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
        };
        VkDescriptorSetLayoutCreateInfo dslInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 5, bindings };
        if (vkCreateDescriptorSetLayout(m_impl->device, &dslInfo, nullptr, &m_impl->dslSwigluBack) == VK_SUCCESS) {
            VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) };
            VkPipelineLayoutCreateInfo plInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &m_impl->dslSwigluBack, 1, &pcr };
            if (vkCreatePipelineLayout(m_impl->device, &plInfo, nullptr, &m_impl->plSwigluBack) == VK_SUCCESS) {
                VkComputePipelineCreateInfo pipeInfo{};
                pipeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
                pipeInfo.layout = m_impl->plSwigluBack;
                pipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                pipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
                pipeInfo.stage.module = m_impl->shSwigluBack;
                pipeInfo.stage.pName = "main";
                vkCreateComputePipelines(m_impl->device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &m_impl->pipeSwigluBack);
            }
        }
    }

    // (e) RMSNorm (x, weights, output, inverse standard deviation)
    m_impl->shRmsNorm = m_impl->createShaderModule(spv_rmsnorm, spv_rmsnorm_size);
    if (m_impl->shRmsNorm != VK_NULL_HANDLE) {
        VkDescriptorSetLayoutBinding bindings[4] = {
            { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
        };
        VkDescriptorSetLayoutCreateInfo dslInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 4, bindings };
        if (vkCreateDescriptorSetLayout(m_impl->device, &dslInfo, nullptr, &m_impl->dslRmsNorm) == VK_SUCCESS) {
            struct RmsPushLayout { uint32_t D; float eps; uint32_t totalTokens; };
            VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RmsPushLayout) };
            VkPipelineLayoutCreateInfo plInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &m_impl->dslRmsNorm, 1, &pcr };
            if (vkCreatePipelineLayout(m_impl->device, &plInfo, nullptr, &m_impl->plRmsNorm) == VK_SUCCESS) {
                VkComputePipelineCreateInfo pipeInfo{};
                pipeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
                pipeInfo.layout = m_impl->plRmsNorm;
                pipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                pipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
                pipeInfo.stage.module = m_impl->shRmsNorm;
                pipeInfo.stage.pName = "main";
                vkCreateComputePipelines(m_impl->device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &m_impl->pipeRmsNorm);
            }
        }
    }

    // (f) Create shared mapped memory buffers used by all compute pipelines.
    m_impl->createGpuBuffer(Impl::MAX_BUF_SIZE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, m_impl->bufA, m_impl->memA, &m_impl->mapA);
    m_impl->createGpuBuffer(Impl::MAX_BUF_SIZE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, m_impl->bufB, m_impl->memB, &m_impl->mapB);
    m_impl->createGpuBuffer(Impl::MAX_BUF_SIZE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, m_impl->bufC, m_impl->memC, &m_impl->mapC);
    m_impl->createGpuBuffer(Impl::MAX_BUF_SIZE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, m_impl->bufD, m_impl->memD, &m_impl->mapD);
    m_impl->createGpuBuffer(Impl::MAX_BUF_SIZE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, m_impl->bufE, m_impl->memE, &m_impl->mapE);

    m_info.available = (m_impl->pipeMatmul != VK_NULL_HANDLE);
    LOGD("Vulkan Engine Initialized. Device: %s, Available: %d", m_info.deviceName.c_str(), m_info.available);
    return m_info.available;
}

std::string VulkanComputeEngine::getStatusString() const {
    if (!m_info.available) {
        return "Vulkan GPU compute inactive; CPU fallback active";
    }

    // Display the physical device API version reported by the Vulkan runtime.
    // Do not label a device as Vulkan 1.2+ unless the runtime actually reports it.
    const uint32_t version = m_info.apiVersion;
    std::ostringstream ss;
    if (version != 0) {
        ss << "Vulkan device API "
           << VK_VERSION_MAJOR(version) << "."
           << VK_VERSION_MINOR(version) << "."
           << VK_VERSION_PATCH(version);
    } else {
        ss << "Vulkan device API version unavailable";
    }
    ss << " | Compute device: " << m_info.deviceName;
    return ss.str();
}

bool VulkanComputeEngine::matmul(const float* A, const float* B, float* C, uint32_t M, uint32_t K, uint32_t N, uint32_t mode, bool useGpu) {
    if (!useGpu || !m_info.available || !m_impl->pipeMatmul || !m_impl->mapA || !m_impl->mapB || !m_impl->mapC) {
        // Fallback fast C++20 SIMD matrix multiplication
        if (mode == 0) { // C = A * B
            for (uint32_t m = 0; m < M; ++m) {
                const float* aRow = A + (m * K);
                float* cRow = C + (m * N);
                for (uint32_t n = 0; n < N; ++n) cRow[n] = 0.0f;
                for (uint32_t k = 0; k < K; ++k) {
                    float ak = aRow[k];
                    const float* bRow = B + (k * N);
                    #pragma clang loop vectorize(enable)
                    for (uint32_t n = 0; n < N; ++n) {
                        cRow[n] += ak * bRow[n];
                    }
                }
            }
        } else if (mode == 1) { // C = A * B^T (B is N x K)
            for (uint32_t m = 0; m < M; ++m) {
                const float* aRow = A + (m * K);
                float* cRow = C + (m * N);
                for (uint32_t n = 0; n < N; ++n) {
                    const float* bRow = B + (n * K);
                    float sum = 0.0f;
                    #pragma clang loop vectorize(enable)
                    for (uint32_t k = 0; k < K; ++k) {
                        sum += aRow[k] * bRow[k];
                    }
                    cRow[n] = sum;
                }
            }
        } else if (mode == 2) { // C = A^T * B (A is K x M, B is K x N)
            for (uint32_t m = 0; m < M; ++m) {
                float* cRow = C + (m * N);
                for (uint32_t n = 0; n < N; ++n) cRow[n] = 0.0f;
            }
            for (uint32_t k = 0; k < K; ++k) {
                const float* aRow = A + (k * M);
                const float* bRow = B + (k * N);
                for (uint32_t m = 0; m < M; ++m) {
                    float am = aRow[m];
                    float* cRow = C + (m * N);
                    #pragma clang loop vectorize(enable)
                    for (uint32_t n = 0; n < N; ++n) {
                        cRow[n] += am * bRow[n];
                    }
                }
            }
        }
        return true;
    }

    size_t sizeA = (mode == 2 ? K * M : M * K) * sizeof(float);
    size_t sizeB = (mode == 1 ? N * K : K * N) * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    // Oversized matrices still produce a valid result via CPU fallback. Returning
    // false here used to leave output buffers untouched for long sequences.
    if (sizeA > Impl::MAX_BUF_SIZE || sizeB > Impl::MAX_BUF_SIZE || sizeC > Impl::MAX_BUF_SIZE) {
        return matmul(A, B, C, M, K, N, mode, false);
    }

    // Direct Host-Visible UMA memory copy
    std::memcpy(m_impl->mapA, A, sizeA);
    std::memcpy(m_impl->mapB, B, sizeB);

    // Allocate Descriptor Set
    VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, m_impl->descPool, 1, &m_impl->dslMatmul };
    VkDescriptorSet descSet = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(m_impl->device, &allocInfo, &descSet) != VK_SUCCESS) {
        return matmul(A, B, C, M, K, N, mode, false);
    }

    VkDescriptorBufferInfo bInfoA{ m_impl->bufA, 0, sizeA };
    VkDescriptorBufferInfo bInfoB{ m_impl->bufB, 0, sizeB };
    VkDescriptorBufferInfo bInfoC{ m_impl->bufC, 0, sizeC };

    VkWriteDescriptorSet writes[3] = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bInfoA, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bInfoB, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bInfoC, nullptr }
    };
    vkUpdateDescriptorSets(m_impl->device, 3, writes, 0, nullptr);

    // Command Recording: never report GPU use unless submission and completion succeed.
    bool ok = vkResetCommandBuffer(m_impl->cmdBuffer, 0) == VK_SUCCESS;
    VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr };
    if (ok) ok = vkBeginCommandBuffer(m_impl->cmdBuffer, &beginInfo) == VK_SUCCESS;
    if (ok) {
        vkCmdBindPipeline(m_impl->cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_impl->pipeMatmul);
        vkCmdBindDescriptorSets(m_impl->cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_impl->plMatmul, 0, 1, &descSet, 0, nullptr);
        uint32_t pushConsts[4] = { M, K, N, mode };
        vkCmdPushConstants(m_impl->cmdBuffer, m_impl->plMatmul, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConsts), pushConsts);
        vkCmdDispatch(m_impl->cmdBuffer, (N + 15) / 16, (M + 15) / 16, 1);
        ok = vkEndCommandBuffer(m_impl->cmdBuffer) == VK_SUCCESS;
    }
    if (ok) {
        VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &m_impl->cmdBuffer, 0, nullptr };
        ok = vkQueueSubmit(m_impl->computeQueue, 1, &submitInfo, VK_NULL_HANDLE) == VK_SUCCESS;
        if (ok) ok = vkQueueWaitIdle(m_impl->computeQueue) == VK_SUCCESS;
    }
    vkFreeDescriptorSets(m_impl->device, m_impl->descPool, 1, &descSet);
    if (!ok) {
        LOGE("Vulkan matmul dispatch failed; running CPU fallback.");
        return matmul(A, B, C, M, K, N, mode, false);
    }
    std::memcpy(C, m_impl->mapC, sizeC);
    ++m_gpuDispatchCount;
    return true;
}

bool VulkanComputeEngine::swiglu(const float* gate, const float* up, float* swigluOut, uint32_t count, bool useGpu) {
    auto cpu = [&]() -> bool {
        #pragma clang loop vectorize(enable)
        for (uint32_t i = 0; i < count; ++i) {
            const float z = gate[i];
            const float silu = z / (1.0f + std::exp(-z));
            swigluOut[i] = silu * up[i];
        }
        return true;
    };
    const size_t bytes = static_cast<size_t>(count) * sizeof(float);
    if (count == 0) return true;
    if (!useGpu || !m_info.available || !m_impl->pipeSwiglu || bytes > Impl::MAX_BUF_SIZE ||
        !m_impl->mapA || !m_impl->mapB || !m_impl->mapC) return cpu();

    std::memcpy(m_impl->mapA, gate, bytes);
    std::memcpy(m_impl->mapB, up, bytes);
    VkDescriptorSetAllocateInfo alloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, m_impl->descPool, 1, &m_impl->dslSwiglu };
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(m_impl->device, &alloc, &set) != VK_SUCCESS) return cpu();
    VkDescriptorBufferInfo infos[3] = {
        {m_impl->bufA, 0, bytes}, {m_impl->bufB, 0, bytes}, {m_impl->bufC, 0, bytes}
    };
    VkWriteDescriptorSet writes[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(m_impl->device, 3, writes, 0, nullptr);
    bool ok = vkResetCommandBuffer(m_impl->cmdBuffer, 0) == VK_SUCCESS;
    VkCommandBufferBeginInfo begin{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr };
    if (ok) ok = vkBeginCommandBuffer(m_impl->cmdBuffer, &begin) == VK_SUCCESS;
    if (ok) {
        vkCmdBindPipeline(m_impl->cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_impl->pipeSwiglu);
        vkCmdBindDescriptorSets(m_impl->cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_impl->plSwiglu, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(m_impl->cmdBuffer, m_impl->plSwiglu, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(count), &count);
        vkCmdDispatch(m_impl->cmdBuffer, (count + 255) / 256, 1, 1);
        ok = vkEndCommandBuffer(m_impl->cmdBuffer) == VK_SUCCESS;
    }
    if (ok) {
        VkSubmitInfo submit{ VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &m_impl->cmdBuffer, 0, nullptr };
        ok = vkQueueSubmit(m_impl->computeQueue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS;
        if (ok) ok = vkQueueWaitIdle(m_impl->computeQueue) == VK_SUCCESS;
    }
    vkFreeDescriptorSets(m_impl->device, m_impl->descPool, 1, &set);
    if (!ok) {
        LOGE("Vulkan SwiGLU dispatch failed; using CPU fallback.");
        return cpu();
    }
    std::memcpy(swigluOut, m_impl->mapC, bytes);
    ++m_gpuDispatchCount;
    return true;
}

bool VulkanComputeEngine::swigluBackward(const float* gate, const float* up, const float* dSwiglu,
                                        float* dGate, float* dUp, uint32_t count, bool useGpu) {
    auto cpu = [&]() -> bool {
        #pragma clang loop vectorize(enable)
        for (uint32_t i = 0; i < count; ++i) {
            const float z = gate[i];
            const float sig = 1.0f / (1.0f + std::exp(-z));
            const float silu = z * sig;
            const float ds = dSwiglu[i];
            dUp[i] = ds * silu;
            const float dsilu = ds * up[i];
            dGate[i] = dsilu * (sig + silu * (1.0f - sig));
        }
        return true;
    };
    const size_t bytes = static_cast<size_t>(count) * sizeof(float);
    if (count == 0) return true;
    if (!useGpu || !m_info.available || !m_impl->pipeSwigluBack || bytes > Impl::MAX_BUF_SIZE ||
        !m_impl->mapA || !m_impl->mapB || !m_impl->mapC || !m_impl->mapD || !m_impl->mapE) return cpu();

    std::memcpy(m_impl->mapA, gate, bytes);
    std::memcpy(m_impl->mapB, up, bytes);
    std::memcpy(m_impl->mapC, dSwiglu, bytes);
    VkDescriptorSetAllocateInfo alloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, m_impl->descPool, 1, &m_impl->dslSwigluBack };
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(m_impl->device, &alloc, &set) != VK_SUCCESS) return cpu();
    VkDescriptorBufferInfo infos[5] = {
        {m_impl->bufA, 0, bytes}, {m_impl->bufB, 0, bytes}, {m_impl->bufC, 0, bytes},
        {m_impl->bufD, 0, bytes}, {m_impl->bufE, 0, bytes}
    };
    VkWriteDescriptorSet writes[5]{};
    for (uint32_t i = 0; i < 5; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(m_impl->device, 5, writes, 0, nullptr);
    bool ok = vkResetCommandBuffer(m_impl->cmdBuffer, 0) == VK_SUCCESS;
    VkCommandBufferBeginInfo begin{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr };
    if (ok) ok = vkBeginCommandBuffer(m_impl->cmdBuffer, &begin) == VK_SUCCESS;
    if (ok) {
        vkCmdBindPipeline(m_impl->cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_impl->pipeSwigluBack);
        vkCmdBindDescriptorSets(m_impl->cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_impl->plSwigluBack, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(m_impl->cmdBuffer, m_impl->plSwigluBack, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(count), &count);
        vkCmdDispatch(m_impl->cmdBuffer, (count + 255) / 256, 1, 1);
        ok = vkEndCommandBuffer(m_impl->cmdBuffer) == VK_SUCCESS;
    }
    if (ok) {
        VkSubmitInfo submit{ VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &m_impl->cmdBuffer, 0, nullptr };
        ok = vkQueueSubmit(m_impl->computeQueue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS;
        if (ok) ok = vkQueueWaitIdle(m_impl->computeQueue) == VK_SUCCESS;
    }
    vkFreeDescriptorSets(m_impl->device, m_impl->descPool, 1, &set);
    if (!ok) {
        LOGE("Vulkan SwiGLU backward dispatch failed; using CPU fallback.");
        return cpu();
    }
    std::memcpy(dGate, m_impl->mapD, bytes);
    std::memcpy(dUp, m_impl->mapE, bytes);
    ++m_gpuDispatchCount;
    return true;
}

bool VulkanComputeEngine::rmsNorm(const float* x, const float* w, float* outNorm, float* outInvStd,
                                  uint32_t D, float eps, uint32_t totalTokens, bool useGpu) {
    auto cpu = [&]() -> bool {
        for (uint32_t t = 0; t < totalTokens; ++t) {
            const float* xRow = x + (static_cast<size_t>(t) * D);
            float* outRow = outNorm + (static_cast<size_t>(t) * D);
            float sumSq = 0.0f;
            #pragma clang loop vectorize(enable)
            for (uint32_t i = 0; i < D; ++i) sumSq += xRow[i] * xRow[i];
            const float invStd = 1.0f / std::sqrt((sumSq / static_cast<float>(D)) + eps);
            if (outInvStd) outInvStd[t] = invStd;
            #pragma clang loop vectorize(enable)
            for (uint32_t i = 0; i < D; ++i) outRow[i] = xRow[i] * invStd * w[i];
        }
        return true;
    };
    if (D == 0 || totalTokens == 0) return true;
    const size_t inputBytes = static_cast<size_t>(D) * totalTokens * sizeof(float);
    const size_t weightBytes = static_cast<size_t>(D) * sizeof(float);
    const size_t invStdBytes = static_cast<size_t>(totalTokens) * sizeof(float);
    if (!useGpu || !m_info.available || !m_impl->pipeRmsNorm || D > 256 ||
        inputBytes > Impl::MAX_BUF_SIZE || weightBytes > Impl::MAX_BUF_SIZE ||
        invStdBytes > Impl::MAX_BUF_SIZE || !m_impl->mapA || !m_impl->mapB ||
        !m_impl->mapC || !m_impl->mapD) return cpu();

    std::memcpy(m_impl->mapA, x, inputBytes);
    std::memcpy(m_impl->mapB, w, weightBytes);
    VkDescriptorSetAllocateInfo alloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, m_impl->descPool, 1, &m_impl->dslRmsNorm };
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(m_impl->device, &alloc, &set) != VK_SUCCESS) return cpu();
    VkDescriptorBufferInfo infos[4] = {
        {m_impl->bufA, 0, inputBytes}, {m_impl->bufB, 0, weightBytes},
        {m_impl->bufC, 0, inputBytes}, {m_impl->bufD, 0, invStdBytes}
    };
    VkWriteDescriptorSet writes[4]{};
    for (uint32_t i = 0; i < 4; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(m_impl->device, 4, writes, 0, nullptr);
    struct RmsPush { uint32_t D; float eps; uint32_t totalTokens; };
    const RmsPush push{D, eps, totalTokens};
    bool ok = vkResetCommandBuffer(m_impl->cmdBuffer, 0) == VK_SUCCESS;
    VkCommandBufferBeginInfo begin{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr };
    if (ok) ok = vkBeginCommandBuffer(m_impl->cmdBuffer, &begin) == VK_SUCCESS;
    if (ok) {
        vkCmdBindPipeline(m_impl->cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_impl->pipeRmsNorm);
        vkCmdBindDescriptorSets(m_impl->cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_impl->plRmsNorm, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(m_impl->cmdBuffer, m_impl->plRmsNorm, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(m_impl->cmdBuffer, totalTokens, 1, 1);
        ok = vkEndCommandBuffer(m_impl->cmdBuffer) == VK_SUCCESS;
    }
    if (ok) {
        VkSubmitInfo submit{ VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &m_impl->cmdBuffer, 0, nullptr };
        ok = vkQueueSubmit(m_impl->computeQueue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS;
        if (ok) ok = vkQueueWaitIdle(m_impl->computeQueue) == VK_SUCCESS;
    }
    vkFreeDescriptorSets(m_impl->device, m_impl->descPool, 1, &set);
    if (!ok) {
        LOGE("Vulkan RMSNorm dispatch failed; using CPU fallback.");
        return cpu();
    }
    std::memcpy(outNorm, m_impl->mapC, inputBytes);
    if (outInvStd) std::memcpy(outInvStd, m_impl->mapD, invStdBytes);
    ++m_gpuDispatchCount;
    return true;
}

bool VulkanComputeEngine::adamWUpdate(float* param, float* grad, float* m, float* v, uint32_t count,
                                     float beta1, float beta2, float alphaT, float epsCorrected, float learningRate,
                                     float weightDecay, float clipScale, bool is2D, bool useGpu) {
    const size_t bytes = static_cast<size_t>(count) * sizeof(float);
    auto cpu = [&]() -> bool {
        const float decayFactor = (is2D && weightDecay > 0.0f) ? (1.0f - learningRate * weightDecay) : 1.0f;
        #pragma clang loop vectorize(enable)
        for (uint32_t i = 0; i < count; ++i) {
            if (is2D && weightDecay > 0.0f) param[i] *= decayFactor;
            const float g = grad[i] * clipScale;
            const float mVal = beta1 * m[i] + (1.0f - beta1) * g;
            const float vVal = beta2 * v[i] + (1.0f - beta2) * (g * g);
            m[i] = mVal;
            v[i] = vVal;
            param[i] -= alphaT * mVal / (std::sqrt(vVal) + epsCorrected);
            grad[i] = 0.0f;
        }
        return true;
    };
    if (count == 0) return true;
    if (!useGpu || !m_info.available || !m_impl->pipeAdamW || bytes > Impl::MAX_BUF_SIZE ||
        !m_impl->mapA || !m_impl->mapB || !m_impl->mapC || !m_impl->mapD) return cpu();

    std::memcpy(m_impl->mapA, param, bytes);
    std::memcpy(m_impl->mapB, grad, bytes);
    std::memcpy(m_impl->mapC, m, bytes);
    std::memcpy(m_impl->mapD, v, bytes);
    VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, m_impl->descPool, 1, &m_impl->dslAdamW };
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(m_impl->device, &allocInfo, &set) != VK_SUCCESS) return cpu();
    VkDescriptorBufferInfo infos[4] = {
        {m_impl->bufA,0,bytes}, {m_impl->bufB,0,bytes},
        {m_impl->bufC,0,bytes}, {m_impl->bufD,0,bytes}
    };
    VkWriteDescriptorSet writes[4]{};
    for (uint32_t i = 0; i < 4; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(m_impl->device, 4, writes, 0, nullptr);
    struct AdamPush {
        uint32_t count;
        float beta1;
        float beta2;
        float alphaT;
        float eps;
        float weightDecay;
        float clipScale;
        uint32_t is2D;
        float learningRate;
    };
    const AdamPush push{count, beta1, beta2, alphaT, eps, weightDecay, clipScale, is2D ? 1u : 0u, learningRate};
    bool ok = vkResetCommandBuffer(m_impl->cmdBuffer, 0) == VK_SUCCESS;
    VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr };
    if (ok) ok = vkBeginCommandBuffer(m_impl->cmdBuffer, &beginInfo) == VK_SUCCESS;
    if (ok) {
        vkCmdBindPipeline(m_impl->cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_impl->pipeAdamW);
        vkCmdBindDescriptorSets(m_impl->cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_impl->plAdamW, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(m_impl->cmdBuffer, m_impl->plAdamW, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(m_impl->cmdBuffer, (count + 255) / 256, 1, 1);
        ok = vkEndCommandBuffer(m_impl->cmdBuffer) == VK_SUCCESS;
    }
    if (ok) {
        VkSubmitInfo submit{ VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &m_impl->cmdBuffer, 0, nullptr };
        ok = vkQueueSubmit(m_impl->computeQueue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS;
        if (ok) ok = vkQueueWaitIdle(m_impl->computeQueue) == VK_SUCCESS;
    }
    vkFreeDescriptorSets(m_impl->device, m_impl->descPool, 1, &set);
    if (!ok) return cpu();
    std::memcpy(param, m_impl->mapA, bytes);
    std::memcpy(grad, m_impl->mapB, bytes);
    std::memcpy(m, m_impl->mapC, bytes);
    std::memcpy(v, m_impl->mapD, bytes);
    ++m_gpuDispatchCount;
    return true;
}

void VulkanComputeEngine::cleanup() {
    if (m_impl->device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_impl->device);

        auto destroyBuf = [&](VkBuffer& b, VkDeviceMemory& m) {
            if (b != VK_NULL_HANDLE) { vkDestroyBuffer(m_impl->device, b, nullptr); b = VK_NULL_HANDLE; }
            if (m != VK_NULL_HANDLE) { vkFreeMemory(m_impl->device, m, nullptr); m = VK_NULL_HANDLE; }
        };
        destroyBuf(m_impl->bufA, m_impl->memA);
        destroyBuf(m_impl->bufB, m_impl->memB);
        destroyBuf(m_impl->bufC, m_impl->memC);
        destroyBuf(m_impl->bufD, m_impl->memD);
        destroyBuf(m_impl->bufE, m_impl->memE);

        auto destroyPipe = [&](VkPipeline& p, VkPipelineLayout& pl, VkDescriptorSetLayout& dsl, VkShaderModule& sh) {
            if (p != VK_NULL_HANDLE) { vkDestroyPipeline(m_impl->device, p, nullptr); p = VK_NULL_HANDLE; }
            if (pl != VK_NULL_HANDLE) { vkDestroyPipelineLayout(m_impl->device, pl, nullptr); pl = VK_NULL_HANDLE; }
            if (dsl != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(m_impl->device, dsl, nullptr); dsl = VK_NULL_HANDLE; }
            if (sh != VK_NULL_HANDLE) { vkDestroyShaderModule(m_impl->device, sh, nullptr); sh = VK_NULL_HANDLE; }
        };

        destroyPipe(m_impl->pipeMatmul, m_impl->plMatmul, m_impl->dslMatmul, m_impl->shMatmul);
        destroyPipe(m_impl->pipeAdamW, m_impl->plAdamW, m_impl->dslAdamW, m_impl->shAdamW);
        destroyPipe(m_impl->pipeSwiglu, m_impl->plSwiglu, m_impl->dslSwiglu, m_impl->shSwiglu);
        destroyPipe(m_impl->pipeSwigluBack, m_impl->plSwigluBack, m_impl->dslSwigluBack, m_impl->shSwigluBack);
        destroyPipe(m_impl->pipeRmsNorm, m_impl->plRmsNorm, m_impl->dslRmsNorm, m_impl->shRmsNorm);

        if (m_impl->descPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(m_impl->device, m_impl->descPool, nullptr);
            m_impl->descPool = VK_NULL_HANDLE;
        }

        if (m_impl->cmdPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(m_impl->device, m_impl->cmdPool, nullptr);
            m_impl->cmdPool = VK_NULL_HANDLE;
        }

        vkDestroyDevice(m_impl->device, nullptr);
        m_impl->device = VK_NULL_HANDLE;
    }

    if (m_impl->instance != VK_NULL_HANDLE) {
        vkDestroyInstance(m_impl->instance, nullptr);
        m_impl->instance = VK_NULL_HANDLE;
    }

    if (m_vulkanLib) {
        dlclose(m_vulkanLib);
        m_vulkanLib = nullptr;
    }
    m_info.available = false;
}

} // namespace nexora
