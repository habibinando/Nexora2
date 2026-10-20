// Nexora Vulkan compute backend.
// Runtime shader compilation and OpenGL ES have deliberately been removed;
// SPIR-V is packaged in assets by the Android Gradle Plugin.
#include <jni.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "NexoraVulkan", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "NexoraVulkan", __VA_ARGS__)

struct HostBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize capacity = 0;
    bool coherent = false;
};

struct VulkanState {
    bool ok = false;
    std::string info = "Vulkan compute belum diinisialisasi";
    std::string error;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamily = UINT32_MAX;
    VkDescriptorSetLayout descriptorLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    HostBuffer a, b, c;

    ~VulkanState() { destroy(); }

    void destroyBuffer(HostBuffer& item) {
        if (device != VK_NULL_HANDLE) {
            if (item.mapped && item.memory != VK_NULL_HANDLE) vkUnmapMemory(device, item.memory);
            if (item.buffer != VK_NULL_HANDLE) vkDestroyBuffer(device, item.buffer, nullptr);
            if (item.memory != VK_NULL_HANDLE) vkFreeMemory(device, item.memory, nullptr);
        }
        item = HostBuffer{};
    }

    void destroy() {
        if (device != VK_NULL_HANDLE) {
            destroyBuffer(a);
            destroyBuffer(b);
            destroyBuffer(c);
            if (fence != VK_NULL_HANDLE) vkDestroyFence(device, fence, nullptr);
            if (commandPool != VK_NULL_HANDLE) vkDestroyCommandPool(device, commandPool, nullptr);
            if (descriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, descriptorPool, nullptr);
            if (pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, pipeline, nullptr);
            if (pipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            if (descriptorLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, descriptorLayout, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance != VK_NULL_HANDLE) vkDestroyInstance(instance, nullptr);
        device = VK_NULL_HANDLE;
        instance = VK_NULL_HANDLE;
        ok = false;
    }

    bool fail(const char* where, VkResult result) {
        error = std::string(where) + " gagal (VkResult=" + std::to_string(static_cast<int>(result)) + ")";
        LOGE("%s", error.c_str());
        return false;
    }

    bool loadSpirv(AAssetManager* assets, std::vector<uint32_t>& words) {
        if (!assets) {
            error = "AssetManager tidak tersedia untuk memuat SPIR-V";
            return false;
        }
        AAsset* file = AAssetManager_open(assets, "shaders/matmul.comp.spv", AASSET_MODE_BUFFER);
        if (!file) {
            error = "Shader shaders/matmul.comp.spv tidak ditemukan dalam APK";
            LOGE("%s", error.c_str());
            return false;
        }
        const off_t len = AAsset_getLength(file);
        if (len <= 0 || (len % 4) != 0) {
            AAsset_close(file);
            error = "Ukuran file SPIR-V tidak valid";
            return false;
        }
        words.resize(static_cast<size_t>(len) / sizeof(uint32_t));
        const int read = AAsset_read(file, words.data(), static_cast<size_t>(len));
        AAsset_close(file);
        if (read != len || words.empty() || words[0] != 0x07230203u) {
            error = "File shader bukan modul SPIR-V yang valid";
            words.clear();
            return false;
        }
        return true;
    }

    bool init(AAssetManager* assets) {
        if (ok) return true;
        VkApplicationInfo app{};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "Nexora";
        app.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        app.pEngineName = "Nexora On-Device ML";
        app.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        app.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo instanceInfo{};
        instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instanceInfo.pApplicationInfo = &app;
        VkResult result = vkCreateInstance(&instanceInfo, nullptr, &instance);
        if (result != VK_SUCCESS) return fail("vkCreateInstance", result);

        uint32_t deviceCount = 0;
        result = vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        if (result != VK_SUCCESS || deviceCount == 0) {
            error = "Perangkat tidak menyediakan Vulkan GPU";
            return false;
        }
        std::vector<VkPhysicalDevice> devices(deviceCount);
        result = vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
        if (result != VK_SUCCESS) return fail("vkEnumeratePhysicalDevices", result);

        int bestScore = -1;
        for (VkPhysicalDevice candidate : devices) {
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(candidate, &props);
            uint32_t familyCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &familyCount, nullptr);
            std::vector<VkQueueFamilyProperties> families(familyCount);
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &familyCount, families.data());

            for (uint32_t i = 0; i < familyCount; ++i) {
                if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) continue;
                if (props.limits.maxComputeWorkGroupInvocations < 256 ||
                    props.limits.maxComputeWorkGroupSize[0] < 16 ||
                    props.limits.maxComputeWorkGroupSize[1] < 16 ||
                    props.limits.maxStorageBufferRange < sizeof(float) * 256) continue;

                int score = 1;
                if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score = 4;
                else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) score = 3;
                else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU) score = 2;

                if (score > bestScore) {
                    bestScore = score;
                    physical = candidate;
                    properties = props;
                    queueFamily = i;
                }
                break;
            }
        }
        if (physical == VK_NULL_HANDLE || queueFamily == UINT32_MAX) {
            error = "Tidak ada Vulkan compute queue yang memenuhi kebutuhan shader";
            return false;
        }
        vkGetPhysicalDeviceMemoryProperties(physical, &memoryProperties);

        float priority = 1.0f;
        VkDeviceQueueCreateInfo queueInfo{};
        queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueInfo.queueFamilyIndex = queueFamily;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &priority;

        VkDeviceCreateInfo deviceInfo{};
        deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceInfo.queueCreateInfoCount = 1;
        deviceInfo.pQueueCreateInfos = &queueInfo;
        result = vkCreateDevice(physical, &deviceInfo, nullptr, &device);
        if (result != VK_SUCCESS) return fail("vkCreateDevice", result);
        vkGetDeviceQueue(device, queueFamily, 0, &queue);

        std::vector<uint32_t> shaderWords;
        if (!loadSpirv(assets, shaderWords)) return false;

        VkShaderModuleCreateInfo shaderInfo{};
        shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shaderInfo.codeSize = shaderWords.size() * sizeof(uint32_t);
        shaderInfo.pCode = shaderWords.data();
        VkShaderModule shader = VK_NULL_HANDLE;
        result = vkCreateShaderModule(device, &shaderInfo, nullptr, &shader);
        if (result != VK_SUCCESS) return fail("vkCreateShaderModule", result);

        VkDescriptorSetLayoutBinding bindings[3]{};
        for (uint32_t i = 0; i < 3; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 3;
        layoutInfo.pBindings = bindings;
        result = vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorLayout);
        if (result != VK_SUCCESS) {
            vkDestroyShaderModule(device, shader, nullptr);
            return fail("vkCreateDescriptorSetLayout", result);
        }

        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushRange.offset = 0;
        pushRange.size = sizeof(int32_t) * 3;
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushRange;
        result = vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout);
        if (result != VK_SUCCESS) {
            vkDestroyShaderModule(device, shader, nullptr);
            return fail("vkCreatePipelineLayout", result);
        }

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = shader;
        stage.pName = "main";
        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage = stage;
        pipelineInfo.layout = pipelineLayout;
        result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
        vkDestroyShaderModule(device, shader, nullptr);
        if (result != VK_SUCCESS) return fail("vkCreateComputePipelines", result);

        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = 3;
        VkDescriptorPoolCreateInfo descriptorPoolInfo{};
        descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptorPoolInfo.maxSets = 1;
        descriptorPoolInfo.poolSizeCount = 1;
        descriptorPoolInfo.pPoolSizes = &poolSize;
        result = vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool);
        if (result != VK_SUCCESS) return fail("vkCreateDescriptorPool", result);

        VkDescriptorSetAllocateInfo setInfo{};
        setInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        setInfo.descriptorPool = descriptorPool;
        setInfo.descriptorSetCount = 1;
        setInfo.pSetLayouts = &descriptorLayout;
        result = vkAllocateDescriptorSets(device, &setInfo, &descriptorSet);
        if (result != VK_SUCCESS) return fail("vkAllocateDescriptorSets", result);

        VkCommandPoolCreateInfo commandPoolInfo{};
        commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        commandPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        commandPoolInfo.queueFamilyIndex = queueFamily;
        result = vkCreateCommandPool(device, &commandPoolInfo, nullptr, &commandPool);
        if (result != VK_SUCCESS) return fail("vkCreateCommandPool", result);

        VkCommandBufferAllocateInfo commandBufferInfo{};
        commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandBufferInfo.commandPool = commandPool;
        commandBufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandBufferInfo.commandBufferCount = 1;
        result = vkAllocateCommandBuffers(device, &commandBufferInfo, &commandBuffer);
        if (result != VK_SUCCESS) return fail("vkAllocateCommandBuffers", result);

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        result = vkCreateFence(device, &fenceInfo, nullptr, &fence);
        if (result != VK_SUCCESS) return fail("vkCreateFence", result);

        info = std::string("Vulkan 1.0 | ") + properties.deviceName +
               " | max SSBO " + std::to_string(properties.limits.maxStorageBufferRange / (1024 * 1024)) + " MiB";
        ok = true;
        LOGI("Initialized: %s", info.c_str());
        return true;
    }

    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred) {
        int best = -1;
        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
            if ((typeBits & (1u << i)) == 0) continue;
            const VkMemoryPropertyFlags flags = memoryProperties.memoryTypes[i].propertyFlags;
            if ((flags & required) != required) continue;
            const int score = (flags & preferred) == preferred ? 2 : 1;
            if (score > best) best = score, lastMemoryType = i;
        }
        return best < 0 ? UINT32_MAX : lastMemoryType;
    }

    uint32_t lastMemoryType = UINT32_MAX;

    bool ensureBuffer(HostBuffer& target, VkDeviceSize requested) {
        if (requested == 0 || requested > properties.limits.maxStorageBufferRange) {
            error = "Matriks melewati batas maxStorageBufferRange Vulkan";
            return false;
        }
        if (target.buffer != VK_NULL_HANDLE && target.capacity >= requested) return true;

        HostBuffer fresh;
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = requested;
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkResult result = vkCreateBuffer(device, &bufferInfo, nullptr, &fresh.buffer);
        if (result != VK_SUCCESS) return fail("vkCreateBuffer", result);

        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device, fresh.buffer, &requirements);
        const VkMemoryPropertyFlags required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        const VkMemoryPropertyFlags preferred = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        const uint32_t memoryType = findMemoryType(requirements.memoryTypeBits, required, preferred);
        if (memoryType == UINT32_MAX) {
            vkDestroyBuffer(device, fresh.buffer, nullptr);
            fresh.buffer = VK_NULL_HANDLE;
            error = "GPU tidak menyediakan host-visible memory untuk buffer compute";
            return false;
        }
        fresh.coherent = (memoryProperties.memoryTypes[memoryType].propertyFlags &
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
        VkMemoryAllocateInfo allocation{};
        allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = memoryType;
        result = vkAllocateMemory(device, &allocation, nullptr, &fresh.memory);
        if (result != VK_SUCCESS) {
            vkDestroyBuffer(device, fresh.buffer, nullptr);
            fresh.buffer = VK_NULL_HANDLE;
            return fail("vkAllocateMemory", result);
        }
        result = vkBindBufferMemory(device, fresh.buffer, fresh.memory, 0);
        if (result != VK_SUCCESS) {
            vkDestroyBuffer(device, fresh.buffer, nullptr);
            vkFreeMemory(device, fresh.memory, nullptr);
            fresh.buffer = VK_NULL_HANDLE;
            fresh.memory = VK_NULL_HANDLE;
            return fail("vkBindBufferMemory", result);
        }
        result = vkMapMemory(device, fresh.memory, 0, VK_WHOLE_SIZE, 0, &fresh.mapped);
        if (result != VK_SUCCESS) {
            vkDestroyBuffer(device, fresh.buffer, nullptr);
            vkFreeMemory(device, fresh.memory, nullptr);
            fresh.buffer = VK_NULL_HANDLE;
            fresh.memory = VK_NULL_HANDLE;
            return fail("vkMapMemory", result);
        }
        fresh.capacity = requested;
        destroyBuffer(target);
        target = fresh;
        return true;
    }

    bool flushIfNeeded(const HostBuffer& buffer) {
        if (buffer.coherent) return true;
        VkMappedMemoryRange range{};
        range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = buffer.memory;
        range.offset = 0;
        range.size = VK_WHOLE_SIZE;
        const VkResult result = vkFlushMappedMemoryRanges(device, 1, &range);
        return result == VK_SUCCESS || fail("vkFlushMappedMemoryRanges", result);
    }

    bool invalidateIfNeeded(const HostBuffer& buffer) {
        if (buffer.coherent) return true;
        VkMappedMemoryRange range{};
        range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = buffer.memory;
        range.offset = 0;
        range.size = VK_WHOLE_SIZE;
        const VkResult result = vkInvalidateMappedMemoryRanges(device, 1, &range);
        return result == VK_SUCCESS || fail("vkInvalidateMappedMemoryRanges", result);
    }

    bool matmul(const float* A, const float* B, float* C, int M, int K, int N) {
        if (!ok || M <= 0 || K <= 0 || N <= 0) return false;
        const uint64_t na = static_cast<uint64_t>(M) * static_cast<uint64_t>(K) * sizeof(float);
        const uint64_t nb = static_cast<uint64_t>(K) * static_cast<uint64_t>(N) * sizeof(float);
        const uint64_t nc = static_cast<uint64_t>(M) * static_cast<uint64_t>(N) * sizeof(float);
        if (na > SIZE_MAX || nb > SIZE_MAX || nc > SIZE_MAX) return false;
        if (!ensureBuffer(a, na) || !ensureBuffer(b, nb) || !ensureBuffer(c, nc)) return false;

        std::memcpy(a.mapped, A, static_cast<size_t>(na));
        std::memcpy(b.mapped, B, static_cast<size_t>(nb));
        if (!flushIfNeeded(a) || !flushIfNeeded(b)) return false;

        VkDescriptorBufferInfo infos[3]{};
        infos[0] = {a.buffer, 0, na};
        infos[1] = {b.buffer, 0, nb};
        infos[2] = {c.buffer, 0, nc};
        VkWriteDescriptorSet writes[3]{};
        for (uint32_t i = 0; i < 3; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = descriptorSet;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);

        VkResult result = vkResetCommandBuffer(commandBuffer, 0);
        if (result != VK_SUCCESS) return fail("vkResetCommandBuffer", result);
        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        result = vkBeginCommandBuffer(commandBuffer, &begin);
        if (result != VK_SUCCESS) return fail("vkBeginCommandBuffer", result);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
        const int32_t dims[3] = {M, K, N};
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(dims), dims);
        vkCmdDispatch(commandBuffer, static_cast<uint32_t>((N + 15) / 16),
                      static_cast<uint32_t>((M + 15) / 16), 1);

        VkMemoryBarrier memoryBarrier{};
        memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &memoryBarrier,
                             0, nullptr, 0, nullptr);
        result = vkEndCommandBuffer(commandBuffer);
        if (result != VK_SUCCESS) return fail("vkEndCommandBuffer", result);

        result = vkResetFences(device, 1, &fence);
        if (result != VK_SUCCESS) return fail("vkResetFences", result);
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &commandBuffer;
        result = vkQueueSubmit(queue, 1, &submit, fence);
        if (result != VK_SUCCESS) return fail("vkQueueSubmit", result);
        result = vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
        if (result != VK_SUCCESS) return fail("vkWaitForFences", result);
        if (!invalidateIfNeeded(c)) return false;
        std::memcpy(C, c.mapped, static_cast<size_t>(nc));
        return true;
    }
};

static thread_local VulkanState* t_state = nullptr;
static VulkanState* state() {
    if (!t_state) t_state = new VulkanState();
    return t_state;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_Nexora_ai_ml_NativeOps_nativeVulkanInit(JNIEnv* env, jclass, jobject assetManager) {
    AAssetManager* assets = AAssetManager_fromJava(env, assetManager);
    return state()->init(assets) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_Nexora_ai_ml_NativeOps_nativeVulkanInfo(JNIEnv* env, jclass) {
    auto* s = state();
    const std::string msg = s->ok ? s->info : (s->error.empty() ? "Vulkan compute tidak tersedia" : s->error);
    return env->NewStringUTF(msg.c_str());
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_Nexora_ai_ml_NativeOps_nativeVulkanBench(JNIEnv*, jclass, jint M, jint K, jint N, jint iter) {
    auto* s = state();
    if (!s->ok || M <= 0 || K <= 0 || N <= 0 || iter <= 0) return -1.0;
    std::vector<float> A(static_cast<size_t>(M) * K, 0.01f);
    std::vector<float> B(static_cast<size_t>(K) * N, 0.01f);
    std::vector<float> C(static_cast<size_t>(M) * N, 0.0f);
    for (int i = 0; i < 3; ++i) if (!s->matmul(A.data(), B.data(), C.data(), M, K, N)) return -1.0;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iter; ++i) if (!s->matmul(A.data(), B.data(), C.data(), M, K, N)) return -1.0;
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iter;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_Nexora_ai_ml_NativeOps_nativeVulkanMatMul(
        JNIEnv* env, jclass, jfloatArray aArr, jfloatArray bArr, jfloatArray cArr,
        jint M, jint K, jint N) {
    if (M <= 0 || K <= 0 || N <= 0) return JNI_FALSE;
    const jlong aNeed = static_cast<jlong>(M) * K;
    const jlong bNeed = static_cast<jlong>(K) * N;
    const jlong cNeed = static_cast<jlong>(M) * N;
    if (aNeed <= 0 || bNeed <= 0 || cNeed <= 0 ||
        env->GetArrayLength(aArr) < aNeed ||
        env->GetArrayLength(bArr) < bNeed ||
        env->GetArrayLength(cArr) < cNeed) return JNI_FALSE;

    auto* s = state();
    if (!s->ok) return JNI_FALSE;

    jfloat* A = env->GetFloatArrayElements(aArr, nullptr);
    if (!A) return JNI_FALSE;
    jfloat* B = env->GetFloatArrayElements(bArr, nullptr);
    if (!B) {
        env->ReleaseFloatArrayElements(aArr, A, JNI_ABORT);
        return JNI_FALSE;
    }
    jfloat* C = env->GetFloatArrayElements(cArr, nullptr);
    if (!C) {
        env->ReleaseFloatArrayElements(aArr, A, JNI_ABORT);
        env->ReleaseFloatArrayElements(bArr, B, JNI_ABORT);
        return JNI_FALSE;
    }

    const bool success = s->matmul(A, B, C, M, K, N);
    env->ReleaseFloatArrayElements(aArr, A, JNI_ABORT);
    env->ReleaseFloatArrayElements(bArr, B, JNI_ABORT);
    env->ReleaseFloatArrayElements(cArr, C, success ? 0 : JNI_ABORT);
    return success ? JNI_TRUE : JNI_FALSE;
}
