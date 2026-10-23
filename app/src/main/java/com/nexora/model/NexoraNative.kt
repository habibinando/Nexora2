package com.nexora.model

/**
 * JNI Bridge untuk Akselerasi Native NDK r30 C++ pada Nexora.
 * Menghubungkan kalkulasi tensor, backward pass sejati, gradient clipping, dan optimizer (AdamW, Lion, SGD-M).
 */
object NexoraNative {

    var isNdkLoaded: Boolean = false
        private set

    init {
        try {
            System.loadLibrary("nexora_native")
            isNdkLoaded = true
        } catch (e: Throwable) {
            isNdkLoaded = false
        }
    }

    external fun getNdkVersion(): String

    fun isNdkAvailable(): Boolean = isNdkLoaded

    fun getActiveNdkStatus(): String {
        return if (isNdkLoaded) {
            try {
                getNdkVersion()
            } catch (e: Throwable) {
                "NDK loaded; native status unavailable"
            }
        } else {
            "JVM Math Fallback (NDK Not Loaded)"
        }
    }

    external fun nativeEmbeddingLogits(
        h: FloatArray,
        emb: FloatArray,
        vocabSize: Int,
        dModel: Int,
        logits: FloatArray
    )

    external fun nativeMatmulVec(
        x: FloatArray,
        w: FloatArray,
        dIn: Int,
        dOut: Int,
        out: FloatArray
    )

    external fun nativeLoraBackwardExact(
        x: FloatArray,
        aMat: FloatArray,
        bMat: FloatArray,
        dOutGrad: FloatArray,
        gradA: FloatArray,
        gradB: FloatArray,
        dxOut: FloatArray?,
        dIn: Int,
        dOut: Int,
        rank: Int,
        scale: Float
    ): Float

    external fun nativeClipGradients(
        grad: FloatArray,
        clipFactor: Float,
        size: Int
    )

    external fun nativeAdamWUpdate(
        param: FloatArray,
        grad: FloatArray,
        m: FloatArray,
        v: FloatArray,
        beta1: Float,
        beta2: Float,
        lr: Float,
        weightDecay: Float,
        eps: Float,
        step: Int,
        size: Int
    )

    external fun nativeLionUpdate(
        param: FloatArray,
        grad: FloatArray,
        m: FloatArray,
        beta1: Float,
        beta2: Float,
        lr: Float,
        weightDecay: Float,
        size: Int
    )

    external fun nativeSgdMomentumUpdate(
        param: FloatArray,
        grad: FloatArray,
        v: FloatArray,
        momentum: Float,
        lr: Float,
        weightDecay: Float,
        size: Int
    )

    // BPE (Byte Pair Encoding) Tokenizer Engine
    external fun nativeBpeEncode(
        textBytes: ByteArray,
        addBos: Boolean,
        addEos: Boolean
    ): IntArray

    external fun nativeBpeDecode(
        tokens: IntArray
    ): ByteArray

    external fun nativeBpeTrainFromBytes(
        textBytes: ByteArray,
        maxNewTokens: Int
    ): Int

    external fun nativeBpeExportVocabulary(): Array<String>

    external fun nativeBpeImportVocabulary(vocabList: Array<String>)

    external fun nativeBpeGetVocabSize(): Int

    external fun nativeBpeRegisterWord(word: String): Int

    // Runtime Vulkan compute status and training-step bridge.
    external fun nativeVulkanInit(): Boolean
    external fun nativeVulkanGetStatus(): String
    external fun nativeVulkanIsActive(): Boolean

    fun isVulkanAvailable(): Boolean {
        if (!isNdkLoaded) return false
        return try {
            nativeVulkanIsActive()
        } catch (e: Throwable) {
            false
        }
    }

    fun getVulkanStatus(): String {
        if (!isNdkLoaded) return "NDK belum dimuat"
        return try {
            nativeVulkanGetStatus()
        } catch (e: Throwable) {
            "CPU Fallback (SIMD C++20)"
        }
    }

    // Native LM5M Training Step (Forward, Backward, Norm, AdamW dalam C++ / Vulkan)
    external fun nativeTrainStepLM5M(
        tokens: IntArray,
        allWeights: FloatArray,
        allGrads: FloatArray,
        allM: FloatArray,
        allV: FloatArray,
        learningRate: Float,
        weightDecay: Float,
        maxGradNorm: Float,
        step: Int,
        preferVulkan: Boolean
    ): FloatArray
}
