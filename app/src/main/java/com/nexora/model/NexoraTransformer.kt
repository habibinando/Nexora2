package com.nexora.model

import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.random.Random

enum class TrainingMode(val displayName: String, val description: String) {
    FULL_PARAMETERS("Full Parameters (Pretrain)", "Melatih seluruh 5.07M parameter LM5M"),
    LORA_ADAPTER("LoRA Adapters (Fine-Tune)", "Melatih 4,096 parameter rank-4 (Hemat Memori)")
}

/**
 * Arsitektur LM5M Transformer Autoregressive Causal Language Model Nexora:
 * - Decoder-only, Pre-RMSNorm, Causal Attention, RoPE (rotate_half), SwiGLU MLP, Weight Tying.
 * - Total Parameter Eksak: 5,065,472 float32.
 * - Dropout = 0.0 (tanpa dropout).
 * - Perhitungan Loss: Cross-Entropy Mean atas posisi.
 * - Optimizer: AdamW Decoupled (Weight decay 0.1 pada tensor 2D; bukan pada norm), Gradient Clipping 1.0.
 */
class NexoraTransformer(
    val config: NexoraConfig = NexoraConfig(),
    val random: Random = Random(2026)
) {
    var trainingMode: TrainingMode = TrainingMode.FULL_PARAMETERS
    val adamStepCount = AtomicLong(config.stepsTrained.toLong())
    private var activeOptimizerType: OptimizerType = OptimizerType.ADAMW

    // 1. Matriks Token Embeddings [4096, 256] = 1,048,576
    val tokEmb = FloatArray(config.vocabSize * config.dModel)
    val gradTokEmb = FloatArray(config.vocabSize * config.dModel)
    val mTokEmb = FloatArray(config.vocabSize * config.dModel)
    val vTokEmb = FloatArray(config.vocabSize * config.dModel)

    // RoPE precomputed inverse frequencies: hd = 64 -> half = 32
    val invFreq = FloatArray(config.headDim / 2) { i ->
        1.0f / Math.pow(config.ropeTheta.toDouble(), (2.0 * i / config.headDim)).toFloat()
    }

    class LayerWeights(
        val dModel: Int,
        val dFf: Int,
        val rank: Int,
        random: Random
    ) {
        val attnNorm = FloatArray(dModel) { 1.0f }
        val gradAttnNorm = FloatArray(dModel)
        val mAttnNorm = FloatArray(dModel)
        val vAttnNorm = FloatArray(dModel)

        // wq, wk, wv, wo: [256, 256]
        val wq = NexoraTensorOps.createRandomWeights(dModel * dModel, 0.02f, random)
        val wk = NexoraTensorOps.createRandomWeights(dModel * dModel, 0.02f, random)
        val wv = NexoraTensorOps.createRandomWeights(dModel * dModel, 0.02f, random)
        val wo = NexoraTensorOps.createRandomWeights(dModel * dModel, 0.02f, random)

        val gradWq = FloatArray(dModel * dModel)
        val gradWk = FloatArray(dModel * dModel)
        val gradWv = FloatArray(dModel * dModel)
        val gradWo = FloatArray(dModel * dModel)

        val mWq = FloatArray(dModel * dModel)
        val vWq = FloatArray(dModel * dModel)
        val mWk = FloatArray(dModel * dModel)
        val vWk = FloatArray(dModel * dModel)
        val mWv = FloatArray(dModel * dModel)
        val vWv = FloatArray(dModel * dModel)
        val mWo = FloatArray(dModel * dModel)
        val vWo = FloatArray(dModel * dModel)

        val ffnNorm = FloatArray(dModel) { 1.0f }
        val gradFfnNorm = FloatArray(dModel)
        val mFfnNorm = FloatArray(dModel)
        val vFfnNorm = FloatArray(dModel)

        // SwiGLU: w1 [704, 256], w3 [704, 256], w2 [256, 704]
        val w1 = NexoraTensorOps.createRandomWeights(dFf * dModel, 0.02f, random)
        val w3 = NexoraTensorOps.createRandomWeights(dFf * dModel, 0.02f, random)
        val w2 = NexoraTensorOps.createRandomWeights(dModel * dFf, 0.02f, random)

        val gradW1 = FloatArray(dFf * dModel)
        val gradW3 = FloatArray(dFf * dModel)
        val gradW2 = FloatArray(dModel * dFf)

        val mW1 = FloatArray(dFf * dModel)
        val vW1 = FloatArray(dFf * dModel)
        val mW3 = FloatArray(dFf * dModel)
        val vW3 = FloatArray(dFf * dModel)
        val mW2 = FloatArray(dModel * dFf)
        val vW2 = FloatArray(dModel * dFf)

        // LoRA Adapters rank-4 untuk fine-tuning adaptif
        val loraA_Q = NexoraTensorOps.createRandomWeights(dModel * rank, 0.02f, random)
        val loraB_Q = FloatArray(rank * dModel)
        val loraA_V = NexoraTensorOps.createRandomWeights(dModel * rank, 0.02f, random)
        val loraB_V = FloatArray(rank * dModel)

        val gradLoraA_Q = FloatArray(dModel * rank)
        val gradLoraB_Q = FloatArray(rank * dModel)
        val gradLoraA_V = FloatArray(dModel * rank)
        val gradLoraB_V = FloatArray(rank * dModel)

        val mLoraA_Q = FloatArray(dModel * rank)
        val vLoraA_Q = FloatArray(dModel * rank)
        val mLoraB_Q = FloatArray(rank * dModel)
        val vLoraB_Q = FloatArray(rank * dModel)
        val mLoraA_V = FloatArray(dModel * rank)
        val vLoraA_V = FloatArray(dModel * rank)
        val mLoraB_V = FloatArray(rank * dModel)
        val vLoraB_V = FloatArray(rank * dModel)
    }

    val layers: Array<LayerWeights> = Array(config.numLayers) {
        LayerWeights(config.dModel, config.dFf, config.loraRank, random)
    }

    val finalNorm = FloatArray(config.dModel) { 1.0f }
    val gradFinalNorm = FloatArray(config.dModel)
    val mFinalNorm = FloatArray(config.dModel)
    val vFinalNorm = FloatArray(config.dModel)

    // Flat backing arrays untuk akselerasi zero-copy JNI / Vulkan compute
    val allWeights = FloatArray(config.totalParams.toInt())
    val allGrads = FloatArray(config.totalParams.toInt())
    val allM = FloatArray(config.totalParams.toInt())
    val allV = FloatArray(config.totalParams.toInt())
    var preferVulkan: Boolean = true
    var lastStepUsedVulkan: Boolean = false

    var isPretrainedLoaded: Boolean = false
        private set

    fun syncWeightsToFlat() {
        var off = 0
        System.arraycopy(tokEmb, 0, allWeights, off, tokEmb.size)
        off += tokEmb.size
        for (lw in layers) {
            System.arraycopy(lw.attnNorm, 0, allWeights, off, lw.attnNorm.size); off += lw.attnNorm.size
            System.arraycopy(lw.wq, 0, allWeights, off, lw.wq.size); off += lw.wq.size
            System.arraycopy(lw.wk, 0, allWeights, off, lw.wk.size); off += lw.wk.size
            System.arraycopy(lw.wv, 0, allWeights, off, lw.wv.size); off += lw.wv.size
            System.arraycopy(lw.wo, 0, allWeights, off, lw.wo.size); off += lw.wo.size
            System.arraycopy(lw.ffnNorm, 0, allWeights, off, lw.ffnNorm.size); off += lw.ffnNorm.size
            System.arraycopy(lw.w1, 0, allWeights, off, lw.w1.size); off += lw.w1.size
            System.arraycopy(lw.w3, 0, allWeights, off, lw.w3.size); off += lw.w3.size
            System.arraycopy(lw.w2, 0, allWeights, off, lw.w2.size); off += lw.w2.size
        }
        System.arraycopy(finalNorm, 0, allWeights, off, finalNorm.size)
    }

    fun syncFlatToWeights() {
        var off = 0
        System.arraycopy(allWeights, off, tokEmb, 0, tokEmb.size)
        off += tokEmb.size
        for (lw in layers) {
            System.arraycopy(allWeights, off, lw.attnNorm, 0, lw.attnNorm.size); off += lw.attnNorm.size
            System.arraycopy(allWeights, off, lw.wq, 0, lw.wq.size); off += lw.wq.size
            System.arraycopy(allWeights, off, lw.wk, 0, lw.wk.size); off += lw.wk.size
            System.arraycopy(allWeights, off, lw.wv, 0, lw.wv.size); off += lw.wv.size
            System.arraycopy(allWeights, off, lw.wo, 0, lw.wo.size); off += lw.wo.size
            System.arraycopy(allWeights, off, lw.ffnNorm, 0, lw.ffnNorm.size); off += lw.ffnNorm.size
            System.arraycopy(allWeights, off, lw.w1, 0, lw.w1.size); off += lw.w1.size
            System.arraycopy(allWeights, off, lw.w3, 0, lw.w3.size); off += lw.w3.size
            System.arraycopy(allWeights, off, lw.w2, 0, lw.w2.size); off += lw.w2.size
        }
        System.arraycopy(allWeights, off, finalNorm, 0, finalNorm.size)
    }

    fun syncMomentsToFlat() {
        var off = 0
        System.arraycopy(mTokEmb, 0, allM, off, mTokEmb.size)
        System.arraycopy(vTokEmb, 0, allV, off, vTokEmb.size)
        off += tokEmb.size
        for (lw in layers) {
            System.arraycopy(lw.mAttnNorm, 0, allM, off, lw.mAttnNorm.size)
            System.arraycopy(lw.vAttnNorm, 0, allV, off, lw.vAttnNorm.size)
            off += lw.attnNorm.size
            System.arraycopy(lw.mWq, 0, allM, off, lw.mWq.size)
            System.arraycopy(lw.vWq, 0, allV, off, lw.vWq.size)
            off += lw.wq.size
            System.arraycopy(lw.mWk, 0, allM, off, lw.mWk.size)
            System.arraycopy(lw.vWk, 0, allV, off, lw.vWk.size)
            off += lw.wk.size
            System.arraycopy(lw.mWv, 0, allM, off, lw.mWv.size)
            System.arraycopy(lw.vWv, 0, allV, off, lw.vWv.size)
            off += lw.wv.size
            System.arraycopy(lw.mWo, 0, allM, off, lw.mWo.size)
            System.arraycopy(lw.vWo, 0, allV, off, lw.vWo.size)
            off += lw.wo.size
            System.arraycopy(lw.mFfnNorm, 0, allM, off, lw.mFfnNorm.size)
            System.arraycopy(lw.vFfnNorm, 0, allV, off, lw.vFfnNorm.size)
            off += lw.ffnNorm.size
            System.arraycopy(lw.mW1, 0, allM, off, lw.mW1.size)
            System.arraycopy(lw.vW1, 0, allV, off, lw.vW1.size)
            off += lw.w1.size
            System.arraycopy(lw.mW3, 0, allM, off, lw.mW3.size)
            System.arraycopy(lw.vW3, 0, allV, off, lw.vW3.size)
            off += lw.w3.size
            System.arraycopy(lw.mW2, 0, allM, off, lw.mW2.size)
            System.arraycopy(lw.vW2, 0, allV, off, lw.vW2.size)
            off += lw.w2.size
        }
        System.arraycopy(mFinalNorm, 0, allM, off, mFinalNorm.size)
        System.arraycopy(vFinalNorm, 0, allV, off, vFinalNorm.size)
    }

    fun syncFlatToMoments() {
        var off = 0
        System.arraycopy(allM, off, mTokEmb, 0, mTokEmb.size)
        System.arraycopy(allV, off, vTokEmb, 0, vTokEmb.size)
        off += tokEmb.size
        for (lw in layers) {
            System.arraycopy(allM, off, lw.mAttnNorm, 0, lw.mAttnNorm.size)
            System.arraycopy(allV, off, lw.vAttnNorm, 0, lw.vAttnNorm.size)
            off += lw.attnNorm.size
            System.arraycopy(allM, off, lw.mWq, 0, lw.mWq.size)
            System.arraycopy(allV, off, lw.vWq, 0, lw.vWq.size)
            off += lw.wq.size
            System.arraycopy(allM, off, lw.mWk, 0, lw.mWk.size)
            System.arraycopy(allV, off, lw.vWk, 0, lw.vWk.size)
            off += lw.wk.size
            System.arraycopy(allM, off, lw.mWv, 0, lw.mWv.size)
            System.arraycopy(allV, off, lw.vWv, 0, lw.vWv.size)
            off += lw.wv.size
            System.arraycopy(allM, off, lw.mWo, 0, lw.mWo.size)
            System.arraycopy(allV, off, lw.vWo, 0, lw.vWo.size)
            off += lw.wo.size
            System.arraycopy(allM, off, lw.mFfnNorm, 0, lw.mFfnNorm.size)
            System.arraycopy(allV, off, lw.vFfnNorm, 0, lw.vFfnNorm.size)
            off += lw.ffnNorm.size
            System.arraycopy(allM, off, lw.mW1, 0, lw.mW1.size)
            System.arraycopy(allV, off, lw.vW1, 0, lw.vW1.size)
            off += lw.w1.size
            System.arraycopy(allM, off, lw.mW3, 0, lw.mW3.size)
            System.arraycopy(allV, off, lw.vW3, 0, lw.vW3.size)
            off += lw.w3.size
            System.arraycopy(allM, off, lw.mW2, 0, lw.mW2.size)
            System.arraycopy(allV, off, lw.vW2, 0, lw.vW2.size)
            off += lw.w2.size
        }
        System.arraycopy(allM, off, mFinalNorm, 0, mFinalNorm.size)
        System.arraycopy(allV, off, vFinalNorm, 0, vFinalNorm.size)
    }

    init {
        val stdDev = 0.02f
        val jRandom = java.util.Random(2026)
        for (i in tokEmb.indices) {
            tokEmb[i] = jRandom.nextGaussian().toFloat() * stdDev
        }
        syncWeightsToFlat()
        syncMomentsToFlat()
    }

    private fun resetOptimizerState() {
        mTokEmb.fill(0f)
        vTokEmb.fill(0f)
        mFinalNorm.fill(0f)
        vFinalNorm.fill(0f)
        allM.fill(0f)
        allV.fill(0f)
        for (lw in layers) {
            arrayOf(
                lw.mAttnNorm, lw.vAttnNorm, lw.mWq, lw.vWq, lw.mWk, lw.vWk,
                lw.mWv, lw.vWv, lw.mWo, lw.vWo, lw.mFfnNorm, lw.vFfnNorm,
                lw.mW1, lw.vW1, lw.mW3, lw.vW3, lw.mW2, lw.vW2,
                lw.mLoraA_Q, lw.vLoraA_Q, lw.mLoraB_Q, lw.vLoraB_Q,
                lw.mLoraA_V, lw.vLoraA_V, lw.mLoraB_V, lw.vLoraB_V
            ).forEach { it.fill(0f) }
        }
    }

    fun calculateTotalParameters(): Long = config.totalParams

    fun calculateTrainableParameters(): Long {
        return if (trainingMode == TrainingMode.FULL_PARAMETERS) {
            config.totalParams
        } else {
            // LoRA rank 4 pada q & v per layer: 5 * (256*4 + 4*256) * 2 = 5 * 2048 * 2 = 20,480
            (config.numLayers * 2 * (config.dModel * config.loraRank + config.loraRank * config.dModel)).toLong()
        }
    }

    fun calculateAdapterSizeBytes(): Long {
        return (config.numLayers * 2 * (config.dModel * config.loraRank + config.loraRank * config.dModel) * 4).toLong()
    }

    /**
     * Memuat bobot float32 little-endian dari weights.f32 sesuai manifest LM5M.
     */
    @Synchronized
    fun loadWeightsFromStream(stream: InputStream): Boolean {
        return try {
            val totalBytes = config.totalParams.toInt() * 4
            val buffer = ByteArray(totalBytes)
            var bytesRead = 0
            while (bytesRead < totalBytes) {
                val r = stream.read(buffer, bytesRead, totalBytes - bytesRead)
                if (r < 0) break
                bytesRead += r
            }
            if (bytesRead < totalBytes) return false

            val bb = ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN)
            val fb = bb.asFloatBuffer()

            fb.get(allWeights)
            syncFlatToWeights()

            isPretrainedLoaded = true
            true
        } catch (e: Exception) {
            false
        }
    }

    /**
     * Memuat Adam moments m dan v dari adam_m.f32 dan adam_v.f32.
     */
    @Synchronized
    fun loadAdamState(mStream: InputStream, vStream: InputStream): Boolean {
        return try {
            val totalBytes = config.totalParams.toInt() * 4
            val mBuffer = ByteArray(totalBytes)
            val vBuffer = ByteArray(totalBytes)

            var readM = 0
            while (readM < totalBytes) {
                val r = mStream.read(mBuffer, readM, totalBytes - readM)
                if (r < 0) break
                readM += r
            }
            var readV = 0
            while (readV < totalBytes) {
                val r = vStream.read(vBuffer, readV, totalBytes - readV)
                if (r < 0) break
                readV += r
            }

            val mFb = ByteBuffer.wrap(mBuffer).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            val vFb = ByteBuffer.wrap(vBuffer).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()

            mFb.get(allM)
            vFb.get(allV)
            syncFlatToMoments()
            true
        } catch (e: Exception) {
            false
        }
    }

    /**
     * RMSNorm: x / sqrt(mean(x^2) + eps) * w
     */
    private fun applyRmsNorm(
        x: FloatArray,
        w: FloatArray,
        out: FloatArray,
        d: Int,
        eps: Float = config.eps
    ): Float {
        var sumSq = 0.0f
        for (i in 0 until d) {
            val v = x[i]
            sumSq += v * v
        }
        val invStd = 1.0f / sqrt((sumSq / d) + eps)
        for (i in 0 until d) {
            out[i] = x[i] * invStd * w[i]
        }
        return invStd
    }

    /**
     * RoPE (rotate_half) pada vektor panjang 64.
     */
    private fun applyRope(
        vec: FloatArray,
        offset: Int,
        pos: Int,
        hd: Int = config.headDim
    ) {
        val half = hd / 2
        for (i in 0 until half) {
            val ang = pos * invFreq[i]
            val c = cos(ang.toDouble()).toFloat()
            val s = sin(ang.toDouble()).toFloat()
            val v1 = vec[offset + i]
            val v2 = vec[offset + i + half]
            vec[offset + i] = v1 * c - v2 * s
            vec[offset + i + half] = v2 * c + v1 * s
        }
    }

    /**
     * Perkalian matriks W [outDim, inDim] dengan vektor x [inDim]:
     * out[j] = sum_i x[i] * W[j * inDim + i]
     */
    private fun matmulVec(
        w: FloatArray,
        x: FloatArray,
        out: FloatArray,
        outDim: Int,
        inDim: Int
    ) {
        if (NexoraNative.isNdkLoaded) {
            try {
                NexoraNative.nativeMatmulVec(x, w, inDim, outDim, out)
                return
            } catch (e: Throwable) {}
        }
        for (o in 0 until outDim) {
            val rowStart = o * inDim
            var s = 0.0f
            for (i in 0 until inDim) {
                s += w[rowStart + i] * x[i]
            }
            out[o] = s
        }
    }

    /**
     * Logits via Weight Tying: logits = x @ tok_emb^T
     */
    private fun computeLogits(
        x: FloatArray,
        logits: FloatArray
    ) {
        if (NexoraNative.isNdkLoaded) {
            try {
                NexoraNative.nativeEmbeddingLogits(x, tokEmb, config.vocabSize, config.dModel, logits)
                return
            } catch (e: Throwable) {}
        }
        val dModel = config.dModel
        for (v in 0 until config.vocabSize) {
            val rowStart = v * dModel
            var dot = 0.0f
            for (i in 0 until dModel) {
                dot += x[i] * tokEmb[rowStart + i]
            }
            logits[v] = dot
        }
    }

    /**
     * Forward pass murni tanpa menyimpan aktivasi (untuk inferensi & evaluasi loss validasi).
     */
    fun evaluateSequenceLoss(tokens: IntArray): Float {
        val tLen = tokens.size
        if (tLen < 2) return 0.0f

        val d = config.dModel
        val hCount = config.numHeads
        val hd = config.headDim
        val scale = 1.0f / sqrt(hd.toFloat())

        // x: [T, d]
        val x = Array(tLen) { FloatArray(d) }
        for (t in 0 until tLen) {
            val rowStart = tokens[t] * d
            System.arraycopy(tokEmb, rowStart, x[t], 0, d)
        }

        val hTemp = FloatArray(d)
        val qAll = Array(tLen) { FloatArray(d) }
        val kAll = Array(tLen) { FloatArray(d) }
        val vAll = Array(tLen) { FloatArray(d) }
        val outHeads = Array(tLen) { FloatArray(d) }
        val gate = FloatArray(config.dFf)
        val up = FloatArray(config.dFf)
        val swiglu = FloatArray(config.dFf)
        val down = FloatArray(d)
        val scores = FloatArray(tLen)
        val probs = FloatArray(tLen)

        for (l in 0 until config.numLayers) {
            val lw = layers[l]

            // 1. Attention
            for (t in 0 until tLen) {
                applyRmsNorm(x[t], lw.attnNorm, hTemp, d)
                matmulVec(lw.wq, hTemp, qAll[t], d, d)
                matmulVec(lw.wk, hTemp, kAll[t], d, d)
                matmulVec(lw.wv, hTemp, vAll[t], d, d)

                // RoPE per head
                for (h in 0 until hCount) {
                    applyRope(qAll[t], h * hd, t, hd)
                    applyRope(kAll[t], h * hd, t, hd)
                }
            }

            // Multi-head Causal Attention
            for (t in 0 until tLen) {
                outHeads[t].fill(0.0f)
            }

            for (h in 0 until hCount) {
                val hOffset = h * hd
                for (t in 0 until tLen) {
                    var maxScore = -Float.MAX_VALUE
                    for (s in 0..t) {
                        var dot = 0.0f
                        for (i in 0 until hd) {
                            dot += qAll[t][hOffset + i] * kAll[s][hOffset + i]
                        }
                        val sc = dot * scale
                        scores[s] = sc
                        if (sc > maxScore) maxScore = sc
                    }

                    var sumExp = 0.0f
                    for (s in 0..t) {
                        val p = exp(scores[s] - maxScore)
                        probs[s] = p
                        sumExp += p
                    }
                    val invSum = 1.0f / sumExp
                    for (s in 0..t) {
                        probs[s] *= invSum
                    }

                    for (i in 0 until hd) {
                        var vVal = 0.0f
                        for (s in 0..t) {
                            vVal += probs[s] * vAll[s][hOffset + i]
                        }
                        outHeads[t][hOffset + i] = vVal
                    }
                }
            }

            // Wo projection + Residual
            for (t in 0 until tLen) {
                matmulVec(lw.wo, outHeads[t], hTemp, d, d)
                for (i in 0 until d) {
                    x[t][i] += hTemp[i]
                }
            }

            // 2. SwiGLU FFN
            for (t in 0 until tLen) {
                applyRmsNorm(x[t], lw.ffnNorm, hTemp, d)
                matmulVec(lw.w1, hTemp, gate, config.dFf, d)
                matmulVec(lw.w3, hTemp, up, config.dFf, d)

                for (i in 0 until config.dFf) {
                    val z = gate[i]
                    val silu = z / (1.0f + exp(-z))
                    swiglu[i] = silu * up[i]
                }

                matmulVec(lw.w2, swiglu, down, d, config.dFf)
                for (i in 0 until d) {
                    x[t][i] += down[i]
                }
            }
        }

        // Final RMSNorm
        for (t in 0 until tLen) {
            applyRmsNorm(x[t], finalNorm, x[t], d)
        }

        // Cross-entropy loss: ids[t] memprediksi ids[t+1]
        val logits = FloatArray(config.vocabSize)
        var totalLoss = 0.0f
        val numTargets = tLen - 1

        for (t in 0 until numTargets) {
            computeLogits(x[t], logits)
            val target = tokens[t + 1]

            var maxLogit = -Float.MAX_VALUE
            for (v in 0 until config.vocabSize) {
                if (logits[v] > maxLogit) maxLogit = logits[v]
            }
            var sumExp = 0.0f
            for (v in 0 until config.vocabSize) {
                sumExp += exp(logits[v] - maxLogit)
            }
            val logSumExp = maxLogit + ln(sumExp)
            totalLoss += (logSumExp - logits[target])
        }

        return totalLoss / numTargets
    }

    /**
     * Melakukan sampling autoregresif token berikutnya dari konteks:
     * - Menghitung logits posisi terakhir.
     * - Temperature scaling & Nucleus Top-P Sampling.
     */
    fun sampleNextToken(
        contextTokens: IntArray,
        temperature: Float = 0.8f,
        topP: Float = 0.9f
    ): Int {
        val tLen = contextTokens.size
        if (tLen == 0) return NexoraTokenizer.EOS_TOKEN

        val d = config.dModel
        val hCount = config.numHeads
        val hd = config.headDim
        val scale = 1.0f / sqrt(hd.toFloat())

        val x = Array(tLen) { FloatArray(d) }
        for (t in 0 until tLen) {
            val rowStart = contextTokens[t] * d
            System.arraycopy(tokEmb, rowStart, x[t], 0, d)
        }

        val hTemp = FloatArray(d)
        val qAll = Array(tLen) { FloatArray(d) }
        val kAll = Array(tLen) { FloatArray(d) }
        val vAll = Array(tLen) { FloatArray(d) }
        val outHeads = Array(tLen) { FloatArray(d) }
        val gate = FloatArray(config.dFf)
        val up = FloatArray(config.dFf)
        val swiglu = FloatArray(config.dFf)
        val down = FloatArray(d)
        val scores = FloatArray(tLen)
        val probs = FloatArray(tLen)

        for (l in 0 until config.numLayers) {
            val lw = layers[l]

            for (t in 0 until tLen) {
                applyRmsNorm(x[t], lw.attnNorm, hTemp, d)
                matmulVec(lw.wq, hTemp, qAll[t], d, d)
                matmulVec(lw.wk, hTemp, kAll[t], d, d)
                matmulVec(lw.wv, hTemp, vAll[t], d, d)

                for (h in 0 until hCount) {
                    applyRope(qAll[t], h * hd, t, hd)
                    applyRope(kAll[t], h * hd, t, hd)
                }
            }

            for (t in 0 until tLen) {
                outHeads[t].fill(0.0f)
            }

            for (h in 0 until hCount) {
                val hOffset = h * hd
                for (t in 0 until tLen) {
                    var maxScore = -Float.MAX_VALUE
                    for (s in 0..t) {
                        var dot = 0.0f
                        for (i in 0 until hd) {
                            dot += qAll[t][hOffset + i] * kAll[s][hOffset + i]
                        }
                        val sc = dot * scale
                        scores[s] = sc
                        if (sc > maxScore) maxScore = sc
                    }

                    var sumExp = 0.0f
                    for (s in 0..t) {
                        val p = exp(scores[s] - maxScore)
                        probs[s] = p
                        sumExp += p
                    }
                    val invSum = 1.0f / sumExp
                    for (s in 0..t) {
                        probs[s] *= invSum
                    }

                    for (i in 0 until hd) {
                        var vVal = 0.0f
                        for (s in 0..t) {
                            vVal += probs[s] * vAll[s][hOffset + i]
                        }
                        outHeads[t][hOffset + i] = vVal
                    }
                }
            }

            for (t in 0 until tLen) {
                matmulVec(lw.wo, outHeads[t], hTemp, d, d)
                for (i in 0 until d) {
                    x[t][i] += hTemp[i]
                }
            }

            for (t in 0 until tLen) {
                applyRmsNorm(x[t], lw.ffnNorm, hTemp, d)
                matmulVec(lw.w1, hTemp, gate, config.dFf, d)
                matmulVec(lw.w3, hTemp, up, config.dFf, d)

                for (i in 0 until config.dFf) {
                    val z = gate[i]
                    val silu = z / (1.0f + exp(-z))
                    swiglu[i] = silu * up[i]
                }

                matmulVec(lw.w2, swiglu, down, d, config.dFf)
                for (i in 0 until d) {
                    x[t][i] += down[i]
                }
            }
        }

        // Ambil token terakhir untuk inferensi
        val lastIdx = tLen - 1
        applyRmsNorm(x[lastIdx], finalNorm, hTemp, d)

        val logits = FloatArray(config.vocabSize)
        computeLogits(hTemp, logits)

        // Argmax jika temperature sangat kecil
        if (temperature <= 0.05f) {
            var bestId = 0
            var bestVal = logits[0]
            for (v in 1 until config.vocabSize) {
                if (logits[v] > bestVal) {
                    bestVal = logits[v]
                    bestId = v
                }
            }
            return bestId
        }

        // Temperature scaling & Top-P Nucleus Sampling
        val invTemp = 1.0f / max(0.1f, temperature)
        var maxL = -Float.MAX_VALUE
        for (v in 0 until config.vocabSize) {
            logits[v] *= invTemp
            if (logits[v] > maxL) maxL = logits[v]
        }

        var sumExp = 0.0f
        val expLogits = FloatArray(config.vocabSize)
        for (v in 0 until config.vocabSize) {
            val e = exp(logits[v] - maxL)
            expLogits[v] = e
            sumExp += e
        }
        for (v in 0 until config.vocabSize) {
            expLogits[v] /= sumExp
        }

        // Top-P filter
        val indexed = Array(config.vocabSize) { it }
        indexed.sortWith { a, b -> expLogits[b].compareTo(expLogits[a]) }

        var cumSum = 0.0f
        var cutoff = config.vocabSize
        for (i in 0 until config.vocabSize) {
            cumSum += expLogits[indexed[i]]
            if (cumSum >= topP) {
                cutoff = i + 1
                break
            }
        }

        var r = random.nextFloat() * cumSum
        for (i in 0 until cutoff) {
            val token = indexed[i]
            r -= expLogits[token]
            if (r <= 0.0f) {
                return token
            }
        }
        return indexed[0]
    }

    /**
     * Pelatihan Sekuensial On-Device Eksak:
     * Menjalankan Forward Pass, Menghitung Train Loss per Batch, Menghitung Gradien Matematis Nyata via Backpropagation,
     * Melakukan Gradient Clipping (global norm 1.0), dan Melakukan Update Parameter dengan Decoupled AdamW.
     */
    @Synchronized
    fun trainSequenceStep(
        tokens: IntArray,
        learningRate: Float = 1e-4f,
        weightDecay: Float = 0.1f,
        optimizerType: OptimizerType = OptimizerType.ADAMW,
        maxGradNorm: Float = 1.0f
    ): Pair<Float, Float> {
        // Reset first: a failed native call must never inherit a previous step's Vulkan status.
        lastStepUsedVulkan = false
        val tLen = tokens.size
        if (tLen < 2) return Pair(0.0f, 0.0f)

        // Switching optimizer invalidates its accumulated moment semantics; reset them
        // once at the transition rather than mixing AdamW/Lion/SGD momentum states.
        if (optimizerType != activeOptimizerType) {
            resetOptimizerState()
            adamStepCount.set(0L)
            activeOptimizerType = optimizerType
            // The Kotlin path keeps parameter arrays authoritative. Repack them before
            // entering the native path again after an optimizer change.
            syncWeightsToFlat()
            syncMomentsToFlat()
        }

        // Native Vulkan trainer currently implements AdamW only. Lion and SGD Momentum
        // intentionally use the real Kotlin optimizer implementation below instead of
        // silently pretending the native AdamW kernel implements them.
        if (optimizerType == OptimizerType.ADAMW && NexoraNative.isNdkLoaded) {
            try {
                val step = (adamStepCount.get() + 1L).coerceAtMost(Int.MAX_VALUE.toLong()).toInt()
                val res = NexoraNative.nativeTrainStepLM5M(
                    tokens = tokens,
                    allWeights = allWeights,
                    allGrads = allGrads,
                    allM = allM,
                    allV = allV,
                    learningRate = learningRate,
                    weightDecay = weightDecay,
                    maxGradNorm = maxGradNorm,
                    step = step,
                    preferVulkan = preferVulkan
                )
                if (res.size >= 3 && res[0].isFinite() && res[1].isFinite()) {
                    adamStepCount.set(step.toLong())
                    lastStepUsedVulkan = (res[2] > 0.5f)
                    syncFlatToWeights()
                    return Pair(res[0], res[1])
                }
            } catch (e: Throwable) {
                // The step counter advances only after a valid native result. Use the
                // explicit Kotlin backprop/optimizer path when native execution fails.
            }
        }

        val d = config.dModel
        val hCount = config.numHeads
        val hd = config.headDim
        val scale = 1.0f / sqrt(hd.toFloat())

        // 1. FORWARD PASS DENGAN PENYIMPANAN AKTIVASI LENGKAP
        val xIn = Array(tLen) { FloatArray(d) }
        for (t in 0 until tLen) {
            System.arraycopy(tokEmb, tokens[t] * d, xIn[t], 0, d)
        }

        // Cache per layer
        class LayerCache {
            val hAttn = Array(tLen) { FloatArray(d) }
            val invStdAttn = FloatArray(tLen)
            val q = Array(tLen) { FloatArray(d) }
            val k = Array(tLen) { FloatArray(d) }
            val v = Array(tLen) { FloatArray(d) }
            val attnWeights = Array(hCount) { Array(tLen) { FloatArray(tLen) } }
            val outHeads = Array(tLen) { FloatArray(d) }
            val xMid = Array(tLen) { FloatArray(d) }
            val hFfn = Array(tLen) { FloatArray(d) }
            val invStdFfn = FloatArray(tLen)
            val gate = Array(tLen) { FloatArray(config.dFf) }
            val up = Array(tLen) { FloatArray(config.dFf) }
            val silu = Array(tLen) { FloatArray(config.dFf) }
            val swiglu = Array(tLen) { FloatArray(config.dFf) }
            val xOut = Array(tLen) { FloatArray(d) }
        }

        val caches = Array(config.numLayers) { LayerCache() }
        var currentX = xIn

        for (l in 0 until config.numLayers) {
            val lw = layers[l]
            val lc = caches[l]

            // Attention Pre-RMSNorm
            for (t in 0 until tLen) {
                lc.invStdAttn[t] = applyRmsNorm(currentX[t], lw.attnNorm, lc.hAttn[t], d)
                matmulVec(lw.wq, lc.hAttn[t], lc.q[t], d, d)
                matmulVec(lw.wk, lc.hAttn[t], lc.k[t], d, d)
                matmulVec(lw.wv, lc.hAttn[t], lc.v[t], d, d)

                for (h in 0 until hCount) {
                    applyRope(lc.q[t], h * hd, t, hd)
                    applyRope(lc.k[t], h * hd, t, hd)
                }
            }

            // Multi-head Attention
            for (h in 0 until hCount) {
                val hOffset = h * hd
                for (t in 0 until tLen) {
                    var maxScore = -Float.MAX_VALUE
                    val scores = FloatArray(t + 1)
                    for (s in 0..t) {
                        var dot = 0.0f
                        for (i in 0 until hd) {
                            dot += lc.q[t][hOffset + i] * lc.k[s][hOffset + i]
                        }
                        val sc = dot * scale
                        scores[s] = sc
                        if (sc > maxScore) maxScore = sc
                    }

                    var sumExp = 0.0f
                    for (s in 0..t) {
                        val p = exp(scores[s] - maxScore)
                        lc.attnWeights[h][t][s] = p
                        sumExp += p
                    }
                    val invSum = 1.0f / sumExp
                    for (s in 0..t) {
                        lc.attnWeights[h][t][s] *= invSum
                    }

                    for (i in 0 until hd) {
                        var vVal = 0.0f
                        for (s in 0..t) {
                            vVal += lc.attnWeights[h][t][s] * lc.v[s][hOffset + i]
                        }
                        lc.outHeads[t][hOffset + i] = vVal
                    }
                }
            }

            // Wo Projection + Residual -> xMid
            val hTemp = FloatArray(d)
            for (t in 0 until tLen) {
                matmulVec(lw.wo, lc.outHeads[t], hTemp, d, d)
                for (i in 0 until d) {
                    lc.xMid[t][i] = currentX[t][i] + hTemp[i]
                }
            }

            // SwiGLU FFN Pre-RMSNorm
            for (t in 0 until tLen) {
                lc.invStdFfn[t] = applyRmsNorm(lc.xMid[t], lw.ffnNorm, lc.hFfn[t], d)
                matmulVec(lw.w1, lc.hFfn[t], lc.gate[t], config.dFf, d)
                matmulVec(lw.w3, lc.hFfn[t], lc.up[t], config.dFf, d)

                for (i in 0 until config.dFf) {
                    val z = lc.gate[t][i]
                    val sVal = z / (1.0f + exp(-z))
                    lc.silu[t][i] = sVal
                    lc.swiglu[t][i] = sVal * lc.up[t][i]
                }

                matmulVec(lw.w2, lc.swiglu[t], hTemp, d, config.dFf)
                for (i in 0 until d) {
                    lc.xOut[t][i] = lc.xMid[t][i] + hTemp[i]
                }
            }

            currentX = lc.xOut
        }

        // Final RMSNorm
        val xFinal = Array(tLen) { FloatArray(d) }
        val invStdFinal = FloatArray(tLen)
        for (t in 0 until tLen) {
            invStdFinal[t] = applyRmsNorm(currentX[t], finalNorm, xFinal[t], d)
        }

        // 2. LOSS & OUTPUT HEAD LOGITS GRADIENT
        val numTargets = tLen - 1
        var batchLoss = 0.0f
        val dXFinal = Array(tLen) { FloatArray(d) }
        val logits = FloatArray(config.vocabSize)

        // Reset gradien
        gradTokEmb.fill(0.0f)
        gradFinalNorm.fill(0.0f)
        for (l in 0 until config.numLayers) {
            val lw = layers[l]
            lw.gradAttnNorm.fill(0.0f)
            lw.gradWq.fill(0.0f); lw.gradWk.fill(0.0f); lw.gradWv.fill(0.0f); lw.gradWo.fill(0.0f)
            lw.gradFfnNorm.fill(0.0f)
            lw.gradW1.fill(0.0f); lw.gradW3.fill(0.0f); lw.gradW2.fill(0.0f)
        }

        val invNormFactor = 1.0f / numTargets
        for (t in 0 until numTargets) {
            computeLogits(xFinal[t], logits)
            val target = tokens[t + 1]

            var maxL = -Float.MAX_VALUE
            for (v in 0 until config.vocabSize) {
                if (logits[v] > maxL) maxL = logits[v]
            }
            var sumExp = 0.0f
            for (v in 0 until config.vocabSize) {
                val e = exp(logits[v] - maxL)
                logits[v] = e
                sumExp += e
            }
            val invSum = 1.0f / sumExp
            for (v in 0 until config.vocabSize) {
                logits[v] *= invSum
            }

            val targetProb = logits[target]
            batchLoss += -ln(max(1e-12f, targetProb))

            // dLogits = (p_v - 1{v == target}) / numTargets
            logits[target] -= 1.0f
            for (v in 0 until config.vocabSize) {
                val dL = logits[v] * invNormFactor
                val rowStart = v * d
                for (i in 0 until d) {
                    dXFinal[t][i] += dL * tokEmb[rowStart + i]
                    gradTokEmb[rowStart + i] += dL * xFinal[t][i]
                }
            }
        }
        val finalLoss = batchLoss / numTargets

        // 3. BACKPROPAGATION TENSOR SEJATI DARI ATAS KE BAWAH
        var dXCurrent = Array(tLen) { FloatArray(d) }

        // Backprop through final_norm
        for (t in 0 until tLen) {
            val invStd = invStdFinal[t]
            val xRow = currentX[t]
            var sumDxW = 0.0f
            for (i in 0 until d) {
                gradFinalNorm[i] += dXFinal[t][i] * xRow[i] * invStd
                sumDxW += dXFinal[t][i] * finalNorm[i] * xRow[i]
            }
            val factor = sumDxW / (d * ((1.0f / (invStd * invStd))))
            for (i in 0 until d) {
                dXCurrent[t][i] = (dXFinal[t][i] * finalNorm[i] - xRow[i] * factor) * invStd
            }
        }

        // Backprop through layers (4 down to 0)
        val dSwiglu = FloatArray(config.dFf)
        val dGate = FloatArray(config.dFf)
        val dUp = FloatArray(config.dFf)
        val dHFfn = FloatArray(d)
        val dHAttn = FloatArray(d)
        val dOutHeads = Array(tLen) { FloatArray(d) }

        for (l in config.numLayers - 1 downTo 0) {
            val lw = layers[l]
            val lc = caches[l]

            // 1. Backprop through SwiGLU FFN
            val dXMid = Array(tLen) { FloatArray(d) }
            for (t in 0 until tLen) {
                // Residual branch: dXMid receives dXCurrent directly
                System.arraycopy(dXCurrent[t], 0, dXMid[t], 0, d)

                // Down projection: down = swiglu @ w2^T
                // dswiglu = dXCurrent @ w2
                dSwiglu.fill(0.0f)
                for (j in 0 until config.dFf) {
                    var s = 0.0f
                    for (i in 0 until d) {
                        s += dXCurrent[t][i] * lw.w2[i * config.dFf + j]
                        lw.gradW2[i * config.dFf + j] += dXCurrent[t][i] * lc.swiglu[t][j]
                    }
                    dSwiglu[j] = s
                }

                // SwiGLU activation derivative:
                // silu = gate * sig(gate)
                // swiglu = silu * up
                // dup = dswiglu * silu
                // dsilu = dswiglu * up
                // dgate = dsilu * (sig(gate) + gate * sig(gate) * (1 - sig(gate)))
                dHFfn.fill(0.0f)
                for (j in 0 until config.dFf) {
                    val z = lc.gate[t][j]
                    val sig = 1.0f / (1.0f + exp(-z))
                    val siluVal = z * sig
                    val dupVal = dSwiglu[j] * siluVal
                    val dsiluVal = dSwiglu[j] * lc.up[t][j]
                    val dgateVal = dsiluVal * (sig + siluVal * (1.0f - sig))

                    dGate[j] = dgateVal
                    dUp[j] = dupVal
                }

                // Up & Gate projections:
                // gradW1 += dgate @ hFfn^T ; gradW3 += dup @ hFfn^T
                // dHFfn = dgate @ w1 + dup @ w3
                for (j in 0 until config.dFf) {
                    val dg = dGate[j]
                    val du = dUp[j]
                    val rowStart = j * d
                    for (i in 0 until d) {
                        lw.gradW1[rowStart + i] += dg * lc.hFfn[t][i]
                        lw.gradW3[rowStart + i] += du * lc.hFfn[t][i]
                        dHFfn[i] += dg * lw.w1[rowStart + i] + du * lw.w3[rowStart + i]
                    }
                }

                // Backprop through FFN RMSNorm
                val invStd = lc.invStdFfn[t]
                val xRow = lc.xMid[t]
                var sumDxW = 0.0f
                for (i in 0 until d) {
                    lw.gradFfnNorm[i] += dHFfn[i] * xRow[i] * invStd
                    sumDxW += dHFfn[i] * lw.ffnNorm[i] * xRow[i]
                }
                val factor = sumDxW / (d * (1.0f / (invStd * invStd)))
                for (i in 0 until d) {
                    dXMid[t][i] += (dHFfn[i] * lw.ffnNorm[i] - xRow[i] * factor) * invStd
                }
            }

            // 2. Backprop through Multi-Head Attention
            val dXPrev = Array(tLen) { FloatArray(d) }
            for (t in 0 until tLen) {
                // Residual branch
                System.arraycopy(dXMid[t], 0, dXPrev[t], 0, d)

                // wo projection:
                // dOutHeads = dXMid @ wo
                // gradWo += dXMid @ outHeads^T
                for (j in 0 until d) {
                    var s = 0.0f
                    for (i in 0 until d) {
                        s += dXMid[t][i] * lw.wo[i * d + j]
                        lw.gradWo[i * d + j] += dXMid[t][i] * lc.outHeads[t][j]
                    }
                    dOutHeads[t][j] = s
                }
            }

            // Attention heads backward
            val dQ = Array(tLen) { FloatArray(d) }
            val dK = Array(tLen) { FloatArray(d) }
            val dV = Array(tLen) { FloatArray(d) }

            for (h in 0 until hCount) {
                val hOffset = h * hd
                for (t in 0 until tLen) {
                    val dScore = FloatArray(t + 1)
                    for (s in 0..t) {
                        var dValSum = 0.0f
                        for (i in 0 until hd) {
                            val dOut = dOutHeads[t][hOffset + i]
                            dV[s][hOffset + i] += lc.attnWeights[h][t][s] * dOut
                            dValSum += dOut * lc.v[s][hOffset + i]
                        }
                        dScore[s] = dValSum
                    }

                    // Softmax backward
                    var sumDScoreP = 0.0f
                    for (s in 0..t) {
                        sumDScoreP += dScore[s] * lc.attnWeights[h][t][s]
                    }
                    for (s in 0..t) {
                        val dAttn = lc.attnWeights[h][t][s] * (dScore[s] - sumDScoreP) * scale
                        for (i in 0 until hd) {
                            dQ[t][hOffset + i] += dAttn * lc.k[s][hOffset + i]
                            dK[s][hOffset + i] += dAttn * lc.q[t][hOffset + i]
                        }
                    }
                }
            }

            // Inverse RoPE on dQ and dK (orthogonal rotation, transpose is inverse angle)
            for (t in 0 until tLen) {
                for (h in 0 until hCount) {
                    val half = hd / 2
                    val hOffset = h * hd
                    for (i in 0 until half) {
                        val ang = t * invFreq[i]
                        val c = cos(ang.toDouble()).toFloat()
                        val s = sin(ang.toDouble()).toFloat()

                        // dQ inverse
                        val dq1 = dQ[t][hOffset + i]
                        val dq2 = dQ[t][hOffset + i + half]
                        dQ[t][hOffset + i] = dq1 * c + dq2 * s
                        dQ[t][hOffset + i + half] = dq2 * c - dq1 * s

                        // dK inverse
                        val dk1 = dK[t][hOffset + i]
                        val dk2 = dK[t][hOffset + i + half]
                        dK[t][hOffset + i] = dk1 * c + dk2 * s
                        dK[t][hOffset + i + half] = dk2 * c - dk1 * s
                    }
                }

                // Projections wq, wk, wv backward:
                // gradWq += dq @ hAttn^T, etc.
                // dHAttn = dq @ wq + dk @ wk + dv @ wv
                dHAttn.fill(0.0f)
                for (j in 0 until d) {
                    val dq = dQ[t][j]
                    val dk = dK[t][j]
                    val dv = dV[t][j]
                    val rowStart = j * d
                    for (i in 0 until d) {
                        lw.gradWq[rowStart + i] += dq * lc.hAttn[t][i]
                        lw.gradWk[rowStart + i] += dk * lc.hAttn[t][i]
                        lw.gradWv[rowStart + i] += dv * lc.hAttn[t][i]
                        dHAttn[i] += dq * lw.wq[rowStart + i] + dk * lw.wk[rowStart + i] + dv * lw.wv[rowStart + i]
                    }
                }

                // Attention Pre-RMSNorm backward
                val invStd = lc.invStdAttn[t]
                val prevX = if (l == 0) xIn[t] else caches[l - 1].xOut[t]
                var sumDxW = 0.0f
                for (i in 0 until d) {
                    lw.gradAttnNorm[i] += dHAttn[i] * prevX[i] * invStd
                    sumDxW += dHAttn[i] * lw.attnNorm[i] * prevX[i]
                }
                val factor = sumDxW / (d * (1.0f / (invStd * invStd)))
                for (i in 0 until d) {
                    dXPrev[t][i] += (dHAttn[i] * lw.attnNorm[i] - prevX[i] * factor) * invStd
                }
            }

            dXCurrent = dXPrev
        }

        // Backprop to Token Embeddings tokEmb
        for (t in 0 until tLen) {
            val rowStart = tokens[t] * d
            for (i in 0 until d) {
                gradTokEmb[rowStart + i] += dXCurrent[t][i]
            }
        }

        // 4. GLOBAL GRADIENT CLIPPING (MAX NORM = 1.0f)
        var sumSqGrad = 0.0
        for (v in gradTokEmb) sumSqGrad += (v * v).toDouble()
        for (v in gradFinalNorm) sumSqGrad += (v * v).toDouble()
        for (lw in layers) {
            for (v in lw.gradAttnNorm) sumSqGrad += (v * v).toDouble()
            for (v in lw.gradWq) sumSqGrad += (v * v).toDouble()
            for (v in lw.gradWk) sumSqGrad += (v * v).toDouble()
            for (v in lw.gradWv) sumSqGrad += (v * v).toDouble()
            for (v in lw.gradWo) sumSqGrad += (v * v).toDouble()
            for (v in lw.gradFfnNorm) sumSqGrad += (v * v).toDouble()
            for (v in lw.gradW1) sumSqGrad += (v * v).toDouble()
            for (v in lw.gradW3) sumSqGrad += (v * v).toDouble()
            for (v in lw.gradW2) sumSqGrad += (v * v).toDouble()
        }
        val globalNorm = sqrt(sumSqGrad).toFloat()
        val clipScale = if (globalNorm > maxGradNorm) (maxGradNorm / globalNorm) else 1.0f

        // 5. Apply the selected optimizer with decoupled decay on matrix parameters only.
        val step = adamStepCount.incrementAndGet().coerceAtLeast(1L)
        val b1 = 0.9f
        val b2 = 0.95f
        val eps = 1e-8f
        val b1Pow = Math.pow(b1.toDouble(), step.toDouble()).toFloat()
        val b2Pow = Math.pow(b2.toDouble(), step.toDouble()).toFloat()
        val alphaT = learningRate * sqrt((1.0f - b2Pow).coerceAtLeast(0f)) / (1.0f - b1Pow)
        val epsCorrected = eps * sqrt((1.0f - b2Pow).coerceAtLeast(0f))

        fun applyOptimizer(
            param: FloatArray,
            grad: FloatArray,
            m: FloatArray,
            v: FloatArray,
            is2D: Boolean
        ) {
            val wd = if (is2D) weightDecay else 0.0f
            val decayFactor = 1.0f - learningRate * wd
            when (optimizerType) {
                OptimizerType.ADAMW -> {
                    for (i in param.indices) {
                        if (wd > 0f) param[i] *= decayFactor
                        val g = grad[i] * clipScale
                        val mVal = b1 * m[i] + (1.0f - b1) * g
                        val vVal = b2 * v[i] + (1.0f - b2) * g * g
                        m[i] = mVal
                        v[i] = vVal
                        param[i] -= alphaT * mVal / (sqrt(vVal) + epsCorrected)
                        grad[i] = 0f
                    }
                }
                OptimizerType.LION -> {
                    val beta1Lion = 0.9f
                    val beta2Lion = 0.99f
                    for (i in param.indices) {
                        if (wd > 0f) param[i] *= decayFactor
                        val g = grad[i] * clipScale
                        val update = beta1Lion * m[i] + (1.0f - beta1Lion) * g
                        val sign = when {
                            update > 0f -> 1f
                            update < 0f -> -1f
                            else -> 0f
                        }
                        param[i] -= learningRate * sign
                        m[i] = beta2Lion * m[i] + (1.0f - beta2Lion) * g
                        grad[i] = 0f
                    }
                }
                OptimizerType.SGD_MOMENTUM -> {
                    val momentum = 0.9f
                    for (i in param.indices) {
                        if (wd > 0f) param[i] *= decayFactor
                        val g = grad[i] * clipScale
                        m[i] = momentum * m[i] + g
                        param[i] -= learningRate * m[i]
                        grad[i] = 0f
                    }
                }
            }
        }

        applyOptimizer(tokEmb, gradTokEmb, mTokEmb, vTokEmb, is2D = true)
        applyOptimizer(finalNorm, gradFinalNorm, mFinalNorm, vFinalNorm, is2D = false)

        for (lw in layers) {
            applyOptimizer(lw.attnNorm, lw.gradAttnNorm, lw.mAttnNorm, lw.vAttnNorm, is2D = false)
            applyOptimizer(lw.wq, lw.gradWq, lw.mWq, lw.vWq, is2D = true)
            applyOptimizer(lw.wk, lw.gradWk, lw.mWk, lw.vWk, is2D = true)
            applyOptimizer(lw.wv, lw.gradWv, lw.mWv, lw.vWv, is2D = true)
            applyOptimizer(lw.wo, lw.gradWo, lw.mWo, lw.vWo, is2D = true)
            applyOptimizer(lw.ffnNorm, lw.gradFfnNorm, lw.mFfnNorm, lw.vFfnNorm, is2D = false)
            applyOptimizer(lw.w1, lw.gradW1, lw.mW1, lw.vW1, is2D = true)
            applyOptimizer(lw.w3, lw.gradW3, lw.mW3, lw.vW3, is2D = true)
            applyOptimizer(lw.w2, lw.gradW2, lw.mW2, lw.vW2, is2D = true)
        }

        return Pair(finalLoss, globalNorm)
    }

    fun cloneIsolated(): NexoraTransformer {
        val copy = NexoraTransformer(config, Random(42))
        System.arraycopy(allWeights, 0, copy.allWeights, 0, allWeights.size)
        System.arraycopy(allM, 0, copy.allM, 0, allM.size)
        System.arraycopy(allV, 0, copy.allV, 0, allV.size)
        copy.syncFlatToWeights()
        copy.syncFlatToMoments()
        return copy
    }
}
