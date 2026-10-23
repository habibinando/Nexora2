package com.nexora.model

import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Operasi Matematika Tensor & Vektor CPU Multi-Core yang Dioptimalkan untuk Android.
 * Menggunakan array float 1D datar untuk meminimalkan beban Garbage Collector (GC)
 * dan memaksimalkan cache L1/L2 prosesor smartphone.
 */
object NexoraTensorOps {

    /**
     * Inisialisasi bobot dengan distribusi normal Gauss (Xavier/He style).
     */
    fun createRandomWeights(size: Int, stdDev: Float, random: Random = Random(42)): FloatArray {
        val array = FloatArray(size)
        for (i in 0 until size) {
            // Box-Muller transform untuk Gaussian random nyata
            val u1 = max(1e-7f, random.nextFloat())
            val u2 = random.nextFloat()
            val z = sqrt(-2.0f * ln(u1)) * kotlin.math.cos(2.0f * Math.PI.toFloat() * u2)
            array[i] = (z * stdDev).toFloat()
        }
        return array
    }

    /**
     * Perkalian matriks-vektor: out = x * W
     * Didukung akselerasi C++ NDK r30 jika tersedia.
     */
    fun matmulVec(x: FloatArray, W: FloatArray, dIn: Int, dOut: Int, out: FloatArray) {
        if (NexoraNative.isNdkLoaded) {
            try {
                NexoraNative.nativeMatmulVec(x, W, dIn, dOut, out)
                return
            } catch (e: Throwable) {
                // Fallback otomatis ke JVM
            }
        }

        out.fill(0f)
        for (r in 0 until dIn) {
            val xr = x[r]
            val wRow = r * dOut
            for (c in 0 until dOut) {
                out[c] += xr * W[wRow + c]
            }
        }
    }

    /**
     * LoRA Forward Pass dengan efisiensi O(dIn*r + r*dOut):
     * out = x * W_base + scale * ((x * A) * B)
     * Didukung akselerasi C++ NDK r30 jika tersedia.
     */
    fun loraForwardVec(
        x: FloatArray,
        wBase: FloatArray,
        aMat: FloatArray,
        bMat: FloatArray,
        dIn: Int,
        dOut: Int,
        rank: Int,
        scale: Float,
        tempRankBuf: FloatArray,
        out: FloatArray
    ) {
        // 1. Base linear projection: out = x * W_base
        matmulVec(x, wBase, dIn, dOut, out)

        // 2. LoRA low-rank path: temp = x * A (dimensi: rank)
        for (r in 0 until rank) {
            var sum = 0.0f
            var aIdx = r
            for (i in 0 until dIn) {
                sum += x[i] * aMat[aIdx]
                aIdx += rank
            }
            tempRankBuf[r] = sum
        }

        // 3. LoRA projection: out += scale * (temp * B)
        for (c in 0 until dOut) {
            var loraSum = 0.0f
            var bIdx = c
            for (r in 0 until rank) {
                loraSum += tempRankBuf[r] * bMat[bIdx]
                bIdx += dOut
            }
            out[c] += scale * loraSum
        }
    }

    /**
     * RMSNorm (Root Mean Square Layer Normalization) in-place:
     * x = x / sqrt(mean(x^2) + eps) * gamma
     */
    fun rmsnorm(x: FloatArray, gamma: FloatArray, eps: Float = 1e-6f) {
        val dim = x.size
        var sumSquares = 0.0f
        for (i in 0 until dim) {
            val v = x[i]
            sumSquares += v * v
        }
        val rms = sqrt(sumSquares / dim + eps)
        val invRms = 1.0f / rms

        for (i in 0 until dim) {
            x[i] = x[i] * invRms * gamma[i]
        }
    }

    /**
     * Backward pass derivatif eksak RMSNorm:
     * dy: gradien dari output norm [dim]
     * x: aktivasi input sebelum norm [dim]
     * gamma: parameter scaling [dim]
     * dGamma: akumulator gradien gamma (nullable)
     * dx: gradien terpropagasi ke input sebelum norm [dim]
     */
    fun rmsnormBackward(
        dy: FloatArray,
        x: FloatArray,
        gamma: FloatArray,
        dGamma: FloatArray?,
        dx: FloatArray,
        eps: Float = 1e-6f
    ) {
        val dim = x.size
        var sumSq = 0f
        for (i in 0 until dim) {
            sumSq += x[i] * x[i]
        }
        val rms = sqrt(sumSq / dim + eps)
        val invRms = 1.0f / rms
        val invRms3 = invRms * invRms * invRms

        var sumGyx = 0f
        for (i in 0 until dim) {
            val gy = dy[i] * gamma[i]
            sumGyx += gy * x[i]
            if (dGamma != null) {
                dGamma[i] += dy[i] * (x[i] * invRms)
            }
        }

        val factor = (sumGyx / dim.toFloat()) * invRms3
        for (i in 0 until dim) {
            dx[i] = (dy[i] * gamma[i]) * invRms - x[i] * factor
        }
    }

    /**
     * Perkalian head logits dengan tokenEmbeddings:
     * tokenEmbeddings berukuran [vocabSize, dModel], disimpan row-major (v * dModel + d).
     * logits[v] = h • tokenEmbeddings[v]
     */
    fun computeEmbeddingLogits(
        h: FloatArray,
        tokenEmbeddings: FloatArray,
        vocabSize: Int,
        dModel: Int,
        logits: FloatArray
    ) {
        for (v in 0 until vocabSize) {
            var sum = 0f
            val off = v * dModel
            for (d in 0 until dModel) {
                sum += h[d] * tokenEmbeddings[off + d]
            }
            logits[v] = sum
        }
    }

    /**
     * Backward pass untuk embedding logits:
     * Mengalirkan gradien dLogits ke dFinalH dan gradTokenEmbeddings secara konsisten.
     */
    fun computeEmbeddingLogitsBackward(
        dLogits: FloatArray,
        finalH: FloatArray,
        tokenEmbeddings: FloatArray,
        vocabSize: Int,
        dModel: Int,
        dFinalH: FloatArray,
        gradTokenEmbeddings: FloatArray?,
        isFullTrain: Boolean
    ) {
        for (v in 0 until vocabSize) {
            val g = dLogits[v]
            if (g == 0f) continue
            val off = v * dModel
            for (d in 0 until dModel) {
                dFinalH[d] += g * tokenEmbeddings[off + d]
                if (isFullTrain && gradTokenEmbeddings != null) {
                    gradTokenEmbeddings[off + d] += g * finalH[d]
                }
            }
        }
    }

    fun rmsNorm(x: FloatArray, gamma: FloatArray, eps: Float, out: FloatArray) {
        val dim = x.size
        var sumSquares = 0.0f
        for (i in 0 until dim) {
            val v = x[i]
            sumSquares += v * v
        }
        val rms = sqrt(sumSquares / dim + eps)
        val invRms = 1.0f / rms

        for (i in 0 until dim) {
            out[i] = x[i] * invRms * gamma[i]
        }
    }

    /**
     * Aktivasi SiLU / Swish: x * sigmoid(x)
     */
    fun silu(x: Float): Float {
        return x / (1.0f + exp(-x))
    }

    /**
     * Softmax stabil numerik dengan pembagi temperatur.
     */
    fun softmaxInPlace(logits: FloatArray, temp: Float = 1.0f) {
        val size = logits.size
        var maxVal = Float.NEGATIVE_INFINITY
        for (i in 0 until size) {
            if (logits[i] > maxVal) maxVal = logits[i]
        }

        var sumExp = 0.0f
        val invT = 1.0f / max(1e-4f, temp)
        for (i in 0 until size) {
            val e = exp((logits[i] - maxVal) * invT)
            logits[i] = e
            sumExp += e
        }

        val invSum = 1.0f / max(1e-12f, sumExp)
        for (i in 0 until size) {
            logits[i] *= invSum
        }
    }

    /**
     * Hitung Cross-Entropy Loss sesungguhnya untuk 1 token target:
     * Loss = -ln(softmax_probs[targetToken])
     */
    fun computeCrossEntropy(probs: FloatArray, targetToken: Int): Float {
        val p = if (targetToken in probs.indices) probs[targetToken] else 1e-12f
        val clampedP = max(1e-12f, minOf(1.0f, p))
        return -ln(clampedP)
    }

    /**
     * Sampling Top-K dan Top-P (Nucleus Sampling) dari distribusi probabilitas.
     */
    fun sampleTopKTopP(
        probs: FloatArray,
        topK: Int = 40,
        topP: Float = 0.9f,
        random: Random = Random
    ): Int {
        // Ambil indeks dengan probabilitas tertinggi
        val indexedProbs = ArrayList<Pair<Int, Float>>(probs.size)
        for (i in probs.indices) {
            indexedProbs.add(Pair(i, probs[i]))
        }
        indexedProbs.sortByDescending { it.second }

        // Batasi ke Top-K
        val limited = if (topK in 1..indexedProbs.size) indexedProbs.take(topK) else indexedProbs

        // Terapkan Top-P cut-off
        var cumSum = 0.0f
        val nucleus = ArrayList<Pair<Int, Float>>()
        for (item in limited) {
            nucleus.add(item)
            cumSum += item.second
            if (cumSum >= topP && nucleus.size >= 1) break
        }

        // Normalisasi ulang dan sampling
        val r = random.nextFloat() * cumSum
        var running = 0.0f
        for (item in nucleus) {
            running += item.second
            if (running >= r) {
                return item.first
            }
        }
        return nucleus.firstOrNull()?.first ?: 0
    }
}
