package com.Nexora.ai.ml

import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.sin
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random

/** Pure-Kotlin math helpers (elementwise ops; heavy matmul goes through NativeOps). */
object MathUtil {

    fun fillNormal(arr: FloatArray, mean: Float, std: Float, rng: Random) {
        var i = 0
        while (i < arr.size) {
            // Box-Muller
            val u1 = rng.nextFloat().coerceAtLeast(1e-7f)
            val u2 = rng.nextFloat()
            val mag = std * sqrt(-2f * ln(u1))
            val z0 = mag * cos(2f * Math.PI.toFloat() * u2)
            val z1 = mag * sin(2f * Math.PI.toFloat() * u2)
            arr[i] = mean + z0
            if (i + 1 < arr.size) arr[i + 1] = mean + z1
            i += 2
        }
    }

    fun xavierLinear(w: FloatArray, inFeatures: Int, rng: Random) {
        val std = sqrt(2.0 / (inFeatures + w.size / inFeatures)).toFloat() * 0.6f
        fillNormal(w, 0f, std, rng)
    }

    fun zeros(arr: FloatArray) { arr.fill(0f) }

    /** Softmax in place for one row of length n (Kotlin fallback). */
    fun softmaxRow(x: FloatArray, offset: Int, n: Int) {
        var mx = x[offset]
        for (i in 1 until n) if (x[offset + i] > mx) mx = x[offset + i]
        var s = 0f
        for (i in 0 until n) {
            val e = exp(x[offset + i] - mx)
            x[offset + i] = e; s += e
        }
        val inv = 1f / s
        for (i in 0 until n) x[offset + i] *= inv
    }

    fun scale(arr: FloatArray, s: Float) {
        for (i in arr.indices) arr[i] *= s
    }

    fun add(dst: FloatArray, src: FloatArray) {
        for (i in dst.indices) dst[i] += src[i]
    }

    fun gradNorm(vararg grads: FloatArray): Float {
        // Float accumulation can overflow on large models even while individual gradients
        // are finite. Double accumulation delays overflow and makes clipping more stable.
        var sumSquares = 0.0
        for (gradient in grads) {
            for (value in gradient) {
                val v = value.toDouble()
                sumSquares += v * v
            }
        }
        return sqrt(sumSquares).toFloat()
    }

    fun clipGradNorm(grads: List<FloatArray>, maxNorm: Float) {
        val total = gradNorm(*grads.toTypedArray())
        if (total > maxNorm) {
            val scale = maxNorm / total
            for (g in grads) scale(g, scale)
        }
    }
}
