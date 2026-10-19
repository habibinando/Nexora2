package com.Nexora.ai.ml

import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sqrt

/** Real AdamW optimizer with decoupled weight decay. Operates on (param, grad) pairs. */
class AdamW(
    private val params: List<Pair<FloatArray, FloatArray>>,
    val beta1: Float = 0.9f,
    val beta2: Float = 0.999f,
    val eps: Float = 1e-8f,
    val weightDecay: Float = 0.01f
) {
    private val m = params.map { FloatArray(it.first.size) }
    private val v = params.map { FloatArray(it.first.size) }
    var step: Int = 0
        private set

    fun step(lr: Float) {
        step++
        val bc1 = 1f - beta1.pow(step)
        val bc2 = 1f - beta2.pow(step)
        for (idx in params.indices) {
            val (p, g) = params[idx]
            val mi = m[idx]; val vi = v[idx]
            for (i in p.indices) {
                val gi = g[i]
                mi[i] = beta1 * mi[i] + (1f - beta1) * gi
                vi[i] = beta2 * vi[i] + (1f - beta2) * gi * gi
                val mhat = mi[i] / bc1
                val vhat = vi[i] / bc2
                p[i] -= lr * (mhat / (sqrt(vhat) + eps) + weightDecay * p[i])
            }
        }
    }

    fun state(): List<Pair<FloatArray, FloatArray>> = m.zip(v)
    fun loadState(state: List<Pair<FloatArray, FloatArray>>, step: Int) {
        for (i in m.indices) { state[i].first.copyInto(m[i]); state[i].second.copyInto(v[i]) }
        this.step = step
    }
}

/** Cosine learning-rate schedule with linear warmup. Real, no mock. */
class CosineWarmupScheduler(
    val maxLR: Float,
    val minLR: Float = maxLR * 0.1f,
    val warmupSteps: Int,
    val totalSteps: Int
) {
    fun lrAt(step: Int): Float {
        if (step < warmupSteps) return maxLR * (step + 1) / warmupSteps
        val progress = ((step - warmupSteps).toFloat() / (totalSteps - warmupSteps).coerceAtLeast(1)).coerceIn(0f, 1f)
        return minLR + 0.5f * (maxLR - minLR) * (1f + cos(Math.PI * progress).toFloat())
    }
}
