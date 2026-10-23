package com.nexora.model

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.max

enum class OptimizerType(val displayName: String, val memoryEfficiency: String) {
    ADAMW("AdamW", "Standar (2 Buffer Momentum)"),
    LION("Lion (2026)", "EvoLved Sign Momentum (Hemat 50% RAM)"),
    SGD_MOMENTUM("SGD-M", "Ringan (1 Buffer)")
}

enum class LrSchedulerType(val displayName: String) {
    COSINE_ANNEALING("Cosine Annealing + Warmup"),
    LINEAR_DECAY("Linear Decay + Warmup"),
    CONSTANT("Constant + Warmup")
}

data class TrainingConfig2026(
    val optimizerType: OptimizerType = OptimizerType.ADAMW,
    val lrSchedulerType: LrSchedulerType = LrSchedulerType.COSINE_ANNEALING,
    val baseLr: Float = 1e-4f,
    val minLr: Float = 1e-5f,
    val warmupRatio: Float = 0.05f, // 5% langkah pertama untuk warmup
    val weightDecay: Float = 0.1f, // 0.1 per train_state.json Colab
    val valSplitRatio: Float = 0.10f, // 10% data untuk evaluasi Val Loss murni
    val autoCheckpointEverySteps: Int = 50,
    val earlyStoppingPatience: Int = 3,
    val isNdkEnabled: Boolean = true
) {
    /**
     * Menghitung Learning Rate dinamis pada step tertentu berdasarkan schedule matematika 2026.
     */
    fun calculateLr(currentStep: Int, totalSteps: Int): Float {
        if (totalSteps <= 0) return baseLr
        val warmupSteps = max(1, (totalSteps * warmupRatio).toInt())

        // 1. Linear Warmup Phase
        if (currentStep < warmupSteps) {
            val progress = currentStep.toFloat() / warmupSteps
            return minLr + progress * (baseLr - minLr)
        }

        val decaySteps = totalSteps - warmupSteps
        val stepInDecay = (currentStep - warmupSteps).coerceIn(0, decaySteps)
        val progress = stepInDecay.toFloat() / max(1, decaySteps)

        return when (lrSchedulerType) {
            LrSchedulerType.COSINE_ANNEALING -> {
                // eta_t = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(pi * progress))
                minLr + 0.5f * (baseLr - minLr) * (1.0f + cos(PI.toFloat() * progress))
            }
            LrSchedulerType.LINEAR_DECAY -> {
                baseLr - progress * (baseLr - minLr)
            }
            LrSchedulerType.CONSTANT -> {
                baseLr
            }
        }
    }
}
