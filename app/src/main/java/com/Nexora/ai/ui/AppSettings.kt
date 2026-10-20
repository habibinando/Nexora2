package com.Nexora.ai.ui

import android.content.Context
import android.content.SharedPreferences
import com.Nexora.ai.ml.ModelConfig
import com.Nexora.ai.train.HyperParams
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/** Hyperparameters persist locally; draft architecture applies only on the next dataset import. */
object AppSettings {
    private var preferences: SharedPreferences? = null

    private val _modelCfg = MutableStateFlow(
        ModelConfig(vocabSize = 16000, seqLen = 128, dModel = 128, ffnDim = 512,
            numHeads = 8, numLayers = 32, dropout = 0.1f, useLora = false, loraRank = 8, loraAlpha = 16)
    )
    val modelCfg: StateFlow<ModelConfig> = _modelCfg.asStateFlow()

    private val _hp = MutableStateFlow(HyperParams())
    val hp: StateFlow<HyperParams> = _hp.asStateFlow()

    private fun normalizeHyperParams(input: HyperParams): HyperParams {
        val cpuLimit = Runtime.getRuntime().availableProcessors().coerceIn(1, 64)
        fun finiteIn(value: Float, fallback: Float, range: ClosedFloatingPointRange<Float>): Float =
            (if (value.isFinite()) value else fallback).coerceIn(range)

        return input.copy(
            learningRate = finiteIn(input.learningRate, 3e-4f, 1e-5f..1e-2f),
            batchSize = input.batchSize.coerceIn(1, 16),
            warmupSteps = input.warmupSteps.coerceIn(0, 500),
            maxSteps = input.maxSteps.coerceIn(500, 20_000),
            saveEvery = input.saveEvery.coerceIn(25, 1_000),
            evalEvery = input.evalEvery.coerceIn(25, 500),
            patience = input.patience.coerceIn(1, 20),
            gradClip = finiteIn(input.gradClip, 1f, 0.5f..5f),
            weightDecay = finiteIn(input.weightDecay, 0.01f, 0f..0.1f),
            numThreads = input.numThreads.coerceIn(0, cpuLimit)
        )
    }

    fun initialize(context: Context) {
        preferences = context.applicationContext.getSharedPreferences("nexora_settings", Context.MODE_PRIVATE)
        val p = preferences ?: return
        _hp.value = normalizeHyperParams(HyperParams(
            learningRate = p.getFloat("learning_rate", 3e-4f),
            batchSize = p.getInt("batch_size", 4),
            warmupSteps = p.getInt("warmup_steps", 100),
            maxSteps = p.getInt("max_steps", 5000),
            saveEvery = p.getInt("save_every", 200),
            evalEvery = p.getInt("eval_every", 100),
            patience = p.getInt("patience", 5),
            gradClip = p.getFloat("grad_clip", 1f),
            weightDecay = p.getFloat("weight_decay", 0.01f),
            numThreads = p.getInt("cpu_threads", 0)
        ))
    }

    fun updateModel(transform: (ModelConfig) -> ModelConfig) {
        _modelCfg.value = transform(_modelCfg.value)
    }

    fun updateHp(transform: (HyperParams) -> HyperParams) {
        val next = normalizeHyperParams(transform(_hp.value))
        _hp.value = next
        val editor = preferences?.edit() ?: return
        editor.putFloat("learning_rate", next.learningRate)
        editor.putInt("batch_size", next.batchSize)
        editor.putInt("warmup_steps", next.warmupSteps)
        editor.putInt("max_steps", next.maxSteps)
        editor.putInt("save_every", next.saveEvery)
        editor.putInt("eval_every", next.evalEvery)
        editor.putInt("patience", next.patience)
        editor.putFloat("grad_clip", next.gradClip)
        editor.putFloat("weight_decay", next.weightDecay)
        editor.putInt("cpu_threads", next.numThreads)
        editor.apply()
    }
}
