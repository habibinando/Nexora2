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
            numHeads = 8, numLayers = 32, dropout = 0.1f, useLora = true, loraRank = 8, loraAlpha = 16)
    )
    val modelCfg: StateFlow<ModelConfig> = _modelCfg.asStateFlow()

    private val _hp = MutableStateFlow(HyperParams())
    val hp: StateFlow<HyperParams> = _hp.asStateFlow()

    fun initialize(context: Context) {
        preferences = context.applicationContext.getSharedPreferences("nexora_settings", Context.MODE_PRIVATE)
        val p = preferences ?: return
        _hp.value = HyperParams(
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
        )
    }

    fun updateModel(transform: (ModelConfig) -> ModelConfig) {
        _modelCfg.value = transform(_modelCfg.value)
    }

    fun updateHp(transform: (HyperParams) -> HyperParams) {
        val next = transform(_hp.value)
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
