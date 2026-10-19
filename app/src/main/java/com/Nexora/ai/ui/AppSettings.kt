package com.Nexora.ai.ui

import com.Nexora.ai.ml.ModelConfig
import com.Nexora.ai.train.HyperParams
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/** Global, in-memory app settings (no cloud). Persisted only via checkpoint. */
object AppSettings {
    private val _modelCfg = MutableStateFlow(
        ModelConfig(vocabSize = 16000, seqLen = 128, dModel = 128, ffnDim = 512,
            numHeads = 8, numLayers = 32, dropout = 0.1f, useLora = true, loraRank = 8, loraAlpha = 16)
    )
    val modelCfg: StateFlow<ModelConfig> = _modelCfg.asStateFlow()
    fun updateModel(transform: (ModelConfig) -> ModelConfig) { _modelCfg.value = transform(_modelCfg.value) }

    private val _hp = MutableStateFlow(HyperParams())
    val hp: StateFlow<HyperParams> = _hp.asStateFlow()
    fun updateHp(transform: (HyperParams) -> HyperParams) { _hp.value = transform(_hp.value) }
}
