package com.nexora.model

import android.content.Context
import com.nexora.data.DatasetManager
import com.nexora.service.TrainingForegroundService
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.util.concurrent.Executors
import kotlin.math.exp
import kotlin.math.roundToInt

data class TrainingState(
    val isTraining: Boolean = false,
    val isPaused: Boolean = false,
    val currentStep: Int = 0,
    val totalSteps: Int = 0,
    val currentEpoch: Int = 0,
    val totalEpochs: Int = 3,
    val currentLoss: Float? = null,
    val valLoss: Float? = null,
    val currentPerplexity: Float? = null,
    val valPerplexity: Float? = null,
    val currentLr: Float? = null,
    val tokensPerSecond: Float? = null,
    val gradientNorm: Float? = null,
    val stepTimeMs: Long? = null,
    val lossHistory: List<Float> = emptyList(),
    val valLossHistory: List<Float> = emptyList(),
    val totalParams: Long = 0L,
    val trainableParams: Long = 0L,
    val adapterSizeBytes: Long = 0L,
    val isLoRAActive: Boolean = true,
    val trainingMode: TrainingMode = TrainingMode.LORA_ADAPTER,
    val optimizerType: OptimizerType = OptimizerType.ADAMW,
    val lrSchedulerType: LrSchedulerType = LrSchedulerType.COSINE_ANNEALING,
    val lastCheckpoint: CheckpointInfo? = null,
    val isNdkActive: Boolean = true,
    val isVulkanActive: Boolean = false,
    val vulkanStatus: String = "Vulkan Compute Initializing...",
    val statusMessage: String = "Belum ada pelatihan aktif"
)

/**
 * Koordinator Pelatihan Neural Network On-Device Nexora:
 * - Thread-Safe dengan atomic state updates (`_state.update { ... }`).
 * - `@Volatile` isPaused flag.
 * - Sequence-Level Training & Exact Validation Loss over full validation split.
 * - Real AdamW, Lion (2026), and SGD with Momentum.
 * - Gradient Clipping & Early Stopping Detection yang benar-benar menghentikan loop saat patience tercapai.
 * - Safe Checkpointing with Atomic CRC32 & Tokenizer Vocabulary Preservation.
 */
class NexoraTrainer(
    val transformer: NexoraTransformer,
    val tokenizer: NexoraTokenizer,
    val datasetManager: DatasetManager,
    private val scope: CoroutineScope
) {
    private val _state = MutableStateFlow(
        TrainingState(
            totalParams = transformer.calculateTotalParameters(),
            trainableParams = transformer.calculateTrainableParameters(),
            adapterSizeBytes = transformer.calculateAdapterSizeBytes(),
            isLoRAActive = (transformer.trainingMode == TrainingMode.LORA_ADAPTER),
            trainingMode = transformer.trainingMode,
            isNdkActive = NexoraNative.isNdkLoaded,
            isVulkanActive = NexoraNative.isVulkanAvailable(),
            vulkanStatus = NexoraNative.getVulkanStatus()
        )
    )
    val state: StateFlow<TrainingState> = _state.asStateFlow()

    private var trainingJob: kotlinx.coroutines.Job? = null
    @Volatile
    private var isPaused = false

    fun refreshStats() {
        _state.update {
            it.copy(
                totalParams = transformer.calculateTotalParameters(),
                trainableParams = transformer.calculateTrainableParameters(),
                adapterSizeBytes = transformer.calculateAdapterSizeBytes(),
                isLoRAActive = (transformer.trainingMode == TrainingMode.LORA_ADAPTER),
                trainingMode = transformer.trainingMode,
                isNdkActive = NexoraNative.isNdkLoaded,
                isVulkanActive = NexoraNative.isVulkanAvailable(),
                vulkanStatus = NexoraNative.getVulkanStatus()
            )
        }
    }

    fun setTrainingMode(mode: TrainingMode) {
        transformer.trainingMode = mode
        refreshStats()
    }

    /**
     * Memulai proses training sekuensial dengan error boundary lengkap.
     */
    fun startTraining(
        context: Context,
        epochs: Int = 3,
        config2026: TrainingConfig2026 = TrainingConfig2026(),
        threads: Int = Runtime.getRuntime().availableProcessors()
    ) {
        if (!datasetManager.hasDataset) {
            _state.update { it.copy(statusMessage = "Gagal: Belum ada dataset yang dimuat!") }
            return
        }

        if (trainingJob?.isActive == true && isPaused) {
            isPaused = false
            _state.update { it.copy(isTraining = true, isPaused = false, statusMessage = "Melanjutkan pelatihan...") }
            return
        }

        trainingJob?.cancel()
        isPaused = false

        val trainCount = datasetManager.totalTrainSamples
        val totalStepsToRun = trainCount * epochs

        _state.update {
            it.copy(
                isTraining = true,
                isPaused = false,
                currentStep = 0,
                totalSteps = totalStepsToRun,
                currentEpoch = 1,
                totalEpochs = epochs,
                optimizerType = config2026.optimizerType,
                lrSchedulerType = config2026.lrSchedulerType,
                trainingMode = transformer.trainingMode,
                isNdkActive = NexoraNative.isNdkLoaded,
                isVulkanActive = NexoraNative.isVulkanAvailable(),
                vulkanStatus = NexoraNative.getVulkanStatus(),
                statusMessage = "Pelatihan dimulai (${transformer.trainingMode.displayName}, ${config2026.optimizerType.displayName})"
            )
        }

        try {
            TrainingForegroundService.startService(context)
        } catch (e: Exception) {}

        val workerThreadPool = Executors.newFixedThreadPool(threads.coerceIn(1, Runtime.getRuntime().availableProcessors()))
        val customDispatcher = workerThreadPool.asCoroutineDispatcher()

        trainingJob = scope.launch(customDispatcher) {
            var globalStep = 0
            val recentTrainLosses = ArrayList<Float>()
            val recentValLosses = ArrayList<Float>()
            var bestValLoss = Float.MAX_VALUE
            var valLossPatienceCounter = 0
            var stoppedByEarlyStopping = false

            try {
                for (epoch in 1..epochs) {
                    if (!isActive || stoppedByEarlyStopping) break

                    // Acak data latih setiap awal epoch untuk mencegah overfitting urutan
                    datasetManager.shuffleTrainSequences(epoch * 1000L + 42L)

                    for (idx in 0 until trainCount) {
                        if (!isActive || stoppedByEarlyStopping) break
                        while (isPaused && isActive) {
                            delay(200)
                        }

                        val seq = datasetManager.getTrainSequence(idx) ?: continue
                        val stepStartTime = System.nanoTime()

                        // 1. Dynamic Learning Rate
                        val currentLr = config2026.calculateLr(globalStep, totalStepsToRun)

                        // 2. Training Sequence Step (Causal Multi-Head Backprop Eksak)
                        val (stepLoss, gradNorm) = transformer.trainSequenceStep(
                            tokens = seq,
                            learningRate = currentLr,
                            weightDecay = config2026.weightDecay,
                            optimizerType = config2026.optimizerType,
                            maxGradNorm = 1.0f
                        )

                        val elapsedNanos = System.nanoTime() - stepStartTime
                        val stepMs = (elapsedNanos / 1_000_000).coerceAtLeast(1)
                        val tokensThisStep = seq.size.coerceAtLeast(1)
                        val tokensPerSec = (tokensThisStep / (elapsedNanos / 1_000_000_000.0f))

                        globalStep++
                        val ppl = exp(stepLoss.coerceIn(0.01f, 20.0f))

                        recentTrainLosses.add(stepLoss)
                        if (recentTrainLosses.size > 60) recentTrainLosses.removeAt(0)

                        // 3. Evaluasi Validation Loss Eksak per epoch (pada akhir setiap epoch) pada SELURUH split validasi
                        var currentValLoss = _state.value.valLoss
                        var currentValPpl = _state.value.valPerplexity

                        if (datasetManager.totalValSamples > 0 && idx == trainCount - 1) {
                            val valSeqs = datasetManager.getAllValSequences()
                            var sumVal = 0.0f
                            for (vSeq in valSeqs) {
                                sumVal += transformer.evaluateSequenceLoss(vSeq)
                            }
                            val meanVal = if (valSeqs.isNotEmpty()) (sumVal / valSeqs.size) else stepLoss
                            val roundedVal = (meanVal * 1000f).roundToInt() / 1000f
                            currentValLoss = roundedVal
                            currentValPpl = (exp(roundedVal.coerceIn(0.01f, 20.0f)) * 100f).roundToInt() / 100f

                            recentValLosses.add(roundedVal)
                            if (recentValLosses.size > 60) recentValLosses.removeAt(0)

                            // Simpan Checkpoint Best-Val jika loss validasi mencapai rekor terendah
                            if (roundedVal < bestValLoss) {
                                bestValLoss = roundedVal
                                valLossPatienceCounter = 0
                                try {
                                    CheckpointManager.saveCheckpoint(
                                        context = context,
                                        transformer = transformer,
                                        tokenizer = tokenizer,
                                        step = globalStep,
                                        epoch = epoch,
                                        trainLoss = stepLoss,
                                        valLoss = roundedVal,
                                        dataIndex = idx,
                                        currentLr = currentLr,
                                        customName = "Best-Val-Step$globalStep",
                                        isBestVal = true
                                    )
                                } catch (e: Exception) {}
                            } else {
                                valLossPatienceCounter++
                                // Early Stopping: Hentikan pelatihan jika val loss tidak membaik
                                if (valLossPatienceCounter >= config2026.earlyStoppingPatience) {
                                    stoppedByEarlyStopping = true
                                    _state.update {
                                        it.copy(
                                            isTraining = false,
                                            statusMessage = "Pelatihan dihentikan oleh Early Stopping: Val Loss tidak membaik selama ${config2026.earlyStoppingPatience} evaluasi."
                                        )
                                    }
                                    break
                                }
                            }
                        }

                        // 4. Update Foreground Service Notification
                        if (globalStep % 5 == 0 || globalStep == totalStepsToRun) {
                            try {
                                TrainingForegroundService.updateProgress(
                                    context = context,
                                    step = globalStep,
                                    totalSteps = totalStepsToRun,
                                    epoch = epoch,
                                    totalEpochs = epochs,
                                    loss = stepLoss,
                                    valLoss = currentValLoss,
                                    speedTokSec = tokensPerSec
                                )
                            } catch (e: Exception) {}
                        }

                        // 5. Checkpoint berkala
                        var ckptRecorded = _state.value.lastCheckpoint
                        if (globalStep % config2026.autoCheckpointEverySteps == 0 || (epoch == epochs && idx == trainCount - 1)) {
                            try {
                                ckptRecorded = CheckpointManager.saveCheckpoint(
                                    context = context,
                                    transformer = transformer,
                                    tokenizer = tokenizer,
                                    step = globalStep,
                                    epoch = epoch,
                                    trainLoss = stepLoss,
                                    valLoss = currentValLoss,
                                    dataIndex = idx,
                                    currentLr = currentLr,
                                    customName = "Auto-Step$globalStep"
                                )
                            } catch (e: Exception) {}
                        }

                        val overfitWarning = if (valLossPatienceCounter >= 2) " [Val Loss Naik - Early Stop $valLossPatienceCounter/${config2026.earlyStoppingPatience}]" else ""

                        _state.update {
                            it.copy(
                                currentStep = globalStep,
                                currentEpoch = epoch,
                                currentLoss = (stepLoss * 1000f).roundToInt() / 1000f,
                                valLoss = currentValLoss,
                                currentPerplexity = (ppl * 100f).roundToInt() / 100f,
                                valPerplexity = currentValPpl,
                                currentLr = currentLr,
                                tokensPerSecond = (tokensPerSec * 10f).roundToInt() / 10f,
                                gradientNorm = (gradNorm * 1000f).roundToInt() / 1000f, // Perbaikan: pembagian 1000f tepat (bukan 100f)
                                stepTimeMs = stepMs,
                                lossHistory = recentTrainLosses.toList(),
                                valLossHistory = recentValLosses.toList(),
                                lastCheckpoint = ckptRecorded,
                                isVulkanActive = transformer.lastStepUsedVulkan || NexoraNative.isVulkanAvailable(),
                                vulkanStatus = NexoraNative.getVulkanStatus(),
                                statusMessage = "Epoch $epoch/$epochs • Step $globalStep/$totalStepsToRun$overfitWarning"
                            )
                        }

                        if (globalStep % 4 == 0) {
                            delay(2)
                        }
                    }
                }

                if (!stoppedByEarlyStopping) {
                    _state.update {
                        it.copy(
                            isTraining = false,
                            isPaused = false,
                            statusMessage = "Pelatihan selesai! Checkpoint otomatis tersimpan."
                        )
                    }
                }
            } catch (e: Exception) {
                _state.update {
                    it.copy(
                        isTraining = false,
                        isPaused = false,
                        statusMessage = "Error pelatihan: ${e.message}"
                    )
                }
            } finally {
                workerThreadPool.shutdown()
                try {
                    TrainingForegroundService.stopService(context)
                } catch (e: Exception) {}
            }
        }
    }

    fun pauseTraining() {
        if (_state.value.isTraining) {
            isPaused = true
            _state.update { it.copy(isPaused = true, statusMessage = "Pelatihan dijeda") }
        }
    }

    fun stopTraining(context: Context? = null) {
        trainingJob?.cancel()
        trainingJob = null
        isPaused = false
        _state.update {
            it.copy(
                isTraining = false,
                isPaused = false,
                statusMessage = "Pelatihan dihentikan oleh pengguna"
            )
        }
        if (context != null) {
            try {
                TrainingForegroundService.stopService(context)
            } catch (e: Exception) {}
        }
    }

    fun saveManualCheckpoint(context: Context, name: String): CheckpointInfo? {
        return try {
            val ckpt = CheckpointManager.saveCheckpoint(
                context = context,
                transformer = transformer,
                tokenizer = tokenizer,
                step = _state.value.currentStep,
                epoch = _state.value.currentEpoch,
                trainLoss = _state.value.currentLoss,
                valLoss = _state.value.valLoss,
                dataIndex = 0,
                currentLr = _state.value.currentLr ?: 1e-3f,
                customName = name,
                isManual = true
            )
            _state.update { it.copy(lastCheckpoint = ckpt) }
            ckpt
        } catch (e: Exception) {
            null
        }
    }

    fun loadCheckpoint(context: Context, ckpt: CheckpointInfo): Boolean {
        val loaded = CheckpointManager.loadCheckpoint(context, transformer, tokenizer, ckpt)
        if (loaded != null) {
            refreshStats()
            _state.update {
                it.copy(
                    currentStep = loaded.step,
                    currentEpoch = loaded.epoch,
                    currentLr = loaded.currentLr,
                    currentLoss = ckpt.trainLoss,
                    valLoss = ckpt.valLoss,
                    lastCheckpoint = ckpt,
                    statusMessage = "Checkpoint '${ckpt.name}' berhasil dimuat (Step ${loaded.step}, Mode: ${loaded.trainingMode.displayName})"
                )
            }
            return true
        }
        return false
    }
}
