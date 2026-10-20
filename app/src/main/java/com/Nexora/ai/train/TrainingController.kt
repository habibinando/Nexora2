package com.Nexora.ai.train

import android.app.ActivityManager
import android.content.Context
import android.net.Uri
import com.Nexora.ai.data.CheckpointManager
import com.Nexora.ai.data.DatasetImporter
import com.Nexora.ai.data.Tokenizer
import com.Nexora.ai.ml.AdamW
import com.Nexora.ai.ml.CosineWarmupScheduler
import com.Nexora.ai.ml.GPT
import com.Nexora.ai.ml.MathUtil
import com.Nexora.ai.ml.ModelConfig
import com.Nexora.ai.ml.NativeOps
import com.Nexora.ai.ml.crossEntropy
import com.Nexora.ai.ui.AppSettings
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlin.random.Random

enum class Phase { IDLE, IMPORTING, READY, RUNNING, PAUSED, EARLY_STOPPED, FINISHED, ERROR }

data class HyperParams(
    val learningRate: Float = 3e-4f,
    val batchSize: Int = 4,
    val epochs: Int = 3,
    val warmupSteps: Int = 100,
    /** Legacy preference retained for compatibility; the training loop no longer stops by maxSteps. */
    val maxSteps: Int = 5000,
    val saveEvery: Int = 200,
    val evalEvery: Int = 100,
    val patience: Int = 5,
    val gradClip: Float = 1.0f,
    val weightDecay: Float = 0.01f,
    val numThreads: Int = 0 // 0 = all cores
)

data class TrainState(
    val phase: Phase = Phase.IDLE,
    val step: Int = 0,
    /** Number of fully completed epochs, persisted to the checkpoint. */
    val epoch: Int = 0,
    /** 1-based epoch currently being processed; zero means no active epoch. */
    val currentEpoch: Int = 0,
    val batchInEpoch: Int = 0,
    val batchesInEpoch: Int = 0,
    val totalEpochs: Int = 3,
    val loss: Float = Float.NaN,
    val valLoss: Float = Float.NaN,
    val bestValLoss: Float = Float.MAX_VALUE,
    val lr: Float = 0f,
    val gradNorm: Float = 0f,
    val tokensPerSec: Float = 0f,
    val elapsedMs: Long = 0L,
    val etaMs: Long = 0L,
    val lossHistory: List<Float> = emptyList(),
    val message: String = "",
    val datasetName: String = "",
    val numTokens: Int = 0,
    val numExamples: Long = 0,
    val vocabSize: Int = 16000,
    val totalParams: Long = 0,
    val trainableParams: Long = 0,
    val importProgress: Float = 0f
)

/**
 * Singleton training controller — the real training loop lives here.
 * UI and foreground service both observe [state]. No mock, no cloud API.
 */
object TrainingController {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val mutex = Mutex()
    private var job: Job? = null

    private val _state = MutableStateFlow(TrainState())
    val state: StateFlow<TrainState> = _state.asStateFlow()

    @Volatile var model: GPT? = null; private set
    @Volatile var tokenizer: Tokenizer? = null; private set
    private var corpus: IntArray = IntArray(0)
    private lateinit var checkpointManager: CheckpointManager
    private lateinit var appContext: Context

    @Volatile private var paused = false
    @Volatile private var stopped = false
    @Volatile private var resumeCheckpointCompatible = false

    private fun estimateModelParams(cfg: ModelConfig): Long {
        val d = cfg.dModel.toLong()
        val f = cfg.ffnDim.toLong()
        val rank = if (cfg.useLora) cfg.loraRank.toLong() else 0L
        val embedding = cfg.vocabSize.toLong() * d + cfg.seqLen.toLong() * d
        val perLayer = 4L * d * d + 2L * d * f + 9L * d + f + rank * (8L * d + 2L * f)
        return embedding + cfg.numLayers.toLong() * perLayer + 2L * d
    }

    /**
     * Approximate peak memory before constructing a model. Deliberately conservative:
     * parameters, gradients, AdamW moments, logits, per-layer attention/FFN activations,
     * and the in-memory corpus all compete for the app's large heap.
     */
    private fun memoryBudgetMessage(
        context: Context,
        cfg: ModelConfig,
        hp: HyperParams,
        corpusTokens: Int,
        training: Boolean
    ): String? {
        val manager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memoryInfo = ActivityManager.MemoryInfo().also { manager.getMemoryInfo(it) }
        val mib = 1024L * 1024L
        val processHeapBytes = manager.largeMemoryClass.toLong() * mib
        val safeHeap = processHeapBytes * 90L / 100L
        val safeAvailable = memoryInfo.availMem * 65L / 100L
        val budget = minOf(safeHeap, safeAvailable)

        val d = cfg.dModel.toLong()
        val f = cfg.ffnDim.toLong()
        val v = cfg.vocabSize.toLong()
        val s = cfg.seqLen.toLong()
        val b = if (training) hp.batchSize.toLong() else 1L
        val layers = cfg.numLayers.toLong()
        val heads = cfg.numHeads.toLong()
        val params = estimateModelParams(cfg)
        val parameterBytes = params * (if (training) 16L else 8L)
        val logitsBytes = b * s * v * (if (training) 8L else 4L)
        val attentionBytes = b * heads * s * s * layers * (if (training) 8L else 4L)
        val hiddenBytes = b * s * (d + f) * layers * (if (training) 8L else 4L)
        val corpusBytes = corpusTokens.toLong() * 4L
        val scratchBytes = 32L * mib
        val estimated = parameterBytes + logitsBytes + attentionBytes + hiddenBytes + corpusBytes + scratchBytes

        if (estimated <= budget) return null
        return "Perkiraan puncak memori model ${estimated / mib} MiB, melebihi anggaran aman ${budget / mib} MiB pada perangkat ini. " +
            "Kurangi layer, dimensi embedding/FFN, context, batch size, atau ukuran dataset."
    }

    fun init(context: Context) {
        appContext = context.applicationContext
        checkpointManager = CheckpointManager(context.filesDir.resolve("checkpoint"))
        NativeOps.nativeSetThreads(NativeOps.nativeCpuCores())
        // Resume weights if present ("bobot terus bertambah"), but never let a
        // corrupt or oversized checkpoint crash the whole application at startup.
        val cfg = checkpointManager.loadConfig()
        if (cfg != null) {
            AppSettings.updateModel { cfg }
            val memoryWarning = memoryBudgetMessage(context, cfg, AppSettings.hp.value, 0, training = false)
            if (memoryWarning != null) {
                _state.value = _state.value.copy(
                    phase = Phase.ERROR,
                    message = "Checkpoint tidak dimuat: $memoryWarning Hapus checkpoint atau gunakan arsitektur lebih kecil."
                )
                return
            }
            try {
                val restoredModel = GPT(cfg)
                if (checkpointManager.loadTensors(restoredModel)) {
                    model = restoredModel
                    tokenizer = checkpointManager.loadTokenizer()
                    val meta = checkpointManager.loadMeta()
                    _state.value = _state.value.copy(
                        phase = Phase.READY,
                        step = meta?.step ?: 0,
                        epoch = meta?.epoch ?: 0,
                        currentEpoch = (meta?.epoch ?: 0) + 1,
                        totalEpochs = AppSettings.hp.value.epochs,
                        bestValLoss = meta?.bestValLoss ?: Float.MAX_VALUE,
                        lossHistory = meta?.lossHistory ?: emptyList(),
                        vocabSize = cfg.vocabSize,
                        totalParams = restoredModel.numParams,
                        trainableParams = restoredModel.trainableParams,
                        message = "Checkpoint dimuat — bobot tersimpan siap dilanjutkan."
                    )
                } else {
                    _state.value = _state.value.copy(
                        phase = Phase.ERROR,
                        message = "Checkpoint bobot tidak lengkap atau tidak cocok dengan arsitektur."
                    )
                }
            } catch (_: OutOfMemoryError) {
                model = null
                tokenizer = null
                _state.value = _state.value.copy(
                    phase = Phase.ERROR,
                    message = "Checkpoint terlalu besar untuk memori aplikasi. Hapus checkpoint atau pilih arsitektur lebih kecil."
                )
            } catch (e: Exception) {
                model = null
                tokenizer = null
                _state.value = _state.value.copy(
                    phase = Phase.ERROR,
                    message = "Gagal memuat checkpoint: ${e.message ?: e::class.java.simpleName}"
                )
            }
        } else if (checkpointManager.hasCheckpoint()) {
            _state.value = _state.value.copy(
                phase = Phase.ERROR,
                message = "Metadata checkpoint rusak atau tidak didukung. Hapus checkpoint untuk mulai ulang."
            )
        }
    }

    fun importDataset(uri: Uri, modelCfg: ModelConfig) {
        scope.launch {
            mutex.withLock {
                if (_state.value.phase == Phase.RUNNING || _state.value.phase == Phase.PAUSED) {
                    _state.value = _state.value.copy(
                        message = "Impor dataset ditolak saat training aktif atau dijeda. Hentikan training terlebih dahulu."
                    )
                    return@withLock
                }
                _state.value = _state.value.copy(phase = Phase.IMPORTING, message = "Mengimpor dataset…", importProgress = 0f)
                try {
                    val savedCfg = checkpointManager.loadConfig()
                    val savedTokenizer = checkpointManager.loadTokenizer()
                    val canReuseCheckpoint = savedCfg != null && savedTokenizer != null &&
                        savedCfg.vocabSize == modelCfg.vocabSize &&
                        savedTokenizer.vocabSize == savedCfg.vocabSize &&
                        savedCfg == modelCfg.copy(vocabSize = savedCfg.vocabSize)
                    val result = DatasetImporter.import(
                        appContext,
                        uri,
                        modelCfg.vocabSize,
                        tokenizerOverride = if (canReuseCheckpoint) savedTokenizer else null
                    ) { p, msg ->
                        _state.value = _state.value.copy(importProgress = p, message = msg)
                    }
                    val actualCfg = modelCfg.copy(vocabSize = result.tokenizer.vocabSize)
                    val memoryWarning = memoryBudgetMessage(
                        appContext, actualCfg, AppSettings.hp.value, result.tokens.size, training = true
                    )
                    if (memoryWarning != null) throw IllegalArgumentException(memoryWarning)
                    tokenizer = result.tokenizer
                    corpus = result.tokens
                    var m = GPT(actualCfg)
                    val checkpointCompatible = canReuseCheckpoint &&
                        savedCfg == actualCfg && checkpointManager.loadTensors(m)
                    if (canReuseCheckpoint && !checkpointCompatible) m = GPT(actualCfg)
                    resumeCheckpointCompatible = checkpointCompatible
                    model = m
                    val meta = if (checkpointCompatible) checkpointManager.loadMeta() else null
                    _state.value = _state.value.copy(
                        phase = Phase.READY,
                        step = meta?.step ?: 0,
                        epoch = meta?.epoch ?: 0,
                        currentEpoch = (meta?.epoch ?: 0) + 1,
                        batchInEpoch = 0,
                        batchesInEpoch = 0,
                        totalEpochs = AppSettings.hp.value.epochs,
                        loss = Float.NaN,
                        valLoss = Float.NaN,
                        bestValLoss = meta?.bestValLoss ?: Float.MAX_VALUE,
                        elapsedMs = 0L,
                        etaMs = 0L,
                        lossHistory = meta?.lossHistory ?: emptyList(),
                        message = "Dataset siap: ${result.numExamples} contoh, ${result.tokens.size} token" +
                            if (result.truncated) " (dipotong)" else "",
                        numTokens = result.tokens.size,
                        numExamples = result.numExamples,
                        vocabSize = result.tokenizer.vocabSize,
                        totalParams = m.numParams,
                        trainableParams = m.trainableParams,
                        datasetName = uri.lastPathSegment?.substringAfterLast('/') ?: "dataset"
                    )
                } catch (_: OutOfMemoryError) {
                    _state.value = _state.value.copy(
                        phase = Phase.ERROR,
                        message = "RAM tidak cukup untuk mengimpor dataset atau membuat model. Gunakan dataset lebih kecil atau kurangi dimensi/layer."
                    )
                } catch (e: Exception) {
                    _state.value = _state.value.copy(phase = Phase.ERROR, message = "Gagal impor: ${e.message}")
                }
            }
        }
    }

    fun startTraining(hp: HyperParams) {
        if (_state.value.phase in setOf(Phase.RUNNING, Phase.PAUSED, Phase.IMPORTING)) {
            _state.value = _state.value.copy(message = "Training baru tidak dapat dimulai saat training atau impor lain sedang aktif.")
            return
        }
        val m = model ?: run {
            _state.value = _state.value.copy(phase = Phase.ERROR, message = "Belum ada model/dataset. Impor dataset dulu.")
            return
        }
        val minimumTokens = 2L * (m.cfg.seqLen + 1L)
        if (corpus.size.toLong() < minimumTokens) {
            _state.value = _state.value.copy(
                phase = Phase.ERROR,
                message = "Dataset memerlukan minimal $minimumTokens token agar tersedia data training dan validasi."
            )
            return
        }

        NativeOps.resetComputeStats()
        if (hp.numThreads > 0) NativeOps.nativeSetThreads(hp.numThreads)
        else NativeOps.nativeSetThreads(NativeOps.nativeCpuCores())
        stopped = false
        paused = false
        job?.cancel()
        _state.value = _state.value.copy(
            phase = Phase.RUNNING,
            currentEpoch = (_state.value.epoch + 1).coerceAtMost(hp.epochs),
            batchInEpoch = 0,
            batchesInEpoch = 0,
            totalEpochs = hp.epochs,
            message = "Menyiapkan training per epoch…"
        )
        job = scope.launch {
            try {
                trainLoop(m, hp)
            } catch (e: CancellationException) {
                throw e
            } catch (_: OutOfMemoryError) {
                _state.value = _state.value.copy(
                    phase = Phase.ERROR,
                    message = "RAM tidak cukup saat training. Kurangi batch size, context, dimensi model, atau jumlah layer."
                )
            } catch (e: Exception) {
                _state.value = _state.value.copy(
                    phase = Phase.ERROR,
                    message = "Training gagal: ${e.message ?: e::class.java.simpleName}"
                )
            }
        }
    }

    fun pause() {
        paused = true
        _state.value = _state.value.copy(phase = Phase.PAUSED, message = "Jeda diminta; batch aktif diselesaikan terlebih dahulu…")
    }

    fun resume() {
        paused = false
        _state.value = _state.value.copy(phase = Phase.RUNNING, message = "Training dilanjutkan.")
    }

    fun stop() {
        stopped = true
        paused = false
        _state.value = _state.value.copy(message = "Stop diminta; batch aktif diselesaikan lalu checkpoint disimpan…")
    }

    /**
     * One epoch is one pass through shuffled fixed-length windows from the training
     * portion of the token corpus. The final partial batch is processed instead of
     * silently dropping examples. Loss/gradient/optimizer work remains the real model loop.
     */
    private suspend fun trainLoop(m: GPT, hp: HyperParams) {
        val tok = tokenizer ?: throw IllegalStateException("Tokenizer belum tersedia.")
        val S = m.cfg.seqLen
        val B = hp.batchSize.coerceIn(1, 16)
        val vocab = m.cfg.vocabSize

        val valSize = maxOf((corpus.size * 0.05).toInt(), S + 1)
            .coerceAtMost(corpus.size - (S + 1))
        val trainEnd = corpus.size - valSize
        if (trainEnd < S + 1 || valSize < S + 1) {
            _state.value = _state.value.copy(phase = Phase.ERROR, message = "Dataset tidak cukup untuk memisahkan training dan validasi.")
            return
        }

        // A window contains S input tokens and the shifted S target tokens.
        // Stride S gives a bounded, repeatable pass instead of sampling forever with replacement.
        val windowCount = ((trainEnd - 1) / S).coerceAtLeast(0)
        if (windowCount <= 0) {
            _state.value = _state.value.copy(phase = Phase.ERROR, message = "Dataset training tidak cukup untuk satu sequence.")
            return
        }
        val stepsPerEpoch = (windowCount + B - 1) / B

        val pairs = m.trainablePairs()
        val optimizer = AdamW(pairs, weightDecay = hp.weightDecay)
        var metadata = if (resumeCheckpointCompatible) checkpointManager.loadMeta() else null
        if (resumeCheckpointCompatible) {
            val savedOptimizer = checkpointManager.loadOptimizerState(optimizer)
            if (savedOptimizer != null) {
                optimizer.loadState(savedOptimizer.second, savedOptimizer.first)
            } else {
                // A weights-only checkpoint is still usable, but stale epoch/optimizer
                // counters must not make the app skip the requested training.
                resumeCheckpointCompatible = false
                metadata = null
            }
        }

        var completedEpochs = if (resumeCheckpointCompatible) {
            metadata?.epoch ?: _state.value.epoch
        } else 0
        completedEpochs = completedEpochs.coerceIn(0, hp.epochs)
        if (completedEpochs >= hp.epochs) {
            _state.value = _state.value.copy(
                phase = Phase.FINISHED,
                epoch = completedEpochs,
                currentEpoch = completedEpochs,
                message = "Target $completedEpochs epoch sudah tercapai. Naikkan pengaturan Epoch untuk training tambahan."
            )
            return
        }

        var step = optimizer.step
        var scheduleStep = 0
        val remainingEpochs = hp.epochs - completedEpochs
        val plannedStepsLong = step.toLong() + remainingEpochs.toLong() * stepsPerEpoch.toLong()
        val plannedTotalSteps = plannedStepsLong.coerceAtMost(Int.MAX_VALUE.toLong()).toInt().coerceAtLeast(step + 1)
        // Small datasets can have fewer than the requested warmup steps. Cap warmup
        // to 10% of the remaining update count so LR does not stay near zero for all epochs.
        val effectiveWarmup = minOf(hp.warmupSteps, ((plannedTotalSteps - step).coerceAtLeast(0) / 10))
        val scheduler = CosineWarmupScheduler(
            maxLR = hp.learningRate,
            warmupSteps = effectiveWarmup,
            totalSteps = plannedTotalSteps
        )
        var bestVal = metadata?.bestValLoss ?: Float.MAX_VALUE
        var patienceCount = metadata?.patienceCount ?: 0
        val history = (metadata?.lossHistory ?: _state.value.lossHistory).toMutableList()
        var tokensSeen = metadata?.tokensSeen ?: 0L
        val runStartMs = System.currentTimeMillis() - _state.value.elapsedMs

        fun saveLastValidCheckpoint(completed: Int) {
            try {
                saveCheckpoint(m, optimizer, tok, step, completed, bestVal, patienceCount, history, tokensSeen)
            } catch (_: Exception) {
                // Keep the previous atomically-published checkpoint if saving fails.
            }
        }

        for (epochIndex in completedEpochs until hp.epochs) {
            if (stopped) break
            val starts = IntArray(windowCount) { it * S }
            val random = Random(2026 + epochIndex)
            for (i in starts.lastIndex downTo 1) {
                val j = random.nextInt(i + 1)
                val tmp = starts[i]
                starts[i] = starts[j]
                starts[j] = tmp
            }

            for (batchIndex in 0 until stepsPerEpoch) {
                while (paused && !stopped) {
                    if (_state.value.phase != Phase.PAUSED) {
                        _state.value = _state.value.copy(phase = Phase.PAUSED, message = "Training dijeda.")
                    }
                    delay(200)
                }
                if (stopped) break

                val begin = batchIndex * B
                val actualBatch = minOf(B, starts.size - begin)
                if (actualBatch <= 0) break
                val T = actualBatch * S
                val input = IntArray(T)
                val target = IntArray(T)
                for (b in 0 until actualBatch) {
                    val start = starts[begin + b]
                    for (s in 0 until S) {
                        input[b * S + s] = corpus[start + s]
                        target[b * S + s] = corpus[start + s + 1]
                    }
                }
                val gradLogits = FloatArray(T * vocab)
                val isLastBatch = batchIndex == stepsPerEpoch - 1
                _state.value = _state.value.copy(
                    phase = Phase.RUNNING,
                    epoch = completedEpochs,
                    currentEpoch = epochIndex + 1,
                    batchInEpoch = batchIndex + 1,
                    batchesInEpoch = stepsPerEpoch,
                    step = step,
                    message = "Epoch ${epochIndex + 1}/${hp.epochs} · batch ${batchIndex + 1}/$stepsPerEpoch — forward pass…"
                )

                synchronized(m) {
                    val stepStart = System.currentTimeMillis()
                    m.zeroGrad()
                    val logits = m.forward(input, actualBatch, S, training = true)
                    val loss = crossEntropy(logits, target, vocab, gradLogits)
                    if (!loss.isFinite()) {
                        _state.value = _state.value.copy(
                            phase = Phase.ERROR,
                            message = "Loss tidak valid pada epoch ${epochIndex + 1}, batch ${batchIndex + 1}; optimizer dihentikan."
                        )
                        saveLastValidCheckpoint(completedEpochs)
                        return
                    }

                    _state.value = _state.value.copy(
                        message = "Epoch ${epochIndex + 1}/${hp.epochs} · batch ${batchIndex + 1}/$stepsPerEpoch — backpropagation…"
                    )
                    m.backward(gradLogits)

                    val grads = pairs.map { it.second }
                    val gNorm = MathUtil.gradNorm(*grads.toTypedArray())
                    if (!gNorm.isFinite() || gNorm < 0f) {
                        _state.value = _state.value.copy(
                            phase = Phase.ERROR,
                            message = "Gradien tidak valid pada epoch ${epochIndex + 1}; optimizer tidak dijalankan."
                        )
                        saveLastValidCheckpoint(completedEpochs)
                        return
                    }
                    if (gNorm > hp.gradClip) {
                        val scale = hp.gradClip / gNorm
                        for (g in grads) MathUtil.scale(g, scale)
                    }

                    _state.value = _state.value.copy(
                        message = "Epoch ${epochIndex + 1}/${hp.epochs} · batch ${batchIndex + 1}/$stepsPerEpoch — AdamW memperbarui bobot…"
                    )
                    // Scheduler progress is local to this requested training run; the
                    // optimizer keeps its own global step for AdamW bias correction.
                    val lr = scheduler.lrAt(scheduleStep)
                    optimizer.step(lr)
                    scheduleStep++
                    step = optimizer.step
                    tokensSeen += T.toLong()

                    val stepMs = (System.currentTimeMillis() - stepStart).coerceAtLeast(1)
                    val tps = T * 1000f / stepMs
                    val elapsed = System.currentTimeMillis() - runStartMs
                    val batchesAfterCurrent = (stepsPerEpoch - batchIndex - 1).coerceAtLeast(0) +
                        ((hp.epochs - epochIndex - 1).coerceAtLeast(0) * stepsPerEpoch)
                    val eta = if (tps.isFinite() && tps > 0f) {
                        (batchesAfterCurrent.toDouble() * T.toDouble() * 1000.0 / tps.toDouble())
                            .coerceIn(0.0, Long.MAX_VALUE.toDouble()).toLong()
                    } else 0L

                    history += loss
                    if (history.size > 1000) history.removeAt(0)

                    var valLoss = _state.value.valLoss
                    var shouldStopEarly = false
                    if (step % hp.evalEvery == 0 || isLastBatch) {
                        _state.value = _state.value.copy(
                            message = "Epoch ${epochIndex + 1}/${hp.epochs} · evaluasi validation loss pada step $step…"
                        )
                        valLoss = evaluate(m, vocab, trainEnd, B, S)
                        if (!valLoss.isFinite()) {
                            _state.value = _state.value.copy(phase = Phase.ERROR, message = "Validation loss tidak valid; training dihentikan.")
                            saveLastValidCheckpoint(completedEpochs)
                            return
                        }
                        if (valLoss < bestVal - 1e-4f) {
                            bestVal = valLoss
                            patienceCount = 0
                        } else {
                            patienceCount++
                            shouldStopEarly = patienceCount >= hp.patience
                        }
                    }

                    val activePhase = if (paused) Phase.PAUSED else Phase.RUNNING
                    val activeMessage = if (paused) {
                        "Epoch ${epochIndex + 1}/${hp.epochs}: batch ${batchIndex + 1}/$stepsPerEpoch selesai; training dijeda."
                    } else {
                        "Training aktif — epoch ${epochIndex + 1}/${hp.epochs}, batch ${batchIndex + 1}/$stepsPerEpoch."
                    }
                    _state.value = buildState(
                        step = step,
                        completedEpochs = completedEpochs,
                        currentEpoch = epochIndex + 1,
                        batchIndex = batchIndex + 1,
                        batchesPerEpoch = stepsPerEpoch,
                        loss = loss,
                        valLoss = valLoss,
                        best = bestVal,
                        lr = lr,
                        gNorm = gNorm,
                        tps = tps,
                        elapsed = elapsed,
                        eta = eta,
                        hist = history,
                        phase = if (shouldStopEarly) Phase.EARLY_STOPPED else activePhase,
                        msg = if (shouldStopEarly) "Early stop: validation loss tidak membaik setelah $patienceCount evaluasi." else activeMessage
                    )

                    if (step % hp.saveEvery == 0) {
                        _state.value = _state.value.copy(message = "Menyimpan checkpoint di step $step…")
                        saveCheckpoint(m, optimizer, tok, step, completedEpochs, bestVal, patienceCount, history, tokensSeen)
                    }
                    if (shouldStopEarly) {
                        saveCheckpoint(m, optimizer, tok, step, completedEpochs, bestVal, patienceCount, history, tokensSeen)
                        return
                    }
                }
            }

            if (stopped) break
            completedEpochs = epochIndex + 1
            _state.value = _state.value.copy(
                phase = Phase.RUNNING,
                epoch = completedEpochs,
                currentEpoch = completedEpochs,
                batchInEpoch = stepsPerEpoch,
                batchesInEpoch = stepsPerEpoch,
                step = step,
                message = "Epoch $completedEpochs/${hp.epochs} selesai. Menyimpan checkpoint bobot dan optimizer…"
            )
            // Always publish a complete-epoch checkpoint so a restart never mistakes
            // the optimizer step count for the completed epoch count.
            saveCheckpoint(m, optimizer, tok, step, completedEpochs, bestVal, patienceCount, history, tokensSeen)
        }

        _state.value = _state.value.copy(message = "Menyimpan checkpoint akhir…")
        saveCheckpoint(m, optimizer, tok, step, completedEpochs, bestVal, patienceCount, history, tokensSeen)
        _state.value = _state.value.copy(
            phase = if (stopped) Phase.READY else Phase.FINISHED,
            epoch = completedEpochs,
            currentEpoch = completedEpochs,
            batchInEpoch = if (completedEpochs > 0) stepsPerEpoch else 0,
            batchesInEpoch = stepsPerEpoch,
            step = step,
            message = if (stopped) {
                "Training dihentikan pengguna. Checkpoint tersimpan pada epoch $completedEpochs."
            } else {
                "Training selesai: $completedEpochs/${hp.epochs} epoch. Checkpoint final tersimpan."
            }
        )
    }

    private fun buildState(
        step: Int,
        completedEpochs: Int,
        currentEpoch: Int,
        batchIndex: Int,
        batchesPerEpoch: Int,
        loss: Float,
        valLoss: Float,
        best: Float,
        lr: Float,
        gNorm: Float,
        tps: Float,
        elapsed: Long,
        eta: Long,
        hist: List<Float>,
        phase: Phase,
        msg: String
    ) = _state.value.copy(
        phase = phase,
        step = step,
        epoch = completedEpochs,
        currentEpoch = currentEpoch,
        batchInEpoch = batchIndex,
        batchesInEpoch = batchesPerEpoch,
        loss = loss,
        valLoss = valLoss,
        bestValLoss = best,
        lr = lr,
        gradNorm = gNorm,
        tokensPerSec = tps,
        elapsedMs = elapsed,
        etaMs = eta,
        lossHistory = hist.toList(),
        message = msg
    )

    /** Evaluate a bounded number of deterministic windows from the held-out corpus tail. */
    private fun evaluate(m: GPT, vocab: Int, trainEnd: Int, B: Int, S: Int): Float {
        val valWindowCount = ((corpus.size - trainEnd - 1) / S).coerceAtLeast(0)
        if (valWindowCount <= 0) return Float.NaN
        val maxWindows = minOf(valWindowCount, B * 4)
        var totalLoss = 0.0
        var totalTokens = 0L
        var offset = 0
        while (offset < maxWindows) {
            val actualBatch = minOf(B, maxWindows - offset)
            val T = actualBatch * S
            val input = IntArray(T)
            val target = IntArray(T)
            for (b in 0 until actualBatch) {
                val start = trainEnd + (offset + b) * S
                for (s in 0 until S) {
                    input[b * S + s] = corpus[start + s]
                    target[b * S + s] = corpus[start + s + 1]
                }
            }
            val logits = m.forward(input, actualBatch, S, training = false)
            val gradLogits = FloatArray(T * vocab)
            val loss = crossEntropy(logits, target, vocab, gradLogits)
            if (!loss.isFinite()) return Float.NaN
            totalLoss += loss.toDouble() * T
            totalTokens += T
            offset += actualBatch
        }
        return if (totalTokens > 0L) (totalLoss / totalTokens).toFloat() else Float.NaN
    }

    private fun saveCheckpoint(
        m: GPT,
        opt: AdamW,
        tok: Tokenizer,
        step: Int,
        completedEpochs: Int,
        best: Float,
        patience: Int,
        hist: List<Float>,
        tokensSeen: Long
    ) {
        checkpointManager.save(
            m, opt, tok,
            CheckpointManager.Meta(step, completedEpochs, best, patience, hist.takeLast(500), tokensSeen)
        )
        resumeCheckpointCompatible = true
    }

    fun clearCheckpoint() {
        if (_state.value.phase in setOf(Phase.RUNNING, Phase.PAUSED, Phase.IMPORTING)) {
            _state.value = _state.value.copy(
                message = "Checkpoint tidak dapat dihapus saat training atau impor dataset masih aktif."
            )
            return
        }
        checkpointManager.clear()
        resumeCheckpointCompatible = false
        model = null
        tokenizer = null
        corpus = IntArray(0)
        _state.value = TrainState()
    }
}
