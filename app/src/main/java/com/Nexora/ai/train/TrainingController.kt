package com.Nexora.ai.train

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
    val warmupSteps: Int = 100,
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
    val epoch: Int = 0,
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

    fun init(context: Context) {
        appContext = context.applicationContext
        checkpointManager = CheckpointManager(context.filesDir.resolve("checkpoint"))
        NativeOps.nativeSetThreads(NativeOps.nativeCpuCores())
        // Resume weights if present ("bobot terus bertambah")
        val cfg = checkpointManager.loadConfig()
        if (cfg != null) {
            AppSettings.updateModel { cfg }
            val m = GPT(cfg)
            if (checkpointManager.loadTensors(m)) {
                model = m
                tokenizer = checkpointManager.loadTokenizer()
                val meta = checkpointManager.loadMeta()
                _state.value = _state.value.copy(
                    phase = Phase.READY,
                    step = meta?.step ?: 0,
                    epoch = meta?.epoch ?: 0,
                    bestValLoss = meta?.bestValLoss ?: Float.MAX_VALUE,
                    lossHistory = meta?.lossHistory ?: emptyList(),
                    vocabSize = cfg.vocabSize,
                    totalParams = m.numParams,
                    trainableParams = m.trainableParams,
                    message = "Checkpoint dimuat — bobot tersimpan siap dilanjutkan."
                )
            }
        }
    }

    fun importDataset(uri: Uri, modelCfg: ModelConfig) {
        scope.launch {
            mutex.withLock {
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
                    tokenizer = result.tokenizer
                    corpus = result.tokens
                    val actualCfg = modelCfg.copy(vocabSize = result.tokenizer.vocabSize)
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
                } catch (e: Exception) {
                    _state.value = _state.value.copy(phase = Phase.ERROR, message = "Gagal impor: ${e.message}")
                }
            }
        }
    }

    fun startTraining(hp: HyperParams) {
        val m = model ?: run { _state.value = _state.value.copy(phase = Phase.ERROR, message = "Belum ada model/dataset. Impor dataset dulu."); return }
        val span = m.cfg.seqLen + 1
        val requiredTokens = span.toLong() * (hp.batchSize + 1L)
        if (corpus.size.toLong() < requiredTokens) {
            _state.value = _state.value.copy(
                phase = Phase.ERROR,
                message = "Dataset terlalu kecil untuk sequence ${m.cfg.seqLen} dan batch ${hp.batchSize}; diperlukan minimal $requiredTokens token."
            )
            return
        }
        // Scope backend counters to this training run, not an earlier benchmark.
        NativeOps.resetComputeStats()
        if (hp.numThreads > 0) NativeOps.nativeSetThreads(hp.numThreads) else NativeOps.nativeSetThreads(NativeOps.nativeCpuCores())
        stopped = false
        paused = false
        job?.cancel()
        _state.value = _state.value.copy(phase = Phase.RUNNING, message = "Menyiapkan training lokal…")
        job = scope.launch {
            try {
                trainLoop(m, hp)
            } catch (e: CancellationException) {
                throw e
            } catch (e: OutOfMemoryError) {
                _state.value = _state.value.copy(
                    phase = Phase.ERROR,
                    message = "Memori tidak cukup. Kurangi batch size, context, dimensi model, atau jumlah layer."
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
        _state.value = _state.value.copy(phase = Phase.PAUSED, message = "Jeda diminta; step aktif diselesaikan terlebih dahulu…")
    }
    fun resume() {
        paused = false
        _state.value = _state.value.copy(phase = Phase.RUNNING, message = "Training dilanjutkan.")
    }
    fun stop() {
        stopped = true
        paused = false
        _state.value = _state.value.copy(message = "Stop diminta; menyelesaikan step aktif lalu menyimpan checkpoint…")
    }

    private suspend fun trainLoop(m: GPT, hp: HyperParams) {
        _state.value = _state.value.copy(phase = Phase.RUNNING, message = "Training dimulai…")
        val tok = tokenizer!!
        val S = m.cfg.seqLen
        val B = hp.batchSize
        val vocab = m.cfg.vocabSize
        val T = B * S
        val rng = Random(2026)

        // Train / validation split (95/5)
        val span = S + 1
        val valSize = maxOf((corpus.size * 0.05).toInt(), B * span)
        val trainEnd = corpus.size - valSize
        if (trainEnd < span || valSize < B * span) {
            _state.value = _state.value.copy(phase = Phase.ERROR, message = "Dataset tidak cukup untuk train/validation split.")
            return
        }

        val pairs = m.trainablePairs()
        val optimizer = AdamW(pairs, weightDecay = hp.weightDecay)
        // Resume optimizer state if available
        if (resumeCheckpointCompatible) {
            checkpointManager.loadOptimizerState(optimizer)?.let { (step, state) ->
                optimizer.loadState(state, step)
            }
        }
        val startStep = optimizer.step
        val scheduler = CosineWarmupScheduler(hp.learningRate, warmupSteps = hp.warmupSteps, totalSteps = hp.maxSteps)

        var bestVal = if (resumeCheckpointCompatible) _state.value.bestValLoss else Float.MAX_VALUE
        var patienceCount = if (resumeCheckpointCompatible) checkpointManager.loadMeta()?.patienceCount ?: 0 else 0
        val history = if (resumeCheckpointCompatible) _state.value.lossHistory.toMutableList() else mutableListOf()
        val gradLogits = FloatArray(T * vocab)
        val startMs = System.currentTimeMillis() - _state.value.elapsedMs
        var tokensSeen = if (resumeCheckpointCompatible) checkpointManager.loadMeta()?.tokensSeen ?: 0L else 0L
        var step = startStep

        while (step < hp.maxSteps && !stopped) {
            while (paused && !stopped) delay(200)
            if (stopped) break

            val stepStart = System.currentTimeMillis()
            // Sample random batch windows
            val input = IntArray(T); val target = IntArray(T)
            for (b in 0 until B) {
                val start = rng.nextInt(trainEnd - span + 1)
                for (s in 0 until S) {
                    input[b * S + s] = corpus[start + s]
                    target[b * S + s] = corpus[start + s + 1]
                }
            }
            _state.value = _state.value.copy(message = "Forward pass — step ${step + 1}/${hp.maxSteps}. Menghitung prediksi model…")
            m.zeroGrad()
            val logits = m.forward(input, B, S, training = true)
            val loss = crossEntropy(logits, target, vocab, gradLogits)
            if (!loss.isFinite()) {
                _state.value = _state.value.copy(phase = Phase.ERROR, message = "Loss bukan angka valid; training dihentikan.")
                return
            }

            _state.value = _state.value.copy(message = "Backpropagation — step ${step + 1}/${hp.maxSteps}. Menghitung gradien…")
            m.backward(gradLogits)

            // Gradient clipping
            val grads = pairs.map { it.second }
            val gNorm = MathUtil.gradNorm(*grads.toTypedArray())
            if (!gNorm.isFinite() || gNorm < 0f) {
                _state.value = _state.value.copy(
                    phase = Phase.ERROR,
                    message = "Gradien tidak finite; optimizer tidak dijalankan agar bobot tidak rusak."
                )
                return
            }
            if (gNorm > hp.gradClip) {
                val sc = hp.gradClip / gNorm
                for (g in grads) MathUtil.scale(g, sc)
            }
            _state.value = _state.value.copy(message = "Optimizer — menerapkan perubahan bobot step ${step + 1}…")
            val lr = scheduler.lrAt(step)
            optimizer.step(lr)
            step = optimizer.step
            tokensSeen += T.toLong()

            val stepMs = (System.currentTimeMillis() - stepStart).coerceAtLeast(1)
            val tps = T * 1000f / stepMs
            val elapsed = System.currentTimeMillis() - startMs
            val stepsLeft = hp.maxSteps - step
            val eta = if (tps > 0) (stepsLeft * T * 1000L / tps.toLong()).coerceAtLeast(0L) else 0L

            history += loss
            if (history.size > 1000) history.removeAt(0)

            var valLoss = _state.value.valLoss
            if (step % hp.evalEvery == 0) {
                _state.value = _state.value.copy(message = "Evaluasi validation loss — step $step…")
                valLoss = evaluate(m, vocab, trainEnd, valSize, B, S, gradLogits)
                if (valLoss < bestVal - 1e-4f) { bestVal = valLoss; patienceCount = 0 }
                else {
                    patienceCount++
                    if (patienceCount >= hp.patience) {
                        _state.value = buildState(step, loss, valLoss, bestVal, lr, gNorm, tps, elapsed, eta, history, Phase.EARLY_STOPPED, "Early stop — val loss tidak membaik setelah $patienceCount evaluasi.")
                        saveCheckpoint(m, optimizer, tok, step, bestVal, patienceCount, history, tokensSeen)
                        return
                    }
                }
            }

            val activePhase = if (paused) Phase.PAUSED else Phase.RUNNING
            val activeMessage = if (paused) "Step $step selesai; training dijeda." else "Training aktif — step $step/${hp.maxSteps}"
            _state.value = buildState(step, loss, valLoss, bestVal, lr, gNorm, tps, elapsed, eta, history, activePhase, activeMessage)

            if (step % hp.saveEvery == 0) {
                _state.value = _state.value.copy(message = "Menyimpan checkpoint — step $step…")
                saveCheckpoint(m, optimizer, tok, step, bestVal, patienceCount, history, tokensSeen)
                _state.value = _state.value.copy(message = "Checkpoint disimpan (step $step). Bobot terus bertambah.")
            }
        }
        _state.value = _state.value.copy(message = "Menyimpan checkpoint akhir…")
        saveCheckpoint(m, optimizer, tok, step, bestVal, patienceCount, history, tokensSeen)
        _state.value = _state.value.copy(
            phase = if (stopped) Phase.READY else Phase.FINISHED,
            message = if (stopped) "Dihentikan pengguna. Checkpoint disimpan." else "Selesai! Checkpoint final disimpan."
        )
    }

    private fun buildState(step: Int, loss: Float, valLoss: Float, best: Float, lr: Float, gNorm: Float, tps: Float, elapsed: Long, eta: Long, hist: List<Float>, phase: Phase, msg: String) =
        _state.value.copy(phase = phase, step = step, loss = loss, valLoss = valLoss, bestValLoss = best, lr = lr, gradNorm = gNorm, tokensPerSec = tps, elapsedMs = elapsed, etaMs = eta, lossHistory = hist.toList(), message = msg)

    private fun evaluate(m: GPT, vocab: Int, trainEnd: Int, valSize: Int, B: Int, S: Int, gradLogits: FloatArray): Float {
        val T = B * S
        var total = 0f; var n = 0
        val input = IntArray(T); val target = IntArray(T)
        val batches = (valSize / (B * (S + 1))).coerceIn(1, 4)
        for (it in 0 until batches) {
            val base = trainEnd + it * B * (S + 1)
            for (b in 0 until B) {
                val st = base + b * (S + 1)
                if (st + S >= corpus.size) break
                for (s in 0 until S) { input[b * S + s] = corpus[st + s]; target[b * S + s] = corpus[st + s + 1] }
            }
            val logits = m.forward(input, B, S, training = false)
            total += crossEntropy(logits, target, vocab, gradLogits); n++
        }
        return if (n > 0) total / n else 0f
    }

    private fun saveCheckpoint(m: GPT, opt: AdamW, tok: Tokenizer, step: Int, best: Float, patience: Int, hist: List<Float>, tokensSeen: Long) {
        checkpointManager.save(m, opt, tok, CheckpointManager.Meta(step, step, best, patience, hist.takeLast(500), tokensSeen))
        resumeCheckpointCompatible = true
    }

    fun clearCheckpoint() {
        checkpointManager.clear()
        resumeCheckpointCompatible = false
        model = null
        tokenizer = null
        corpus = IntArray(0)
        _state.value = TrainState()
    }
}
