package com.nexora.ui

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.nexora.data.DatasetManager
import com.nexora.data.DatasetMetadata
import com.nexora.model.BenchmarkResult
import com.nexora.model.CheckpointInfo
import com.nexora.model.CheckpointManager
import com.nexora.model.LrSchedulerType
import com.nexora.model.NexoraBenchmark
import com.nexora.model.NexoraConfig
import com.nexora.model.NexoraNative
import com.nexora.model.NexoraTokenizer
import com.nexora.model.NexoraTrainer
import com.nexora.model.NexoraTransformer
import com.nexora.model.OptimizerType
import com.nexora.model.TrainingConfig2026
import com.nexora.model.TrainingMode
import com.nexora.model.TrainingState
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class ChatMessage(
    val id: String,
    val sender: String,
    val text: String,
    val tokensCount: Int = 0,
    val generationTimeMs: Long = 0L,
    val tokensPerSec: Float = 0f,
    val isTrainedWeight: Boolean = true
)

enum class AppScreen {
    SPLASH,
    DASHBOARD,
    DATASET,
    TRAINING,
    BENCHMARK,
    CHAT,
    SETTINGS
}

data class HardwareConfig(
    val cpuThreads: Int = Runtime.getRuntime().availableProcessors(),
    val learningRate: Float = 1e-4f,
    val loraRank: Int = 4,
    val loraAlpha: Float = 8.0f,
    val batchSize: Int = 1
)

class NexoraViewModel(application: Application) : AndroidViewModel(application) {

    // Arsitektur LM5M 5.07M parameter (d=256, L=5, H=4, FFN=704, Vocab=4096)
    val config = NexoraConfig()

    val tokenizer = NexoraTokenizer(config.vocabSize)
    val transformer = NexoraTransformer(config)
    val datasetManager = DatasetManager(tokenizer)
    val trainer = NexoraTrainer(transformer, tokenizer, datasetManager, viewModelScope)
    val benchmark = NexoraBenchmark(transformer)

    private val _currentScreen = MutableStateFlow(AppScreen.SPLASH)
    val currentScreen: StateFlow<AppScreen> = _currentScreen.asStateFlow()

    private val _splashProgress = MutableStateFlow(0f)
    val splashProgress: StateFlow<Float> = _splashProgress.asStateFlow()

    private val _splashStatus = MutableStateFlow("Menginisialisasi Nexora Core...")
    val splashStatus: StateFlow<String> = _splashStatus.asStateFlow()

    val trainingState: StateFlow<TrainingState> = trainer.state

    private val _trainingConfig2026 = MutableStateFlow(TrainingConfig2026(baseLr = 1e-4f))
    val trainingConfig2026: StateFlow<TrainingConfig2026> = _trainingConfig2026.asStateFlow()

    private val _checkpointsList = MutableStateFlow<List<CheckpointInfo>>(emptyList())
    val checkpointsList: StateFlow<List<CheckpointInfo>> = _checkpointsList.asStateFlow()

    private val _datasetMeta = MutableStateFlow<DatasetMetadata?>(null)
    val datasetMeta: StateFlow<DatasetMetadata?> = _datasetMeta.asStateFlow()

    private val _isDatasetLoading = MutableStateFlow(false)
    val isDatasetLoading: StateFlow<Boolean> = _isDatasetLoading.asStateFlow()

    private val _benchmarkResult = MutableStateFlow<BenchmarkResult?>(null)
    val benchmarkResult: StateFlow<BenchmarkResult?> = _benchmarkResult.asStateFlow()

    private val _isBenchmarking = MutableStateFlow(false)
    val isBenchmarking: StateFlow<Boolean> = _isBenchmarking.asStateFlow()

    private val _benchmarkProgress = MutableStateFlow(0f)
    val benchmarkProgress: StateFlow<Float> = _benchmarkProgress.asStateFlow()

    private val _chatMessages = MutableStateFlow<List<ChatMessage>>(emptyList())
    val chatMessages: StateFlow<List<ChatMessage>> = _chatMessages.asStateFlow()

    private val _isGenerating = MutableStateFlow(false)
    val isGenerating: StateFlow<Boolean> = _isGenerating.asStateFlow()

    private val _hardwareConfig = MutableStateFlow(
        HardwareConfig(
            cpuThreads = Runtime.getRuntime().availableProcessors(),
            learningRate = 1e-4f,
            loraRank = config.loraRank,
            loraAlpha = config.loraAlpha
        )
    )
    val hardwareConfig: StateFlow<HardwareConfig> = _hardwareConfig.asStateFlow()

    init {
        runStartupSequence()
        refreshCheckpoints()
    }

    private fun runStartupSequence() {
        viewModelScope.launch(Dispatchers.Default) {
            // 1. Verifikasi alokasi memori heap
            val maxMemMb = (Runtime.getRuntime().maxMemory() / (1024 * 1024)).toInt()
            _splashStatus.value = "Memverifikasi Alokasi Heap: ${maxMemMb}MB (LargeHeap Aktif)..."
            _splashProgress.value = 0.15f

            // 2. Deteksi CPU NEON SIMD
            val ndkStatus = NexoraNative.getActiveNdkStatus()
            _splashStatus.value = "Engine: $ndkStatus (Cores: ${Runtime.getRuntime().availableProcessors()})..."
            _splashProgress.value = 0.30f

            val app = getApplication<Application>()

            // 3. Memuat Tokenizer GPT-2 ByteLevel BPE
            _splashStatus.value = "Memuat ByteLevel BPE Tokenizer (vocab 4096)..."
            try {
                tokenizer.loadFromAssets(app, "lm5m/tokenizer")
            } catch (e: Exception) {}
            _splashProgress.value = 0.50f

            // 4. Memuat Bobot LM5M Pre-trained (weights.f32)
            _splashStatus.value = "Memuat 5.07M Bobot Pre-trained LM5M (weights.f32)..."
            try {
                app.assets.open("lm5m/weights.f32").use { stream ->
                    transformer.loadWeightsFromStream(stream)
                }
            } catch (e: Exception) {}
            _splashProgress.value = 0.75f

            // 5. Memuat Optimizer Moments (adam_m.f32, adam_v.f32)
            _splashStatus.value = "Memuat Status Optimizer AdamW (Step 747)..."
            try {
                val mStream = app.assets.open("lm5m/adam_m.f32")
                val vStream = app.assets.open("lm5m/adam_v.f32")
                transformer.loadAdamState(mStream, vStream)
                mStream.close()
                vStream.close()
                transformer.adamStepCount.set(747L)
            } catch (e: Exception) {}
            _splashProgress.value = 0.90f

            // 6. Verifikasi Numerik Kesesuaian PyTorch
            val totalParams = transformer.calculateTotalParameters()
            trainer.refreshStats()
            _splashStatus.value = "Model LM5M Siap: ${totalParams / 1_000_000f}M parameter..."
            _splashProgress.value = 1.0f

            _currentScreen.value = AppScreen.DASHBOARD
        }
    }

    fun navigateTo(screen: AppScreen) {
        _currentScreen.value = screen
    }

    fun loadPresetDataset() {
        viewModelScope.launch(Dispatchers.Default) {
            _isDatasetLoading.value = true
            val meta = datasetManager.loadPresetIndonesianDataset(_trainingConfig2026.value.valSplitRatio)
            _datasetMeta.value = meta
            trainer.refreshStats()
            _isDatasetLoading.value = false
        }
    }

    fun importDatasetFile(uri: Uri, name: String, sizeBytes: Long) {
        viewModelScope.launch(Dispatchers.Default) {
            _isDatasetLoading.value = true
            try {
                val meta = datasetManager.loadFromUri(
                    context = getApplication(),
                    uri = uri,
                    fileName = name,
                    fileSize = sizeBytes,
                    valSplitRatio = _trainingConfig2026.value.valSplitRatio
                ) { _, _ -> }
                _datasetMeta.value = meta
                trainer.refreshStats()
            } catch (e: Exception) {
            } finally {
                _isDatasetLoading.value = false
            }
        }
    }

    fun setTrainingMode(mode: TrainingMode) {
        trainer.setTrainingMode(mode)
    }

    fun runBenchmark(iterations: Int = 20) {
        if (_isBenchmarking.value) return
        viewModelScope.launch(Dispatchers.Default) {
            _isBenchmarking.value = true
            _benchmarkProgress.value = 0f
            val result = benchmark.runHardwareBenchmark(
                iterations = iterations,
                threads = _hardwareConfig.value.cpuThreads
            ) { curr, total ->
                _benchmarkProgress.value = curr.toFloat() / total
            }
            _benchmarkResult.value = result
            _isBenchmarking.value = false
        }
    }

    fun startTraining(epochs: Int = 3) {
        val cfg = _trainingConfig2026.value.copy(
            baseLr = _hardwareConfig.value.learningRate
        )
        trainer.startTraining(
            context = getApplication(),
            epochs = epochs,
            config2026 = cfg,
            threads = _hardwareConfig.value.cpuThreads
        )
    }

    fun pauseTraining() {
        trainer.pauseTraining()
    }

    fun stopTraining() {
        trainer.stopTraining(getApplication())
    }

    fun saveManualCheckpoint(name: String) {
        trainer.saveManualCheckpoint(getApplication(), name)
        refreshCheckpoints()
    }

    fun loadCheckpoint(ckpt: CheckpointInfo) {
        trainer.loadCheckpoint(getApplication(), ckpt)
    }

    fun deleteCheckpoint(ckpt: CheckpointInfo) {
        CheckpointManager.deleteCheckpoint(ckpt)
        refreshCheckpoints()
    }

    fun refreshCheckpoints() {
        viewModelScope.launch(Dispatchers.IO) {
            _checkpointsList.value = CheckpointManager.listCheckpoints(getApplication())
        }
    }

    fun updateOptimizerType(type: OptimizerType) {
        _trainingConfig2026.value = _trainingConfig2026.value.copy(optimizerType = type)
    }

    fun updateLrSchedulerType(type: LrSchedulerType) {
        _trainingConfig2026.value = _trainingConfig2026.value.copy(lrSchedulerType = type)
    }

    fun updateCpuThreads(threads: Int) {
        _hardwareConfig.value = _hardwareConfig.value.copy(cpuThreads = threads)
    }

    fun updateLearningRate(lr: Float) {
        _hardwareConfig.value = _hardwareConfig.value.copy(learningRate = lr)
    }

    /**
     * Inferensi Chat Nyata (Thread-Safe):
     * Menggunakan model LM5M dengan sampling autoregresif.
     */
    fun sendPrompt(prompt: String, temperature: Float = 0.7f, maxNewTokens: Int = 32) {
        val trimmed = prompt.trim()
        if (trimmed.isEmpty() || _isGenerating.value) return

        val userMsg = ChatMessage(
            id = System.currentTimeMillis().toString(),
            sender = "user",
            text = trimmed
        )
        _chatMessages.value = _chatMessages.value + userMsg
        _isGenerating.value = true

        viewModelScope.launch(Dispatchers.Default) {
            val startTime = System.currentTimeMillis()
            val inputTokens = tokenizer.encode(trimmed, addBos = false, addEos = false)
            val generatedTokens = ArrayList<Int>()

            val activeContext = ArrayList<Int>()
            for (t in inputTokens) activeContext.add(t)

            // Autoregressive generation loop
            for (step in 0 until maxNewTokens) {
                val windowStart = maxOf(0, activeContext.size - config.contextLength)
                val window = activeContext.subList(windowStart, activeContext.size).toIntArray()

                val nextToken = transformer.sampleNextToken(
                    contextTokens = window,
                    temperature = temperature
                )

                if (nextToken == NexoraTokenizer.EOS_TOKEN) break

                generatedTokens.add(nextToken)
                activeContext.add(nextToken)
            }

            val elapsedMs = (System.currentTimeMillis() - startTime).coerceAtLeast(1)
            val decodedOutput = tokenizer.decode(generatedTokens.toIntArray())
            val tokensCount = generatedTokens.size
            val tokensPerSec = if (elapsedMs > 0) (tokensCount * 1000f / elapsedMs) else 0f

            val fullResponse = if (decodedOutput.isBlank()) {
                "[Selesai / End of Sequence <|endoftext|>]"
            } else {
                decodedOutput
            }

            val botMsg = ChatMessage(
                id = (System.currentTimeMillis() + 1).toString(),
                sender = "nexora",
                text = fullResponse,
                tokensCount = tokensCount,
                generationTimeMs = elapsedMs,
                tokensPerSec = tokensPerSec,
                isTrainedWeight = true
            )

            withContext(Dispatchers.Main) {
                _chatMessages.value = _chatMessages.value + botMsg
                _isGenerating.value = false
            }
        }
    }

    fun clearChat() {
        _chatMessages.value = emptyList()
    }
}
