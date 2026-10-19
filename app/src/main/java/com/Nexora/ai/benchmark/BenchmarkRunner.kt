package com.Nexora.ai.benchmark

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import com.Nexora.ai.ml.AdamW
import com.Nexora.ai.ml.GPT
import com.Nexora.ai.ml.ModelConfig
import com.Nexora.ai.ml.NativeOps
import com.Nexora.ai.ml.crossEntropy
import kotlin.math.abs
import kotlin.system.measureNanoTime

data class BenchmarkResult(
    val cpuCores: Int,
    val cpuGflops: Double,
    val cpuMsPerMatMul: Double,
    val gpuAvailable: Boolean,
    val gpuInfo: String,
    val gpuGflops: Double,
    val gpuMsPerMatMul: Double,
    val stepMsPerStep: Double,
    val tokensPerSec: Double,
    val stepLoss: Float,
    val stepCpuGemmCalls: Long,
    val stepGpuGemmCalls: Long,
    val stepGpuFallbacks: Long,
    val benchmarkModel: String,
    val ramTotalMb: Long,
    val ramAvailMb: Long,
    val modelParams: Long,
    val deviceModel: String,
    val androidVersion: String
)

object BenchmarkRunner {
    fun run(context: Context, cfg: ModelConfig, onProgress: (String) -> Unit): BenchmarkResult {
        val cores = NativeOps.nativeCpuCores()
        NativeOps.nativeSetThreads(cores)
        val m = 512; val k = 512; val n = 512
        val flops = 2.0 * m * k * n

        onProgress("1/3 · Tes CPU 512×512, 20 iterasi, hingga $cores thread…")
        val cpuMs = NativeOps.nativeBenchMatMul(m, k, n, 20).coerceAtLeast(0.001)
        val cpuGflops = flops / (cpuMs * 1e6)

        onProgress("2/3 · Inisialisasi GPU GLES dan tes GPU 512×512…")
        val gpuAvailable = NativeOps.gpuInit()
        val gpuInfo = NativeOps.gpuInfo()
        val gpuMs = if (gpuAvailable) NativeOps.gpuBench(m, k, n, 20) else -1.0
        val gpuGflops = if (gpuMs > 0.0) flops / (gpuMs * 1e6) else 0.0

        onProgress("3/3 · Mini-training: forward + loss + backward + optimizer…")
        val benchDim = (cfg.dModel.coerceIn(64, 128) / 8) * 8
        val benchVocab = cfg.vocabSize.coerceIn(512, 4096)
        val benchFfn = cfg.ffnDim.coerceIn(128, 512)
        val benchHeads = (2..16).filter { benchDim % it == 0 }.minByOrNull { abs(it - cfg.numHeads) } ?: 2
        val tinyCfg = cfg.copy(
            vocabSize = benchVocab, seqLen = 32, dModel = benchDim,
            ffnDim = benchFfn, numHeads = benchHeads, numLayers = 2, dropout = 0f
        )
        val model = GPT(tinyCfg)
        val optimizer = AdamW(model.trainablePairs())
        val batch = 1; val seq = tinyCfg.seqLen; val tokenCount = batch * seq
        val sample = IntArray(tokenCount + 1) { (0 until tinyCfg.vocabSize).random() }
        val input = sample.copyOfRange(0, tokenCount)
        val targets = sample.copyOfRange(1, tokenCount + 1)
        val gradLogits = FloatArray(tokenCount * tinyCfg.vocabSize)
        var lastLoss = 0f

        fun trainStep() {
            model.zeroGrad()
            val logits = model.forward(input, batch, seq, training = true)
            lastLoss = crossEntropy(logits, targets, tinyCfg.vocabSize, gradLogits)
            model.backward(gradLogits)
            optimizer.step(1e-4f)
            model.zeroGrad()
        }

        trainStep()
        NativeOps.resetComputeStats()
        val steps = 3
        val elapsedNs = measureNanoTime { repeat(steps) { trainStep() } }
        val msPerStep = elapsedNs.toDouble() / steps / 1_000_000.0
        val tps = if (msPerStep > 0.0) tokenCount * 1000.0 / msPerStep else 0.0
        val compute = NativeOps.computeStats()
        val manager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memory = ActivityManager.MemoryInfo().also { manager.getMemoryInfo(it) }

        return BenchmarkResult(
            cpuCores = cores, cpuGflops = cpuGflops, cpuMsPerMatMul = cpuMs,
            gpuAvailable = gpuAvailable, gpuInfo = gpuInfo, gpuGflops = gpuGflops, gpuMsPerMatMul = gpuMs,
            stepMsPerStep = msPerStep, tokensPerSec = tps, stepLoss = lastLoss,
            stepCpuGemmCalls = compute.cpuGemmCalls, stepGpuGemmCalls = compute.gpuGemmCalls,
            stepGpuFallbacks = compute.gpuFallbacks,
            benchmarkModel = "2 layer · dModel $benchDim · vocab $benchVocab · batch $batch · context $seq",
            ramTotalMb = memory.totalMem / (1024 * 1024), ramAvailMb = memory.availMem / (1024 * 1024),
            modelParams = estimateParameterCount(cfg),
            deviceModel = "${Build.MANUFACTURER} ${Build.MODEL}",
            androidVersion = "API ${Build.VERSION.SDK_INT} (Android ${Build.VERSION.RELEASE})"
        )
    }

    /** Count parameters without allocating a duplicate model instance. */
    fun estimateParameterCount(cfg: ModelConfig): Long {
        val d = cfg.dModel.toLong()
        val f = cfg.ffnDim.toLong()
        val vocabAndPosition = cfg.vocabSize.toLong() * d + cfg.seqLen.toLong() * d
        val rank = if (cfg.useLora) cfg.loraRank.toLong() else 0L
        val perBlock = 4L * d * d + 2L * d * f + 9L * d + f + rank * (8L * d + 2L * f)
        return vocabAndPosition + cfg.numLayers.toLong() * perBlock + 2L * d
    }
}
