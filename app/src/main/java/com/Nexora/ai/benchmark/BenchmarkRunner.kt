package com.Nexora.ai.benchmark

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import com.Nexora.ai.ml.GPT
import com.Nexora.ai.ml.ModelConfig
import com.Nexora.ai.ml.NativeOps
import kotlin.system.measureTimeMillis

data class BenchmarkResult(
    val cpuCores: Int,
    val cpuGflops: Double,
    val cpuMsPerStep: Double,
    val gpuAvailable: Boolean,
    val gpuInfo: String,
    val gpuGflops: Double,
    val tokensPerSec: Double,
    val ramTotalMb: Long,
    val ramAvailMb: Long,
    val modelParams: Long,
    val deviceModel: String,
    val androidVersion: String
)

/** Real benchmark: native matmul GFLOPS, GPU compute, training-step throughput. */
object BenchmarkRunner {

    fun run(context: Context, cfg: ModelConfig, onProgress: (String) -> Unit): BenchmarkResult {
        val cores = NativeOps.nativeCpuCores()
        NativeOps.nativeSetThreads(cores)

        onProgress("CPU matmul benchmark…")
        val M = 512; val K = 512; val N = 512
        val ms = NativeOps.nativeBenchMatMul(M, K, N, 20)
        val flops = 2.0 * M * K * N
        val cpuGflops = flops / (ms * 1e6) // GFLOP/s

        onProgress("GPU compute probe…")
        val gpuAvail = NativeOps.nativeGpuInit()
        val gpuInfo = NativeOps.nativeGpuInfo()
        val gpuMs = if (gpuAvail) NativeOps.nativeGpuBench(M, K, N, 20) else -1.0
        val gpuGflops = if (gpuMs > 0) flops / (gpuMs * 1e6) else 0.0

        onProgress("Training-step throughput (real forward+backward)…")
        val tinyCfg = cfg.copy(numLayers = 2, seqLen = 64)
        val model = GPT(tinyCfg)
        val B = 2; val S = tinyCfg.seqLen; val T = B * S
        val tokens = IntArray(T) { (0 until cfg.vocabSize).random() }
        val gradLogits = FloatArray(T * cfg.vocabSize)
        // warmup
        repeat(2) {
            val logits = model.forward(tokens, B, S, true)
            model.backward(gradLogits); model.zeroGrad()
        }
        val steps = 3
        val elapsed = measureTimeMillis {
            repeat(steps) {
                val logits = model.forward(tokens, B, S, true)
                model.backward(gradLogits); model.zeroGrad()
            }
        }
        val msPerStep = elapsed.toDouble() / steps
        val tokensPerSec = (T * 1000.0) / msPerStep

        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memInfo = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }

        return BenchmarkResult(
            cpuCores = cores,
            cpuGflops = cpuGflops,
            cpuMsPerStep = msPerStep,
            gpuAvailable = gpuAvail,
            gpuInfo = gpuInfo,
            gpuGflops = gpuGflops,
            tokensPerSec = tokensPerSec,
            ramTotalMb = memInfo.totalMem / (1024 * 1024),
            ramAvailMb = memInfo.availMem / (1024 * 1024),
            modelParams = GPT(cfg).numParams,
            deviceModel = "${Build.MANUFACTURER} ${Build.MODEL}",
            androidVersion = "API ${Build.VERSION.SDK_INT} (Android ${Build.VERSION.RELEASE})"
        )
    }
}
