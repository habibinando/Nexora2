package com.nexora.model

import android.os.Build
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.roundToInt

data class BenchmarkResult(
    val iterations: Int,
    val totalTimeMs: Long,
    val tokensProcessed: Int,
    val tokensPerSecond: Float,
    val latencyPerTokenMs: Float,
    val gflops: Float,
    val cpuCoresUsed: Int,
    val memoryUsedMb: Float,
    val deviceModel: String
)

/**
 * Benchmark Hardware On-Device Sejati:
 * - Menggunakan instance Transformer KLONINGAN TERISOLASI sehingga bobot & momen model pengguna TIDAK TERNODAI!
 * - Benar-benar mengeksekusi beban kerja komputasi paralel pada thread pool sejumlah `threads` yang dipilih pengguna.
 * - Menghitung FLOPs matematis eksak untuk arsitektur Transformer Causal 128-128-32 Vocab 16k.
 */
class NexoraBenchmark(
    private val mainTransformer: NexoraTransformer
) {
    var lastResult: BenchmarkResult? = null
        private set

    /**
     * Menjalankan benchmark hardware dengan Multi-threading aktual dan isolasi penuh.
     */
    fun runHardwareBenchmark(
        iterations: Int = 20,
        threads: Int = Runtime.getRuntime().availableProcessors(),
        onProgress: (current: Int, total: Int) -> Unit
    ): BenchmarkResult {
        val numThreads = threads.coerceIn(1, Runtime.getRuntime().availableProcessors())
        val executor = Executors.newFixedThreadPool(numThreads)

        // Buat kloning terisolasi per worker thread agar tidak race condition dan tidak mencemari model utama!
        val isolatedTransformers = Array(numThreads) { mainTransformer.cloneIsolated() }

        val seqLen = 32
        val dummyTokens = IntArray(seqLen) { ((it * 43) % 400) + 4 }

        val completedCount = AtomicInteger(0)

        // Warm up JIT compiler
        isolatedTransformers[0].trainSequenceStep(dummyTokens, learningRate = 1e-4f)

        val startTime = System.nanoTime()

        val tasks = (1..iterations).map { idx ->
            Callable {
                val threadIndex = (idx % numThreads)
                val worker = isolatedTransformers[threadIndex]

                // Eksekusi forward + backward sequence sejati
                worker.trainSequenceStep(dummyTokens, learningRate = 1e-4f)

                val done = completedCount.incrementAndGet()
                onProgress(done, iterations)
            }
        }

        // Eksekusi di thread pool nyata
        val futures = executor.invokeAll(tasks)
        for (f in futures) {
            f.get() // Menunggu semua task selesai dan memvalidasi tidak ada exception
        }

        executor.shutdown()
        executor.awaitTermination(30, TimeUnit.SECONDS)

        val elapsedNanos = System.nanoTime() - startTime
        val elapsedSeconds = elapsedNanos / 1_000_000_000.0f
        val elapsedMs = (elapsedNanos / 1_000_000).coerceAtLeast(1)

        val totalTokens = iterations * seqLen
        val tokensPerSec = totalTokens / maxOf(0.001f, elapsedSeconds)
        val latencyMs = elapsedMs.toFloat() / totalTokens

        // Kalkulasi FLOPs Eksak untuk 1 Sequence (panjang S=32):
        // S = 32, dModel = 128, dFf = 128, V = 16,384, L = 2 layers
        // Forward per sequence:
        // 1. Attention Q, K, V: 3 * (2 * S * dModel * dModel) = 3 * (2 * 32 * 128 * 128) = 3,145,728
        // 2. Q K^T + Softmax + SV: ~4 * S^2 * dModel = 4 * 1024 * 128 = 524,288
        // 3. W_O output projection: 2 * S * dModel * dModel = 1,048,576
        // 4. FFN (W1 + W2): 4 * S * dModel * dFf = 4 * 32 * 128 * 128 = 2,097,152
        // Per layer total forward = ~6,815,744 FLOPs
        // Untuk 2 layer: 13,631,488 FLOPs
        // 5. Final Output Head Logits: 2 * S * dModel * Vocab = 2 * 32 * 128 * 16384 = 134,217,728 FLOPs
        // Total Forward = ~147,849,216 FLOPs per sequence step.
        // Backward pass = 2x Forward pass = ~295,698,432 FLOPs.
        // Total Forward + Backward per step = ~443,547,648 FLOPs (0.443 GFLOPs).
        // Kalkulasi FLOPs Eksak untuk 1 Sequence (panjang S=32):
        // S = 32, dModel = 128, dFf = 128, V = 16,384, L = 2 layers
        // Forward per layer:
        // 1. Proyeksi Q, K, V: 3 * (2 * S * dModel * dModel) = 6 * S * dModel^2
        // 2. Causal Attention Q*K^T & Softmax & V-sum: 4 * S^2 * dModel
        // 3. Output Projection W_O: 2 * S * dModel^2
        //    Total linear attention = 8 * S * dModel^2 + 4 * S^2 * dModel
        // 4. FFN (W1 & W2): 2 * (2 * S * dModel * dFf) = 4 * S * dModel * dFf
        // 5. RMSNorm (Attn & FFN): ~4 * S * dModel
        // Per layer total forward = 8*S*d^2 + 4*S^2*d + 4*S*d*f
        // 6. Output Logits Head: 2 * S * dModel * Vocab
        // Backward pass memerlukan 2x komputasi forward (gradien aktivasi + gradien bobot)
        val s = seqLen.toLong()
        val d = mainTransformer.config.dModel.toLong()
        val f = mainTransformer.config.dFf.toLong()
        val v = mainTransformer.config.vocabSize.toLong()
        val l = mainTransformer.config.numLayers.toLong()

        val layerForwardFlops = l * (8L * s * d * d + 4L * s * s * d + 6L * s * d * f)
        val headForwardFlops = 2L * s * d * v
        val forwardFlopsPerStep = layerForwardFlops + headForwardFlops
        val totalFlopsPerStep = forwardFlopsPerStep * 3L // 1x Forward + 2x Backward
        val totalFlopsAll = totalFlopsPerStep * iterations

        val gflops = (totalFlopsAll / maxOf(0.001f, elapsedSeconds)) / 1_000_000_000.0f

        val runtime = Runtime.getRuntime()
        val memUsedMb = (runtime.totalMemory() - runtime.freeMemory()) / (1024f * 1024f)

        val manufacturer = try { Build.MANUFACTURER?.replaceFirstChar { it.uppercase() } ?: "Android" } catch (e: Throwable) { "Android" }
        val model = try { Build.MODEL ?: "Device" } catch (e: Throwable) { "Device" }
        val deviceModel = "$manufacturer $model"

        val result = BenchmarkResult(
            iterations = iterations,
            totalTimeMs = elapsedMs,
            tokensProcessed = totalTokens,
            tokensPerSecond = (tokensPerSec * 10f).roundToInt() / 10f,
            latencyPerTokenMs = (latencyMs * 100f).roundToInt() / 100f,
            gflops = (gflops * 100f).roundToInt() / 100f,
            cpuCoresUsed = numThreads,
            memoryUsedMb = (memUsedMb * 10f).roundToInt() / 10f,
            deviceModel = deviceModel
        )

        lastResult = result
        return result
    }
}
