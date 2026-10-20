package com.Nexora.ai.ml

import android.content.Context
import android.content.res.AssetManager
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.abs

/**
 * Native CPU + Vulkan compute.
 * Large GEMMs use Vulkan only after runtime shader/device initialization and numerical
 * comparison against the CPU implementation. Unsupported devices always fall back to CPU.
 */
object NativeOps {
    init { System.loadLibrary("nexora-native") }

    const val GPU_GEMM_MIN_WORK: Long = 8_000_000L

    data class ComputeStats(
        val cpuGemmCalls: Long,
        val gpuGemmCalls: Long,
        val gpuFallbacks: Long
    )

    private val cpuGemmCalls = AtomicLong()
    private val gpuGemmCalls = AtomicLong()
    private val gpuFallbacks = AtomicLong()

    // Keep Vulkan device/queue/command-buffer calls serialized on one thread.
    private val gpuExecutor = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "Nexora-Vulkan").apply { isDaemon = true }
    }
    private var appAssets: AssetManager? = null
    private var gpuValidationResult: Boolean? = null

    /** Store application assets for loading the packaged SPIR-V shader. No network or API key. */
    fun initialize(context: Context) {
        appAssets = context.applicationContext.assets
    }

    private fun <T> gpuCall(fallback: T, block: () -> T): T = try {
        gpuExecutor.submit<T> { block() }.get()
    } catch (e: InterruptedException) {
        Thread.currentThread().interrupt()
        fallback
    } catch (_: Exception) {
        fallback
    }

    @JvmStatic external fun nativeSetThreads(n: Int)
    @JvmStatic external fun nativeCpuCores(): Int

    @JvmStatic private external fun jniMatMul(
        a: FloatArray, b: FloatArray, c: FloatArray,
        M: Int, K: Int, N: Int, transA: Boolean, transB: Boolean, alpha: Float, beta: Float
    )

    /** Counts CPU matrix operations used by forward/backward/training. */
    fun nativeMatMul(
        a: FloatArray, b: FloatArray, c: FloatArray,
        M: Int, K: Int, N: Int, transA: Boolean, transB: Boolean, alpha: Float, beta: Float
    ) {
        cpuGemmCalls.incrementAndGet()
        jniMatMul(a, b, c, M, K, N, transA, transB, alpha, beta)
    }

    @JvmStatic external fun nativeGeluInplace(x: FloatArray, n: Int)
    @JvmStatic external fun nativeGeluBackwardInplace(grad: FloatArray, x: FloatArray, n: Int)
    @JvmStatic external fun nativeSoftmaxRows(x: FloatArray, rows: Int, cols: Int)
    @JvmStatic external fun nativeAddBias(x: FloatArray, bias: FloatArray, rows: Int, cols: Int)
    @JvmStatic external fun nativeAxpy(y: FloatArray, x: FloatArray, alpha: Float, n: Int)
    @JvmStatic external fun nativeBenchMatMul(M: Int, K: Int, N: Int, iter: Int): Double

    @JvmStatic private external fun nativeVulkanInit(assetManager: AssetManager): Boolean
    @JvmStatic private external fun nativeVulkanInfo(): String
    @JvmStatic private external fun nativeVulkanBench(M: Int, K: Int, N: Int, iter: Int): Double
    @JvmStatic private external fun nativeVulkanMatMul(
        a: FloatArray, b: FloatArray, c: FloatArray, M: Int, K: Int, N: Int, accumulate: Boolean
    )

    /**
     * Must be called on gpuExecutor. Shader creation alone isn't enough: compare Vulkan
     * results against the CPU for non-multiple-of-16 edge tiles and every transpose mode.
     */
    private fun validateGpuOnExecutor(): Boolean {
        gpuValidationResult?.let { return it }

        val assets = appAssets
        if (assets == null) {
            gpuValidationResult = false
            return false
        }
        val ready = try {
            nativeVulkanInit(assets)
        } catch (_: Exception) {
            false
        }
        if (!ready) {
            gpuValidationResult = false
            return false
        }

        val m = 17
        val k = 19
        val n = 13
        val a = FloatArray(m * k) { ((it % 13) - 6) * 0.125f }
        val b = FloatArray(k * n) { ((it % 9) - 4) * 0.2f }
        val aTransposed = transposeRowMajor(a, m, k)
        val bTransposed = transposeRowMajor(b, k, n)

        fun casePasses(transA: Boolean, transB: Boolean): Boolean {
            val inputA = if (transA) aTransposed else a
            val inputB = if (transB) bTransposed else b
            val gpuA = if (transA) transposeRowMajor(inputA, k, m) else inputA
            val gpuB = if (transB) transposeRowMajor(inputB, n, k) else inputB
            val expected = FloatArray(m * n)
            val actual = FloatArray(m * n)

            jniMatMul(inputA, inputB, expected, m, k, n, transA, transB, 1f, 0f)
            if (!nativeVulkanMatMul(gpuA, gpuB, actual, m, k, n, false)) return false

            var maxError = 0f
            var maxReference = 0f
            for (i in expected.indices) {
                maxError = maxOf(maxError, abs(expected[i] - actual[i]))
                maxReference = maxOf(maxReference, abs(expected[i]))
            }
            return maxError <= 1e-3f * maxOf(1f, maxReference)
        }

        val passed = try {
            casePasses(false, false) &&
                casePasses(true, false) &&
                casePasses(false, true) &&
                casePasses(true, true)
        } catch (_: Exception) {
            false
        }
        gpuValidationResult = passed
        return passed
    }

    fun gpuInit(): Boolean = gpuCall(false) { validateGpuOnExecutor() }

    fun gpuInfo(): String = gpuCall("Vulkan tidak tersedia atau gagal diinisialisasi") {
        val passed = validateGpuOnExecutor()
        val info = nativeVulkanInfo()
        if (passed) "$info | tes akurasi GEMM: LULUS"
        else "$info | tes akurasi GEMM: GAGAL; operasi besar dialihkan ke CPU"
    }

    fun gpuBench(M: Int, K: Int, N: Int, iter: Int): Double = gpuCall(-1.0) {
        if (validateGpuOnExecutor()) nativeVulkanBench(M, K, N, iter) else -1.0
    }

    fun computeStats() = ComputeStats(cpuGemmCalls.get(), gpuGemmCalls.get(), gpuFallbacks.get())

    fun resetComputeStats() {
        cpuGemmCalls.set(0)
        gpuGemmCalls.set(0)
        gpuFallbacks.set(0)
    }

    /** C = op(A) * op(B). Vulkan receives row-major matrices with transposes prepared. */
    fun matmul(
        a: FloatArray, b: FloatArray, M: Int, K: Int, N: Int,
        transA: Boolean = false, transB: Boolean = false
    ): FloatArray {
        require(M > 0 && K > 0 && N > 0) { "Matrix dimensions must be positive" }
        val expectedA = M.toLong() * K
        val expectedB = K.toLong() * N
        require(a.size.toLong() >= expectedA && b.size.toLong() >= expectedB) {
            "Matrix buffer is too small for requested dimensions"
        }

        val c = FloatArray(M * N)
        val work = M.toLong() * K.toLong() * N.toLong()
        if (work >= GPU_GEMM_MIN_WORK) {
            val gpuA = if (transA) transposeRowMajor(a, K, M) else a
            val gpuB = if (transB) transposeRowMajor(b, N, K) else b
            val gpuOk = gpuCall(false) {
                validateGpuOnExecutor() && nativeVulkanMatMul(gpuA, gpuB, c, M, K, N, false)
            }
            if (gpuOk) {
                gpuGemmCalls.incrementAndGet()
                return c
            }
            gpuFallbacks.incrementAndGet()
        }
        nativeMatMul(a, b, c, M, K, N, transA, transB, 1f, 0f)
        return c
    }

    /**
     * Accumulate op(A) * op(B) into the existing C buffer.
     * Large gradient matrices use the same accuracy-validated Vulkan path and let the
     * shader read the existing C value before adding; CPU fallback preserves beta=1 semantics.
     */
    fun matmulAccumulate(
        a: FloatArray, b: FloatArray, c: FloatArray,
        M: Int, K: Int, N: Int,
        transA: Boolean = false, transB: Boolean = false
    ) {
        require(M > 0 && K > 0 && N > 0) { "Matrix dimensions must be positive" }
        require(a.size.toLong() >= M.toLong() * K && b.size.toLong() >= K.toLong() * N &&
            c.size.toLong() >= M.toLong() * N) { "Matrix buffer is too small for accumulating GEMM" }

        val work = M.toLong() * K.toLong() * N.toLong()
        if (work >= GPU_GEMM_MIN_WORK) {
            val gpuA = if (transA) transposeRowMajor(a, K, M) else a
            val gpuB = if (transB) transposeRowMajor(b, N, K) else b
            val gpuOk = gpuCall(false) {
                validateGpuOnExecutor() && nativeVulkanMatMul(gpuA, gpuB, c, M, K, N, true)
            }
            if (gpuOk) {
                gpuGemmCalls.incrementAndGet()
                return
            }
            gpuFallbacks.incrementAndGet()
        }
        nativeMatMul(a, b, c, M, K, N, transA, transB, 1f, 1f)
    }

    /** Transpose row-major [rows x cols] into [cols x rows]. */
    private fun transposeRowMajor(src: FloatArray, rows: Int, cols: Int): FloatArray {
        require(src.size.toLong() >= rows.toLong() * cols.toLong()) {
            "Matrix buffer too small for transpose ($rows x $cols)"
        }
        val dst = FloatArray(rows * cols)
        for (r in 0 until rows) {
            val offset = r * cols
            for (c in 0 until cols) dst[c * rows + r] = src[offset + c]
        }
        return dst
    }
}
