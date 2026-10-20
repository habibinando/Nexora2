package com.Nexora.ai.ml

import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.abs

/** JNI compute. Large GEMMs attempt GLES on one EGL-safe thread and fall back to CPU. */
object NativeOps {
    init { System.loadLibrary("nexora-native") }

    data class ComputeStats(
        val cpuGemmCalls: Long,
        val gpuGemmCalls: Long,
        val gpuFallbacks: Long
    )

    private val cpuGemmCalls = AtomicLong()
    private val gpuGemmCalls = AtomicLong()
    private val gpuFallbacks = AtomicLong()

    // EGL contexts are thread-affine. All GPU operations run on the same persistent thread.
    private val gpuExecutor = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "Nexora-GLES").apply { isDaemon = true }
    }

    // Accessed only from gpuExecutor; cached so validation runs once per process.
    private var gpuValidationResult: Boolean? = null

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

    /** Counts CPU matrix operations from both forward and backward passes. */
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

    @JvmStatic private external fun nativeGpuInit(): Boolean
    @JvmStatic private external fun nativeGpuInfo(): String
    @JvmStatic private external fun nativeGpuBench(M: Int, K: Int, N: Int, iter: Int): Double
    @JvmStatic private external fun nativeGpuMatMul(a: FloatArray, b: FloatArray, c: FloatArray, M: Int, K: Int, N: Int): Boolean

    /**
     * Runs exclusively on gpuExecutor. Shader compilation alone does not prove that
     * the result is correct, so compare a non-tile-aligned GPU GEMM against the CPU.
     * Dimensions 17x19x13 intentionally exercise the shader's edge-tile bounds.
     */
    private fun validateGpuOnExecutor(): Boolean {
        gpuValidationResult?.let { return it }

        val ready = try {
            nativeGpuInit()
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

            // CPU reference uses the same transposition contract as the training code.
            jniMatMul(inputA, inputB, expected, m, k, n, transA, transB, 1f, 0f)
            if (!nativeGpuMatMul(gpuA, gpuB, actual, m, k, n)) return false

            var maxError = 0f
            var maxReference = 0f
            for (i in expected.indices) {
                maxError = maxOf(maxError, abs(expected[i] - actual[i]))
                maxReference = maxOf(maxReference, abs(expected[i]))
            }
            return maxError <= 1e-3f * maxOf(1f, maxReference)
        }

        // Include edge tiles and every transA/transB combination, since the training
        // code uses transposed matrices during backward propagation and weight updates.
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

    fun gpuInfo(): String = gpuCall("GPU tidak tersedia atau gagal diinisialisasi") {
        val passed = validateGpuOnExecutor()
        val info = nativeGpuInfo()
        if (passed) "$info | tes numerik GPU: LULUS"
        else "$info | tes numerik GPU: GAGAL; GEMM besar dialihkan ke CPU"
    }

    fun gpuBench(M: Int, K: Int, N: Int, iter: Int): Double =
        gpuCall(-1.0) {
            if (validateGpuOnExecutor()) nativeGpuBench(M, K, N, iter) else -1.0
        }

    fun computeStats() = ComputeStats(cpuGemmCalls.get(), gpuGemmCalls.get(), gpuFallbacks.get())

    /** Clear counters so displayed numbers describe only the next measured phase. */
    fun resetComputeStats() {
        cpuGemmCalls.set(0)
        gpuGemmCalls.set(0)
        gpuFallbacks.set(0)
    }

    /** C = op(A) * op(B). Transposed inputs are prepared for the GLES row-major shader. */
    fun matmul(
        a: FloatArray, b: FloatArray, M: Int, K: Int, N: Int,
        transA: Boolean = false, transB: Boolean = false
    ): FloatArray {
        require(M > 0 && K > 0 && N > 0) { "Matrix dimensions must be positive" }
        require(a.size >= (if (transA) K * M else M * K)) { "Matrix A buffer is too small" }
        require(b.size >= (if (transB) N * K else K * N)) { "Matrix B buffer is too small" }

        val c = FloatArray(M * N)
        val work = M.toLong() * K.toLong() * N.toLong()
        if (work >= 8_000_000L) {
            val gpuA = if (transA) transposeRowMajor(a, K, M) else a
            val gpuB = if (transB) transposeRowMajor(b, N, K) else b
            val gpuOk = gpuCall(false) {
                validateGpuOnExecutor() && nativeGpuMatMul(gpuA, gpuB, c, M, K, N)
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

    /** Transpose row-major [rows x cols] into [cols x rows]. */
    private fun transposeRowMajor(src: FloatArray, rows: Int, cols: Int): FloatArray {
        require(src.size >= rows * cols) { "Matrix buffer too small for transpose ($rows x $cols)" }
        val dst = FloatArray(rows * cols)
        for (r in 0 until rows) {
            val offset = r * cols
            for (c in 0 until cols) dst[c * rows + r] = src[offset + c]
        }
        return dst
    }
}
