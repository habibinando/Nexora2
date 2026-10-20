package com.Nexora.ai.ml

import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong

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

    fun gpuInit(): Boolean = gpuCall(false) { nativeGpuInit() }
    fun gpuInfo(): String = gpuCall("GPU tidak tersedia atau gagal diinisialisasi") { nativeGpuInfo() }
    fun gpuBench(M: Int, K: Int, N: Int, iter: Int): Double =
        gpuCall(-1.0) { nativeGpuBench(M, K, N, iter) }

    fun computeStats() = ComputeStats(cpuGemmCalls.get(), gpuGemmCalls.get(), gpuFallbacks.get())

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
        val c = FloatArray(M * N)
        val work = M.toLong() * K.toLong() * N.toLong()
        if (work >= 8_000_000L) {
            val gpuA = if (transA) transposeRowMajor(a, K, M) else a
            val gpuB = if (transB) transposeRowMajor(b, N, K) else b
            if (gpuCall(false) { nativeGpuMatMul(gpuA, gpuB, c, M, K, N) }) {
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
