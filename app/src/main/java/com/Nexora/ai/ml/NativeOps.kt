package com.Nexora.ai.ml

/**
 * JNI bindings to the Nexora native compute core (nexora-native.so).
 * Real CPU: ARM NEON + pthread pool. Real GPU: OpenGL ES 3.1 compute.
 * No mock, no cloud API.
 */
object NativeOps {
    init {
        System.loadLibrary("nexora-native")
    }

    @JvmStatic external fun nativeSetThreads(n: Int)
    @JvmStatic external fun nativeCpuCores(): Int
    @JvmStatic external fun nativeMatMul(
        a: FloatArray, b: FloatArray, c: FloatArray,
        M: Int, K: Int, N: Int,
        transA: Boolean, transB: Boolean, alpha: Float, beta: Float
    )
    @JvmStatic external fun nativeGeluInplace(x: FloatArray, n: Int)
    @JvmStatic external fun nativeGeluBackwardInplace(grad: FloatArray, x: FloatArray, n: Int)
    @JvmStatic external fun nativeSoftmaxRows(x: FloatArray, rows: Int, cols: Int)
    @JvmStatic external fun nativeAddBias(x: FloatArray, bias: FloatArray, rows: Int, cols: Int)
    @JvmStatic external fun nativeAxpy(y: FloatArray, x: FloatArray, alpha: Float, n: Int)
    @JvmStatic external fun nativeBenchMatMul(M: Int, K: Int, N: Int, iter: Int): Double
    @JvmStatic external fun nativeGpuInit(): Boolean
    @JvmStatic external fun nativeGpuInfo(): String
    @JvmStatic external fun nativeGpuBench(M: Int, K: Int, N: Int, iter: Int): Double
    @JvmStatic external fun nativeGpuMatMul(a: FloatArray, b: FloatArray, c: FloatArray, M: Int, K: Int, N: Int): Boolean

    /** C = A(MxK) * B(KxN), freshly allocated. */
    fun matmul(a: FloatArray, b: FloatArray, M: Int, K: Int, N: Int,
               transA: Boolean = false, transB: Boolean = false): FloatArray {
        val c = FloatArray(M * N)
        // The GLES backend accepts ordinary row-major A*B. Avoid the transfer/dispatch
        // overhead for small matrices and keep transposed GEMMs on the CPU path.
        val work = M.toLong() * K.toLong() * N.toLong()
        if (!transA && !transB && work >= 8_000_000L) {
            if (nativeGpuMatMul(a, b, c, M, K, N)) return c
        }
        nativeMatMul(a, b, c, M, K, N, transA, transB, 1f, 0f)
        return c
    }
}
