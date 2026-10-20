// Nexora native compute core — real, no mock.
// Optimized CPU: ARM NEON + pthread pool. GPU path in gles_compute.cpp.
#include <jni.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <pthread.h>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <android/log.h>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "NexoraNative", __VA_ARGS__)

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#define USE_NEON 1
#else
#define USE_NEON 0
#endif

// ---------------------------------------------------------------------------
// Tiny static thread pool (avoids pthread_create overhead per matmul)
// ---------------------------------------------------------------------------
struct Pool {
    int n = 1;
    std::vector<pthread_t> threads;
    std::mutex mtx;
    std::condition_variable cv;
    std::condition_variable done_cv;
    std::atomic<int> next_tid{0};
    std::atomic<int> ready{0};
    int pending = 0;
    int active = 1;
    bool shutdown = false;
    void (*fn)(int tid, int nthreads, void* arg) = nullptr;
    void* arg = nullptr;

    static void* worker(void* p) {
        auto* pool = (Pool*)p;
        int tid = pool->next_tid.fetch_add(1, std::memory_order_seq_cst);
        pool->ready.fetch_add(1, std::memory_order_seq_cst);
        while (true) {
            std::unique_lock<std::mutex> lk(pool->mtx);
            pool->cv.wait(lk, [&]{ return pool->fn != nullptr || pool->shutdown; });
            if (pool->shutdown) return nullptr;
            auto fn = pool->fn; auto* arg = pool->arg;
            int nthreads = pool->active;
            const bool shouldRun = tid < nthreads;
            lk.unlock();
            if (shouldRun) fn(tid, nthreads, arg);
            lk.lock();
            if (shouldRun) {
                int rem = --pool->pending;
                if (rem == 0) pool->done_cv.notify_one();
            }
            pool->cv.wait(lk, [&]{ return pool->fn == nullptr || pool->shutdown; });
        }
    }
    void start(int nthreads) {
        n = nthreads;
        for (int i = 0; i < n; i++) {
            pthread_t t;
            pthread_create(&t, nullptr, worker, this);
            threads.push_back(t);
        }
        while (ready.load(std::memory_order_seq_cst) < n) std::this_thread::yield();
    }
    void run(void (*fn)(int,int,void*), void* arg, int requestedThreads) {
        std::unique_lock<std::mutex> lk(mtx);
        active = requestedThreads < 1 ? 1 : (requestedThreads > n ? n : requestedThreads);
        this->fn = fn; this->arg = arg; pending = active;
        cv.notify_all();
        done_cv.wait(lk, [&]{ return pending == 0; });
        this->fn = nullptr; this->arg = nullptr;
        cv.notify_all();
    }
};
static Pool g_pool;
static int g_num_threads = 1;
static bool g_pool_started = false;

static int detect_cores() {
    int c = std::thread::hardware_concurrency();
    return c > 0 ? c : 4;
}

extern "C" JNIEXPORT void JNICALL
Java_com_Nexora_ai_ml_NativeOps_nativeSetThreads(JNIEnv*, jclass, jint n) {
    g_num_threads = n > 0 ? n : detect_cores();
    if (!g_pool_started) {
        g_pool.start(g_num_threads);
        g_pool_started = true;
        LOGI("Thread pool started with %d threads", g_num_threads);
    }
}

// ---------------------------------------------------------------------------
// Generalized GEMM: C = alpha * op(A) * op(B) + beta * C
// A layout: if !transA A is MxK row-major; if transA A is KxM (A^T = MxK)
// B layout: if !transB B is KxN; if transB B is NxK (B^T = KxN)
// ---------------------------------------------------------------------------
struct GemmArg {
    const float* A; const float* B; float* C;
    int M, K, N; bool transA, transB; float alpha, beta;
};

static inline float a_get(const GemmArg* g, int i, int k) {
    return g->transA ? g->A[k * g->M + i] : g->A[i * g->K + k];
}
static inline float b_get(const GemmArg* g, int k, int j) {
    return g->transB ? g->B[j * g->K + k] : g->B[k * g->N + j];
}

static void gemm_worker(int tid, int nthreads, void* arg) {
    auto* g = (GemmArg*)arg;
    int rows_per = (g->M + nthreads - 1) / nthreads;
    int r0 = tid * rows_per;
    int r1 = r0 + rows_per; if (r1 > g->M) r1 = g->M;
    const int K = g->K, N = g->N;
    for (int i = r0; i < r1; i++) {
        float* Crow = g->C + i * N;
        if (g->beta == 0.f) {
            memset(Crow, 0, N * sizeof(float));
        } else if (g->beta != 1.f) {
            for (int j = 0; j < N; j++) Crow[j] *= g->beta;
        }
        // ikj loop with NEON inner vector over N
        for (int k = 0; k < K; k++) {
            float a = a_get(g, i, k) * g->alpha;
            const float* Brow = g->transB ? nullptr : (g->B + k * N);
#if USE_NEON
            float32x4_t va = vdupq_n_f32(a);
            int j = 0;
            if (!g->transB) {
                for (; j + 4 <= N; j += 4) {
                    float32x4_t vb = vld1q_f32(Brow + j);
                    float32x4_t vc = vld1q_f32(Crow + j);
                    vc = vfmaq_f32(vc, va, vb);
                    vst1q_f32(Crow + j, vc);
                }
            }
            for (; j < N; j++) Crow[j] += a * b_get(g, k, j);
#else
            for (int j = 0; j < N; j++) Crow[j] += a * b_get(g, k, j);
#endif
        }
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_Nexora_ai_ml_NativeOps_jniMatMul(
        JNIEnv* env, jclass, jfloatArray aArr, jfloatArray bArr, jfloatArray cArr,
        jint M, jint K, jint N, jboolean transA, jboolean transB,
        jfloat alpha, jfloat beta) {
    if (M <= 0 || K <= 0 || N <= 0 || aArr == nullptr || bArr == nullptr || cArr == nullptr) return;
    const jlong aNeed = static_cast<jlong>(M) * K;
    const jlong bNeed = static_cast<jlong>(K) * N;
    const jlong cNeed = static_cast<jlong>(M) * N;
    if (env->GetArrayLength(aArr) < aNeed ||
        env->GetArrayLength(bArr) < bNeed ||
        env->GetArrayLength(cArr) < cNeed) return;

    if (!g_pool_started) {
        Java_com_Nexora_ai_ml_NativeOps_nativeSetThreads(env, nullptr, 0);
    }
    float* A = env->GetFloatArrayElements(aArr, nullptr);
    if (A == nullptr) return;
    float* B = env->GetFloatArrayElements(bArr, nullptr);
    if (B == nullptr) {
        env->ReleaseFloatArrayElements(aArr, A, JNI_ABORT);
        return;
    }
    float* C = env->GetFloatArrayElements(cArr, nullptr);
    if (C == nullptr) {
        env->ReleaseFloatArrayElements(aArr, A, JNI_ABORT);
        env->ReleaseFloatArrayElements(bArr, B, JNI_ABORT);
        return;
    }

    GemmArg arg{A, B, C, M, K, N, transA != JNI_FALSE, transB != JNI_FALSE, alpha, beta};
    long long work = static_cast<long long>(M) * K * N;
    if (work < (1LL << 18) || g_num_threads <= 1) {
        gemm_worker(0, 1, &arg);
    } else {
        g_pool.run(gemm_worker, &arg, g_num_threads);
    }
    env->ReleaseFloatArrayElements(aArr, A, JNI_ABORT);
    env->ReleaseFloatArrayElements(bArr, B, JNI_ABORT);
    env->ReleaseFloatArrayElements(cArr, C, 0);
}

// ---------------------------------------------------------------------------
// Elementwise kernels (real, vectorized where possible)
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT void JNICALL
Java_com_Nexora_ai_ml_NativeOps_nativeGeluInplace(JNIEnv* env, jclass, jfloatArray xArr, jint n) {
    if (xArr == nullptr || n < 0 || env->GetArrayLength(xArr) < n) return;
    float* x = env->GetFloatArrayElements(xArr, nullptr);
    if (x == nullptr) return;
    const float c = 0.7978845608028654f; // sqrt(2/pi)
    for (int i = 0; i < n; i++) {
        float v = x[i];
        x[i] = 0.5f * v * (1.f + tanhf(c * (v + 0.044715f * v * v * v)));
    }
    env->ReleaseFloatArrayElements(xArr, x, 0);
}

extern "C" JNIEXPORT void JNICALL
Java_com_Nexora_ai_ml_NativeOps_nativeGeluBackwardInplace(
        JNIEnv* env, jclass, jfloatArray gradArr, jfloatArray xArr, jint n) {
    if (gradArr == nullptr || xArr == nullptr || n < 0 ||
        env->GetArrayLength(gradArr) < n || env->GetArrayLength(xArr) < n) return;
    float* g = env->GetFloatArrayElements(gradArr, nullptr);
    if (g == nullptr) return;
    float* x = env->GetFloatArrayElements(xArr, nullptr);
    if (x == nullptr) {
        env->ReleaseFloatArrayElements(gradArr, g, JNI_ABORT);
        return;
    }
    const float c = 0.7978845608028654f;
    for (int i = 0; i < n; i++) {
        float v = x[i];
        float t = tanhf(c * (v + 0.044715f * v * v * v));
        float d = 0.5f * (1.f + t) + 0.5f * v * (1.f - t * t) * c * (1.f + 3.f * 0.044715f * v * v);
        g[i] *= d;
    }
    env->ReleaseFloatArrayElements(gradArr, g, 0);
    env->ReleaseFloatArrayElements(xArr, x, JNI_ABORT);
}

extern "C" JNIEXPORT void JNICALL
Java_com_Nexora_ai_ml_NativeOps_nativeSoftmaxRows(
        JNIEnv* env, jclass, jfloatArray xArr, jint rows, jint cols) {
    if (xArr == nullptr || rows <= 0 || cols <= 0 ||
        static_cast<jlong>(rows) * cols > env->GetArrayLength(xArr)) return;
    float* x = env->GetFloatArrayElements(xArr, nullptr);
    if (x == nullptr) return;
    for (int r = 0; r < rows; r++) {
        float* row = x + r * cols;
        float mx = row[0];
        for (int j = 1; j < cols; j++) if (row[j] > mx) mx = row[j];
        float s = 0.f;
        for (int j = 0; j < cols; j++) { row[j] = expf(row[j] - mx); s += row[j]; }
        float inv = 1.f / s;
        for (int j = 0; j < cols; j++) row[j] *= inv;
    }
    env->ReleaseFloatArrayElements(xArr, x, 0);
}

extern "C" JNIEXPORT void JNICALL
Java_com_Nexora_ai_ml_NativeOps_nativeAddBias(
        JNIEnv* env, jclass, jfloatArray xArr, jfloatArray biasArr, jint rows, jint cols) {
    if (xArr == nullptr || biasArr == nullptr || rows < 0 || cols <= 0 ||
        static_cast<jlong>(rows) * cols > env->GetArrayLength(xArr) ||
        env->GetArrayLength(biasArr) < cols) return;
    float* x = env->GetFloatArrayElements(xArr, nullptr);
    if (x == nullptr) return;
    float* b = env->GetFloatArrayElements(biasArr, nullptr);
    if (b == nullptr) {
        env->ReleaseFloatArrayElements(xArr, x, JNI_ABORT);
        return;
    }
    for (int r = 0; r < rows; r++) {
        float* row = x + r * cols;
        for (int j = 0; j < cols; j++) row[j] += b[j];
    }
    env->ReleaseFloatArrayElements(xArr, x, 0);
    env->ReleaseFloatArrayElements(biasArr, b, JNI_ABORT);
}

// y += alpha * x
extern "C" JNIEXPORT void JNICALL
Java_com_Nexora_ai_ml_NativeOps_nativeAxpy(
        JNIEnv* env, jclass, jfloatArray yArr, jfloatArray xArr, jfloat alpha, jint n) {
    if (yArr == nullptr || xArr == nullptr || n < 0 ||
        env->GetArrayLength(yArr) < n || env->GetArrayLength(xArr) < n) return;
    float* y = env->GetFloatArrayElements(yArr, nullptr);
    if (y == nullptr) return;
    float* x = env->GetFloatArrayElements(xArr, nullptr);
    if (x == nullptr) {
        env->ReleaseFloatArrayElements(yArr, y, JNI_ABORT);
        return;
    }
#if USE_NEON
    float32x4_t va = vdupq_n_f32(alpha);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t vy = vld1q_f32(y + i);
        vy = vfmaq_f32(vy, va, vx);
        vst1q_f32(y + i, vy);
    }
    for (; i < n; i++) y[i] += alpha * x[i];
#else
    for (int i = 0; i < n; i++) y[i] += alpha * x[i];
#endif
    env->ReleaseFloatArrayElements(yArr, y, 0);
    env->ReleaseFloatArrayElements(xArr, x, JNI_ABORT);
}

// CPU benchmark: returns ms per iteration for MxK * KxN
extern "C" JNIEXPORT jdouble JNICALL
Java_com_Nexora_ai_ml_NativeOps_nativeBenchMatMul(
        JNIEnv* env, jclass, jint M, jint K, jint N, jint iter) {
    if (!g_pool_started) Java_com_Nexora_ai_ml_NativeOps_nativeSetThreads(env, nullptr, 0);
    std::vector<float> A((size_t)M * K, 0.01f);
    std::vector<float> B((size_t)K * N, 0.01f);
    std::vector<float> C((size_t)M * N, 0.f);
    GemmArg arg{A.data(), B.data(), C.data(), M, K, N, false, false, 1.f, 0.f};
    // warmup
    for (int i = 0; i < 3; i++) gemm_worker(0, 1, &arg);
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iter; i++) {
        if (g_num_threads > 1) g_pool.run(gemm_worker, &arg, g_num_threads); else gemm_worker(0, 1, &arg);
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iter;
    return ms;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_Nexora_ai_ml_NativeOps_nativeCpuCores(JNIEnv*, jclass) {
    return detect_cores();
}
