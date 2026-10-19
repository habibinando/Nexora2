// Nexora GPU compute backend — real OpenGL ES 3.1 compute shaders.
// No mock: EGL context + tiled SSBO matmul, compiled at runtime by the driver.
#include <jni.h>
#include <EGL/egl.h>
#include <GLES3/gl31.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <android/log.h>
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "NexoraGpu", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "NexoraGpu", __VA_ARGS__)

static const char* COMPUTE_SRC = R"(#version 310 es
layout(local_size_x=16, local_size_y=16, local_size_z=1) in;
layout(std430, binding=0) readonly buffer BufA { float A[]; };
layout(std430, binding=1) readonly buffer BufB { float B[]; };
layout(std430, binding=2) buffer BufC { float C[]; };
uniform int uM; uniform int uK; uniform int uN;
shared float As[16][16];
shared float Bs[16][16];
void main(){
  int row = int(gl_WorkGroupID.y)*16 + int(gl_LocalInvocationID.y);
  int col = int(gl_WorkGroupID.x)*16 + int(gl_LocalInvocationID.x);
  int ly = int(gl_LocalInvocationID.y);
  int lx = int(gl_LocalInvocationID.x);
  float sum = 0.0;
  int numTiles = (uK + 15) / 16;
  for (int t = 0; t < numTiles; t++) {
    int k = t*16 + lx;
    As[ly][lx] = (row < uM && k < uK) ? A[row*uK + k] : 0.0;
    int k2 = t*16 + ly;
    Bs[ly][lx] = (k2 < uK && col < uN) ? B[k2*uN + col] : 0.0;
    barrier();
    for (int e = 0; e < 16; e++) sum += As[ly][e] * Bs[e][lx];
    barrier();
  }
  if (row < uM && col < uN) C[row*uN + col] = sum;
})";

struct GpuState {
    bool ok = false;
    EGLDisplay dpy = EGL_NO_DISPLAY;
    EGLContext ctx = EGL_NO_CONTEXT;
    EGLSurface surf = EGL_NO_SURFACE;
    GLuint prog = 0;
    GLint uM=0, uK=0, uN=0;
    GLuint bufA=0, bufB=0, bufC=0;
    size_t szA=0, szB=0, szC=0;
    std::string info;

    bool init() {
        dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (dpy == EGL_NO_DISPLAY) return false;
        EGLint maj, min;
        if (!eglInitialize(dpy, &maj, &min)) return false;
        eglBindAPI(EGL_OPENGL_ES_API);
        EGLint cfgAttr[] = {
            EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
            EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
            EGL_NONE
        };
        EGLConfig cfg; EGLint ncfg;
        if (!eglChooseConfig(dpy, cfgAttr, &cfg, 1, &ncfg) || ncfg < 1) return false;
        EGLint ctxAttr[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
        ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, ctxAttr);
        if (ctx == EGL_NO_CONTEXT) return false;
        EGLint surfAttr[] = { EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE };
        surf = eglCreatePbufferSurface(dpy, cfg, surfAttr);
        if (surf == EGL_NO_SURFACE) return false;
        if (!eglMakeCurrent(dpy, surf, surf, ctx)) return false;

        const char* ver = (const char*)glGetString(GL_VERSION);
        const char* rend = (const char*)glGetString(GL_RENDERER);
        info = std::string(ver ? ver : "?") + " | " + (rend ? rend : "?");
        int major = 0, minor = 0;
        if (ver) sscanf(ver, "OpenGL ES %d.%d", &major, &minor);
        if (major < 3 || (major == 3 && minor < 1)) { eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT); return false; }

        GLuint sh = glCreateShader(GL_COMPUTE_SHADER);
        glShaderSource(sh, 1, &COMPUTE_SRC, nullptr);
        glCompileShader(sh);
        GLint compiled = 0; glGetShaderiv(sh, GL_COMPILE_STATUS, &compiled);
        if (!compiled) {
            char log[2048]; glGetShaderInfoLog(sh, sizeof(log), nullptr, log);
            LOGE("Compute shader compile failed: %s", log);
            glDeleteShader(sh); return false;
        }
        prog = glCreateProgram();
        glAttachShader(prog, sh);
        glLinkProgram(prog);
        glDeleteShader(sh);
        GLint linked = 0; glGetProgramiv(prog, GL_LINK_STATUS, &linked);
        if (!linked) return false;
        uM = glGetUniformLocation(prog, "uM");
        uK = glGetUniformLocation(prog, "uK");
        uN = glGetUniformLocation(prog, "uN");
        glGenBuffers(1, &bufA); glGenBuffers(1, &bufB); glGenBuffers(1, &bufC);
        ok = true;
        LOGI("GPU compute initialized: %s", info.c_str());
        return true;
    }

    void ensureBuf(GLuint& buf, size_t& cur, size_t need) {
        if (need > cur) {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
            glBufferData(GL_SHADER_STORAGE_BUFFER, need, nullptr, GL_DYNAMIC_DRAW);
            cur = need;
        }
    }

    // C = A * B ; A MxK, B KxN, C MxN. Returns true on success.
    bool matmul(const float* A, const float* B, float* C, int M, int K, int N) {
        if (!ok) return false;
        size_t na = (size_t)M*K*sizeof(float), nb = (size_t)K*N*sizeof(float), nc = (size_t)M*N*sizeof(float);
        ensureBuf(bufA, szA, na); ensureBuf(bufB, szB, nb); ensureBuf(bufC, szC, nc);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufA);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, na, A);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufB);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, nb, B);
        glUseProgram(prog);
        glUniform1i(uM, M); glUniform1i(uK, K); glUniform1i(uN, N);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufA);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bufB);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, bufC);
        GLuint gx = (N + 15) / 16, gy = (M + 15) / 16;
        glDispatchCompute(gx, gy, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufC);
        void* mapped = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, static_cast<GLsizeiptr>(nc), GL_MAP_READ_BIT);
        if (!mapped) return false;
        std::memcpy(C, mapped, nc);
        const GLboolean unmapped = glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        return unmapped == GL_TRUE;
    }
};

static thread_local GpuState* t_state = nullptr;
static GpuState* state() {
    if (!t_state) {
        t_state = new GpuState();
        t_state->init();
    }
    return t_state;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_Nexora_ai_ml_NativeOps_nativeGpuInit(JNIEnv*, jclass) {
    return state()->ok ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_Nexora_ai_ml_NativeOps_nativeGpuInfo(JNIEnv* env, jclass) {
    auto* s = state();
    return env->NewStringUTF(s->ok ? s->info.c_str() : "GPU compute unavailable");
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_Nexora_ai_ml_NativeOps_nativeGpuBench(
        JNIEnv*, jclass, jint M, jint K, jint N, jint iter) {
    auto* s = state();
    if (!s->ok) return -1.0;
    std::vector<float> A((size_t)M*K, 0.01f);
    std::vector<float> B((size_t)K*N, 0.01f);
    std::vector<float> C((size_t)M*N, 0.f);
    for (int i = 0; i < 3; i++) s->matmul(A.data(), B.data(), C.data(), M, K, N);
    glFinish();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iter; i++) s->matmul(A.data(), B.data(), C.data(), M, K, N);
    glFinish();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iter;
}


// Large forward GEMMs may use the GLES backend. The caller falls back to the
// optimized CPU implementation when initialization or execution is unavailable.
extern "C" JNIEXPORT jboolean JNICALL
Java_com_Nexora_ai_ml_NativeOps_nativeGpuMatMul(
        JNIEnv* env, jclass, jfloatArray aArr, jfloatArray bArr, jfloatArray cArr,
        jint M, jint K, jint N) {
    if (M <= 0 || K <= 0 || N <= 0) return JNI_FALSE;
    const jlong aNeed = static_cast<jlong>(M) * K;
    const jlong bNeed = static_cast<jlong>(K) * N;
    const jlong cNeed = static_cast<jlong>(M) * N;
    if (env->GetArrayLength(aArr) < aNeed ||
        env->GetArrayLength(bArr) < bNeed ||
        env->GetArrayLength(cArr) < cNeed) return JNI_FALSE;

    auto* s = state();
    if (!s->ok) return JNI_FALSE;

    jfloat* A = env->GetFloatArrayElements(aArr, nullptr);
    if (!A) return JNI_FALSE;
    jfloat* B = env->GetFloatArrayElements(bArr, nullptr);
    if (!B) {
        env->ReleaseFloatArrayElements(aArr, A, JNI_ABORT);
        return JNI_FALSE;
    }
    jfloat* C = env->GetFloatArrayElements(cArr, nullptr);
    if (!C) {
        env->ReleaseFloatArrayElements(aArr, A, JNI_ABORT);
        env->ReleaseFloatArrayElements(bArr, B, JNI_ABORT);
        return JNI_FALSE;
    }

    const bool ok = s->matmul(A, B, C, M, K, N);
    env->ReleaseFloatArrayElements(aArr, A, JNI_ABORT);
    env->ReleaseFloatArrayElements(bArr, B, JNI_ABORT);
    env->ReleaseFloatArrayElements(cArr, C, ok ? 0 : JNI_ABORT);
    return ok ? JNI_TRUE : JNI_FALSE;
}
