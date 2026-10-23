#include <jni.h>
#include <string>
#include <vector>
#include <cmath>
#include <cstring>
#include <android/log.h>
#include "nexora_bpe.h"
#include "nexora_vulkan.h"
#include "nexora_lm5m_trainer.h"

#define TAG "NexoraNative"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)

extern "C" {

/**
 * Mengembalikan informasi compiler dan toolchain NDK aktual saat kompilasi.
 */
JNIEXPORT jstring JNICALL
Java_com_nexora_model_NexoraNative_getNdkVersion(JNIEnv *env, jobject /* this */) {
#ifdef __clang_version__
    std::string info = "NDK r30 Clang " __clang_version__;
#elif defined(__VERSION__)
    std::string info = "NDK r30 " __VERSION__;
#else
    std::string info = "NDK r30 (Native C++20)";
#endif
    return env->NewStringUTF(info.c_str());
}

JNIEXPORT jboolean JNICALL
Java_com_nexora_model_NexoraNative_isNdkAvailable(JNIEnv *env, jobject /* this */) {
    return JNI_TRUE;
}

/**
 * Perkalian matriks-vektor cache-friendly dengan contiguous row access:
 * out = x * W
 * x: [dIn]
 * W: [dIn, dOut] disimpan row-major (indeks: r * dOut + c)
 * out: [dOut]
 * Menggunakan GetPrimitiveArrayCritical untuk zero-copy memory access.
 */
JNIEXPORT void JNICALL
Java_com_nexora_model_NexoraNative_nativeMatmulVec(
        JNIEnv *env,
        jobject /* this */,
        jfloatArray xArr,
        jfloatArray wArr,
        jint dIn,
        jint dOut,
        jfloatArray outArr) {

    jfloat *x = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(xArr, nullptr));
    jfloat *w = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(wArr, nullptr));
    jfloat *out = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(outArr, nullptr));

    if (x && w && out) {
        std::memset(out, 0, sizeof(float) * dOut);

        // Akses baris berurutan untuk memaksimalkan auto-vectorization SIMD dan L1 cache hit rate
        for (int r = 0; r < dIn; ++r) {
            float xr = x[r];
            const float *wRow = w + (r * dOut);
            #pragma clang loop vectorize(enable)
            for (int c = 0; c < dOut; ++c) {
                out[c] += xr * wRow[c];
            }
        }
    }

    if (out) env->ReleasePrimitiveArrayCritical(outArr, out, 0);
    if (w) env->ReleasePrimitiveArrayCritical(wArr, w, JNI_ABORT);
    if (x) env->ReleasePrimitiveArrayCritical(xArr, x, JNI_ABORT);
}

/**
 * Output head logits: logits[v] = dot_product(h, tokenEmbeddings[v])
 * tokenEmbeddings: [vocabSize, dModel] contiguous row-major
 */
JNIEXPORT void JNICALL
Java_com_nexora_model_NexoraNative_nativeEmbeddingLogits(
        JNIEnv *env,
        jobject /* this */,
        jfloatArray hArr,
        jfloatArray embArr,
        jint vocabSize,
        jint dModel,
        jfloatArray logitsArr) {

    jfloat *h = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(hArr, nullptr));
    jfloat *emb = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(embArr, nullptr));
    jfloat *logits = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(logitsArr, nullptr));

    if (h && emb && logits) {
        for (int v = 0; v < vocabSize; ++v) {
            float sum = 0.0f;
            const float *embRow = emb + (v * dModel);
            #pragma clang loop vectorize(enable)
            for (int d = 0; d < dModel; ++d) {
                sum += h[d] * embRow[d];
            }
            logits[v] = sum;
        }
    }

    if (logits) env->ReleasePrimitiveArrayCritical(logitsArr, logits, 0);
    if (emb) env->ReleasePrimitiveArrayCritical(embArr, emb, JNI_ABORT);
    if (h) env->ReleasePrimitiveArrayCritical(hArr, h, JNI_ABORT);
}

/**
 * Backpropagation LoRA Matematis Nyata:
 * Input:
 * - x: [dIn] (aktivasi forward input layer)
 * - loraA: [dIn, rank]
 * - loraB: [rank, dOut]
 * - dOutGrad: [dOut] (gradien dari layer di atasnya: dL / dy)
 * - scale: alpha / rank
 *
 * Output (diperbarui secara in-place):
 * - gradA: [dIn, rank] += scale * (x^T * (dOutGrad * B^T))
 * - gradB: [rank, dOut] += scale * ((x * A)^T * dOutGrad)
 * - dxOut: [dIn] += scale * ( (dOutGrad * B^T) * A^T ) -> gradien yang dialirkan ke layer sebelumnya
 *
 * Return: sum of squared gradients untuk norm calculation.
 */
JNIEXPORT jfloat JNICALL
Java_com_nexora_model_NexoraNative_nativeLoraBackwardExact(
        JNIEnv *env,
        jobject /* this */,
        jfloatArray xArr,
        jfloatArray aMatArr,
        jfloatArray bMatArr,
        jfloatArray dOutGradArr,
        jfloatArray gradAArr,
        jfloatArray gradBArr,
        jfloatArray dxOutArr,
        jint dIn,
        jint dOut,
        jint rank,
        jfloat scale) {

    jfloat *x = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(xArr, nullptr));
    jfloat *loraA = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(aMatArr, nullptr));
    jfloat *loraB = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(bMatArr, nullptr));
    jfloat *dOutGrad = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(dOutGradArr, nullptr));
    jfloat *gradA = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(gradAArr, nullptr));
    jfloat *gradB = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(gradBArr, nullptr));
    jfloat *dxOut = dxOutArr ? static_cast<jfloat*>(env->GetPrimitiveArrayCritical(dxOutArr, nullptr)) : nullptr;

    float sumSq = 0.0f;

    if (x && loraA && loraB && dOutGrad && gradA && gradB) {
        // 1. Hitung forward intermediate: tempRank = x * A (dimensi: rank)
        std::vector<float> tempRank(rank, 0.0f);
        for (int i = 0; i < dIn; ++i) {
            float xi = x[i];
            const float *aRow = loraA + (i * rank);
            for (int r = 0; r < rank; ++r) {
                tempRank[r] += xi * aRow[r];
            }
        }

        // 2. Hitung backward intermediate: gradTempRank = dOutGrad * B^T (dimensi: rank)
        std::vector<float> gradTempRank(rank, 0.0f);
        for (int r = 0; r < rank; ++r) {
            float sum = 0.0f;
            const float *bRow = loraB + (r * dOut);
            #pragma clang loop vectorize(enable)
            for (int c = 0; c < dOut; ++c) {
                sum += dOutGrad[c] * bRow[c];
            }
            gradTempRank[r] = sum * scale;
        }

        // 3. Akumulasi gradB: dL/dB = scale * (tempRank^T * dOutGrad)
        for (int r = 0; r < rank; ++r) {
            float tr = tempRank[r] * scale;
            float *gbRow = gradB + (r * dOut);
            #pragma clang loop vectorize(enable)
            for (int c = 0; c < dOut; ++c) {
                float g = tr * dOutGrad[c];
                gbRow[c] += g;
                sumSq += g * g;
            }
        }

        // 4. Akumulasi gradA: dL/dA = x^T * gradTempRank
        for (int i = 0; i < dIn; ++i) {
            float xi = x[i];
            float *gaRow = gradA + (i * rank);
            #pragma clang loop vectorize(enable)
            for (int r = 0; r < rank; ++r) {
                float g = xi * gradTempRank[r];
                gaRow[r] += g;
                sumSq += g * g;
            }
        }

        // 5. Backpropagate gradien ke input aktivasi: dxOut += gradTempRank * A^T
        if (dxOut) {
            for (int i = 0; i < dIn; ++i) {
                float sum = 0.0f;
                const float *aRow = loraA + (i * rank);
                for (int r = 0; r < rank; ++r) {
                    sum += gradTempRank[r] * aRow[r];
                }
                dxOut[i] += sum;
            }
        }
    }

    if (dxOut) env->ReleasePrimitiveArrayCritical(dxOutArr, dxOut, 0);
    if (gradB) env->ReleasePrimitiveArrayCritical(gradBArr, gradB, 0);
    if (gradA) env->ReleasePrimitiveArrayCritical(gradAArr, gradA, 0);
    if (dOutGrad) env->ReleasePrimitiveArrayCritical(dOutGradArr, dOutGrad, JNI_ABORT);
    if (loraB) env->ReleasePrimitiveArrayCritical(bMatArr, loraB, JNI_ABORT);
    if (loraA) env->ReleasePrimitiveArrayCritical(aMatArr, loraA, JNI_ABORT);
    if (x) env->ReleasePrimitiveArrayCritical(xArr, x, JNI_ABORT);

    return sumSq;
}

/**
 * Native Gradient Clipping by L2 Norm:
 * Jika totalNorm > maxNorm, skalakan gradArr: grad = grad * (maxNorm / totalNorm).
 */
JNIEXPORT void JNICALL
Java_com_nexora_model_NexoraNative_nativeClipGradients(
        JNIEnv *env,
        jobject /* this */,
        jfloatArray gradArr,
        jfloat clipFactor,
        jint size) {

    jfloat *grad = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(gradArr, nullptr));
    if (grad) {
        #pragma clang loop vectorize(enable)
        for (int i = 0; i < size; ++i) {
            grad[i] *= clipFactor;
        }
    }
    if (grad) env->ReleasePrimitiveArrayCritical(gradArr, grad, 0);
}

/**
 * Native AdamW Optimizer:
 */
JNIEXPORT void JNICALL
Java_com_nexora_model_NexoraNative_nativeAdamWUpdate(
        JNIEnv *env,
        jobject /* this */,
        jfloatArray paramArr,
        jfloatArray gradArr,
        jfloatArray mArr,
        jfloatArray vArr,
        jfloat beta1,
        jfloat beta2,
        jfloat lr,
        jfloat weightDecay,
        jfloat eps,
        jint step,
        jint size) {

    jfloat *param = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(paramArr, nullptr));
    jfloat *grad = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(gradArr, nullptr));
    jfloat *m = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(mArr, nullptr));
    jfloat *v = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(vArr, nullptr));

    if (param && grad && m && v) {
        float b1Correction = 1.0f - std::pow(beta1, (float)step);
        float b2Correction = 1.0f - std::pow(beta2, (float)step);
        float invB1 = (b1Correction > 1e-12f) ? (1.0f / b1Correction) : 1.0f;
        float invB2 = (b2Correction > 1e-12f) ? (1.0f / b2Correction) : 1.0f;

        #pragma clang loop vectorize(enable)
        for (int i = 0; i < size; ++i) {
            float g = grad[i];
            m[i] = beta1 * m[i] + (1.0f - beta1) * g;
            v[i] = beta2 * v[i] + (1.0f - beta2) * (g * g);

            float mHat = m[i] * invB1;
            float vHat = v[i] * invB2;

            param[i] -= lr * (mHat / (std::sqrt(vHat) + eps) + weightDecay * param[i]);
            grad[i] = 0.0f; // Reset gradien setelah update
        }
    }

    if (v) env->ReleasePrimitiveArrayCritical(vArr, v, 0);
    if (m) env->ReleasePrimitiveArrayCritical(mArr, m, 0);
    if (grad) env->ReleasePrimitiveArrayCritical(gradArr, grad, 0);
    if (param) env->ReleasePrimitiveArrayCritical(paramArr, param, 0);
}

/**
 * Native Lion Optimizer (EvoLved Sign Momentum):
 */
JNIEXPORT void JNICALL
Java_com_nexora_model_NexoraNative_nativeLionUpdate(
        JNIEnv *env,
        jobject /* this */,
        jfloatArray paramArr,
        jfloatArray gradArr,
        jfloatArray mArr,
        jfloat beta1,
        jfloat beta2,
        jfloat lr,
        jfloat weightDecay,
        jint size) {

    jfloat *param = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(paramArr, nullptr));
    jfloat *grad = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(gradArr, nullptr));
    jfloat *m = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(mArr, nullptr));

    if (param && grad && m) {
        #pragma clang loop vectorize(enable)
        for (int i = 0; i < size; ++i) {
            float g = grad[i];
            float c = beta1 * m[i] + (1.0f - beta1) * g;
            float signC = (c > 0.0f) ? 1.0f : ((c < 0.0f) ? -1.0f : 0.0f);

            param[i] -= lr * (signC + weightDecay * param[i]);
            m[i] = beta2 * m[i] + (1.0f - beta2) * g;
            grad[i] = 0.0f; // Reset gradien
        }
    }

    if (m) env->ReleasePrimitiveArrayCritical(mArr, m, 0);
    if (grad) env->ReleasePrimitiveArrayCritical(gradArr, grad, 0);
    if (param) env->ReleasePrimitiveArrayCritical(paramArr, param, 0);
}

/**
 * Native SGD with Momentum Optimizer:
 */
JNIEXPORT void JNICALL
Java_com_nexora_model_NexoraNative_nativeSgdMomentumUpdate(
        JNIEnv *env,
        jobject /* this */,
        jfloatArray paramArr,
        jfloatArray gradArr,
        jfloatArray vArr,
        jfloat momentum,
        jfloat lr,
        jfloat weightDecay,
        jint size) {

    jfloat *param = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(paramArr, nullptr));
    jfloat *grad = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(gradArr, nullptr));
    jfloat *v = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(vArr, nullptr));

    if (param && grad && v) {
        #pragma clang loop vectorize(enable)
        for (int i = 0; i < size; ++i) {
            float g = grad[i];
            v[i] = momentum * v[i] + g;
            param[i] -= lr * (v[i] + weightDecay * param[i]);
            grad[i] = 0.0f;
        }
    }

    if (v) env->ReleasePrimitiveArrayCritical(vArr, v, 0);
    if (grad) env->ReleasePrimitiveArrayCritical(gradArr, grad, 0);
    if (param) env->ReleasePrimitiveArrayCritical(paramArr, param, 0);
}

// ============================================================================
// NATIVE BPE (BYTE PAIR ENCODING) TOKENIZER ENGINE
// ============================================================================

static nexora::BPETokenizerEngine g_bpe_engine;

JNIEXPORT jintArray JNICALL
Java_com_nexora_model_NexoraNative_nativeBpeEncode(
        JNIEnv *env,
        jobject /* this */,
        jbyteArray textBytes,
        jboolean addBos,
        jboolean addEos) {
    if (!textBytes) {
        return env->NewIntArray(0);
    }
    jsize len = env->GetArrayLength(textBytes);
    jbyte *bytes = env->GetByteArrayElements(textBytes, nullptr);
    if (!bytes) {
        return env->NewIntArray(0);
    }

    std::vector<uint32_t> tokens = g_bpe_engine.encode(reinterpret_cast<const uint8_t*>(bytes), len, addBos, addEos);
    env->ReleaseByteArrayElements(textBytes, bytes, JNI_ABORT);

    jintArray result = env->NewIntArray(static_cast<jsize>(tokens.size()));
    if (result && !tokens.empty()) {
        env->SetIntArrayRegion(result, 0, static_cast<jsize>(tokens.size()), reinterpret_cast<const jint*>(tokens.data()));
    }
    return result;
}

JNIEXPORT jbyteArray JNICALL
Java_com_nexora_model_NexoraNative_nativeBpeDecode(
        JNIEnv *env,
        jobject /* this */,
        jintArray tokenArr) {
    if (!tokenArr) {
        return env->NewByteArray(0);
    }
    jsize len = env->GetArrayLength(tokenArr);
    jint *tokens = env->GetIntArrayElements(tokenArr, nullptr);
    if (!tokens) {
        return env->NewByteArray(0);
    }

    std::vector<uint8_t> bytes = g_bpe_engine.decode(reinterpret_cast<const uint32_t*>(tokens), len);
    env->ReleaseIntArrayElements(tokenArr, tokens, JNI_ABORT);

    jbyteArray result = env->NewByteArray(static_cast<jsize>(bytes.size()));
    if (result && !bytes.empty()) {
        env->SetByteArrayRegion(result, 0, static_cast<jsize>(bytes.size()), reinterpret_cast<const jbyte*>(bytes.data()));
    }
    return result;
}

JNIEXPORT jint JNICALL
Java_com_nexora_model_NexoraNative_nativeBpeTrainFromBytes(
        JNIEnv *env,
        jobject /* this */,
        jbyteArray textBytes,
        jint maxNewTokens) {
    if (!textBytes || maxNewTokens <= 0) return 0;
    jsize len = env->GetArrayLength(textBytes);
    jbyte *bytes = env->GetByteArrayElements(textBytes, nullptr);
    if (!bytes) return 0;

    int added = g_bpe_engine.trainFromBytes(reinterpret_cast<const uint8_t*>(bytes), len, maxNewTokens);
    env->ReleaseByteArrayElements(textBytes, bytes, JNI_ABORT);
    return added;
}

JNIEXPORT jobjectArray JNICALL
Java_com_nexora_model_NexoraNative_nativeBpeExportVocabulary(
        JNIEnv *env,
        jobject /* this */) {
    std::vector<std::string> vocab = g_bpe_engine.exportVocabulary();
    jclass stringClass = env->FindClass("java/lang/String");
    jobjectArray result = env->NewObjectArray(static_cast<jsize>(vocab.size()), stringClass, nullptr);
    for (size_t i = 0; i < vocab.size(); ++i) {
        jstring str = env->NewStringUTF(vocab[i].c_str());
        env->SetObjectArrayElement(result, static_cast<jsize>(i), str);
        env->DeleteLocalRef(str);
    }
    return result;
}

JNIEXPORT void JNICALL
Java_com_nexora_model_NexoraNative_nativeBpeImportVocabulary(
        JNIEnv *env,
        jobject /* this */,
        jobjectArray vocabList) {
    if (!vocabList) return;
    jsize len = env->GetArrayLength(vocabList);
    std::vector<std::string> list;
    list.reserve(len);
    for (jsize i = 0; i < len; ++i) {
        jstring str = (jstring)env->GetObjectArrayElement(vocabList, i);
        if (str) {
            const char *chars = env->GetStringUTFChars(str, nullptr);
            if (chars) {
                list.emplace_back(chars);
                env->ReleaseStringUTFChars(str, chars);
            }
            env->DeleteLocalRef(str);
        }
    }
    g_bpe_engine.importVocabulary(list);
}

JNIEXPORT jint JNICALL
Java_com_nexora_model_NexoraNative_nativeBpeGetVocabSize(
        JNIEnv *env,
        jobject /* this */) {
    return static_cast<jint>(g_bpe_engine.getVocabSize());
}

JNIEXPORT jint JNICALL
Java_com_nexora_model_NexoraNative_nativeBpeRegisterWord(
        JNIEnv *env,
        jobject /* this */,
        jstring word) {
    if (!word) return 3; // UNK
    const char *chars = env->GetStringUTFChars(word, nullptr);
    std::string w(chars ? chars : "");
    if (chars) env->ReleaseStringUTFChars(word, chars);
    return static_cast<jint>(g_bpe_engine.registerWord(w));
}

JNIEXPORT jboolean JNICALL
Java_com_nexora_model_NexoraNative_nativeVulkanInit(
        JNIEnv *env,
        jobject /* this */) {
    return nexora::VulkanComputeEngine::instance().init() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jstring JNICALL
Java_com_nexora_model_NexoraNative_nativeVulkanGetStatus(
        JNIEnv *env,
        jobject /* this */) {
    std::string status = nexora::VulkanComputeEngine::instance().getStatusString();
    return env->NewStringUTF(status.c_str());
}

JNIEXPORT jboolean JNICALL
Java_com_nexora_model_NexoraNative_nativeVulkanIsActive(
        JNIEnv *env,
        jobject /* this */) {
    return nexora::VulkanComputeEngine::instance().isAvailable() ? JNI_TRUE : JNI_FALSE;
}

/**
 * Pelatihan LM5M Sekuensial On-Device Terakselerasi Vulkan / SIMD C++:
 * Menerima buffer bobot lengkap 5.07M parameter Little-Endian, token sekuens, gradien, dan momen AdamW.
 * Mengembalikan array float berukuran 3: [loss, gradNorm, isVulkanUsed (1.0f atau 0.0f)]
 */
JNIEXPORT jfloatArray JNICALL
Java_com_nexora_model_NexoraNative_nativeTrainStepLM5M(
        JNIEnv *env,
        jobject /* this */,
        jintArray tokensArr,
        jfloatArray allWeightsArr,
        jfloatArray allGradsArr,
        jfloatArray allMArr,
        jfloatArray allVArr,
        jfloat learningRate,
        jfloat weightDecay,
        jfloat maxGradNorm,
        jint step,
        jboolean preferVulkan) {

    if (!tokensArr || !allWeightsArr || !allGradsArr || !allMArr || !allVArr) {
        jfloatArray res = env->NewFloatArray(3);
        return res;
    }

    jsize seqLen = env->GetArrayLength(tokensArr);
    if (seqLen < 2) {
        jfloatArray res = env->NewFloatArray(3);
        return res;
    }

    jint *tokens = static_cast<jint*>(env->GetPrimitiveArrayCritical(tokensArr, nullptr));
    jfloat *weights = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(allWeightsArr, nullptr));
    jfloat *grads = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(allGradsArr, nullptr));
    jfloat *m = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(allMArr, nullptr));
    jfloat *v = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(allVArr, nullptr));

    nexora::TrainStepResult stepRes;

    if (tokens && weights && grads && m && v) {
        stepRes = nexora::executeLM5MTrainStep(
            reinterpret_cast<const int32_t*>(tokens),
            static_cast<uint32_t>(seqLen),
            weights,
            grads,
            m,
            v,
            learningRate,
            weightDecay,
            maxGradNorm,
            static_cast<uint32_t>(step),
            preferVulkan == JNI_TRUE
        );
    }

    if (v) env->ReleasePrimitiveArrayCritical(allVArr, v, 0);
    if (m) env->ReleasePrimitiveArrayCritical(allMArr, m, 0);
    if (grads) env->ReleasePrimitiveArrayCritical(allGradsArr, grads, 0);
    if (weights) env->ReleasePrimitiveArrayCritical(allWeightsArr, weights, 0);
    if (tokens) env->ReleasePrimitiveArrayCritical(tokensArr, tokens, JNI_ABORT);

    jfloatArray resultArr = env->NewFloatArray(3);
    jfloat outVals[3] = { stepRes.loss, stepRes.gradNorm, stepRes.vulkanUsed ? 1.0f : 0.0f };
    env->SetFloatArrayRegion(resultArr, 0, 3, outVals);
    return resultArr;
}

} // extern "C"
