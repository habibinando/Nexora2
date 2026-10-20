package com.Nexora.ai.system

import com.Nexora.ai.ml.ModelConfig
import com.Nexora.ai.train.HyperParams
import java.util.Locale

/**
 * Conservative, device-aware starting points for on-device training.
 * Uses measured physical RAM, currently available RAM, app heap class, CPU cores,
 * and thermal state; it never treats OEM memory extension as physical RAM.
 */
object DeviceModelAdvisor {
    data class Recommendation(
        val profileName: String,
        val reason: String,
        val model: ModelConfig,
        val training: HyperParams,
        val notes: List<String>
    )

    private const val GIB: Long = 1024L * 1024L * 1024L
    private const val MIB: Long = 1024L * 1024L

    fun recommend(device: DeviceDiagnostics.Report): Recommendation {
        val availableGiB = device.availableRamBytes.toDouble() / GIB.toDouble()
        val physicalGiB = device.totalRamBytes.toDouble() / GIB.toDouble()
        val hot = listOf("hangat", "panas", "severe", "critical", "darurat", "shutdown")
            .any { device.thermalStatus.contains(it, ignoreCase = true) }
        val compact = device.lowMemory ||
            device.totalRamBytes < 4L * GIB ||
            device.largeMemoryClassMb <= 256 ||
            device.availableRamBytes < 900L * MIB

        val balanced = !compact && (
            device.totalRamBytes < 6L * GIB ||
            device.largeMemoryClassMb <= 384 ||
            device.availableRamBytes < (1536L * MIB)
        )

        val standard = !compact && !balanced && (
            device.totalRamBytes < 8L * GIB ||
            device.largeMemoryClassMb <= 512 ||
            device.availableRamBytes < (2L * GIB)
        )

        val model: ModelConfig
        val profile: String
        val reason: String
        val baseBatch: Int
        val baseEpochs: Int
        val maxThreads: Int

        when {
            compact -> {
                profile = "Hemat RAM"
                reason = "Arsitektur kecil untuk mengurangi puncak penggunaan RAM dan panas."
                model = ModelConfig(
                    vocabSize = 8_000, seqLen = 32, dModel = 64, ffnDim = 256,
                    numHeads = 4, numLayers = 2, dropout = 0.10f, useLora = false
                )
                baseBatch = 1
                baseEpochs = 3
                maxThreads = 2
            }
            balanced -> {
                profile = "Seimbang"
                reason = "Model kecil-menengah dengan context 64 agar training lebih realistis di ponsel."
                model = ModelConfig(
                    vocabSize = 10_000, seqLen = 64, dModel = 96, ffnDim = 384,
                    numHeads = 4, numLayers = 3, dropout = 0.10f, useLora = false
                )
                baseBatch = if (availableGiB >= 1.8 && !hot) 2 else 1
                baseEpochs = 3
                maxThreads = 3
            }
            standard -> {
                profile = "Standar"
                reason = "Model menengah; batch dibatasi berdasarkan RAM bebas saat pemeriksaan."
                model = ModelConfig(
                    vocabSize = 12_000, seqLen = 64, dModel = 128, ffnDim = 512,
                    numHeads = 8, numLayers = 4, dropout = 0.10f, useLora = false
                )
                baseBatch = if (availableGiB >= 2.5 && !hot) 2 else 1
                baseEpochs = 3
                maxThreads = 4
            }
            else -> {
                profile = "Lebih besar (tetap konservatif)"
                reason = "RAM dan batas heap terdeteksi lebih longgar; tetap gunakan batch kecil untuk menjaga kestabilan ponsel."
                model = ModelConfig(
                    vocabSize = 16_000, seqLen = 128, dModel = 160, ffnDim = 640,
                    numHeads = 8, numLayers = 4, dropout = 0.08f, useLora = false
                )
                baseBatch = if (availableGiB >= 3.0 && !hot) 2 else 1
                baseEpochs = 3
                maxThreads = 4
            }
        }

        val cores = device.cpuCoresNative.coerceAtLeast(1)
        val training = HyperParams(
            learningRate = 2e-4f,
            batchSize = if (hot) 1 else baseBatch,
            epochs = if (hot) 2 else baseEpochs,
            warmupSteps = 10,
            maxSteps = 5_000,
            saveEvery = 25,
            evalEvery = 25,
            patience = 4,
            gradClip = 1.0f,
            weightDecay = 0.01f,
            numThreads = minOf(cores, if (hot) 2 else maxThreads)
        )

        val notes = buildList {
            add("RAM fisik ${"%.1f".format(Locale.US, physicalGiB)} GiB; RAM tersedia saat deteksi ${"%.1f".format(Locale.US, availableGiB)} GiB. RAM virtual/ekspansi OEM tidak dihitung sebagai RAM fisik.")
            add("Batas heap besar aplikasi ${device.largeMemoryClassMb} MiB; rekomendasi menyisakan ruang untuk Android, aktivasi, gradien, dan optimizer.")
            add("Training awal dari nol direkomendasikan dengan LoRA mati agar bobot linear ikut belajar.")
            if (device.cpuCoresRuntime <= 4) add("CPU melaporkan ${device.cpuCoresRuntime} core; jumlah thread dibatasi untuk mengurangi kompetisi CPU.")
            if (device.lowMemory || availableGiB < 1.0) add("RAM bebas sedang rendah. Tutup aplikasi lain sebelum impor dataset dan mulai dengan dataset kecil.")
            if (hot) add("Sensor termal melaporkan perangkat hangat/panas; batch dan thread diturunkan. Dinginkan perangkat jika temperatur terus naik.")
            add("Ini titik awal konservatif, bukan jaminan bebas OOM. Ukuran dataset dan tokenizer juga memengaruhi pemakaian RAM.")
        }

        return Recommendation(profile, reason, model, training, notes)
    }
}
