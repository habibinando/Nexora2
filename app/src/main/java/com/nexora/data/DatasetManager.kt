package com.nexora.data

import android.content.Context
import android.net.Uri
import com.nexora.model.NexoraTokenizer
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStream
import java.io.InputStreamReader
import java.nio.charset.StandardCharsets
import java.util.Collections
import java.util.Random

data class DatasetMetadata(
    val name: String,
    val format: String,
    val sizeBytes: Long,
    val totalLines: Int,
    val totalTokens: Int,
    val trainCount: Int,
    val valCount: Int,
    val sampleSnippets: List<String>,
    val isTruncated: Boolean = false,
    val truncationNotice: String? = null
)

/**
 * Manajer Dataset On-Device Tanpa Kebocoran Data (Zero Data Leakage):
 * - Memisahkan teks mentah ke Train vs Validation sebelum proses pembuatan sequence/sliding window,
 *   sehingga tidak ada sequence validation yang tumpang tindih dengan data pelatihan.
 * - Membatasi buffer memori aktif (maks 5.000 sequence) untuk stabilitas heap JVM pada file besar.
 * - Menyediakan fitur pengacakan sequence per-epoch (shuffle) untuk konvergensi optimal.
 */
class DatasetManager(private val tokenizer: NexoraTokenizer) {

    companion object {
        const val MAX_BUFFERED_SAMPLES = 5000
    }

    private val trainSequences = ArrayList<IntArray>()
    private val valSequences = ArrayList<IntArray>()

    var metadata: DatasetMetadata? = null
        private set

    val hasDataset: Boolean get() = trainSequences.isNotEmpty()
    val totalSamples: Int get() = trainSequences.size + valSequences.size
    val totalTrainSamples: Int get() = trainSequences.size
    val totalValSamples: Int get() = valSequences.size

    fun getTrainSequence(index: Int): IntArray? {
        if (index in 0 until trainSequences.size) return trainSequences[index]
        return null
    }

    fun getValSequence(index: Int): IntArray? {
        if (index in 0 until valSequences.size) return valSequences[index]
        return null
    }

    fun getAllValSequences(): List<IntArray> = valSequences.toList()

    /**
     * Mengacak urutan data latih di awal setiap epoch dengan seed baru.
     */
    fun shuffleTrainSequences(seed: Long) {
        trainSequences.shuffle(kotlin.random.Random(seed))
    }

    /**
     * Memuat file dari URI secara streaming dengan proteksi kebocoran data.
     */
    fun loadFromUri(
        context: Context,
        uri: Uri,
        fileName: String,
        fileSize: Long,
        valSplitRatio: Float = 0.15f,
        onProgress: (linesRead: Int, tokensGenerated: Int) -> Unit
    ): DatasetMetadata {
        val inputStream = context.contentResolver.openInputStream(uri)
            ?: throw IllegalStateException("Tidak dapat membuka stream file: $fileName")

        return processInputStream(inputStream, fileName, fileSize, valSplitRatio, onProgress)
    }

    /**
     * Memproses stream input:
     * 1. Kumpulkan teks mentah per baris/sampel.
     * 2. Pisahkan sampel mentah menjadi Train Set dan Validation Set secara terpisah (Zero Leakage).
     * 3. Tokenize dan buat potongan sequence 32-token secara terisolasi.
     */
    fun processInputStream(
        inputStream: InputStream,
        fileName: String,
        fileSize: Long,
        valSplitRatio: Float = 0.15f,
        onProgress: (linesRead: Int, tokensGenerated: Int) -> Unit
    ): DatasetMetadata {
        trainSequences.clear()
        valSequences.clear()

        val format = when {
            fileName.endsWith(".jsonl", ignoreCase = true) -> "JSONL"
            fileName.endsWith(".csv", ignoreCase = true) -> "CSV"
            else -> "TXT"
        }

        val reader = BufferedReader(InputStreamReader(inputStream, StandardCharsets.UTF_8), 32768)
        var lineCount = 0
        var totalTokens = 0
        val snippets = ArrayList<String>()
        val rawSamples = ArrayList<String>()
        var hitBufferLimit = false

        try {
            var line: String? = reader.readLine()
            var isFirstLine = true

            while (line != null) {
                lineCount++

                // Jika format CSV, abaikan header baris pertama
                if (isFirstLine && format == "CSV") {
                    isFirstLine = false
                    line = reader.readLine()
                    continue
                }
                isFirstLine = false

                val text = extractTextFromLine(line, format)
                if (text.isNotBlank()) {
                    if (snippets.size < 5) {
                        snippets.add(text.take(150))
                    }
                    if (rawSamples.size < MAX_BUFFERED_SAMPLES * 2) {
                        rawSamples.add(text)
                    } else {
                        hitBufferLimit = true
                    }
                    // Pelajari subword baru
                    tokenizer.learnTokensFromText(text, maxNewTokens = 10)
                }

                if (lineCount % 250 == 0) {
                    onProgress(lineCount, totalTokens)
                }

                line = reader.readLine()
            }
        } finally {
            reader.close()
            inputStream.close()
        }

        // Pisahkan sampel teks secara acak deterministik SEBELUM membuat sequence
        val rng = Random(42)
        val shuffledRaw = ArrayList(rawSamples)
        Collections.shuffle(shuffledRaw, rng)

        val valCount = maxOf(1, (shuffledRaw.size * valSplitRatio).toInt())
        val valTexts = shuffledRaw.take(valCount)
        val trainTexts = shuffledRaw.drop(valCount)

        // Buat sequence untuk Data Latih (Train)
        for (text in trainTexts) {
            val tokenIds = tokenizer.encode(text, addBos = false, addEos = true)
            totalTokens += tokenIds.size
            if (tokenIds.size >= 2) {
                var start = 0
                while (start < tokenIds.size && trainSequences.size < MAX_BUFFERED_SAMPLES) {
                    val end = minOf(tokenIds.size, start + 32)
                    val subSeq = tokenIds.copyOfRange(start, end)
                    if (subSeq.size >= 2) {
                        trainSequences.add(subSeq)
                    }
                    start += 16
                }
            }
        }

        // Buat sequence untuk Data Validasi (Validation) secara terpisah murni
        for (text in valTexts) {
            val tokenIds = tokenizer.encode(text, addBos = false, addEos = true)
            totalTokens += tokenIds.size
            if (tokenIds.size >= 2) {
                var start = 0
                while (start < tokenIds.size) {
                    val end = minOf(tokenIds.size, start + 32)
                    val subSeq = tokenIds.copyOfRange(start, end)
                    if (subSeq.size >= 2) {
                        valSequences.add(subSeq)
                    }
                    start += 16
                }
            }
        }

        val notice = if (hitBufferLimit) {
            "Buffer aktif dibatasi hingga ${trainSequences.size + valSequences.size} sequence (~160.000 token) untuk stabilitas memori RAM ponsel."
        } else null

        val meta = DatasetMetadata(
            name = fileName,
            format = format,
            sizeBytes = fileSize,
            totalLines = lineCount,
            totalTokens = totalTokens,
            trainCount = trainSequences.size,
            valCount = valSequences.size,
            sampleSnippets = snippets,
            isTruncated = hitBufferLimit,
            truncationNotice = notice
        )
        metadata = meta
        return meta
    }

    /**
     * Memuat dataset preset Bahasa Indonesia kaya (30 kalimat beragam):
     * Menghasilkan multi-sequence untuk train dan validasi tanpa kebocoran data.
     */
    fun loadPresetIndonesianDataset(valSplitRatio: Float = 0.15f): DatasetMetadata {
        trainSequences.clear()
        valSequences.clear()

        val rawTexts = listOf(
            "Nexora adalah model kecerdasan buatan on-device berbasis transformer untuk pemrosesan bahasa alami.",
            "Pelatihan model bahasa lokal memungkinkan pembelajaran privat tanpa mengirim data sensitif ke server eksternal.",
            "Teknologi LoRA melakukan adaptasi bobot dengan dekomposisi matriks rank rendah yang sangat hemat konsumsi memori.",
            "Kecerdasan buatan membantu memecahkan masalah matematika sains dan rekayasa perangkat lunak modern.",
            "Sistem operasi Android mengoptimalkan komputasi vektor CPU multi-core dengan thread prioritas tinggi.",
            "Model bahasa autoregresif memprediksi token berikutnya berdasarkan distribusi probabilitas softmax.",
            "Bahasa Indonesia memiliki struktur tata bahasa yang kaya dan fleksibel untuk dilatih oleh neural network.",
            "Dengan optimasi tensor array memori heap Android tetap stabil tanpa latensi garbage collector.",
            "Algoritma AdamW memperbarui parameter dengan momentum dan weight decay agar konvergensi optimal.",
            "Kinerja komputasi diukur dalam satuan GFLOPS dan throughput token per detik secara transparan.",
            "Pemrosesan bahasa alami menggunakan representasi vektor kontinu untuk menangkap makna semantik kata.",
            "Mekanisme multi-head causal attention membagi representasi ke dalam beberapa ruang proyeksi paralel.",
            "Layer normalisasi RMSNorm mempertahankan kestabilan gradien sepanjang lapisan jaringan neural yang dalam.",
            "Fungsi aktivasi non-linear seperti ReLU dan SiLU memungkinkan pemetaan representasi non-linear yang kompleks.",
            "Checkpointing berkala melindungi progres komputasi dari gangguan tak terduga seperti baterai habis.",
            "Evaluasi cross-entropy loss mengukur ketepatan prediksi probabilitas terhadap token target sejati.",
            "Perangkat seluler masa kini memiliki akselerator vektor NEON yang mampu mengeksekusi operasi SIMD dengan cepat.",
            "Teknik gradient clipping mencegah fenomena ledakan gradien pada pelatihan transformer bertingkat.",
            "Jadwal laju pembelajaran cosine annealing menurunkan learning rate secara halus hingga akhir masa latih.",
            "Kosa kata byte-level UTF-8 menjamin seluruh aksen karakter dan emoji dapat diproses tanpa mengalami kerusakan.",
            "Tokenisasi subword memecah kata majemuk menjadi unit bermakna yang sering muncul dalam korpus teks.",
            "Perkalian matriks terblokir meningkatkan efisiensi pemanfaatan cache CPU L1 dan L2 secara signifikan.",
            "Pengurangan presisi perhitungan tensor menjaga penggunaan bandwidth memori tetap berada dalam batas hemat daya.",
            "Privasi data pengguna terlindungi sepenuhnya karena seluruh bobot dan gradien berada di memori lokal ponsel.",
            "Inovasi model AI ringkas membuktikan bahwa perangkat mobile sanggup menjalankan inferensi dan training mandiri.",
            "Optimasi arsitektur neural network memungkinkan eksperimen sains kecerdasan buatan kapan saja dan di mana saja.",
            "Pengembang aplikasi dapat mengadaptasi model bahasa untuk kebutuhan spesifik seperti asisten tugas dan penulisan.",
            "Pengujian finite difference memastikan turunan matematis backpropagation terbukti valid dan presisi secara numerik.",
            "Desain antarmuka modern Material Design 3 memberikan visualisasi telemetri hardware yang informatif dan elegan.",
            "Teknologi masa depan berfokus pada efisiensi energi kecerdasan komputasi terdesentralisasi di perangkat pengguna."
        )

        var totalBytes = 0L
        var totalTokens = 0
        val snippets = ArrayList<String>()

        for (t in rawTexts) {
            totalBytes += t.toByteArray(StandardCharsets.UTF_8).size
            tokenizer.learnTokensFromText(t, maxNewTokens = 25)
            if (snippets.size < 5) snippets.add(t)
        }

        // Pisahkan kalimat secara deterministik: 85% Train, 15% Validation
        val rng = Random(42)
        val shuffled = ArrayList(rawTexts)
        Collections.shuffle(shuffled, rng)

        val valCount = maxOf(4, (shuffled.size * valSplitRatio).toInt())
        val valTexts = shuffled.take(valCount)
        val trainTexts = shuffled.drop(valCount)

        for (t in trainTexts) {
            val tokenIds = tokenizer.encode(t, addBos = true, addEos = true)
            totalTokens += tokenIds.size
            if (tokenIds.size >= 2) {
                var start = 0
                while (start < tokenIds.size) {
                    val end = minOf(tokenIds.size, start + 32)
                    val subSeq = tokenIds.copyOfRange(start, end)
                    if (subSeq.size >= 2) {
                        trainSequences.add(subSeq)
                    }
                    start += 16
                }
            }
        }

        for (t in valTexts) {
            val tokenIds = tokenizer.encode(t, addBos = true, addEos = true)
            totalTokens += tokenIds.size
            if (tokenIds.size >= 2) {
                var start = 0
                while (start < tokenIds.size) {
                    val end = minOf(tokenIds.size, start + 32)
                    val subSeq = tokenIds.copyOfRange(start, end)
                    if (subSeq.size >= 2) {
                        valSequences.add(subSeq)
                    }
                    start += 16
                }
            }
        }

        val meta = DatasetMetadata(
            name = "Nexora-ID-Core.jsonl",
            format = "JSONL",
            sizeBytes = totalBytes,
            totalLines = rawTexts.size,
            totalTokens = totalTokens,
            trainCount = trainSequences.size,
            valCount = valSequences.size,
            sampleSnippets = snippets,
            isTruncated = false,
            truncationNotice = null
        )
        metadata = meta
        return meta
    }

    private fun extractTextFromLine(rawLine: String, format: String): String {
        val trimmed = rawLine.trim()
        if (trimmed.isEmpty()) return ""

        return when (format) {
            "JSONL" -> {
                try {
                    val obj = JSONObject(trimmed)
                    when {
                        obj.has("text") -> obj.optString("text")
                        obj.has("prompt") && obj.has("response") ->
                            "${obj.optString("prompt")} ${obj.optString("response")}"
                        obj.has("input") && obj.has("output") ->
                            "${obj.optString("input")} ${obj.optString("output")}"
                        obj.has("content") -> obj.optString("content")
                        else -> {
                            val firstKey = obj.keys().asSequence().firstOrNull()
                            if (firstKey != null) obj.optString(firstKey) else trimmed
                        }
                    }
                } catch (e: Exception) {
                    trimmed
                }
            }
            "CSV" -> {
                val columns = ArrayList<String>()
                var inQuotes = false
                val cur = StringBuilder()
                for (c in trimmed) {
                    when {
                        c == '\"' -> inQuotes = !inQuotes
                        c == ',' && !inQuotes -> {
                            columns.add(cur.toString().trim())
                            cur.clear()
                        }
                        else -> cur.append(c)
                    }
                }
                columns.add(cur.toString().trim())
                columns.maxByOrNull { it.length }?.trim('\"', ' ') ?: trimmed
            }
            else -> trimmed
        }
    }
}
