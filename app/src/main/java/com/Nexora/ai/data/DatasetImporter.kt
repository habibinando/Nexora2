package com.Nexora.ai.data

import android.content.Context
import android.net.Uri
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.BufferedReader
import java.io.InputStreamReader
import java.util.TreeMap
import java.util.LinkedHashSet

/**
 * Two-pass streaming dataset importer.
 * Pass one counts candidate vocabulary across the full file. Pass two reopens it and writes
 * token IDs into primitive storage. This avoids skipping data after 20,000 rows and avoids
 * a boxed ArrayList<Int>, which can consume many times more memory than the token IDs.
 */
object DatasetImporter {
    const val MAX_BYTES = 500L * 1024 * 1024
    const val MAX_TOKENS = 64_000_000

    data class Result(
        val tokenizer: Tokenizer,
        val tokens: IntArray,
        val numExamples: Long,
        val bytesRead: Long,
        val truncated: Boolean
    )

    private data class ScanSummary(
        val bytesRead: Long,
        val textRows: Long,
        val completedNormally: Boolean
    )

    private val json = Json { ignoreUnknownKeys = true; isLenient = true }

    /** Bounded Space-Saving counters keep frequent candidate pieces with bounded memory. */
    private class BoundedFrequencies(private val capacity: Int) {
        private data class Counter(var count: Int)
        private val counts = HashMap<String, Counter>(capacity)
        private val byCount = TreeMap<Int, LinkedHashSet<String>>()

        private fun addToBucket(piece: String, count: Int) {
            byCount.getOrPut(count) { LinkedHashSet() }.add(piece)
        }

        private fun removeFromBucket(piece: String, count: Int) {
            val bucket = byCount[count] ?: return
            bucket.remove(piece)
            if (bucket.isEmpty()) byCount.remove(count)
        }

        fun add(piece: String) {
            val current = counts[piece]
            if (current != null) {
                removeFromBucket(piece, current.count)
                if (current.count < Int.MAX_VALUE) current.count++
                addToBucket(piece, current.count)
                return
            }

            if (counts.size < capacity) {
                counts[piece] = Counter(1)
                addToBucket(piece, 1)
                return
            }

            val minimum = byCount.firstEntry() ?: return
            val bucket = minimum.value
            val victim = bucket.first()
            bucket.remove(victim)
            if (bucket.isEmpty()) byCount.remove(minimum.key)
            counts.remove(victim)

            val estimated = (minimum.key.toLong() + 1L)
                .coerceAtMost(Int.MAX_VALUE.toLong()).toInt()
            counts[piece] = Counter(estimated)
            addToBucket(piece, estimated)
        }

        fun toFrequencyMap(): Map<String, Int> = counts.mapValues { it.value.count }
    }

    /** Growable primitive integer storage. */
    private class IntArrayBuilder(private val maximum: Int) {
        private var data = IntArray(minOf(1 shl 16, maximum))
        var size: Int = 0
            private set

        fun add(value: Int): Boolean {
            if (size >= maximum) return false
            if (size == data.size) {
                val next = minOf(maximum, maxOf(size + 1, data.size * 3 / 2))
                if (next <= size) return false
                data = data.copyOf(next)
            }
            data[size++] = value
            return true
        }

        fun toIntArray(): IntArray = if (size == data.size) data else data.copyOf(size)
    }

    fun import(
        context: Context,
        uri: Uri,
        vocabSize: Int,
        tokenizerOverride: Tokenizer? = null,
        onProgress: (Float, String) -> Unit
    ): Result {
        require(vocabSize > Tokenizer.SPECIAL + Tokenizer.BYTES) {
            "Vocabulary harus lebih besar dari ${Tokenizer.SPECIAL + Tokenizer.BYTES} token."
        }

        val reportedSize = try {
            context.contentResolver.openAssetFileDescriptor(uri, "r")?.use { it.length } ?: -1L
        } catch (_: Exception) {
            -1L
        }
        if (reportedSize > MAX_BYTES) {
            throw IllegalArgumentException("Dataset melebihi batas 500 MiB.")
        }

        val name = uri.lastPathSegment?.substringAfterLast(':')?.lowercase().orEmpty()
        val format = when {
            name.endsWith(".jsonl") || name.endsWith(".json") -> "jsonl"
            name.endsWith(".csv") -> "csv"
            name.endsWith(".tsv") -> "tsv"
            else -> "txt"
        }
        val delimiter = if (format == "tsv") '\t' else ','

        val existingTokenizer = tokenizerOverride?.takeIf { it.vocabSize == vocabSize }
        val frequencies = if (existingTokenizer == null) {
            val candidateLimit = (vocabSize.toLong() * 8L).coerceIn(65_536L, 512_000L).toInt()
            BoundedFrequencies(candidateLimit)
        } else null

        onProgress(0.01f, "Pass 1/2 — membaca seluruh dataset dan menghitung kandidat kosakata…")
        val first = scanTextRows(
            context = context,
            uri = uri,
            format = format,
            delimiter = delimiter,
            expectedBytes = reportedSize,
            progressStart = 0.01f,
            progressEnd = 0.45f,
            phaseName = "Pass 1/2",
            onProgress = onProgress
        ) { text ->
            if (frequencies != null) {
                Tokenizer.forEachPiece(text) { piece -> frequencies.add(piece) }
            }
            true
        }

        if (first.textRows == 0L) {
            throw IllegalArgumentException("Dataset tidak berisi teks yang dapat dibaca. Periksa format dan kolom text/content.")
        }

        onProgress(0.48f, "Membuat tokenizer dari ${first.textRows} contoh teks…")
        val tokenizer = existingTokenizer ?: Tokenizer.build(frequencies!!.toFrequencyMap(), vocabSize)
        val corpusBuilder = IntArrayBuilder(MAX_TOKENS)
        var tokenizedRows = 0L

        onProgress(0.50f, "Pass 2/2 — tokenisasi seluruh dataset…")
        val second = scanTextRows(
            context = context,
            uri = uri,
            format = format,
            delimiter = delimiter,
            expectedBytes = reportedSize,
            progressStart = 0.50f,
            progressEnd = 0.98f,
            phaseName = "Pass 2/2",
            onProgress = onProgress
        ) { text ->
            val completeRow = tokenizer.encodeTo(text, addBos = true, addEos = true) { token ->
                corpusBuilder.add(token)
            }
            if (completeRow) tokenizedRows++
            completeRow
        }

        val tokens = corpusBuilder.toIntArray()
        if (tokens.isEmpty()) throw IllegalArgumentException("Tidak ada token yang berhasil dibuat dari dataset.")
        val truncated = !second.completedNormally
        val finalMessage = if (truncated) {
            "Batas $MAX_TOKENS token tercapai; dataset terpotong. Token tersimpan: ${tokens.size}."
        } else {
            "Selesai: ${tokens.size} token dari $tokenizedRows contoh teks."
        }
        onProgress(1f, finalMessage)

        return Result(
            tokenizer = tokenizer,
            tokens = tokens,
            numExamples = tokenizedRows,
            bytesRead = first.bytesRead,
            truncated = truncated
        )
    }

    private fun scanTextRows(
        context: Context,
        uri: Uri,
        format: String,
        delimiter: Char,
        expectedBytes: Long,
        progressStart: Float,
        progressEnd: Float,
        phaseName: String,
        onProgress: (Float, String) -> Unit,
        onText: (String) -> Boolean
    ): ScanSummary {
        var bytesRead = 0L
        var textRows = 0L
        var lineNumber = 0L
        var completedNormally = true
        var header: List<String>? = null
        var textColumn = 0

        val input = context.contentResolver.openInputStream(uri)
            ?: throw IllegalStateException("Tidak dapat membuka ulang dataset. Pilih file yang dapat dibaca dua kali.")

        input.use { stream ->
            BufferedReader(InputStreamReader(stream, Charsets.UTF_8), 1 shl 20).use { reader ->
                while (true) {
                    val line = reader.readLine() ?: break
                    lineNumber++
                    bytesRead += line.toByteArray(Charsets.UTF_8).size.toLong() + 1L
                    if (bytesRead > MAX_BYTES) {
                        throw IllegalArgumentException("Dataset melebihi batas 500 MiB saat dibaca.")
                    }

                    val text: String? = when (format) {
                        "jsonl" -> extractJsonText(line)
                        "csv", "tsv" -> {
                            val columns = parseCsvLine(line, delimiter)
                            if (header == null) {
                                header = columns
                                textColumn = findTextCol(columns)
                                null
                            } else {
                                columns.getOrNull(textColumn)
                            }
                        }
                        else -> line
                    }

                    if (!text.isNullOrBlank()) {
                        textRows++
                        if (!onText(text)) {
                            completedNormally = false
                            break
                        }
                    }

                    if (lineNumber % 1000L == 0L) {
                        val ratio = if (expectedBytes > 0L) {
                            (bytesRead.toFloat() / expectedBytes.toFloat()).coerceIn(0f, 1f)
                        } else {
                            (textRows.toFloat() / 100_000f).coerceIn(0f, 0.9f)
                        }
                        val progress = progressStart + (progressEnd - progressStart) * ratio
                        onProgress(progress.coerceIn(0f, 0.99f), "$phaseName — $textRows baris teks, $bytesRead byte dibaca")
                    }
                }
            }
        }
        return ScanSummary(bytesRead, textRows, completedNormally)
    }

    private fun extractJsonText(line: String): String? {
        val element = try {
            json.parseToJsonElement(line)
        } catch (_: Exception) {
            return null
        }
        return when (element) {
            is JsonObject -> {
                element["text"]?.jsonPrimitive?.content
                    ?: element["content"]?.jsonPrimitive?.content
                    ?: element["message"]?.jsonPrimitive?.content
                    ?: element.values.firstOrNull {
                        it is kotlinx.serialization.json.JsonPrimitive && it.isString
                    }?.jsonPrimitive?.content
            }
            is JsonArray -> element.firstOrNull {
                it is kotlinx.serialization.json.JsonPrimitive && it.isString
            }?.jsonPrimitive?.content
            else -> if (
                element is kotlinx.serialization.json.JsonPrimitive && element.isString
            ) element.content else null
        }
    }

    private fun findTextCol(columns: List<String>): Int {
        columns.forEachIndexed { index, column ->
            if (column.trim().lowercase() in setOf("text", "content", "message", "prompt", "input", "instruction")) {
                return index
            }
        }
        return (columns.size - 1).coerceAtLeast(0)
    }

    /** CSV/TSV line parser with quoted fields and escaped double quotes. */
    fun parseCsvLine(line: String, delimiter: Char = ','): List<String> {
        val out = mutableListOf<String>()
        val current = StringBuilder()
        var inQuotes = false
        var index = 0
        while (index < line.length) {
            val char = line[index]
            when {
                inQuotes -> {
                    if (char == '"') {
                        if (index + 1 < line.length && line[index + 1] == '"') {
                            current.append('"')
                            index++
                        } else {
                            inQuotes = false
                        }
                    } else {
                        current.append(char)
                    }
                }
                char == '"' -> inQuotes = true
                char == delimiter -> {
                    out += current.toString()
                    current.clear()
                }
                else -> current.append(char)
            }
            index++
        }
        out += current.toString()
        return out
    }
}
