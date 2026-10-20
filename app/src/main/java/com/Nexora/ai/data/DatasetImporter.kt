package com.Nexora.ai.data

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
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
    private val TEXT_HEADERS = setOf(
        "text", "content", "message", "prompt", "input", "instruction",
        "output", "completion", "response", "answer", "question"
    )

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

        val displayName = try {
            context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
                if (cursor.moveToFirst()) cursor.getString(0) else null
            }
        } catch (_: Exception) {
            null
        }
        val name = (displayName?.takeIf { it.isNotBlank() } ?: uri.lastPathSegment.orEmpty())
            .substringAfterLast('/').substringAfterLast(':').lowercase()
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
        var csvHeaderResolved = false
        var csvTextColumns = emptyList<Int>()
        var csvTextLabels = emptyList<String>()

        val input = context.contentResolver.openInputStream(uri)
            ?: throw IllegalStateException("Tidak dapat membuka ulang dataset. Pilih file yang dapat dibaca dua kali.")

        input.use { stream ->
            BufferedReader(InputStreamReader(stream, Charsets.UTF_8), 1 shl 20).use { reader ->
                while (true) {
                    val line = (if (format == "csv" || format == "tsv") readCsvRecord(reader) else reader.readLine()) ?: break
                    lineNumber++
                    bytesRead += line.toByteArray(Charsets.UTF_8).size.toLong() + 1L
                    if (bytesRead > MAX_BYTES) {
                        throw IllegalArgumentException("Dataset melebihi batas 500 MiB saat dibaca.")
                    }

                    val text: String? = when (format) {
                        "jsonl" -> extractJsonText(line)
                        "csv", "tsv" -> {
                            val columns = parseCsvLine(line, delimiter)
                            if (!csvHeaderResolved) {
                                csvHeaderResolved = true
                                val matches = columns.mapIndexedNotNull { index, column ->
                                    index.takeIf { column.trim().lowercase() in TEXT_HEADERS }
                                }
                                if (matches.isNotEmpty()) {
                                    csvTextColumns = matches
                                    csvTextLabels = matches.map { columns[it].trim() }
                                    null // A recognized header row is metadata, not training text.
                                } else {
                                    // No recognized header: treat the first row as actual data.
                                    csvTextColumns = columns.indices.toList()
                                    csvTextLabels = emptyList()
                                    columns.filter { it.isNotBlank() }.joinToString("\n").ifBlank { null }
                                }
                            } else {
                                val values = csvTextColumns.mapIndexedNotNull { labelIndex, columnIndex ->
                                    val value = columns.getOrNull(columnIndex)?.takeIf { it.isNotBlank() }
                                        ?: return@mapIndexedNotNull null
                                    val label = csvTextLabels.getOrNull(labelIndex)
                                    if (label == null) value else "$label: $value"
                                }
                                values.joinToString("\n").ifBlank { null }
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

        fun stringValue(value: kotlinx.serialization.json.JsonElement?): String? {
            val primitive = value as? kotlinx.serialization.json.JsonPrimitive ?: return null
            return if (primitive.isString) primitive.content else null
        }

        fun flattenContent(value: kotlinx.serialization.json.JsonElement?): String? {
            if (value == null) return null
            stringValue(value)?.let { return it }
            return when (value) {
                is JsonArray -> value.mapNotNull { flattenContent(it) }
                    .filter { it.isNotBlank() }.joinToString("\n").ifBlank { null }
                is JsonObject -> {
                    val preferred = listOf("text", "content", "message", "value", "body")
                    preferred.firstNotNullOfOrNull { key ->
                        flattenContent(value[key])
                    } ?: value.values.mapNotNull { stringValue(it) }
                        .filter { it.isNotBlank() }.joinToString("\n").ifBlank { null }
                }
                else -> null
            }
        }

        return when (element) {
            is JsonObject -> {
                val preferred = listOf(
                    "text", "content", "message", "prompt", "instruction",
                    "input", "output", "completion", "response", "answer"
                )
                preferred.firstNotNullOfOrNull { key -> flattenContent(element[key]) }
                    ?: flattenContent(element["messages"])
                    ?: flattenContent(element["conversation"])
                    ?: element.values.mapNotNull { stringValue(it) }
                        .filter { it.isNotBlank() }.joinToString("\n").ifBlank { null }
            }
            is JsonArray -> flattenContent(element)
            else -> stringValue(element)
        }
    }


    /**
     * Read one logical CSV/TSV record, preserving newlines inside quoted fields.
     * The previous physical-line reader could silently split a valid multiline field
     * into unrelated examples. An unclosed quoted field is rejected instead.
     */
    private fun readCsvRecord(reader: BufferedReader): String? {
        val record = StringBuilder()
        var inQuotes = false
        var sawAny = false

        while (true) {
            val nextChar = reader.read()
            if (nextChar == -1) {
                if (!sawAny) return null
                if (inQuotes) {
                    throw IllegalArgumentException("CSV/TSV berisi field bertanda kutip yang tidak ditutup.")
                }
                return record.toString()
            }

            sawAny = true
            when (val ch = nextChar.toChar()) {
                '"' -> {
                    record.append(ch)
                    if (inQuotes) {
                        // Inside a quoted field, a doubled quote is an escaped literal quote.
                        reader.mark(1)
                        val lookahead = reader.read()
                        if (lookahead == '"'.code) {
                            record.append('"')
                        } else {
                            inQuotes = false
                            if (lookahead != -1) reader.reset()
                        }
                    } else {
                        inQuotes = true
                    }
                }
                '\n' -> {
                    if (inQuotes) record.append(ch) else return record.toString()
                }
                '\r' -> {
                    reader.mark(1)
                    val lookahead = reader.read()
                    if (inQuotes) {
                        record.append(ch)
                        if (lookahead == '\n'.code) {
                            record.append('\n')
                        } else if (lookahead != -1) {
                            reader.reset()
                        }
                    } else {
                        if (lookahead != -1 && lookahead != '\n'.code) reader.reset()
                        return record.toString()
                    }
                }
                else -> record.append(ch)
            }
        }
    }

    /** CSV/TSV record parser with quoted fields, embedded newlines, and escaped double quotes. */
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
