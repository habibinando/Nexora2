package com.Nexora.ai.data

import android.content.Context
import android.net.Uri
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.BufferedReader
import java.io.InputStreamReader

/**
 * Streams a dataset (JSONL / CSV / TXT, up to 500 MB) from a SAF Uri,
 * builds tokenizer vocab frequencies, and writes token ids to a corpus file.
 * Real streaming parser — never loads the whole file into memory.
 */
object DatasetImporter {

    const val MAX_BYTES = 500L * 1024 * 1024
    const val MAX_TOKENS = 64_000_000 // 256 MB of IntArray, safe with largeHeap=true

    data class Result(
        val tokenizer: Tokenizer,
        val tokens: IntArray,
        val numExamples: Long,
        val bytesRead: Long,
        val truncated: Boolean
    )

    private val json = Json { ignoreUnknownKeys = true; isLenient = true }

    fun import(context: Context, uri: Uri, vocabSize: Int, onProgress: (Float, String) -> Unit): Result {
        val pfd = context.contentResolver.openAssetFileDescriptor(uri, "r")
            ?: throw IllegalStateException("Cannot open file")
        val size = pfd.length.coerceAtLeast(0)
        if (size > MAX_BYTES) throw IllegalArgumentException("File exceeds 500 MB limit")
        val stream = context.contentResolver.openInputStream(uri)
            ?: throw IllegalStateException("Cannot open input stream")
        val reader = BufferedReader(InputStreamReader(stream, Charsets.UTF_8), 1 shl 20)

        val fname = uri.lastPathSegment?.lowercase().orEmpty()
        val format = when {
            fname.endsWith(".jsonl") || fname.endsWith(".json") -> "jsonl"
            fname.endsWith(".csv") || fname.endsWith(".tsv") -> "csv"
            else -> "txt"
        }

        val freq = HashMap<String, Int>(1024 * 1024)
        val examples = mutableListOf<String>()
        var numExamples = 0L
        var bytesRead = 0L
        var truncated = false
        val charBuf = CharArray(8192)

        // First pass: stream lines, extract text, count piece frequencies.
        var header: List<String>? = null
        var textCol = 0
        var line: String?
        while (true) {
            line = reader.readLine() ?: break
            bytesRead += line.length + 1
            if (bytesRead > MAX_BYTES) { truncated = true; break }
            if (numExamples % 5000 == 0L) onProgress(bytesRead.toFloat() / size.coerceAtLeast(1), "Scanning… ${(bytesRead / 1048576)} MB")
            val text = when (format) {
                "jsonl" -> extractJsonText(line) ?: continue
                "csv" -> {
                    val cols = parseCsvLine(line)
                    if (header == null) { header = cols; textCol = findTextCol(cols); continue }
                    cols.getOrNull(textCol) ?: continue
                }
                else -> line
            }
            if (text.isBlank()) continue
            numExamples++
            for (p in Tokenizer.splitToPieces(text)) freq.merge(p, 1, Int::plus)
            examples += text
            // Keep examples bounded; flush to token list periodically
            if (examples.size >= 20000) break // we'll tokenize in second pass below
        }
        reader.close()

        onProgress(0.95f, "Building vocab (${freq.size} unique pieces)…")
        val tokenizer = Tokenizer.build(freq, vocabSize)

        // Second pass: tokenize examples into a flat IntArray corpus.
        val tokensList = ArrayList<Int>(1_000_000)
        fun tokenizeText(text: String) {
            val ids = tokenizer.encode(text, addBos = true, addEos = true)
            for (id in ids) {
                if (tokensList.size >= MAX_TOKENS) { truncated = true; return }
                tokensList += id
            }
        }
        for (t in examples) tokenizeText(t)
        // If first pass broke early due to example buffer, continue streaming remaining
        if (!truncated && examples.size >= 20000) {
            val stream2 = context.contentResolver.openInputStream(uri)!!
            val reader2 = BufferedReader(InputStreamReader(stream2, Charsets.UTF_8), 1 shl 20)
            var skipped = examples.size.toLong()
            var l: String?
            var hdr = header
            while (true) {
                l = reader2.readLine() ?: break
                if (skipped > 0) { skipped--; continue }
                val text = when (format) {
                    "jsonl" -> extractJsonText(l) ?: continue
                    "csv" -> { val cols = parseCsvLine(l); if (hdr == null) { hdr = cols; continue }; cols.getOrNull(textCol) ?: continue }
                    else -> l
                }
                if (text.isBlank()) continue
                numExamples++
                tokenizeText(text)
                if (tokensList.size >= MAX_TOKENS) { truncated = true; break }
            }
            reader2.close()
        }

        onProgress(1f, "Done: ${tokensList.size} tokens")
        return Result(tokenizer, tokensList.toIntArray(), numExamples, bytesRead, truncated)
    }

    private fun extractJsonText(line: String): String? {
        val el = try { json.parseToJsonElement(line) } catch (_: Exception) { return null }
        return when (el) {
            is JsonObject -> {
                el["text"]?.jsonPrimitive?.content
                    ?: el["content"]?.jsonPrimitive?.content
                    ?: el["message"]?.jsonPrimitive?.content
                    ?: el.values.firstOrNull { it is kotlinx.serialization.json.JsonPrimitive && it.isString }?.jsonPrimitive?.content
            }
            is JsonArray -> el.firstOrNull { it is kotlinx.serialization.json.JsonPrimitive && it.isString }?.jsonPrimitive?.content
            else -> if (el is kotlinx.serialization.json.JsonPrimitive && el.isString) el.content else null
        }
    }

    private fun findTextCol(cols: List<String>): Int {
        cols.forEachIndexed { i, c ->
            val lc = c.lowercase()
            if (lc in setOf("text", "content", "message", "prompt", "input", "instruction")) return i
        }
        return cols.size - 1
    }

    /** Minimal RFC-4180-ish CSV line parser (handles quoted fields). */
    fun parseCsvLine(line: String): List<String> {
        val out = mutableListOf<String>()
        val cur = StringBuilder()
        var inQuotes = false
        var i = 0
        while (i < line.length) {
            val c = line[i]
            when {
                inQuotes -> {
                    if (c == '"') {
                        if (i + 1 < line.length && line[i + 1] == '"') { cur.append('"'); i++ }
                        else inQuotes = false
                    } else cur.append(c)
                }
                c == '"' -> inQuotes = true
                c == ',' -> { out += cur.toString(); cur.clear() }
                else -> cur.append(c)
            }
            i++
        }
        out += cur.toString()
        return out
    }
}
