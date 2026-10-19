package com.Nexora.ai.data

import java.io.DataInputStream
import java.io.DataOutputStream

/**
 * Real on-device tokenizer: word-level vocab built from the imported dataset
 * + byte-level fallback. Vocab size capped at vocabSize (default 16000).
 * No external API, no downloaded vocab.
 */
class Tokenizer private constructor(
    val vocabSize: Int,
    val wordToId: Map<String, Int>,
    val idToWord: Array<String?>
) {
    val padId = 0
    val bosId = 1
    val eosId = 2
    val unkId = 3
    private val byteOffset = 4 // 4..259 = 256 byte fallback tokens

    fun encode(text: String, addBos: Boolean = true, addEos: Boolean = false): IntArray {
        val out = mutableListOf<Int>()
        if (addBos) out += bosId
        val pieces = splitToPieces(text)
        for (p in pieces) {
            val id = wordToId[p]
            if (id != null) {
                out += id
            } else {
                for (b in p.toByteArray(Charsets.UTF_8)) out += byteOffset + (b.toInt() and 0xFF)
            }
        }
        if (addEos) out += eosId
        return out.toIntArray()
    }

    fun decode(ids: IntArray): String {
        val bytes = mutableListOf<Byte>()
        val sb = StringBuilder()
        fun flush() {
            if (bytes.isNotEmpty()) {
                sb.append(String(bytes.toByteArray(), Charsets.UTF_8))
                bytes.clear()
            }
        }
        for (id in ids) {
            when {
                id == bosId || id == eosId || id == padId -> { flush(); }
                id == unkId -> { flush(); sb.append('\uFFFD') }
                id in byteOffset until byteOffset + 256 -> bytes += (id - byteOffset).toByte()
                else -> { flush(); val w = idToWord[id]; if (w != null) sb.append(w) }
            }
        }
        flush()
        return sb.toString()
    }

    fun save(out: DataOutputStream) {
        out.writeInt(vocabSize)
        out.writeInt(idToWord.size)
        for (w in idToWord) {
            out.writeBoolean(w != null)
            if (w != null) out.writeUTF(w)
        }
    }

    companion object {
        const val SPECIAL = 4
        const val BYTES = 256

        private val PIECE_REGEX = Regex("""[A-Za-z0-9]+|\s+|[^\sA-Za-z0-9]+""")

        fun splitToPieces(text: String): List<String> = PIECE_REGEX.findAll(text).map { it.value }.toList()

        /** Build vocab from frequency map; keeps top (vocabSize - SPECIAL - BYTES) words. */
        fun build(freq: Map<String, Int>, vocabSize: Int): Tokenizer {
            val wordSlots = vocabSize - SPECIAL - BYTES
            val top = freq.entries.sortedByDescending { it.value }.take(wordSlots).map { it.key }
            val wordToId = HashMap<String, Int>(top.size * 2)
            val idToWord = arrayOfNulls<String>(vocabSize)
            idToWord[0] = "<pad>"; idToWord[1] = "<bos>"; idToWord[2] = "<eos>"; idToWord[3] = "<unk>"
            var id = SPECIAL + BYTES
            for (w in top) { wordToId[w] = id; idToWord[id] = w; id++ }
            return Tokenizer(vocabSize, wordToId, idToWord)
        }

        fun load(input: DataInputStream): Tokenizer {
            val vocabSize = input.readInt()
            val n = input.readInt()
            val idToWord = arrayOfNulls<String>(n)
            val wordToId = HashMap<String, Int>()
            for (i in 0 until n) {
                if (input.readBoolean()) {
                    val w = input.readUTF()
                    idToWord[i] = w
                    if (i >= SPECIAL + BYTES) wordToId[w] = i
                }
            }
            return Tokenizer(vocabSize, wordToId, idToWord)
        }
    }
}
