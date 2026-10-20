package com.Nexora.ai.data

import com.Nexora.ai.ml.GPT
import com.Nexora.ai.ml.ModelConfig
import com.Nexora.ai.ml.AdamW
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream

/**
 * Persists the full model (including frozen base weights + LoRA adapters),
 * optimizer state, tokenizer, config and training metadata.
 * "Bobot terus bertambah": weights survive across sessions and training resumes.
 */
class CheckpointManager(private val dir: File) {

    init { dir.mkdirs() }

    data class Meta(
        val step: Int, val epoch: Int, val bestValLoss: Float,
        val patienceCount: Int, val lossHistory: List<Float>,
        val tokensSeen: Long
    )

    fun save(model: GPT, optimizer: AdamW, tokenizer: Tokenizer, meta: Meta) {
        DataOutputStream(FileOutputStream(File(dir, "config.bin"))).use { o ->
            val c = model.cfg
            o.writeInt(c.vocabSize); o.writeInt(c.seqLen); o.writeInt(c.dModel); o.writeInt(c.ffnDim)
            o.writeInt(c.numHeads); o.writeInt(c.numLayers); o.writeFloat(c.dropout)
            o.writeBoolean(c.useLora); o.writeInt(c.loraRank); o.writeInt(c.loraAlpha); o.writeFloat(c.layerNormEps)
        }
        DataOutputStream(FileOutputStream(File(dir, "tokenizer.bin"))).use { tokenizer.save(it) }
        DataOutputStream(FileOutputStream(File(dir, "tensors.bin"))).use { o ->
            for (t in model.allTensors()) { o.writeInt(t.size); for (v in t) o.writeFloat(v) }
        }
        DataOutputStream(FileOutputStream(File(dir, "optimizer.bin"))).use { o ->
            o.writeInt(optimizer.step)
            for ((m, v) in optimizer.state()) {
                o.writeInt(m.size); for (x in m) o.writeFloat(x)
                o.writeInt(v.size); for (x in v) o.writeFloat(x)
            }
        }
        DataOutputStream(FileOutputStream(File(dir, "meta.bin"))).use { o ->
            o.writeInt(meta.step); o.writeInt(meta.epoch); o.writeFloat(meta.bestValLoss)
            o.writeInt(meta.patienceCount); o.writeLong(meta.tokensSeen)
            o.writeInt(meta.lossHistory.size); for (l in meta.lossHistory) o.writeFloat(l)
        }
    }

    fun hasCheckpoint(): Boolean = File(dir, "config.bin").exists()

    fun loadConfig(): ModelConfig? {
        val f = File(dir, "config.bin")
        if (!f.exists()) return null
        return try {
            DataInputStream(FileInputStream(f)).use { i ->
                ModelConfig(
                    vocabSize = i.readInt(), seqLen = i.readInt(), dModel = i.readInt(),
                    ffnDim = i.readInt(), numHeads = i.readInt(), numLayers = i.readInt(),
                    dropout = i.readFloat(), useLora = i.readBoolean(),
                    loraRank = i.readInt(), loraAlpha = i.readInt(), layerNormEps = i.readFloat()
                )
            }
        } catch (_: Exception) { null }
    }

    fun loadTokenizer(): Tokenizer? = try {
        DataInputStream(FileInputStream(File(dir, "tokenizer.bin"))).use { Tokenizer.load(it) }
    } catch (_: Exception) { null }

    fun loadTensors(model: GPT): Boolean = try {
        DataInputStream(FileInputStream(File(dir, "tensors.bin"))).use { i ->
            for (t in model.allTensors()) {
                val n = i.readInt()
                require(n == t.size)
                for (idx in t.indices) t[idx] = i.readFloat()
            }
        }
        true
    } catch (_: Exception) { false }

    fun loadOptimizerState(optimizer: AdamW): Pair<Int, List<Pair<FloatArray, FloatArray>>>? = try {
        DataInputStream(FileInputStream(File(dir, "optimizer.bin"))).use { i ->
            val step = i.readInt()
            val state = optimizer.state().map { (m, v) ->
                val nm = FloatArray(i.readInt()) { i.readFloat() }
                val nv = FloatArray(i.readInt()) { i.readFloat() }
                nm to nv
            }
            step to state
        }
    } catch (_: Exception) { null }

    fun loadMeta(): Meta? = try {
        DataInputStream(FileInputStream(File(dir, "meta.bin"))).use { i ->
            val step = i.readInt(); val epoch = i.readInt(); val best = i.readFloat()
            val patience = i.readInt(); val tokens = i.readLong()
            val n = i.readInt(); val hist = List(n) { i.readFloat() }
            Meta(step, epoch, best, patience, hist, tokens)
        }
    } catch (_: Exception) { null }

    fun clear() { dir.listFiles()?.forEach { it.delete() } }
}
