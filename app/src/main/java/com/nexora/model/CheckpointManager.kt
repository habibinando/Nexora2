package com.nexora.model

import android.content.Context
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.zip.CRC32

data class CheckpointInfo(
    val id: String,
    val name: String,
    val timestamp: Long,
    val dateFormatted: String,
    val step: Int,
    val epoch: Int,
    val trainLoss: Float?,
    val valLoss: Float?,
    val sizeBytes: Long,
    val filePath: String,
    val isManual: Boolean = false,
    val isBestVal: Boolean = false,
    val trainingMode: String = "FULL_PARAMETERS"
)

data class LoadedCheckpointState(
    val step: Int,
    val epoch: Int,
    val dataIndex: Int,
    val currentLr: Float,
    val adamStepCount: Long,
    val trainingMode: TrainingMode
)

/**
 * Manajer Checkpoint Pelatihan On-Device Nexora untuk Arsitektur LM5M:
 * - Penulisan & Pembacaan Atomik dengan Verifikasi Checksum CRC32.
 * - Menyimpan SELURUH 5,065,472 bobot dasar LM5M (tokEmb, attnNorm, wq, wk, wv, wo, ffnNorm, w1, w3, w2, finalNorm)
 *   serta bobot adapter LoRA dan optimizer moments.
 * - Menyimpan & memulihkan step, epoch, adamStepCount, learning rate, dan posisi data index.
 * - Kebijakan retensi cerdas: Checkpoint Manual dan Best-Val tidak akan pernah dihapus otomatis.
 */
object CheckpointManager {

    private const val DIR_NAME = "nexora_checkpoints"
    private const val MAGIC_HEADER_V4 = 0x4E585234 // "NXR4"
    private const val MAX_RETAINED_AUTO_CHECKPOINTS = 5

    private fun getCheckpointDir(context: Context): File {
        val dir = File(context.filesDir, DIR_NAME)
        if (!dir.exists()) {
            dir.mkdirs()
        }
        return dir
    }

    /**
     * Menyimpan checkpoint secara atomik dengan checksum CRC32.
     */
    fun saveCheckpoint(
        context: Context,
        transformer: NexoraTransformer,
        tokenizer: NexoraTokenizer,
        step: Int,
        epoch: Int,
        trainLoss: Float?,
        valLoss: Float?,
        dataIndex: Int = 0,
        currentLr: Float = 1e-4f,
        customName: String? = null,
        isManual: Boolean = false,
        isBestVal: Boolean = false
    ): CheckpointInfo {
        val dir = getCheckpointDir(context)
        val timestamp = System.currentTimeMillis()
        val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())
        val dateStr = dateFormat.format(Date(timestamp))

        val prefix = when {
            isBestVal -> "ckpt_best"
            isManual -> "ckpt_manual_${timestamp}"
            else -> "ckpt_${timestamp}_step${step}"
        }
        val name = customName ?: if (isBestVal) "Best-Validation" else if (isManual) "Manual-Step$step" else "Checkpoint-Step$step"
        val fileName = "${prefix}.bin"
        val finalFile = File(dir, fileName)
        val tempFile = File(dir, "$fileName.tmp")

        // 1. Buat payload biner di memori untuk menghitung CRC32
        val payloadStream = ByteArrayOutputStream()
        DataOutputStream(payloadStream).use { dos ->
            dos.writeLong(timestamp)
            dos.writeInt(step)
            dos.writeInt(epoch)
            dos.writeInt(dataIndex)
            dos.writeFloat(currentLr)
            dos.writeFloat(trainLoss ?: -1f)
            dos.writeFloat(valLoss ?: -1f)
            dos.writeUTF(name)
            dos.writeBoolean(isManual)
            dos.writeBoolean(isBestVal)
            dos.writeLong(transformer.adamStepCount.get())
            dos.writeUTF(transformer.trainingMode.name)

            // Arsitektur Config
            dos.writeInt(transformer.config.vocabSize)
            dos.writeInt(transformer.config.dModel)
            dos.writeInt(transformer.config.dFf)
            dos.writeInt(transformer.config.contextLength)
            dos.writeInt(transformer.config.numLayers)
            dos.writeInt(transformer.config.loraRank)

            // Flag mode penyimpanan
            val isFull = (transformer.trainingMode == TrainingMode.FULL_PARAMETERS)
            dos.writeBoolean(isFull)

            // Bobot LM5M
            writeFloatArray(dos, transformer.tokEmb)
            writeFloatArray(dos, transformer.finalNorm)

            if (isFull) {
                writeFloatArray(dos, transformer.mTokEmb)
                writeFloatArray(dos, transformer.vTokEmb)
                writeFloatArray(dos, transformer.mFinalNorm)
                writeFloatArray(dos, transformer.vFinalNorm)
            }

            dos.writeInt(transformer.layers.size)
            for (layer in transformer.layers) {
                writeFloatArray(dos, layer.attnNorm)
                writeFloatArray(dos, layer.wq)
                writeFloatArray(dos, layer.wk)
                writeFloatArray(dos, layer.wv)
                writeFloatArray(dos, layer.wo)
                writeFloatArray(dos, layer.ffnNorm)
                writeFloatArray(dos, layer.w1)
                writeFloatArray(dos, layer.w3)
                writeFloatArray(dos, layer.w2)

                if (isFull) {
                    writeFloatArray(dos, layer.mAttnNorm); writeFloatArray(dos, layer.vAttnNorm)
                    writeFloatArray(dos, layer.mWq); writeFloatArray(dos, layer.vWq)
                    writeFloatArray(dos, layer.mWk); writeFloatArray(dos, layer.vWk)
                    writeFloatArray(dos, layer.mWv); writeFloatArray(dos, layer.vWv)
                    writeFloatArray(dos, layer.mWo); writeFloatArray(dos, layer.vWo)
                    writeFloatArray(dos, layer.mFfnNorm); writeFloatArray(dos, layer.vFfnNorm)
                    writeFloatArray(dos, layer.mW1); writeFloatArray(dos, layer.vW1)
                    writeFloatArray(dos, layer.mW3); writeFloatArray(dos, layer.vW3)
                    writeFloatArray(dos, layer.mW2); writeFloatArray(dos, layer.vW2)
                }

                // LoRA Adapters
                writeFloatArray(dos, layer.loraA_Q)
                writeFloatArray(dos, layer.loraB_Q)
                writeFloatArray(dos, layer.loraA_V)
                writeFloatArray(dos, layer.loraB_V)
            }
        }

        val payloadBytes = payloadStream.toByteArray()
        val crc = CRC32()
        crc.update(payloadBytes)
        val checksum = crc.value

        // 2. Tulis File Atomik (.tmp -> .bin)
        FileOutputStream(tempFile).use { fos ->
            DataOutputStream(fos).use { dos ->
                dos.writeInt(MAGIC_HEADER_V4)
                dos.writeInt(4) // Version 4 (LM5M)
                dos.writeLong(checksum)
                dos.writeInt(payloadBytes.size)
                dos.write(payloadBytes)
            }
        }

        if (tempFile.exists()) {
            if (finalFile.exists()) finalFile.delete()
            tempFile.renameTo(finalFile)
        }

        if (!isManual && !isBestVal) {
            applyRetentionPolicy(context)
        }

        return CheckpointInfo(
            id = fileName,
            name = name,
            timestamp = timestamp,
            dateFormatted = dateStr,
            step = step,
            epoch = epoch,
            trainLoss = trainLoss,
            valLoss = valLoss,
            sizeBytes = finalFile.length(),
            filePath = finalFile.absolutePath,
            isManual = isManual,
            isBestVal = isBestVal,
            trainingMode = transformer.trainingMode.name
        )
    }

    /**
     * Memuat checkpoint dengan verifikasi atomik CRC32 checksum.
     */
    fun loadCheckpoint(
        context: Context,
        transformer: NexoraTransformer,
        tokenizer: NexoraTokenizer,
        checkpoint: CheckpointInfo
    ): LoadedCheckpointState? {
        val file = File(checkpoint.filePath)
        if (!file.exists()) return null

        try {
            val fileBytes = file.readBytes()
            val bais = ByteArrayInputStream(fileBytes)
            val dis = DataInputStream(bais)

            val magic = dis.readInt()
            val version = dis.readInt()

            if (magic == MAGIC_HEADER_V4 && version == 4) {
                val expectedChecksum = dis.readLong()
                val payloadSize = dis.readInt()
                val payloadBytes = ByteArray(payloadSize)
                dis.readFully(payloadBytes)

                val crc = CRC32()
                crc.update(payloadBytes)
                if (crc.value != expectedChecksum) {
                    return null
                }

                val pDis = DataInputStream(ByteArrayInputStream(payloadBytes))
                pDis.readLong() // timestamp
                val step = pDis.readInt()
                val epoch = pDis.readInt()
                val dataIndex = pDis.readInt()
                val currentLr = pDis.readFloat()
                pDis.readFloat() // trainLoss
                pDis.readFloat() // valLoss
                pDis.readUTF()   // name
                pDis.readBoolean() // isManual
                pDis.readBoolean() // isBestVal
                val adamSteps = pDis.readLong()
                val modeName = pDis.readUTF()
                val mode = try { TrainingMode.valueOf(modeName) } catch (e: Exception) { TrainingMode.FULL_PARAMETERS }

                val vocabSize = pDis.readInt()
                val dModel = pDis.readInt()
                val dFf = pDis.readInt()
                val ctxLen = pDis.readInt()
                val numLayers = pDis.readInt()
                val rank = pDis.readInt()

                if (dModel != transformer.config.dModel) {
                    return null
                }

                val isFull = pDis.readBoolean()

                readFloatArray(pDis, transformer.tokEmb)
                readFloatArray(pDis, transformer.finalNorm)

                if (isFull) {
                    readFloatArray(pDis, transformer.mTokEmb)
                    readFloatArray(pDis, transformer.vTokEmb)
                    readFloatArray(pDis, transformer.mFinalNorm)
                    readFloatArray(pDis, transformer.vFinalNorm)
                }

                val layerCount = pDis.readInt()
                for (i in 0 until minOf(layerCount, transformer.layers.size)) {
                    val layer = transformer.layers[i]
                    readFloatArray(pDis, layer.attnNorm)
                    readFloatArray(pDis, layer.wq)
                    readFloatArray(pDis, layer.wk)
                    readFloatArray(pDis, layer.wv)
                    readFloatArray(pDis, layer.wo)
                    readFloatArray(pDis, layer.ffnNorm)
                    readFloatArray(pDis, layer.w1)
                    readFloatArray(pDis, layer.w3)
                    readFloatArray(pDis, layer.w2)

                    if (isFull) {
                        readFloatArray(pDis, layer.mAttnNorm); readFloatArray(pDis, layer.vAttnNorm)
                        readFloatArray(pDis, layer.mWq); readFloatArray(pDis, layer.vWq)
                        readFloatArray(pDis, layer.mWk); readFloatArray(pDis, layer.vWk)
                        readFloatArray(pDis, layer.mWv); readFloatArray(pDis, layer.vWv)
                        readFloatArray(pDis, layer.mWo); readFloatArray(pDis, layer.vWo)
                        readFloatArray(pDis, layer.mFfnNorm); readFloatArray(pDis, layer.vFfnNorm)
                        readFloatArray(pDis, layer.mW1); readFloatArray(pDis, layer.vW1)
                        readFloatArray(pDis, layer.mW3); readFloatArray(pDis, layer.vW3)
                        readFloatArray(pDis, layer.mW2); readFloatArray(pDis, layer.vW2)
                    }

                    readFloatArray(pDis, layer.loraA_Q)
                    readFloatArray(pDis, layer.loraB_Q)
                    readFloatArray(pDis, layer.loraA_V)
                    readFloatArray(pDis, layer.loraB_V)
                }

                transformer.adamStepCount.set(adamSteps)
                transformer.trainingMode = mode

                // Native training consumes the flattened JNI buffers; refresh them
                // after loading per-parameter arrays or it would train stale random weights.
                transformer.syncWeightsToFlat()
                transformer.syncMomentsToFlat()

                return LoadedCheckpointState(
                    step = step,
                    epoch = epoch,
                    dataIndex = dataIndex,
                    currentLr = currentLr,
                    adamStepCount = adamSteps,
                    trainingMode = mode
                )
            }
        } catch (e: Exception) {
            return null
        }
        return null
    }

    fun listCheckpoints(context: Context): List<CheckpointInfo> {
        val dir = getCheckpointDir(context)
        val files = dir.listFiles { f -> f.extension == "bin" } ?: return emptyList()
        val list = ArrayList<CheckpointInfo>()

        for (file in files) {
            try {
                FileInputStream(file).use { fis ->
                    DataInputStream(fis).use { dis ->
                        val magic = dis.readInt()
                        val version = dis.readInt()
                        if (magic == MAGIC_HEADER_V4 && version == 4) {
                            val checksum = dis.readLong()
                            val payloadSize = dis.readInt()
                            val payload = ByteArray(payloadSize)
                            dis.readFully(payload)

                            val pDis = DataInputStream(ByteArrayInputStream(payload))
                            val timestamp = pDis.readLong()
                            val step = pDis.readInt()
                            val epoch = pDis.readInt()
                            pDis.readInt() // dataIndex
                            pDis.readFloat() // lr
                            val trainLossRaw = pDis.readFloat()
                            val valLossRaw = pDis.readFloat()
                            val name = pDis.readUTF()
                            val isManual = pDis.readBoolean()
                            val isBestVal = pDis.readBoolean()
                            pDis.readLong() // adamSteps
                            val mode = pDis.readUTF()

                            val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())
                            list.add(
                                CheckpointInfo(
                                    id = file.name,
                                    name = name,
                                    timestamp = timestamp,
                                    dateFormatted = dateFormat.format(Date(timestamp)),
                                    step = step,
                                    epoch = epoch,
                                    trainLoss = if (trainLossRaw >= 0f) trainLossRaw else null,
                                    valLoss = if (valLossRaw >= 0f) valLossRaw else null,
                                    sizeBytes = file.length(),
                                    filePath = file.absolutePath,
                                    isManual = isManual,
                                    isBestVal = isBestVal,
                                    trainingMode = mode
                                )
                            )
                        }
                    }
                }
            } catch (e: Exception) {}
        }
        return list.sortedByDescending { it.timestamp }
    }

    fun deleteCheckpoint(checkpoint: CheckpointInfo): Boolean {
        return try {
            val file = File(checkpoint.filePath)
            if (file.exists()) file.delete() else false
        } catch (e: Exception) {
            false
        }
    }

    private fun applyRetentionPolicy(context: Context) {
        val all = listCheckpoints(context)
        val autoCheckpoints = all.filter { !it.isManual && !it.isBestVal }.sortedByDescending { it.timestamp }
        if (autoCheckpoints.size > MAX_RETAINED_AUTO_CHECKPOINTS) {
            val toDelete = autoCheckpoints.drop(MAX_RETAINED_AUTO_CHECKPOINTS)
            for (ckpt in toDelete) {
                deleteCheckpoint(ckpt)
            }
        }
    }

    private fun writeFloatArray(dos: DataOutputStream, arr: FloatArray) {
        dos.writeInt(arr.size)
        for (f in arr) {
            dos.writeFloat(f)
        }
    }

    private fun readFloatArray(dis: DataInputStream, arr: FloatArray) {
        val size = dis.readInt()
        val readCount = minOf(size, arr.size)
        for (i in 0 until readCount) {
            arr[i] = dis.readFloat()
        }
        if (size > arr.size) {
            for (i in arr.size until size) {
                dis.readFloat()
            }
        }
    }
}
