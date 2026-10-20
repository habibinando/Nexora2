package com.Nexora.ai.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.Nexora.ai.ml.NativeOps
import com.Nexora.ai.train.Phase
import com.Nexora.ai.train.TrainingController
import com.Nexora.ai.ui.AppSettings
import com.Nexora.ai.ui.components.GlassCard
import com.Nexora.ai.ui.components.GradientLabel
import com.Nexora.ai.ui.components.KeyValueRow
import com.Nexora.ai.ui.components.LossChart
import com.Nexora.ai.ui.components.SectionTitle
import com.Nexora.ai.ui.theme.NexoraAmber
import com.Nexora.ai.ui.theme.NexoraDanger
import com.Nexora.ai.ui.theme.NexoraMuted
import com.Nexora.ai.ui.theme.NexoraPrimary

private fun fmtMs(ms: Long): String {
    val seconds = ms.coerceAtLeast(0L) / 1000L
    return "%02d:%02d:%02d".format(seconds / 3600, (seconds % 3600) / 60, seconds % 60)
}

@Composable
fun TrainScreen() {
    val state by TrainingController.state.collectAsState()
    val hp by AppSettings.hp.collectAsState()
    val stats = NativeOps.computeStats()
    val running = state.phase == Phase.RUNNING
    val paused = state.phase == Phase.PAUSED
    val settingsEnabled = !running && !paused
    val canStart = state.numTokens > 0 &&
        state.phase in setOf(Phase.READY, Phase.IDLE, Phase.FINISHED, Phase.EARLY_STOPPED, Phase.ERROR)

    Column(
        Modifier.fillMaxSize().padding(16.dp).verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        GradientLabel("Training")
        Text(
            when (state.phase) {
                Phase.IDLE -> "Belum siap — impor dataset terlebih dahulu."
                Phase.IMPORTING -> "Dataset sedang diproses."
                Phase.READY -> "Model dan dataset siap untuk training."
                Phase.RUNNING -> "Training aktif di perangkat. Fase kerja ditampilkan di bawah."
                Phase.PAUSED -> "Training dijeda; tekan Resume untuk melanjutkan."
                Phase.EARLY_STOPPED -> "Training berhenti otomatis karena validation loss tidak membaik."
                Phase.FINISHED -> "Batas step tercapai; checkpoint akhir tersimpan."
                Phase.ERROR -> "Training mengalami masalah. Lihat pesan error di bawah."
            },
            color = NexoraMuted,
            fontSize = 12.sp
        )
        Text(state.message, color = NexoraPrimary, fontSize = 13.sp, fontWeight = FontWeight.Medium)
        if (running) LinearProgressIndicator(Modifier.fillMaxWidth())

        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            if (!running && !paused) {
                Button(
                    onClick = { TrainingController.startTraining(hp) },
                    enabled = canStart,
                    modifier = Modifier.weight(1f),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = NexoraPrimary,
                        contentColor = androidx.compose.ui.graphics.Color.Black
                    )
                ) {
                    Text(if (state.step > 0) "Lanjutkan training" else "Mulai training", fontWeight = FontWeight.Bold)
                }
            }
            if (running) {
                OutlinedButton(onClick = { TrainingController.pause() }, modifier = Modifier.weight(1f)) { Text("Jeda") }
            }
            if (paused) {
                Button(
                    onClick = { TrainingController.resume() },
                    modifier = Modifier.weight(1f),
                    colors = ButtonDefaults.buttonColors(containerColor = NexoraAmber, contentColor = androidx.compose.ui.graphics.Color.Black)
                ) { Text("Resume") }
            }
            if (running || paused) {
                OutlinedButton(
                    onClick = { TrainingController.stop() },
                    modifier = Modifier.weight(1f),
                    colors = ButtonDefaults.outlinedButtonColors(contentColor = NexoraDanger)
                ) { Text("Stop") }
            }
        }
        if (state.numTokens <= 0) {
            Text(
                "Checkpoint menyimpan bobot, bukan dataset. Setelah aplikasi dibuka ulang, impor ulang dataset sebelum melanjutkan training.",
                color = NexoraMuted,
                fontSize = 11.sp
            )
        }

        SectionTitle("Progres training")
        GlassCard {
            KeyValueRow("Fase", state.phase.name)
            KeyValueRow("Step selesai / target", "${state.step} / ${hp.maxSteps}")
            KeyValueRow("Loss", if (state.loss.isFinite()) "%.5f".format(state.loss) else "Belum ada step")
            KeyValueRow("Validation loss", if (state.valLoss.isFinite()) "%.5f".format(state.valLoss) else "Belum dievaluasi")
            KeyValueRow("Best validation loss", "%.5f".format(if (state.bestValLoss == Float.MAX_VALUE) 0f else state.bestValLoss))
            KeyValueRow("Learning rate", "%.2e".format(state.lr))
            KeyValueRow("Grad norm", "%.4f".format(state.gradNorm))
            KeyValueRow("Kecepatan", "%.1f token/detik".format(state.tokensPerSec))
            KeyValueRow("Waktu berjalan", fmtMs(state.elapsedMs))
            KeyValueRow("Estimasi tersisa", fmtMs(state.etaMs))
            KeyValueRow("Checkpoint", "Setiap ${hp.saveEvery} step")
        }

        SectionTitle("Backend komputasi yang benar-benar dipakai")
        GlassCard {
            KeyValueRow("GEMM CPU", "${stats.cpuGemmCalls} panggilan")
            KeyValueRow("GEMM GPU berhasil", "${stats.gpuGemmCalls} panggilan")
            KeyValueRow("Fallback GPU ke CPU", "${stats.gpuFallbacks} kali")
            Text(
                "Counter bergerak saat status training diperbarui. Operasi backward tertentu tetap berjalan di CPU; GPU tidak berarti seluruh training memakai GPU.",
                color = NexoraMuted,
                fontSize = 11.sp
            )
        }

        if (state.lossHistory.size > 1) {
            SectionTitle("Kurva loss")
            GlassCard { LossChart(state.lossHistory) }
        }

        SectionTitle("Pengaturan training")
        Text(
            "Pengaturan disimpan otomatis untuk sesi berikutnya. Max steps adalah jumlah update bobot, bukan jumlah epoch. Pengaturan dikunci selama training agar target tidak berubah diam-diam.",
            color = NexoraMuted,
            fontSize = 12.sp
        )
        GlassCard {
            LabeledSlider("Learning rate", hp.learningRate.toDouble(), 1e-5..1e-2, { "%.2e".format(it) }, enabled = settingsEnabled) {
                AppSettings.updateHp { p -> p.copy(learningRate = it.toFloat()) }
            }
            LabeledSlider("Batch size", hp.batchSize.toDouble(), 1.0..16.0, { "%.0f".format(it) }, enabled = settingsEnabled) {
                AppSettings.updateHp { p -> p.copy(batchSize = it.toInt().coerceIn(1, 16)) }
            }
            LabeledSlider("Warmup steps", hp.warmupSteps.toDouble(), 0.0..500.0, { "%.0f".format(it) }, enabled = settingsEnabled) {
                AppSettings.updateHp { p -> p.copy(warmupSteps = it.toInt().coerceIn(0, 500)) }
            }
            LabeledSlider("Max training steps", hp.maxSteps.toDouble(), 500.0..20000.0, { "%.0f".format(it) }, enabled = settingsEnabled) {
                AppSettings.updateHp { p -> p.copy(maxSteps = it.toInt().coerceIn(500, 20000)) }
            }
            LabeledSlider("Validation setiap N step", hp.evalEvery.toDouble(), 25.0..500.0, { "%.0f".format(it) }, enabled = settingsEnabled) {
                AppSettings.updateHp { p -> p.copy(evalEvery = it.toInt().coerceAtLeast(1)) }
            }
            LabeledSlider("Simpan checkpoint setiap N step", hp.saveEvery.toDouble(), 25.0..1000.0, { "%.0f".format(it) }, enabled = settingsEnabled) {
                AppSettings.updateHp { p -> p.copy(saveEvery = it.toInt().coerceAtLeast(1)) }
            }
            LabeledSlider("Early-stop patience", hp.patience.toDouble(), 1.0..20.0, { "%.0f".format(it) }, enabled = settingsEnabled) {
                AppSettings.updateHp { p -> p.copy(patience = it.toInt().coerceIn(1, 20)) }
            }
            LabeledSlider("Gradient clipping", hp.gradClip.toDouble(), 0.5..5.0, { "%.1f".format(it) }, enabled = settingsEnabled) {
                AppSettings.updateHp { p -> p.copy(gradClip = it.toFloat().coerceIn(0.5f, 5f)) }
            }
            LabeledSlider(
                "Thread CPU (0 = otomatis/semua core)",
                hp.numThreads.toDouble(),
                0.0..NativeOps.nativeCpuCores().toDouble(),
                { if (it == 0.0) "Otomatis" else "%.0f".format(it) },
                enabled = settingsEnabled
            ) {
                AppSettings.updateHp { p -> p.copy(numThreads = it.toInt().coerceIn(0, NativeOps.nativeCpuCores())) }
            }
        }
    }
}

@Composable
private fun LabeledSlider(
    label: String,
    value: Double,
    range: ClosedFloatingPointRange<Double>,
    fmt: (Double) -> String,
    enabled: Boolean = true,
    onChange: (Double) -> Unit
) {
    Text("$label: ${fmt(value)}", fontSize = 13.sp, color = com.Nexora.ai.ui.theme.NexoraOnSurface)
    val floatRange = range.start.toFloat()..range.endInclusive.toFloat()
    Slider(
        value = value.toFloat().coerceIn(floatRange),
        valueRange = floatRange,
        enabled = enabled,
        onValueChange = { onChange(it.toDouble()) }
    )
}
