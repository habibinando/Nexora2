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
    val s = ms / 1000
    return "%02d:%02d:%02d".format(s / 3600, (s % 3600) / 60, s % 60)
}

@Composable
fun TrainScreen() {
    val s by TrainingController.state.collectAsState()
    val hp by AppSettings.hp.collectAsState()
    val running = s.phase == Phase.RUNNING || s.phase == Phase.PAUSED
    Column(Modifier.fillMaxSize().padding(16.dp).verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        GradientLabel("Training")
        Text(s.message, color = NexoraMuted, fontSize = 12.sp)

        // Controls
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            if (!running) {
                Button(onClick = { TrainingController.startTraining(hp) },
                    enabled = s.phase == Phase.READY || s.phase == Phase.IDLE || s.phase == Phase.FINISHED || s.phase == Phase.EARLY_STOPPED,
                    modifier = Modifier.weight(1f),
                    colors = ButtonDefaults.buttonColors(containerColor = NexoraPrimary, contentColor = androidx.compose.ui.graphics.Color.Black)) {
                    Text(if (s.step > 0) "Lanjutkan" else "Mulai", fontWeight = FontWeight.Bold)
                }
            }
            if (s.phase == Phase.RUNNING) {
                OutlinedButton(onClick = { TrainingController.pause() }, modifier = Modifier.weight(1f)) { Text("Jeda") }
            }
            if (s.phase == Phase.PAUSED) {
                Button(onClick = { TrainingController.resume() }, modifier = Modifier.weight(1f),
                    colors = ButtonDefaults.buttonColors(containerColor = NexoraAmber, contentColor = androidx.compose.ui.graphics.Color.Black)) { Text("Resume") }
            }
            if (running) {
                OutlinedButton(onClick = { TrainingController.stop() }, modifier = Modifier.weight(1f),
                    colors = ButtonDefaults.outlinedButtonColors(contentColor = NexoraDanger)) { Text("Stop") }
            }
        }

        // Live stats
        SectionTitle("Detail training (real)")
        GlassCard {
            KeyValueRow("Step", "${s.step}")
            KeyValueRow("Loss", "%.5f".format(s.loss))
            KeyValueRow("Val loss", "%.5f".format(s.valLoss))
            KeyValueRow("Best val loss", "%.5f".format(if (s.bestValLoss == Float.MAX_VALUE) 0f else s.bestValLoss))
            KeyValueRow("Learning rate", "%.2e".format(s.lr))
            KeyValueRow("Grad norm", "%.4f".format(s.gradNorm))
            KeyValueRow("Tokens/detik", "%.1f".format(s.tokensPerSec))
            KeyValueRow("Elapsed", fmtMs(s.elapsedMs))
            KeyValueRow("ETA", fmtMs(s.etaMs))
            KeyValueRow("Checkpoint", "otomatis setiap ${hp.saveEvery} step")
        }

        if (s.lossHistory.size > 1) {
            SectionTitle("Kurva loss")
            GlassCard { LossChart(s.lossHistory) }
        }

        // Hyperparameters
        SectionTitle("Hyperparameter")
        GlassCard {
            LabeledSlider("Learning rate", hp.learningRate.toDouble(), 1e-5..1e-2, { "%.2e".format(it) }) {
                AppSettings.updateHp { p -> p.copy(learningRate = it.toFloat()) }
            }
            LabeledSlider("Batch size", hp.batchSize.toDouble(), 1.0..16.0, { "%.0f".format(it) }) {
                AppSettings.updateHp { p -> p.copy(batchSize = it.toInt()) }
            }
            LabeledSlider("Warmup steps", hp.warmupSteps.toDouble(), 0.0..500.0, { "%.0f".format(it) }) {
                AppSettings.updateHp { p -> p.copy(warmupSteps = it.toInt()) }
            }
            LabeledSlider("Max steps", hp.maxSteps.toDouble(), 500.0..20000.0, { "%.0f".format(it) }) {
                AppSettings.updateHp { p -> p.copy(maxSteps = it.toInt()) }
            }
            LabeledSlider("Early-stop patience", hp.patience.toDouble(), 1.0..20.0, { "%.0f".format(it) }) {
                AppSettings.updateHp { p -> p.copy(patience = it.toInt()) }
            }
            LabeledSlider("Grad clip", hp.gradClip.toDouble(), 0.5..5.0, { "%.1f".format(it) }) {
                AppSettings.updateHp { p -> p.copy(gradClip = it.toFloat()) }
            }
            LabeledSlider("Thread CPU", hp.numThreads.toDouble(), 0.0..NativeOps.nativeCpuCores().toDouble(), { if (it == 0.0) "semua core" else "%.0f".format(it) }) {
                AppSettings.updateHp { p -> p.copy(numThreads = it.toInt()) }
            }
        }
        Text("Fitur aktif: LoRA (rank dapat diatur di Lab), dropout, AdamW + cosine LR schedule, gradient clipping, early stopping, checkpoint periodik.",
            color = NexoraMuted, fontSize = 11.sp)
    }
}

@Composable
private fun LabeledSlider(label: String, value: Double, range: ClosedFloatingPointRange<Double>, fmt: (Double) -> String, onChange: (Double) -> Unit) {
    Text("$label: ${fmt(value)}", fontSize = 13.sp, color = com.Nexora.ai.ui.theme.NexoraOnSurface)
    val fr = range.start.toFloat()..range.endInclusive.toFloat()
    Slider(value = value.toFloat().coerceIn(fr), valueRange = fr, onValueChange = { onChange(it.toDouble()) })
}
