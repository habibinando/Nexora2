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
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavController
import com.Nexora.ai.ml.NativeOps
import com.Nexora.ai.train.TrainingController
import com.Nexora.ai.train.Phase
import com.Nexora.ai.ui.AppSettings
import com.Nexora.ai.ui.components.GlassCard
import com.Nexora.ai.ui.components.GradientLabel
import com.Nexora.ai.ui.components.KeyValueRow
import com.Nexora.ai.ui.components.LossChart
import com.Nexora.ai.ui.components.SectionTitle
import com.Nexora.ai.ui.theme.NexoraAmber
import com.Nexora.ai.ui.theme.NexoraDanger
import com.Nexora.ai.ui.theme.NexoraMuted
import com.Nexora.ai.ui.theme.NexoraOnSurface
import com.Nexora.ai.ui.theme.NexoraPrimary

@Composable
fun HomeScreen(nav: NavController) {
    val s by TrainingController.state.collectAsState()
    val hp by AppSettings.hp.collectAsState()
    val progress = (s.step.toFloat() / hp.maxSteps.coerceAtLeast(1)).coerceIn(0f, 1f)
    val phaseColor = when (s.phase) {
        Phase.RUNNING -> Color(0xFF10B981)
        Phase.PAUSED, Phase.IMPORTING -> NexoraAmber
        Phase.ERROR -> NexoraDanger
        Phase.FINISHED, Phase.EARLY_STOPPED -> Color(0xFF10B981)
        Phase.READY -> NexoraPrimary
        Phase.IDLE -> NexoraMuted
    }
    Column(
        Modifier.fillMaxSize().padding(16.dp).verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        GradientLabel("NEXORA")
        Text("Latih model bahasa AI langsung di perangkat Anda — tanpa cloud, tanpa API, tanpa mock.",
            color = NexoraMuted, fontSize = 13.sp, modifier = Modifier.padding(bottom = 8.dp))

        GlassCard {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Column(Modifier.weight(1f)) {
                    Text("STATUS MODEL", color = NexoraMuted, fontSize = 11.sp, fontWeight = FontWeight.SemiBold)
                    Text(s.phase.name.replace('_', ' '), color = phaseColor, fontSize = 19.sp, fontWeight = FontWeight.Bold)
                }
                Text(if (s.totalParams > 0) "%,d".format(s.totalParams) else "Model belum dibuat", color = NexoraOnSurface, fontSize = 12.sp, fontWeight = FontWeight.Medium)
            }
            if (s.numTokens > 0) {
                LinearProgressIndicator(progress = { progress }, modifier = Modifier.fillMaxWidth(), color = phaseColor)
                KeyValueRow("Progres training", s.step.toString() + " / " + hp.maxSteps + " step")
                KeyValueRow("Loss", if (s.loss.isFinite()) "%.5f".format(s.loss) else "Belum ada step")
                KeyValueRow("Validation loss", if (s.valLoss.isFinite()) "%.5f".format(s.valLoss) else "Belum dievaluasi")
            }
            Text(s.message.ifBlank { "Impor dataset untuk mulai membuat model." }, color = NexoraMuted, fontSize = 12.sp)
            if (s.totalParams > 0) {
                KeyValueRow("Parameter trainable", "%,d".format(s.trainableParams))
                KeyValueRow("Vocabulary", "%,d".format(s.vocabSize))
                KeyValueRow("Token corpus", "%,d".format(s.numTokens))
            }
            KeyValueRow("CPU core tersedia", NativeOps.nativeCpuCores().toString())
            KeyValueRow("GEMM CPU / GPU", NativeOps.computeStats().cpuGemmCalls.toString() + " / " + NativeOps.computeStats().gpuGemmCalls)
        }

        SectionTitle("Kurva loss terbaru")
        GlassCard {
            if (s.lossHistory.size > 1) {
                LossChart(s.lossHistory)
                KeyValueRow("Loss terbaru", "%.5f".format(s.lossHistory.last()))
                KeyValueRow("Jumlah titik", s.lossHistory.size.toString())
            } else {
                Text("Belum ada grafik. Kurva akan terbentuk setelah lebih dari satu step training selesai.", color = NexoraMuted, fontSize = 12.sp)
            }
        }

        SectionTitle("Mulai cepat")
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = { nav.navigate("dataset") }, modifier = Modifier.weight(1f),
                colors = ButtonDefaults.buttonColors(containerColor = NexoraPrimary, contentColor = androidx.compose.ui.graphics.Color.Black)) {
                Text("Impor Dataset", fontWeight = FontWeight.Bold)
            }
            Button(onClick = { nav.navigate("train") }, modifier = Modifier.weight(1f),
                colors = ButtonDefaults.buttonColors(containerColor = NexoraAmber, contentColor = androidx.compose.ui.graphics.Color.Black)) {
                Text("Training", fontWeight = FontWeight.Bold)
            }
        }
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = { nav.navigate("chat") }, modifier = Modifier.weight(1f)) { Text("Chat dengan model") }
            Button(onClick = { nav.navigate("lab") }, modifier = Modifier.weight(1f)) { Text("Benchmark & Setelan") }
        }

        Text("Catatan: bobot model disimpan sebagai checkpoint setiap beberapa step dan otomatis dimuat ulang — bobot terus bertambah antar sesi.",
            color = NexoraMuted, fontSize = 11.sp, modifier = Modifier.padding(top = 8.dp))
    }
}
