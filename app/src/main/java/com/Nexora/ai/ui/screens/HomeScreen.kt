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
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavController
import com.Nexora.ai.ml.NativeOps
import com.Nexora.ai.train.TrainingController
import com.Nexora.ai.ui.components.GlassCard
import com.Nexora.ai.ui.components.GradientLabel
import com.Nexora.ai.ui.components.KeyValueRow
import com.Nexora.ai.ui.components.LossChart
import com.Nexora.ai.ui.components.SectionTitle
import com.Nexora.ai.ui.theme.NexoraAmber
import com.Nexora.ai.ui.theme.NexoraMuted
import com.Nexora.ai.ui.theme.NexoraOnSurface
import com.Nexora.ai.ui.theme.NexoraPrimary

@Composable
fun HomeScreen(nav: NavController) {
    val s by TrainingController.state.collectAsState()
    Column(
        Modifier.fillMaxSize().padding(16.dp).verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        GradientLabel("NEXORA")
        Text("Latih model bahasa AI langsung di perangkat Anda — tanpa cloud, tanpa API, tanpa mock.",
            color = NexoraMuted, fontSize = 13.sp, modifier = Modifier.padding(bottom = 8.dp))

        GlassCard {
            KeyValueRow("Status", s.phase.name)
            KeyValueRow("Pesan", s.message.take(60))
            if (s.totalParams > 0) {
                KeyValueRow("Total parameter", "%,d".format(s.totalParams))
                KeyValueRow("Trainable (LoRA + emb + norm)", "%,d".format(s.trainableParams))
                KeyValueRow("Vocab", "%,d".format(s.vocabSize))
                KeyValueRow("Token corpus", "%,d".format(s.numTokens))
            }
            KeyValueRow("CPU core aktif", NativeOps.nativeCpuCores().toString())
        }

        if (s.lossHistory.size > 1) {
            SectionTitle("Kurva loss terbaru")
            GlassCard { LossChart(s.lossHistory) }
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
