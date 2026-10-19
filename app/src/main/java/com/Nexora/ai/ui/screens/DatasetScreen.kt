package com.Nexora.ai.ui.screens

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.Nexora.ai.train.Phase
import com.Nexora.ai.train.TrainingController
import com.Nexora.ai.ui.AppSettings
import com.Nexora.ai.ui.components.GlassCard
import com.Nexora.ai.ui.components.GradientLabel
import com.Nexora.ai.ui.components.KeyValueRow
import com.Nexora.ai.ui.components.SectionTitle
import com.Nexora.ai.ui.theme.NexoraMuted
import com.Nexora.ai.ui.theme.NexoraPrimary

@Composable
fun DatasetScreen() {
    val s by TrainingController.state.collectAsState()
    val cfg by AppSettings.modelCfg.collectAsState()
    var pickedUri by remember { mutableStateOf<Uri?>(null) }
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            pickedUri = uri
            TrainingController.importDataset(uri, cfg)
        }
    }
    Column(Modifier.fillMaxSize().padding(16.dp).verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        GradientLabel("Dataset")
        Text("Impor file .jsonl / .csv / .txt (maks 500 MB). Vocab tokenizer dibangun otomatis dari dataset (maks ${cfg.vocabSize} token).",
            color = NexoraMuted, fontSize = 13.sp)

        Button(
            onClick = { launcher.launch(arrayOf("*/*")) },
            enabled = s.phase != Phase.IMPORTING,
            modifier = Modifier.fillMaxWidth(),
            colors = ButtonDefaults.buttonColors(containerColor = NexoraPrimary, contentColor = androidx.compose.ui.graphics.Color.Black)
        ) { Text("Pilih file dataset", fontWeight = FontWeight.Bold) }

        if (s.phase == Phase.IMPORTING) {
            LinearProgressIndicator(progress = { s.importProgress }, modifier = Modifier.fillMaxWidth())
            Text(s.message, color = NexoraMuted, fontSize = 12.sp)
        }

        SectionTitle("Format yang didukung")
        GlassCard {
            KeyValueRow("JSONL", "baris {\"text\": \"...\"} atau field string pertama")
            KeyValueRow("CSV", "kolom teks terdeteksi otomatis (text/content/message/…)")
            KeyValueRow("TXT", "setiap baris = satu contoh")
            KeyValueRow("Batas ukuran", "500 MB · token corpus dibatasi 64 jt")
        }

        if (s.numTokens > 0) {
            SectionTitle("Dataset aktif")
            GlassCard {
                KeyValueRow("Berkas", s.datasetName)
                KeyValueRow("Contoh", "%,d".format(s.numExamples))
                KeyValueRow("Token", "%,d".format(s.numTokens))
                KeyValueRow("Vocab", "%,d".format(s.vocabSize))
            }
        }
        Text("Setelah impor selesai, buka tab Train untuk mulai melatih model dengan bobot yang terus bertambah.",
            color = NexoraMuted, fontSize = 12.sp)
    }
}
