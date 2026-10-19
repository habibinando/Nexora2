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
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.Nexora.ai.benchmark.BenchmarkResult
import com.Nexora.ai.benchmark.BenchmarkRunner
import com.Nexora.ai.ml.NativeOps
import com.Nexora.ai.train.TrainingController
import com.Nexora.ai.ui.AppSettings
import com.Nexora.ai.ui.components.GlassCard
import com.Nexora.ai.ui.components.GradientLabel
import com.Nexora.ai.ui.components.KeyValueRow
import com.Nexora.ai.ui.components.SectionTitle
import com.Nexora.ai.ui.theme.NexoraDanger
import com.Nexora.ai.ui.theme.NexoraMuted
import com.Nexora.ai.ui.theme.NexoraOnSurface
import com.Nexora.ai.ui.theme.NexoraPrimary
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

@Composable
fun LabScreen() {
    val context = LocalContext.current
    val cfg by AppSettings.modelCfg.collectAsState()
    val scope = rememberCoroutineScope()
    var benchBusy by remember { mutableStateOf(false) }
    var benchProgress by remember { mutableStateOf("") }
    var benchResult by remember { mutableStateOf<BenchmarkResult?>(null) }
    var gpuInfo by remember { mutableStateOf(NativeOps.nativeGpuInfo()) }

    Column(Modifier.fillMaxSize().padding(16.dp).verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        GradientLabel("Lab — Benchmark & Setelan")

        SectionTitle("Benchmark perangkat (real)")
        Button(
            onClick = {
                benchBusy = true
                scope.launch {
                    val r = withContext(Dispatchers.Default) {
                        BenchmarkRunner.run(context, cfg) { msg -> benchProgress = msg }
                    }
                    benchResult = r; gpuInfo = NativeOps.nativeGpuInfo(); benchBusy = false
                }
            },
            enabled = !benchBusy, modifier = Modifier.fillMaxWidth(),
            colors = ButtonDefaults.buttonColors(containerColor = NexoraPrimary, contentColor = androidx.compose.ui.graphics.Color.Black)
        ) { Text("Jalankan benchmark", fontWeight = FontWeight.Bold) }
        if (benchBusy) {
            LinearProgressIndicator(Modifier.fillMaxWidth())
            Text(benchProgress, color = NexoraMuted, fontSize = 12.sp)
        }
        benchResult?.let { r ->
            GlassCard {
                KeyValueRow("Perangkat", r.deviceModel)
                KeyValueRow("Android", r.androidVersion)
                KeyValueRow("CPU core", "${r.cpuCores}")
                KeyValueRow("CPU matmul", "%.2f GFLOPS".format(r.cpuGflops))
                KeyValueRow("GPU compute", if (r.gpuAvailable) "tersedia" else "tidak tersedia (fallback CPU)")
                KeyValueRow("GPU", r.gpuInfo.take(60))
                if (r.gpuGflops > 0) KeyValueRow("GPU matmul", "%.2f GFLOPS".format(r.gpuGflops))
                KeyValueRow("Throughput training", "%.1f token/detik".format(r.tokensPerSec))
                KeyValueRow("Waktu step (2 layer, B=2)", "%.1f ms".format(r.cpuMsPerStep))
                KeyValueRow("RAM total / tersedia", "${r.ramTotalMb} MB / ${r.ramAvailMb} MB")
                KeyValueRow("Parameter model (konfigurasi Anda)", "%,d".format(r.modelParams))
            }
        }
        Text("Info GPU: $gpuInfo", color = NexoraMuted, fontSize = 11.sp)

        SectionTitle("Arsitektur model (diterapkan pada impor berikutnya)")
        GlassCard {
            IntSlider("dModel (128)", cfg.dModel, 64..512) { AppSettings.updateModel { c -> c.copy(dModel = it) } }
            IntSlider("FFN hidden", cfg.ffnDim, 128..2048) { AppSettings.updateModel { c -> c.copy(ffnDim = it) } }
            IntSlider("Attention heads", cfg.numHeads, 2..16) { AppSettings.updateModel { c -> c.copy(numHeads = it) } }
            IntSlider("Jumlah layer (32)", cfg.numLayers, 1..32) { AppSettings.updateModel { c -> c.copy(numLayers = it) } }
            IntSlider("Context length (128)", cfg.seqLen, 32..256) { AppSettings.updateModel { c -> c.copy(seqLen = it) } }
            IntSlider("Vocab (16k)", cfg.vocabSize / 1000, 4..32) { AppSettings.updateModel { c -> c.copy(vocabSize = it * 1000) } }
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("LoRA aktif", color = NexoraOnSurface, fontSize = 13.sp, modifier = Modifier.weight(1f))
                Switch(checked = cfg.useLora, onCheckedChange = { AppSettings.updateModel { c -> c.copy(useLora = it) } })
            }
            if (cfg.useLora) {
                IntSlider("LoRA rank", cfg.loraRank, 1..32) { AppSettings.updateModel { c -> c.copy(loraRank = it) } }
                IntSlider("LoRA alpha", cfg.loraAlpha, 4..64) { AppSettings.updateModel { c -> c.copy(loraAlpha = it) } }
            }
            FloatSlider("Dropout", cfg.dropout, 0f..0.5f) { AppSettings.updateModel { c -> c.copy(dropout = it) } }
        }

        SectionTitle("Penyimpanan")
        OutlinedButton(
            onClick = { TrainingController.clearCheckpoint(); benchResult = null },
            modifier = Modifier.fillMaxWidth(),
            colors = ButtonDefaults.outlinedButtonColors(contentColor = NexoraDanger)
        ) { Text("Hapus checkpoint & reset model") }
        Text("largeHeap=true aktif di manifest. Checkpoint disimpan di penyimpanan internal aplikasi.",
            color = NexoraMuted, fontSize = 11.sp)
    }
}

@Composable
private fun IntSlider(label: String, value: Int, range: IntRange, onChange: (Int) -> Unit) {
    Text("$label: $value", color = NexoraOnSurface, fontSize = 13.sp)
    Slider(value = value.toFloat().coerceIn(range.first.toFloat(), range.last.toFloat()),
        valueRange = range.first.toFloat()..range.last.toFloat(),
        onValueChange = { onChange(it.toInt()) })
}

@Composable
private fun FloatSlider(label: String, value: Float, range: ClosedFloatingPointRange<Float>, onChange: (Float) -> Unit) {
    Text("$label: %.2f".format(value), color = NexoraOnSurface, fontSize = 13.sp)
    Slider(value = value.coerceIn(range), valueRange = range, onValueChange = onChange)
}
