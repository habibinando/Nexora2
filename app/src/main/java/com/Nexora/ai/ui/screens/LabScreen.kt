package com.Nexora.ai.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
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
import com.Nexora.ai.train.Phase
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
import kotlin.math.abs

@Composable
fun LabScreen() {
    val context = LocalContext.current
    val cfg by AppSettings.modelCfg.collectAsState()
    val train by TrainingController.state.collectAsState()
    val scope = rememberCoroutineScope()
    var benchBusy by remember { mutableStateOf(false) }
    var benchProgress by remember { mutableStateOf("Belum dijalankan") }
    var benchError by remember { mutableStateOf<String?>(null) }
    var benchResult by remember { mutableStateOf<BenchmarkResult?>(null) }
    var gpuInfo by remember { mutableStateOf("GPU belum diperiksa") }
    var confirmReset by remember { mutableStateOf(false) }
    var headsExpanded by remember { mutableStateOf(false) }

    LaunchedEffect(Unit) { gpuInfo = withContext(Dispatchers.IO) { NativeOps.gpuInfo() } }
    val validHeads = (2..16).filter { cfg.dModel % it == 0 }.ifEmpty { listOf(1) }

    Column(
        Modifier.fillMaxSize().padding(16.dp).verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        GradientLabel("Lab • Benchmark & Setelan")
        SectionTitle("Benchmark CPU/GPU")
        Text(
            "Tes CPU 512×512 dan GPU 512×512 dijalankan terpisah. Mini-training mengukur panggilan GEMM CPU/GPU yang benar-benar terjadi. Benchmark memakai CPU intensif; jangan jalankan bersamaan dengan training.",
            color = NexoraMuted, fontSize = 12.sp
        )
        Button(
            onClick = {
                benchBusy = true
                benchError = null
                benchProgress = "Menyiapkan benchmark…"
                scope.launch {
                    try {
                        val result = withContext(Dispatchers.Default) {
                            BenchmarkRunner.run(context, cfg) { message ->
                                scope.launch { benchProgress = message }
                            }
                        }
                        benchResult = result
                        gpuInfo = result.gpuInfo
                        benchProgress = "Selesai"
                    } catch (e: Exception) {
                        benchError = "Benchmark gagal: ${e.message ?: e::class.java.simpleName}"
                        benchProgress = "Gagal"
                    } finally {
                        benchBusy = false
                    }
                }
            },
            enabled = !benchBusy && train.phase != Phase.RUNNING && train.phase != Phase.PAUSED,
            modifier = Modifier.fillMaxWidth(),
            colors = ButtonDefaults.buttonColors(containerColor = NexoraPrimary, contentColor = androidx.compose.ui.graphics.Color.Black)
        ) { Text(if (benchBusy) "Benchmark berjalan…" else "Jalankan benchmark", fontWeight = FontWeight.Bold) }
        if (train.phase == Phase.RUNNING || train.phase == Phase.PAUSED) {
            Text("Benchmark dinonaktifkan saat training aktif/dijeda agar tidak berebut CPU dan memori.", color = NexoraMuted, fontSize = 12.sp)
        }
        if (benchBusy) {
            LinearProgressIndicator(Modifier.fillMaxWidth())
            Text("Sedang dikerjakan: $benchProgress", color = NexoraPrimary, fontSize = 12.sp)
        } else {
            Text("Status: $benchProgress", color = NexoraMuted, fontSize = 12.sp)
        }
        benchError?.let { Text(it, color = NexoraDanger, fontSize = 12.sp) }

        benchResult?.let { r ->
            SectionTitle("Hasil benchmark terakhir")
            GlassCard {
                KeyValueRow("Perangkat", r.deviceModel)
                KeyValueRow("Android", r.androidVersion)
                KeyValueRow("CPU thread tersedia", "${r.cpuCores}")
                KeyValueRow("CPU GEMM 512×512", "%.2f GFLOPS · %.2f ms".format(r.cpuGflops, r.cpuMsPerMatMul))
                KeyValueRow("GPU tersedia", if (r.gpuAvailable) "Ya — GLES compute aktif" else "Tidak — GEMM besar fallback CPU")
                KeyValueRow("GPU GEMM 512×512", if (r.gpuMsPerMatMul > 0) "%.2f GFLOPS · %.2f ms".format(r.gpuGflops, r.gpuMsPerMatMul) else "Tidak dijalankan")
                KeyValueRow("Mini-training step", "%.2f ms · %.1f token/detik".format(r.stepMsPerStep, r.tokensPerSec))
                KeyValueRow("Loss mini-training", "%.5f".format(r.stepLoss))
                KeyValueRow("GEMM CPU saat mini-training", "${r.stepCpuGemmCalls}")
                KeyValueRow("GEMM GPU berhasil saat mini-training", "${r.stepGpuGemmCalls}")
                KeyValueRow("GPU fallback ke CPU", "${r.stepGpuFallbacks}")
                KeyValueRow("Arsitektur mini-training", r.benchmarkModel)
                KeyValueRow("Perkiraan parameter model pilihan", "%,d".format(r.modelParams))
                KeyValueRow("RAM total / tersedia", "${r.ramTotalMb} MB / ${r.ramAvailMb} MB")
            }
        }
        Text("Info backend GPU: $gpuInfo", color = NexoraMuted, fontSize = 11.sp)

        SectionTitle("Arsitektur model")
        Text(
            "Pengaturan ini digunakan saat impor dataset berikutnya. Mengubahnya tidak mengubah bobot model atau checkpoint aktif. Dimensi embedding dibuat kelipatan 8 dan attention heads dibatasi ke pembagi yang valid.",
            color = NexoraMuted, fontSize = 12.sp
        )
        GlassCard {
            IntSlider("Dimensi embedding (kelipatan 8)", cfg.dModel, 64..512, step = 8) { requested ->
                val d = requested.coerceIn(64, 512)
                AppSettings.updateModel { c ->
                    val heads = (2..16).filter { d % it == 0 }.minByOrNull { abs(it - c.numHeads) } ?: 2
                    c.copy(dModel = d, numHeads = heads)
                }
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("Attention heads", color = NexoraOnSurface, fontSize = 13.sp, modifier = Modifier.weight(1f))
                Box {
                    OutlinedButton(onClick = { headsExpanded = true }) { Text("${cfg.numHeads} heads") }
                    DropdownMenu(expanded = headsExpanded, onDismissRequest = { headsExpanded = false }) {
                        validHeads.forEach { count ->
                            DropdownMenuItem(
                                text = { Text("$count heads") },
                                onClick = {
                                    AppSettings.updateModel { c -> c.copy(numHeads = count) }
                                    headsExpanded = false
                                }
                            )
                        }
                    }
                }
            }
            IntSlider("FFN hidden dimension", cfg.ffnDim, 128..2048, step = 32) {
                AppSettings.updateModel { c -> c.copy(ffnDim = it) }
            }
            IntSlider("Jumlah transformer layers", cfg.numLayers, 1..32) {
                AppSettings.updateModel { c -> c.copy(numLayers = it) }
            }
            IntSlider("Context window (token)", cfg.seqLen, 32..256, step = 32) {
                AppSettings.updateModel { c -> c.copy(seqLen = it) }
            }
            IntSlider("Vocabulary size (ribu token)", cfg.vocabSize / 1000, 4..32) {
                AppSettings.updateModel { c -> c.copy(vocabSize = it * 1000) }
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("LoRA aktif", color = NexoraOnSurface, fontSize = 13.sp, modifier = Modifier.weight(1f))
                Switch(
                    checked = cfg.useLora,
                    onCheckedChange = { enabled -> AppSettings.updateModel { c -> c.copy(useLora = enabled) } }
                )
            }
            if (cfg.useLora) {
                IntSlider("LoRA rank", cfg.loraRank, 1..32) { AppSettings.updateModel { c -> c.copy(loraRank = it) } }
                IntSlider("LoRA alpha", cfg.loraAlpha, 4..64) { AppSettings.updateModel { c -> c.copy(loraAlpha = it) } }
            }
            FloatSlider("Dropout", cfg.dropout, 0f..0.5f) { AppSettings.updateModel { c -> c.copy(dropout = it) } }
            KeyValueRow("Perkiraan parameter", "%,d".format(BenchmarkRunner.estimateParameterCount(cfg)))
        }

        SectionTitle("Penyimpanan model")
        OutlinedButton(
            onClick = { confirmReset = true },
            modifier = Modifier.fillMaxWidth(),
            colors = ButtonDefaults.outlinedButtonColors(contentColor = NexoraDanger)
        ) { Text("Hapus checkpoint & reset model") }
        Text("Ini menghapus bobot, optimizer state, tokenizer, dan metadata checkpoint. Dataset sumber tidak dihapus.", color = NexoraMuted, fontSize = 11.sp)
    }

    if (confirmReset) {
        AlertDialog(
            onDismissRequest = { confirmReset = false },
            title = { Text("Hapus checkpoint?") },
            text = { Text("Bobot hasil training dan metadata akan dihapus. Tindakan ini tidak bisa dibatalkan.") },
            confirmButton = {
                TextButton(onClick = {
                    TrainingController.clearCheckpoint()
                    benchResult = null
                    confirmReset = false
                }) { Text("Hapus checkpoint", color = NexoraDanger) }
            },
            dismissButton = { TextButton(onClick = { confirmReset = false }) { Text("Batal") } }
        )
    }
}

@Composable
private fun IntSlider(label: String, value: Int, range: IntRange, step: Int = 1, onChange: (Int) -> Unit) {
    Text("$label: $value", color = NexoraOnSurface, fontSize = 13.sp)
    val steps = if (step > 1) ((range.last - range.first) / step - 1).coerceAtLeast(0) else 0
    Slider(
        value = value.toFloat().coerceIn(range.first.toFloat(), range.last.toFloat()),
        valueRange = range.first.toFloat()..range.last.toFloat(),
        steps = steps,
        onValueChange = { onChange(((it.toInt() - range.first) / step * step + range.first).coerceIn(range)) }
    )
}

@Composable
private fun FloatSlider(label: String, value: Float, range: ClosedFloatingPointRange<Float>, onChange: (Float) -> Unit) {
    Text("$label: %.2f".format(value), color = NexoraOnSurface, fontSize = 13.sp)
    Slider(value = value.coerceIn(range), valueRange = range, onValueChange = onChange)
}
