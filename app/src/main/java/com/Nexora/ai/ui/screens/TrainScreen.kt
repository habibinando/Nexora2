package com.Nexora.ai.ui.screens

import android.Manifest
import android.app.NotificationManager
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.provider.Settings
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.Nexora.ai.ml.NativeOps
import com.Nexora.ai.train.Phase
import com.Nexora.ai.train.TrainingController
import com.Nexora.ai.train.TrainingService
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
import com.Nexora.ai.ui.theme.NexoraSurfaceVar
import com.Nexora.ai.ui.theme.NexoraPrimary
import java.util.Locale

private fun fmtMs(ms: Long): String {
    val seconds = ms.coerceAtLeast(0L) / 1000L
    return "%02d:%02d:%02d".format(seconds / 3600, (seconds % 3600) / 60, seconds % 60)
}

private fun notificationsAllowed(context: Context): Boolean {
    if (!NotificationManagerCompat.from(context).areNotificationsEnabled()) return false
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
        ContextCompat.checkSelfPermission(context, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
    ) return false
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
        val channel = context.getSystemService(NotificationManager::class.java)
            ?.getNotificationChannel(TrainingService.CHANNEL_ID)
        if (channel?.importance == NotificationManager.IMPORTANCE_NONE) return false
    }
    return true
}

private fun openNotificationSettings(context: Context) {
    val intent = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
        Intent(Settings.ACTION_APP_NOTIFICATION_SETTINGS)
            .putExtra(Settings.EXTRA_APP_PACKAGE, context.packageName)
    } else {
        Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.parse("package:${context.packageName}"))
    }
    intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
    context.startActivity(intent)
}

private fun fmtOperationMs(ms: Long): String =
    if (ms < 1000L) "$ms ms" else "%.2f dtk".format(Locale.US, ms / 1000.0)

@Composable
fun TrainScreen() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val state by TrainingController.state.collectAsState()
    val hp by AppSettings.hp.collectAsState()
    var notificationsEnabled by remember { mutableStateOf(notificationsAllowed(context)) }
    var notificationMessage by remember { mutableStateOf<String?>(null) }
    val notificationPermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        notificationsEnabled = notificationsAllowed(context)
        notificationMessage = if (granted && notificationsEnabled) {
            "Izin notifikasi aktif. Pemberitahuan training akan tampil selama layanan training berjalan."
        } else {
            "Izin belum aktif atau channel notifikasi dinonaktifkan. Buka pengaturan notifikasi aplikasi lalu aktifkan Nexora dan channel Training."
        }
    }
    DisposableEffect(lifecycleOwner, context) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_RESUME) notificationsEnabled = notificationsAllowed(context)
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
    }
    val stats = NativeOps.computeStats()
    val running = state.phase == Phase.RUNNING
    val paused = state.phase == Phase.PAUSED
    val settingsEnabled = state.phase !in setOf(Phase.RUNNING, Phase.PAUSED, Phase.IMPORTING)
    val completedBatchFraction = if (state.batchInEpoch > 0 && state.batchesInEpoch > 0) {
        (state.batchInEpoch - 1).coerceAtLeast(0).toFloat() + state.batchProgress.coerceIn(0f, 1f)
    } else 0f
    val runProgress = (state.currentEpoch - 1).coerceAtLeast(0).toFloat() +
        if (state.batchesInEpoch > 0) completedBatchFraction / state.batchesInEpoch else 0f
    val targetReached = state.epoch >= hp.epochs && state.numTokens > 0
    val progress = if (targetReached) 1f else (runProgress / hp.epochs.coerceAtLeast(1)).coerceIn(0f, 1f)
    val progressPercent = (progress * 100).toInt()
    val displayEpoch = if (state.currentEpoch <= state.epoch && state.epoch < hp.epochs) state.epoch + 1
        else maxOf(state.currentEpoch, state.epoch)
    val canStart = state.numTokens > 0 && !targetReached &&
        state.phase in setOf(Phase.READY, Phase.IDLE, Phase.FINISHED, Phase.EARLY_STOPPED, Phase.ERROR)

    Column(
        Modifier.fillMaxSize().padding(16.dp).verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        GradientLabel("Training")
        Text("Pantau update bobot, loss, validasi, dan operasi komputasi yang benar-benar berjalan.", color = NexoraMuted, fontSize = 13.sp)

        SectionTitle("Izin notifikasi training")
        GlassCard {
            Text(
                if (notificationsEnabled) {
                    "Notifikasi aplikasi aktif. Nexora akan memperbarui pemberitahuan saat training berjalan."
                } else {
                    "Notifikasi belum diizinkan atau channel Training dimatikan. Pada Android 13+, izin runtime harus disetujui; jika ditolak, notifikasi training mungkin tidak muncul di panel notifikasi."
                },
                color = if (notificationsEnabled) NexoraPrimary else NexoraMuted,
                fontSize = 12.sp
            )
            notificationMessage?.let { Text(it, color = NexoraMuted, fontSize = 11.sp) }
            if (!notificationsEnabled) {
                val runtimePermissionMissing = Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
                    ContextCompat.checkSelfPermission(context, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
                Button(
                    onClick = {
                        if (runtimePermissionMissing) {
                            notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
                        } else {
                            openNotificationSettings(context)
                        }
                    },
                    modifier = Modifier.fillMaxWidth(),
                    colors = ButtonDefaults.buttonColors(containerColor = NexoraPrimary, contentColor = Color.Black)
                ) { Text(if (runtimePermissionMissing) "Minta izin notifikasi" else "Periksa pengaturan notifikasi") }
            }
            OutlinedButton(
                onClick = { openNotificationSettings(context) },
                modifier = Modifier.fillMaxWidth()
            ) { Text("Buka pengaturan notifikasi aplikasi") }
        }

        GlassCard {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Column(Modifier.weight(1f)) {
                    Text("STATUS TRAINING", color = NexoraMuted, fontSize = 11.sp, fontWeight = FontWeight.SemiBold)
                    Text(
                        when (state.phase) {
                            Phase.IDLE -> "Belum dimulai"
                            Phase.IMPORTING -> "Menyiapkan dataset"
                            Phase.READY -> "Siap dilatih"
                            Phase.RUNNING -> "TRAINING AKTIF"
                            Phase.PAUSED -> "Dijeda"
                            Phase.EARLY_STOPPED -> "Early stop"
                            Phase.FINISHED -> "Selesai"
                            Phase.ERROR -> "Perlu perhatian"
                        },
                        color = phaseColor(state.phase),
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
                Text(progressPercent.toString() + "%", color = phaseColor(state.phase), fontSize = 22.sp, fontWeight = FontWeight.ExtraBold)
            }
            LinearProgressIndicator(
                progress = { progress },
                modifier = Modifier.fillMaxWidth(),
                color = phaseColor(state.phase),
                trackColor = NexoraSurfaceVar
            )
            Text(
                "Epoch " + displayEpoch + " / " + hp.epochs +
                    if (state.batchesInEpoch > 0) " · batch ${state.batchInEpoch}/${state.batchesInEpoch}" else "",
                color = NexoraOnSurface, fontSize = 12.sp, fontWeight = FontWeight.Medium
            )
            Text(state.message.ifBlank { "Impor dataset untuk menginisialisasi model dan tokenizer." }, color = NexoraMuted, fontSize = 12.sp)
            Text(
                "Operasi aktif: " + state.operation.ifBlank { "menunggu perintah training" },
                color = NexoraPrimary, fontSize = 13.sp, fontWeight = FontWeight.SemiBold, maxLines = 3
            )
            if (running || paused) {
                Text(
                    "Durasi tahap: ${fmtOperationMs(state.operationElapsedMs)} · bobot diperbarui ${state.step} kali",
                    color = NexoraOnSurface, fontSize = 12.sp, fontWeight = FontWeight.Medium
                )
                Text(
                    "Kemajuan batch aktif: ${(state.batchProgress.coerceIn(0f, 1f) * 100f).toInt()}% · progress ini perkiraan tahap komputasi, bukan persentase bobot yang sudah diperbarui",
                    color = NexoraMuted, fontSize = 11.sp
                )
                if (state.step == 0 && state.batchInEpoch > 0) {
                    Text(
                        "Belum ada update bobot yang selesai. Step pertama baru bertambah setelah backpropagation dan AdamW untuk batch ini selesai.",
                        color = NexoraAmber, fontSize = 11.sp
                    )
                }
            }

            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                if (!running && !paused) {
                    Button(
                        onClick = { TrainingController.startTraining(hp) },
                        enabled = canStart,
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(containerColor = NexoraPrimary, contentColor = Color.Black)
                    ) {
                        Text(
                            when {
                                state.phase == Phase.IMPORTING -> "Menyiapkan…"
                                state.step > 0 -> "Lanjutkan training"
                                state.phase == Phase.ERROR -> "Coba lagi"
                                else -> "Mulai training"
                            },
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
                if (running) {
                    OutlinedButton(onClick = { TrainingController.pause() }, modifier = Modifier.weight(1f)) { Text("Jeda") }
                }
                if (paused) {
                    Button(
                        onClick = { TrainingController.resume() },
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(containerColor = NexoraAmber, contentColor = Color.Black)
                    ) { Text("Lanjutkan", fontWeight = FontWeight.Bold) }
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
                Text("Belum ada dataset aktif. Buka tab Data untuk mengimpor TXT, CSV, atau JSONL terlebih dahulu.", color = NexoraMuted, fontSize = 11.sp)
            } else if (targetReached && state.phase != Phase.RUNNING && state.phase != Phase.PAUSED) {
                Text("Target epoch sudah tercapai. Naikkan Jumlah epoch untuk meneruskan training.", color = NexoraMuted, fontSize = 11.sp)
            }
        }

        SectionTitle("Metrik training aktual")
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            TrainingMetric("Loss terkini", formatLoss(state.loss), "Training batch terakhir", NexoraPrimary, Modifier.weight(1f))
            TrainingMetric("Validation loss", formatLoss(state.valLoss), "Data validasi", Color(0xFFA855F7), Modifier.weight(1f))
        }
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            TrainingMetric(
                "Kecepatan",
                if (state.tokensPerSec.isFinite() && state.tokensPerSec > 0f) "%.1f tok/s".format(state.tokensPerSec) else "—",
                "Throughput aktual", Color(0xFF10B981), Modifier.weight(1f)
            )
            TrainingMetric(
                "ETA",
                if (running || paused) if (state.etaMs > 0L) fmtMs(state.etaMs) else "Menghitung…" else "—",
                if (state.etaMs > 0L) "Perkiraan dari batch selesai" else "Belum cukup data untuk estimasi",
                NexoraAmber, Modifier.weight(1f)
            )
        }
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            TrainingMetric(
                "Parameter",
                if (state.totalParams > 0) "%,d".format(state.totalParams) else "—",
                "Trainable: " + if (state.trainableParams > 0) "%,d".format(state.trainableParams) else "—",
                NexoraOnSurface, Modifier.weight(1f)
            )
            TrainingMetric(
                "Norm gradien",
                if (state.gradNorm.isFinite() && state.step > 0) "%.4f".format(state.gradNorm) else "—",
                "Setelah backprop", NexoraMuted, Modifier.weight(1f)
            )
        }

        SectionTitle("Progres dan detail")
        GlassCard {
            KeyValueRow("Fase", state.phase.name)
            KeyValueRow("Operasi aktif", state.operation.ifBlank { "—" })
            KeyValueRow("Durasi tahap", fmtOperationMs(state.operationElapsedMs))
            KeyValueRow("Total optimizer updates", state.step.toString())
            KeyValueRow("Epoch selesai", state.epoch.toString() + " / " + hp.epochs)
            KeyValueRow("Loss", formatLoss(state.loss))
            KeyValueRow("Validation loss", formatLoss(state.valLoss))
            KeyValueRow("Best validation loss", formatLoss(state.bestValLoss))
            KeyValueRow("Learning rate", if (state.lr > 0f) "%.2e".format(state.lr) else "—")
            KeyValueRow("Waktu berjalan", fmtMs(state.elapsedMs))
            KeyValueRow("Estimasi tersisa", if (running || paused) fmtMs(state.etaMs) else "—")
            KeyValueRow("Checkpoint", "Setiap " + hp.saveEvery + " step")
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

        SectionTitle("Kurva loss")
        GlassCard {
            if (state.lossHistory.size > 1) {
                Text("Riwayat loss aktual • " + state.lossHistory.size + " titik terakhir", color = NexoraMuted, fontSize = 11.sp)
                LossChart(state.lossHistory)
                KeyValueRow("Loss pertama", formatLoss(state.lossHistory.first()))
                KeyValueRow("Loss terakhir", formatLoss(state.lossHistory.last()))
            } else {
                Text("Grafik muncul setelah beberapa update bobot. Loss tidak ditampilkan sebagai nol sebelum step dihitung.", color = NexoraMuted, fontSize = 12.sp)
            }
        }

        SectionTitle("Pengaturan training")
        Text(
            "Training melewati dataset per epoch; setiap epoch mengacak urutan sequence dan memperbarui bobot melalui backpropagation + AdamW. Pengaturan terkunci saat training.",
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
            LabeledSlider("Jumlah epoch", hp.epochs.toDouble(), 1.0..100.0, { "%.0f".format(it) }, enabled = settingsEnabled) {
                AppSettings.updateHp { p -> p.copy(epochs = it.toInt().coerceIn(1, 100)) }
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
            LabeledSlider("Weight decay", hp.weightDecay.toDouble(), 0.0..0.1, { "%.3f".format(it) }, enabled = settingsEnabled) {
                AppSettings.updateHp { p -> p.copy(weightDecay = it.toFloat().coerceIn(0f, 0.1f)) }
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

private fun phaseColor(phase: Phase): Color = when (phase) {
    Phase.RUNNING -> Color(0xFF10B981)
    Phase.PAUSED, Phase.IMPORTING -> NexoraAmber
    Phase.ERROR -> NexoraDanger
    Phase.FINISHED, Phase.EARLY_STOPPED -> Color(0xFF10B981)
    Phase.READY -> NexoraPrimary
    Phase.IDLE -> NexoraMuted
}

private fun formatLoss(value: Float): String =
    if (value.isFinite() && value != Float.MAX_VALUE) String.format(Locale.US, "%.5f", value) else "—"

@Composable
private fun TrainingMetric(
    label: String,
    value: String,
    subtitle: String,
    accent: Color,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .background(NexoraSurfaceVar, RoundedCornerShape(16.dp))
            .border(1.dp, Color.White.copy(alpha = 0.06f), RoundedCornerShape(16.dp))
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        Text(label, color = NexoraMuted, fontSize = 11.sp)
        Text(value, color = accent, fontSize = 17.sp, fontWeight = FontWeight.Bold, maxLines = 1)
        Text(subtitle, color = NexoraMuted, fontSize = 10.sp, maxLines = 2)
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
