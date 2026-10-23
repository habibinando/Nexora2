package com.nexora.ui.screens

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Bookmark
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.Notifications
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Save
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.nexora.model.CheckpointInfo
import com.nexora.model.LrSchedulerType
import com.nexora.model.OptimizerType
import com.nexora.model.TrainingConfig2026
import com.nexora.model.TrainingState
import com.nexora.ui.components.LossChart
import com.nexora.ui.theme.CyberBg
import com.nexora.ui.theme.CyberCard
import com.nexora.ui.theme.CyberCardBorder
import com.nexora.ui.theme.NeonAmber
import com.nexora.ui.theme.NeonCrimson
import com.nexora.ui.theme.NeonCyan
import com.nexora.ui.theme.NeonEmerald
import com.nexora.ui.theme.NeonViolet
import com.nexora.ui.theme.TextMuted
import com.nexora.ui.theme.TextPrimary
import com.nexora.ui.theme.TextSecondary
import java.text.NumberFormat
import java.util.Locale

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TrainingScreen(
    trainingState: TrainingState,
    trainingConfig: TrainingConfig2026,
    checkpoints: List<CheckpointInfo>,
    datasetMeta: com.nexora.data.DatasetMetadata? = null,
    onUpdateOptimizer: (OptimizerType) -> Unit,
    onUpdateLrScheduler: (LrSchedulerType) -> Unit,
    onUpdateTrainingMode: (com.nexora.model.TrainingMode) -> Unit,
    onStartTraining: (epochs: Int) -> Unit,
    onPauseTraining: () -> Unit,
    onStopTraining: () -> Unit,
    onSaveCheckpoint: (name: String) -> Unit,
    onLoadCheckpoint: (CheckpointInfo) -> Unit,
    onDeleteCheckpoint: (CheckpointInfo) -> Unit,
    onBack: () -> Unit,
    modifier: Modifier = Modifier
) {
    BackHandler(onBack = onBack)
    val context = LocalContext.current
    var selectedEpochs by remember { mutableFloatStateOf(3f) }
    var showSaveDialog by remember { mutableStateOf(false) }
    var checkpointNameInput by remember { mutableStateOf("") }
    val numFormat = NumberFormat.getNumberInstance(Locale.US)

    // Deteksi & Permohonan Izin Notifikasi (Android 13+ / API 33+)
    var hasNotifPermission by remember {
        mutableStateOf(
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                ContextCompat.checkSelfPermission(
                    context,
                    Manifest.permission.POST_NOTIFICATIONS
                ) == PackageManager.PERMISSION_GRANTED
            } else {
                true
            }
        )
    }

    val notifPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        hasNotifPermission = isGranted
    }

    Scaffold(
        modifier = modifier.fillMaxSize(),
        containerColor = CyberBg,
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        text = "Training & LoRA Engine 2026",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold,
                        color = TextPrimary
                    )
                },
                navigationIcon = {
                    IconButton(onClick = onBack, modifier = Modifier.testTag("training_back_button")) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "Kembali ke Beranda",
                            tint = TextPrimary
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(containerColor = CyberBg)
            )
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Banner Izin Notifikasi jika belum diberikan
            if (!hasNotifPermission && Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(12.dp),
                    colors = CardDefaults.cardColors(containerColor = CyberCard),
                    border = androidx.compose.foundation.BorderStroke(1.dp, NeonAmber)
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(14.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            imageVector = Icons.Default.Notifications,
                            contentDescription = "Notification Alert",
                            tint = NeonAmber,
                            modifier = Modifier.size(28.dp)
                        )
                        Spacer(modifier = Modifier.width(12.dp))
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = "Aktifkan Izin Notifikasi",
                                style = MaterialTheme.typography.titleSmall,
                                fontWeight = FontWeight.Bold,
                                color = TextPrimary
                            )
                            Text(
                                text = "Agar pelatihan dapat tetap berjalan di latar belakang (Foreground Service) dan status terlihat di notification bar saat layar mati.",
                                style = MaterialTheme.typography.bodySmall,
                                color = TextSecondary,
                                fontSize = 11.sp
                            )
                        }
                        Spacer(modifier = Modifier.width(8.dp))
                        Button(
                            onClick = { notifPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS) },
                            colors = ButtonDefaults.buttonColors(containerColor = NeonAmber),
                            shape = RoundedCornerShape(8.dp),
                            modifier = Modifier.testTag("grant_notif_button")
                        ) {
                            Text("Izinkan", color = CyberBg, fontWeight = FontWeight.Bold, fontSize = 12.sp)
                        }
                    }
                }
            }

            // Status Banner Utama
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = CyberCard),
                border = androidx.compose.foundation.BorderStroke(1.dp, CyberCardBorder)
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(14.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    val statusDotColor = when {
                        trainingState.isTraining && !trainingState.isPaused -> NeonEmerald
                        trainingState.isPaused -> NeonAmber
                        else -> TextMuted
                    }
                    Box(
                        modifier = Modifier
                            .size(10.dp)
                            .background(statusDotColor, CircleShape)
                    )
                    Spacer(modifier = Modifier.width(10.dp))
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = trainingState.statusMessage,
                            style = MaterialTheme.typography.bodyMedium,
                            fontWeight = FontWeight.SemiBold,
                            color = TextPrimary
                        )
                        if (trainingState.isTraining) {
                            Text(
                                text = "Epoch ${trainingState.currentEpoch}/${trainingState.totalEpochs} • Step ${trainingState.currentStep}/${trainingState.totalSteps} • LR: ${trainingState.currentLr?.let { "%.2e".format(it) } ?: "-"}",
                                style = MaterialTheme.typography.bodySmall,
                                color = NeonCyan,
                                fontFamily = FontFamily.Monospace
                            )
                        }
                    }

                    // Badge akselerasi: versi API diambil dari status Vulkan runtime, bukan angka tetap.
                    val detectedVulkanApi = trainingState.vulkanStatus
                        .takeIf { it.startsWith("Vulkan device API ") }
                        ?.substringAfter("Vulkan device API ")
                        ?.substringBefore(" |")
                    val accelBadgeText = when {
                        trainingState.isVulkanActive -> "Vulkan API ${detectedVulkanApi ?: "detected"}"
                        trainingState.isNdkActive -> "NDK SIMD C++20"
                        else -> "JVM Fallback"
                    }
                    val accelBadgeColor = when {
                        trainingState.isVulkanActive -> NeonEmerald
                        trainingState.isNdkActive -> NeonCyan
                        else -> NeonAmber
                    }
                    Box(
                        modifier = Modifier
                            .background(
                                accelBadgeColor.copy(alpha = 0.15f),
                                RoundedCornerShape(6.dp)
                            )
                            .border(
                                1.dp,
                                accelBadgeColor.copy(alpha = 0.4f),
                                RoundedCornerShape(6.dp)
                            )
                            .padding(horizontal = 8.dp, vertical = 4.dp)
                    ) {
                        Text(
                            text = accelBadgeText,
                            style = MaterialTheme.typography.labelSmall,
                            color = accelBadgeColor,
                            fontWeight = FontWeight.Bold,
                            fontFamily = FontFamily.Monospace,
                            fontSize = 10.sp
                        )
                    }
                }
            }

            // Metrik Pelatihan Nyata (Train Loss & Validation Loss 2026)
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = CyberCard),
                border = androidx.compose.foundation.BorderStroke(1.dp, CyberCardBorder)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        text = "Metrik Pelatihan & Validasi Nyata",
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.SemiBold,
                        color = TextPrimary
                    )
                    Spacer(modifier = Modifier.height(12.dp))

                    Row(modifier = Modifier.fillMaxWidth()) {
                        TrainingMetricBox(
                            label = "Train Loss (Cross-Entropy)",
                            value = trainingState.currentLoss?.let { "%.3f".format(it) } ?: "Belum ada",
                            unit = "Nats",
                            color = NeonViolet,
                            modifier = Modifier.weight(1f)
                        )
                        TrainingMetricBox(
                            label = "Validation Loss (Val Loss)",
                            value = trainingState.valLoss?.let { "%.3f".format(it) } ?: "Belum ada",
                            unit = "Data Uji Mandiri",
                            color = NeonCyan,
                            modifier = Modifier.weight(1f)
                        )
                    }

                    Spacer(modifier = Modifier.height(10.dp))

                    Row(modifier = Modifier.fillMaxWidth()) {
                        TrainingMetricBox(
                            label = "Train Perplexity (PPL)",
                            value = trainingState.currentPerplexity?.let { "%.2f".format(it) } ?: "Belum ada",
                            unit = "exp(train_loss)",
                            color = NeonViolet,
                            modifier = Modifier.weight(1f)
                        )
                        TrainingMetricBox(
                            label = "Val Perplexity (Val PPL)",
                            value = trainingState.valPerplexity?.let { "%.2f".format(it) } ?: "Belum ada",
                            unit = "exp(val_loss)",
                            color = NeonCyan,
                            modifier = Modifier.weight(1f)
                        )
                    }

                    Spacer(modifier = Modifier.height(10.dp))

                    Row(modifier = Modifier.fillMaxWidth()) {
                        TrainingMetricBox(
                            label = "Kecepatan Pelatihan",
                            value = trainingState.tokensPerSecond?.let { "%.1f".format(it) } ?: "Belum ada",
                            unit = "Token / Detik",
                            color = NeonEmerald,
                            modifier = Modifier.weight(1f)
                        )
                        TrainingMetricBox(
                            label = "Gradien Norm (||∇||)",
                            value = trainingState.gradientNorm?.let { "%.4f".format(it) } ?: "Belum ada",
                            unit = "L2 Norm",
                            color = NeonAmber,
                            modifier = Modifier.weight(1f)
                        )
                    }
                }
            }

            // Grafik Real-Time Loss (Menampilkan Train Loss & Val Loss)
            LossChart(
                lossHistory = trainingState.lossHistory,
                valLossHistory = trainingState.valLossHistory
            )

            // Konfigurasi Suite 2026: Optimizer & LR Schedule
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = CyberCard),
                border = androidx.compose.foundation.BorderStroke(1.dp, CyberCardBorder)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        text = "Konfigurasi Optimizer & LR Schedule 2026",
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.SemiBold,
                        color = TextPrimary
                    )

                    Spacer(modifier = Modifier.height(12.dp))

                    Text("Mode Pelatihan:", style = MaterialTheme.typography.labelSmall, color = TextSecondary)
                    Spacer(modifier = Modifier.height(6.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        listOf(com.nexora.model.TrainingMode.LORA_ADAPTER, com.nexora.model.TrainingMode.FULL_PARAMETERS).forEach { mode ->
                            FilterChip(
                                selected = trainingState.trainingMode == mode,
                                onClick = { onUpdateTrainingMode(mode) },
                                label = { Text(mode.displayName, fontSize = 11.sp) },
                                colors = FilterChipDefaults.filterChipColors(
                                    selectedContainerColor = NeonEmerald.copy(alpha = 0.2f),
                                    selectedLabelColor = NeonEmerald
                                ),
                                enabled = !trainingState.isTraining
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(12.dp))

                    Text("Optimizer:", style = MaterialTheme.typography.labelSmall, color = TextSecondary)
                    Spacer(modifier = Modifier.height(6.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        listOf(OptimizerType.ADAMW, OptimizerType.LION, OptimizerType.SGD_MOMENTUM).forEach { opt ->
                            FilterChip(
                                selected = trainingConfig.optimizerType == opt,
                                onClick = { onUpdateOptimizer(opt) },
                                label = { Text(opt.displayName, fontSize = 12.sp) },
                                colors = FilterChipDefaults.filterChipColors(
                                    selectedContainerColor = NeonViolet.copy(alpha = 0.2f),
                                    selectedLabelColor = NeonViolet
                                ),
                                enabled = !trainingState.isTraining
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(12.dp))

                    Text("Learning Rate Scheduler:", style = MaterialTheme.typography.labelSmall, color = TextSecondary)
                    Spacer(modifier = Modifier.height(6.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        listOf(LrSchedulerType.COSINE_ANNEALING, LrSchedulerType.LINEAR_DECAY).forEach { sch ->
                            FilterChip(
                                selected = trainingConfig.lrSchedulerType == sch,
                                onClick = { onUpdateLrScheduler(sch) },
                                label = { Text(if (sch == LrSchedulerType.COSINE_ANNEALING) "Cosine + Warmup" else "Linear Decay", fontSize = 11.sp) },
                                colors = FilterChipDefaults.filterChipColors(
                                    selectedContainerColor = NeonCyan.copy(alpha = 0.2f),
                                    selectedLabelColor = NeonCyan
                                ),
                                enabled = !trainingState.isTraining
                            )
                        }
                    }
                }
            }

            // Kontrol Pelatihan (Start / Pause / Stop)
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = CyberCard),
                border = androidx.compose.foundation.BorderStroke(1.dp, CyberCardBorder)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(
                            text = "Target Epoch: ${selectedEpochs.toInt()}",
                            style = MaterialTheme.typography.bodyMedium,
                            fontWeight = FontWeight.SemiBold,
                            color = TextPrimary
                        )
                        val stepsPerEpoch = datasetMeta?.trainCount ?: (if (trainingState.totalEpochs > 0) trainingState.totalSteps / trainingState.totalEpochs else 0)
                        val estText = if (stepsPerEpoch > 0) "${(selectedEpochs * stepsPerEpoch).toInt()} Step Est." else "Memerlukan Dataset"
                        Text(
                            text = estText,
                            style = MaterialTheme.typography.bodySmall,
                            color = TextSecondary,
                            fontFamily = FontFamily.Monospace
                        )
                    }

                    Slider(
                        value = selectedEpochs,
                        onValueChange = { selectedEpochs = it },
                        valueRange = 1f..10f,
                        steps = 8,
                        colors = SliderDefaults.colors(
                            thumbColor = NeonViolet,
                            activeTrackColor = NeonViolet,
                            inactiveTrackColor = CyberBg
                        ),
                        enabled = !trainingState.isTraining
                    )

                    Spacer(modifier = Modifier.height(12.dp))

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        if (!trainingState.isTraining) {
                            Button(
                                onClick = { onStartTraining(selectedEpochs.toInt()) },
                                modifier = Modifier
                                    .weight(1f)
                                    .testTag("start_training_button"),
                                colors = ButtonDefaults.buttonColors(containerColor = NeonViolet),
                                shape = RoundedCornerShape(8.dp)
                            ) {
                                Icon(Icons.Default.PlayArrow, contentDescription = "Mulai Training", modifier = Modifier.size(18.dp))
                                Spacer(modifier = Modifier.width(6.dp))
                                Text("Mulai Training")
                            }
                        } else {
                            if (trainingState.isPaused) {
                                Button(
                                    onClick = onPauseTraining,
                                    modifier = Modifier
                                        .weight(1f)
                                        .testTag("resume_training_button"),
                                    colors = ButtonDefaults.buttonColors(containerColor = NeonEmerald),
                                    shape = RoundedCornerShape(8.dp)
                                ) {
                                    Icon(Icons.Default.PlayArrow, contentDescription = "Lanjutkan", modifier = Modifier.size(18.dp))
                                    Spacer(modifier = Modifier.width(6.dp))
                                    Text("Lanjutkan")
                                }
                            } else {
                                OutlinedButton(
                                    onClick = onPauseTraining,
                                    modifier = Modifier
                                        .weight(1f)
                                        .testTag("pause_training_button"),
                                    shape = RoundedCornerShape(8.dp)
                                ) {
                                    Icon(Icons.Default.Pause, contentDescription = "Jeda", tint = NeonAmber, modifier = Modifier.size(18.dp))
                                    Spacer(modifier = Modifier.width(6.dp))
                                    Text("Jeda", color = NeonAmber)
                                }
                            }

                            Button(
                                onClick = onStopTraining,
                                modifier = Modifier
                                    .weight(1f)
                                    .testTag("stop_training_button"),
                                colors = ButtonDefaults.buttonColors(containerColor = NeonCrimson),
                                shape = RoundedCornerShape(8.dp)
                            ) {
                                Icon(Icons.Default.Stop, contentDescription = "Hentikan", modifier = Modifier.size(18.dp))
                                Spacer(modifier = Modifier.width(6.dp))
                                Text("Hentikan")
                            }
                        }
                    }
                }
            }

            // Manajemen Checkpoint (Simpan & Muat Bobot Model)
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = CyberCard),
                border = androidx.compose.foundation.BorderStroke(1.dp, CyberCardBorder)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            imageVector = Icons.Default.Bookmark,
                            contentDescription = "Checkpoint",
                            tint = NeonEmerald,
                            modifier = Modifier.size(20.dp)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = "Manajemen Checkpoint Bobot Model",
                            style = MaterialTheme.typography.titleSmall,
                            fontWeight = FontWeight.SemiBold,
                            color = TextPrimary
                        )
                        Spacer(modifier = Modifier.weight(1f))
                        OutlinedButton(
                            onClick = {
                                checkpointNameInput = "Manual-Step${trainingState.currentStep}"
                                showSaveDialog = true
                            },
                            shape = RoundedCornerShape(8.dp),
                            modifier = Modifier.testTag("save_checkpoint_button")
                        ) {
                            Icon(Icons.Default.Save, contentDescription = "Simpan", modifier = Modifier.size(16.dp), tint = NeonEmerald)
                            Spacer(modifier = Modifier.width(4.dp))
                            Text("Simpan", color = NeonEmerald, fontSize = 11.sp)
                        }
                    }

                    Spacer(modifier = Modifier.height(12.dp))

                    if (checkpoints.isEmpty()) {
                        Text(
                            text = "Belum ada checkpoint tersimpan. Checkpoint akan disimpan otomatis saat training atau tekan tombol Simpan.",
                            style = MaterialTheme.typography.bodySmall,
                            color = TextMuted,
                            fontSize = 11.sp
                        )
                    } else {
                        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                            checkpoints.take(5).forEach { ckpt ->
                                Box(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .background(CyberBg, RoundedCornerShape(8.dp))
                                        .border(1.dp, CyberCardBorder, RoundedCornerShape(8.dp))
                                        .padding(10.dp)
                                ) {
                                    Row(
                                        modifier = Modifier.fillMaxWidth(),
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        Column(modifier = Modifier.weight(1f)) {
                                            Text(
                                                text = ckpt.name,
                                                style = MaterialTheme.typography.labelMedium,
                                                fontWeight = FontWeight.Bold,
                                                color = TextPrimary
                                            )
                                            Text(
                                                text = "Step ${ckpt.step} • Loss: ${ckpt.trainLoss?.let { "%.3f".format(it) } ?: "-"} • Val: ${ckpt.valLoss?.let { "%.3f".format(it) } ?: "-"} • ${ckpt.dateFormatted}",
                                                style = MaterialTheme.typography.bodySmall,
                                                color = TextSecondary,
                                                fontSize = 10.sp,
                                                fontFamily = FontFamily.Monospace
                                            )
                                        }

                                        IconButton(
                                            onClick = { onLoadCheckpoint(ckpt) },
                                            modifier = Modifier.testTag("load_ckpt_${ckpt.id}")
                                        ) {
                                            Icon(
                                                imageVector = Icons.Default.Download,
                                                contentDescription = "Muat Checkpoint",
                                                tint = NeonCyan,
                                                modifier = Modifier.size(18.dp)
                                            )
                                        }

                                        IconButton(
                                            onClick = { onDeleteCheckpoint(ckpt) },
                                            modifier = Modifier.testTag("del_ckpt_${ckpt.id}")
                                        ) {
                                            Icon(
                                                imageVector = Icons.Default.Delete,
                                                contentDescription = "Hapus Checkpoint",
                                                tint = NeonCrimson,
                                                modifier = Modifier.size(18.dp)
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Dialog Simpan Checkpoint Manual
    if (showSaveDialog) {
        AlertDialog(
            onDismissRequest = { showSaveDialog = false },
            title = { Text("Simpan Checkpoint", color = TextPrimary) },
            text = {
                Column {
                    Text(
                        text = "Beri nama untuk checkpoint bobot model saat ini:",
                        style = MaterialTheme.typography.bodySmall,
                        color = TextSecondary
                    )
                    Spacer(modifier = Modifier.height(10.dp))
                    OutlinedTextField(
                        value = checkpointNameInput,
                        onValueChange = { checkpointNameInput = it },
                        singleLine = true,
                        modifier = Modifier.fillMaxWidth()
                    )
                }
            },
            confirmButton = {
                Button(
                    onClick = {
                        onSaveCheckpoint(checkpointNameInput.ifBlank { "Checkpoint-Step${trainingState.currentStep}" })
                        showSaveDialog = false
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = NeonEmerald)
                ) {
                    Text("Simpan", color = CyberBg)
                }
            },
            dismissButton = {
                TextButton(onClick = { showSaveDialog = false }) {
                    Text("Batal", color = TextSecondary)
                }
            },
            containerColor = CyberCard
        )
    }
}

@Composable
private fun TrainingMetricBox(
    label: String,
    value: String,
    unit: String,
    color: Color,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier
            .padding(4.dp)
            .background(CyberBg, RoundedCornerShape(8.dp))
            .border(1.dp, CyberCardBorder, RoundedCornerShape(8.dp))
            .padding(10.dp)
    ) {
        Column {
            Text(text = label, style = MaterialTheme.typography.labelSmall, color = TextMuted, fontSize = 10.sp)
            Spacer(modifier = Modifier.height(2.dp))
            Text(
                text = value,
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
                color = color,
                fontFamily = FontFamily.Monospace
            )
            Text(text = unit, style = MaterialTheme.typography.bodySmall, color = TextSecondary, fontSize = 10.sp)
        }
    }
}
