package com.nexora.ui.screens

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
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
import androidx.compose.material.icons.automirrored.filled.Chat
import androidx.compose.material.icons.filled.DataObject
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.ModelTraining
import androidx.compose.material.icons.filled.Speed
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.nexora.R
import com.nexora.model.TrainingState
import com.nexora.ui.AppScreen
import com.nexora.ui.theme.CyberBg
import com.nexora.ui.theme.CyberCard
import com.nexora.ui.theme.CyberCardBorder
import com.nexora.ui.theme.NeonAmber
import com.nexora.ui.theme.NeonCyan
import com.nexora.ui.theme.NeonEmerald
import com.nexora.ui.theme.NeonViolet
import com.nexora.ui.theme.TextMuted
import com.nexora.ui.theme.TextPrimary
import com.nexora.ui.theme.TextSecondary
import java.text.NumberFormat
import java.util.Locale

@Composable
fun DashboardScreen(
    trainingState: TrainingState,
    onNavigate: (AppScreen) -> Unit,
    modifier: Modifier = Modifier
) {
    val scrollState = rememberScrollState()
    val runtime = Runtime.getRuntime()
    val maxMemMb = runtime.maxMemory() / (1024 * 1024)
    val totalMemMb = runtime.totalMemory() / (1024 * 1024)
    val freeMemMb = runtime.freeMemory() / (1024 * 1024)
    val usedMemMb = totalMemMb - freeMemMb
    val cores = runtime.availableProcessors()

    val numFormat = NumberFormat.getNumberInstance(Locale.US)

    Column(
        modifier = modifier
            .fillMaxSize()
            .background(CyberBg)
            .verticalScroll(scrollState)
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Hero Card Visual
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .height(160.dp),
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(containerColor = CyberCard)
        ) {
            Box(modifier = Modifier.fillMaxSize()) {
                Image(
                    painter = painterResource(id = R.drawable.banner_nexora_neural_1792405857050),
                    contentDescription = "Neural Network Tensor Banner",
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Crop
                )

                // Dark gradient overlay
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(
                            Brush.verticalGradient(
                                colors = listOf(Color.Transparent, CyberBg.copy(alpha = 0.85f))
                            )
                        )
                )

                Column(
                    modifier = Modifier
                        .align(Alignment.BottomStart)
                        .padding(16.dp)
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Box(
                            modifier = Modifier
                                .size(8.dp)
                                .background(NeonEmerald, CircleShape)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        val isVulkan = com.nexora.model.NexoraNative.isVulkanAvailable()
                        val isNdkRunning = com.nexora.model.NexoraNative.isNdkLoaded
                        val engineStatusText = when {
                            isVulkan -> "ENGINE ONLINE • VULKAN 1.2+ GPU AKTIF • LARGEHEAP"
                            isNdkRunning -> "ENGINE ONLINE • NDK C++20 SIMD AKTIF • LARGEHEAP"
                            else -> "ENGINE ONLINE • JVM ENGINE • LARGEHEAP"
                        }
                        Text(
                            text = engineStatusText,
                            style = MaterialTheme.typography.labelSmall,
                            color = NeonEmerald,
                            fontWeight = FontWeight.Bold
                        )
                    }
                    Text(
                        text = "Nexora Neural Core",
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                        color = TextPrimary
                    )
                    Text(
                        text = "Transformer LM5M (d=256, L=5, H=4, FFN=704) | Vocab 4,096 | RoPE & SwiGLU",
                        style = MaterialTheme.typography.bodySmall,
                        color = NeonCyan,
                        fontFamily = FontFamily.Monospace
                    )
                }
            }
        }

        // Kartu Spesifikasi Arsitektur Nyata
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(12.dp),
            colors = CardDefaults.cardColors(containerColor = CyberCard),
            border = androidx.compose.foundation.BorderStroke(1.dp, CyberCardBorder)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = "Parameter & Spesifikasi Nyata",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                    color = TextPrimary
                )
                Spacer(modifier = Modifier.height(12.dp))

                Row(modifier = Modifier.fillMaxWidth()) {
                    MetricItem(
                        label = "Total Parameter",
                        value = numFormat.format(trainingState.totalParams),
                        subtext = "%.2fM Bobot Riil".format(trainingState.totalParams / 1_000_000.0),
                        color = NeonViolet,
                        modifier = Modifier.weight(1f)
                    )
                    MetricItem(
                        label = if (trainingState.isLoRAActive) "Trainable (LoRA)" else "Trainable (Full)",
                        value = numFormat.format(trainingState.trainableParams),
                        subtext = if (trainingState.isLoRAActive) {
                            val pct = (trainingState.trainableParams.toDouble() / maxOf(1L, trainingState.totalParams).toDouble()) * 100.0
                            "Rank 4 (%.2f%% Aktif)".format(pct)
                        } else {
                            "100% Parameter Dilatih"
                        },
                        color = NeonCyan,
                        modifier = Modifier.weight(1f)
                    )
                }

                Spacer(modifier = Modifier.height(12.dp))

                Row(modifier = Modifier.fillMaxWidth()) {
                    MetricItem(
                        label = "Ukuran Adapter LoRA",
                        value = "%.2f KB".format(trainingState.adapterSizeBytes / 1024f),
                        subtext = "${trainingState.adapterSizeBytes} Bytes",
                        color = NeonEmerald,
                        modifier = Modifier.weight(1f)
                    )
                    MetricItem(
                        label = "Kosa Kata (Vocab)",
                        value = "4,096",
                        subtext = "Byte-Level BPE",
                        color = NeonAmber,
                        modifier = Modifier.weight(1f)
                    )
                }
            }
        }

        // Hardware Monitor Card
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(12.dp),
            colors = CardDefaults.cardColors(containerColor = CyberCard),
            border = androidx.compose.foundation.BorderStroke(1.dp, CyberCardBorder)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        imageVector = Icons.Default.Memory,
                        contentDescription = "Memory Icon",
                        tint = NeonCyan,
                        modifier = Modifier.size(20.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "Telemetri Hardware & Memori Android",
                        style = MaterialTheme.typography.titleSmall,
                        color = TextPrimary,
                        fontWeight = FontWeight.SemiBold
                    )
                }

                Spacer(modifier = Modifier.height(12.dp))

                Row(modifier = Modifier.fillMaxWidth()) {
                    HardwareItem(
                        label = "CPU Cores",
                        value = "$cores Cores",
                        desc = "Multi-thread SIMD",
                        modifier = Modifier.weight(1f)
                    )
                    HardwareItem(
                        label = "Heap Terpakai",
                        value = "$usedMemMb MB",
                        desc = "dari Max $maxMemMb MB",
                        modifier = Modifier.weight(1f)
                    )
                }

                Spacer(modifier = Modifier.height(10.dp))

                val isNdkRunning = com.nexora.model.NexoraNative.isNdkLoaded
                val ndkStatusText = com.nexora.model.NexoraNative.getActiveNdkStatus()
                val arch = System.getProperty("os.arch") ?: "ARM"

                Row(modifier = Modifier.fillMaxWidth()) {
                    HardwareItem(
                        label = "Native Engine",
                        value = if (isNdkRunning) "C++ Native Aktif" else "JVM Engine",
                        desc = ndkStatusText,
                        modifier = Modifier.weight(1f)
                    )
                    HardwareItem(
                        label = "Arsitektur CPU",
                        value = arch,
                        desc = if (isNdkRunning) "Vector SIMD (-O3)" else "JVM Standard Math",
                        modifier = Modifier.weight(1f)
                    )
                }

                Spacer(modifier = Modifier.height(8.dp))

                // Bar visual memori
                val memFraction = (usedMemMb.toFloat() / maxOf(1f, maxMemMb.toFloat())).coerceIn(0f, 1f)
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(6.dp)
                        .clip(RoundedCornerShape(3.dp))
                        .background(CyberBg)
                ) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth(memFraction)
                            .fillMaxSize()
                            .background(if (memFraction > 0.8f) NeonAmber else NeonCyan)
                    )
                }
            }
        }

        // Navigasi Fitur Cepat
        Text(
            text = "Menu Pelatihan & Eksperimen",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.SemiBold,
            color = TextPrimary
        )

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            FeatureActionCard(
                title = "Pelatihan & LoRA",
                desc = "Loss, PPL, & Step",
                icon = Icons.Default.ModelTraining,
                iconColor = NeonViolet,
                testTag = "nav_training_button",
                onClick = { onNavigate(AppScreen.TRAINING) },
                modifier = Modifier.weight(1f)
            )
            FeatureActionCard(
                title = "Hardware Benchmark",
                desc = "GFLOPS & Throughput",
                icon = Icons.Default.Speed,
                iconColor = NeonCyan,
                testTag = "nav_benchmark_button",
                onClick = { onNavigate(AppScreen.BENCHMARK) },
                modifier = Modifier.weight(1f)
            )
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            FeatureActionCard(
                title = "Chat Model Nyata",
                desc = "Inferensi Autoregresif",
                icon = Icons.AutoMirrored.Filled.Chat,
                iconColor = NeonEmerald,
                testTag = "nav_chat_button",
                onClick = { onNavigate(AppScreen.CHAT) },
                modifier = Modifier.weight(1f)
            )
            FeatureActionCard(
                title = "Dataset Manager",
                desc = "JSONL/CSV/TXT 500MB",
                icon = Icons.Default.DataObject,
                iconColor = NeonAmber,
                testTag = "nav_dataset_button",
                onClick = { onNavigate(AppScreen.DATASET) },
                modifier = Modifier.weight(1f)
            )
        }
    }
}

@Composable
private fun MetricItem(
    label: String,
    value: String,
    subtext: String,
    color: Color,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier) {
        Text(text = label, style = MaterialTheme.typography.labelSmall, color = TextMuted)
        Spacer(modifier = Modifier.height(2.dp))
        Text(
            text = value,
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold,
            color = color,
            fontFamily = FontFamily.Monospace
        )
        Text(text = subtext, style = MaterialTheme.typography.bodySmall, color = TextSecondary, fontSize = 11.sp)
    }
}

@Composable
private fun HardwareItem(
    label: String,
    value: String,
    desc: String,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier) {
        Text(text = label, style = MaterialTheme.typography.labelSmall, color = TextMuted)
        Text(
            text = value,
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.SemiBold,
            color = TextPrimary
        )
        Text(text = desc, style = MaterialTheme.typography.bodySmall, color = TextSecondary, fontSize = 11.sp)
    }
}

@Composable
private fun FeatureActionCard(
    title: String,
    desc: String,
    icon: ImageVector,
    iconColor: Color,
    testTag: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier
            .testTag(testTag)
            .clickable(onClick = onClick),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = CyberCard),
        border = androidx.compose.foundation.BorderStroke(1.dp, CyberCardBorder)
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Box(
                modifier = Modifier
                    .size(36.dp)
                    .background(iconColor.copy(alpha = 0.15f), RoundedCornerShape(8.dp)),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = icon,
                    contentDescription = title,
                    tint = iconColor,
                    modifier = Modifier.size(20.dp)
                )
            }
            Spacer(modifier = Modifier.height(10.dp))
            Text(
                text = title,
                style = MaterialTheme.typography.labelLarge,
                fontWeight = FontWeight.Bold,
                color = TextPrimary
            )
            Text(
                text = desc,
                style = MaterialTheme.typography.bodySmall,
                color = TextSecondary,
                fontSize = 11.sp
            )
        }
    }
}
