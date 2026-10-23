package com.nexora

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Chat
import androidx.compose.material.icons.filled.DataObject
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.ModelTraining
import androidx.compose.material.icons.filled.Speed
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationBarItemDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.nexora.ui.AppScreen
import com.nexora.ui.NexoraViewModel
import com.nexora.ui.screens.BenchmarkScreen
import com.nexora.ui.screens.ChatScreen
import com.nexora.ui.screens.DashboardScreen
import com.nexora.ui.screens.DatasetScreen
import com.nexora.ui.screens.SettingsScreen
import com.nexora.ui.screens.SplashScreen
import com.nexora.ui.screens.TrainingScreen
import com.nexora.ui.theme.CyberBg
import com.nexora.ui.theme.CyberCard
import com.nexora.ui.theme.NexoraTheme
import com.nexora.ui.theme.NeonCyan
import com.nexora.ui.theme.NeonViolet
import com.nexora.ui.theme.TextMuted
import com.nexora.ui.theme.TextPrimary

class MainActivity : ComponentActivity() {

    private val viewModel: NexoraViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        setContent {
            NexoraTheme {
                NexoraApp(viewModel = viewModel)
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun NexoraApp(viewModel: NexoraViewModel) {
    val currentScreen by viewModel.currentScreen.collectAsStateWithLifecycle()
    val splashProgress by viewModel.splashProgress.collectAsStateWithLifecycle()
    val splashStatus by viewModel.splashStatus.collectAsStateWithLifecycle()
    val trainingState by viewModel.trainingState.collectAsStateWithLifecycle()
    val datasetMeta by viewModel.datasetMeta.collectAsStateWithLifecycle()
    val isDatasetLoading by viewModel.isDatasetLoading.collectAsStateWithLifecycle()
    val benchmarkResult by viewModel.benchmarkResult.collectAsStateWithLifecycle()
    val isBenchmarking by viewModel.isBenchmarking.collectAsStateWithLifecycle()
    val benchmarkProgress by viewModel.benchmarkProgress.collectAsStateWithLifecycle()
    val chatMessages by viewModel.chatMessages.collectAsStateWithLifecycle()
    val isGenerating by viewModel.isGenerating.collectAsStateWithLifecycle()
    val hardwareConfig by viewModel.hardwareConfig.collectAsStateWithLifecycle()
    val trainingConfig2026 by viewModel.trainingConfig2026.collectAsStateWithLifecycle()
    val checkpointsList by viewModel.checkpointsList.collectAsStateWithLifecycle()

    val context = androidx.compose.ui.platform.LocalContext.current
    val permissionLauncher = androidx.activity.compose.rememberLauncherForActivityResult(
        contract = androidx.activity.result.contract.ActivityResultContracts.RequestPermission()
    ) { _ -> }

    androidx.compose.runtime.LaunchedEffect(Unit) {
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.TIRAMISU) {
            if (androidx.core.content.ContextCompat.checkSelfPermission(
                    context,
                    android.Manifest.permission.POST_NOTIFICATIONS
                ) != android.content.pm.PackageManager.PERMISSION_GRANTED
            ) {
                permissionLauncher.launch(android.Manifest.permission.POST_NOTIFICATIONS)
            }
        }
    }

    if (currentScreen == AppScreen.SPLASH) {
        SplashScreen(
            progress = splashProgress,
            status = splashStatus
        )
    } else {
        Scaffold(
            modifier = Modifier.fillMaxSize(),
            containerColor = CyberBg,
            topBar = {
                if (currentScreen == AppScreen.DASHBOARD) {
                    TopAppBar(
                        title = {
                            Text(
                                text = "NEXORA AI",
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.Bold,
                                color = TextPrimary
                            )
                        },
                        actions = {
                            IconButton(
                                onClick = { viewModel.navigateTo(AppScreen.SETTINGS) },
                                modifier = Modifier.testTag("dashboard_settings_button")
                            ) {
                                Icon(
                                    imageVector = Icons.Default.Tune,
                                    contentDescription = "Pengaturan Hardware",
                                    tint = NeonCyan
                                )
                            }
                        },
                        colors = TopAppBarDefaults.topAppBarColors(containerColor = CyberBg)
                    )
                }
            },
            bottomBar = {
                NavigationBar(
                    containerColor = CyberCard,
                    tonalElevation = 8.dp
                ) {
                    NavigationBarItem(
                        selected = currentScreen == AppScreen.DASHBOARD,
                        onClick = { viewModel.navigateTo(AppScreen.DASHBOARD) },
                        icon = { Icon(Icons.Default.Home, contentDescription = "Beranda") },
                        label = { Text("Beranda") },
                        colors = NavigationBarItemDefaults.colors(
                            selectedIconColor = NeonViolet,
                            selectedTextColor = NeonViolet,
                            indicatorColor = NeonViolet.copy(alpha = 0.15f),
                            unselectedIconColor = TextMuted,
                            unselectedTextColor = TextMuted
                        ),
                        modifier = Modifier.testTag("bottom_nav_home")
                    )

                    NavigationBarItem(
                        selected = currentScreen == AppScreen.DATASET,
                        onClick = { viewModel.navigateTo(AppScreen.DATASET) },
                        icon = { Icon(Icons.Default.DataObject, contentDescription = "Dataset") },
                        label = { Text("Dataset") },
                        colors = NavigationBarItemDefaults.colors(
                            selectedIconColor = NeonCyan,
                            selectedTextColor = NeonCyan,
                            indicatorColor = NeonCyan.copy(alpha = 0.15f),
                            unselectedIconColor = TextMuted,
                            unselectedTextColor = TextMuted
                        ),
                        modifier = Modifier.testTag("bottom_nav_dataset")
                    )

                    NavigationBarItem(
                        selected = currentScreen == AppScreen.TRAINING,
                        onClick = { viewModel.navigateTo(AppScreen.TRAINING) },
                        icon = { Icon(Icons.Default.ModelTraining, contentDescription = "Training") },
                        label = { Text("Training") },
                        colors = NavigationBarItemDefaults.colors(
                            selectedIconColor = NeonViolet,
                            selectedTextColor = NeonViolet,
                            indicatorColor = NeonViolet.copy(alpha = 0.15f),
                            unselectedIconColor = TextMuted,
                            unselectedTextColor = TextMuted
                        ),
                        modifier = Modifier.testTag("bottom_nav_training")
                    )

                    NavigationBarItem(
                        selected = currentScreen == AppScreen.BENCHMARK,
                        onClick = { viewModel.navigateTo(AppScreen.BENCHMARK) },
                        icon = { Icon(Icons.Default.Speed, contentDescription = "Benchmark") },
                        label = { Text("Benchmark") },
                        colors = NavigationBarItemDefaults.colors(
                            selectedIconColor = NeonCyan,
                            selectedTextColor = NeonCyan,
                            indicatorColor = NeonCyan.copy(alpha = 0.15f),
                            unselectedIconColor = TextMuted,
                            unselectedTextColor = TextMuted
                        ),
                        modifier = Modifier.testTag("bottom_nav_benchmark")
                    )

                    NavigationBarItem(
                        selected = currentScreen == AppScreen.CHAT,
                        onClick = { viewModel.navigateTo(AppScreen.CHAT) },
                        icon = { Icon(Icons.AutoMirrored.Filled.Chat, contentDescription = "Chat") },
                        label = { Text("Chat") },
                        colors = NavigationBarItemDefaults.colors(
                            selectedIconColor = NeonCyan,
                            selectedTextColor = NeonCyan,
                            indicatorColor = NeonCyan.copy(alpha = 0.15f),
                            unselectedIconColor = TextMuted,
                            unselectedTextColor = TextMuted
                        ),
                        modifier = Modifier.testTag("bottom_nav_chat")
                    )
                }
            }
        ) { innerPadding ->
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(innerPadding)
                    .background(CyberBg)
            ) {
                when (currentScreen) {
                    AppScreen.SPLASH -> {}
                    AppScreen.DASHBOARD -> {
                        DashboardScreen(
                            trainingState = trainingState,
                            onNavigate = { screen -> viewModel.navigateTo(screen) }
                        )
                    }
                    AppScreen.DATASET -> {
                        DatasetScreen(
                            datasetMeta = datasetMeta,
                            isLoading = isDatasetLoading,
                            onImportFile = { uri, name, size -> viewModel.importDatasetFile(uri, name, size) },
                            onLoadPreset = { viewModel.loadPresetDataset() },
                            onBack = { viewModel.navigateTo(AppScreen.DASHBOARD) }
                        )
                    }
                    AppScreen.TRAINING -> {
                        TrainingScreen(
                            trainingState = trainingState,
                            trainingConfig = trainingConfig2026,
                            checkpoints = checkpointsList,
                            datasetMeta = datasetMeta,
                            onUpdateOptimizer = { opt -> viewModel.updateOptimizerType(opt) },
                            onUpdateLrScheduler = { sch -> viewModel.updateLrSchedulerType(sch) },
                            onUpdateTrainingMode = { mode -> viewModel.setTrainingMode(mode) },
                            onStartTraining = { epochs -> viewModel.startTraining(epochs) },
                            onPauseTraining = { viewModel.pauseTraining() },
                            onStopTraining = { viewModel.stopTraining() },
                            onSaveCheckpoint = { name -> viewModel.saveManualCheckpoint(name) },
                            onLoadCheckpoint = { ckpt -> viewModel.loadCheckpoint(ckpt) },
                            onDeleteCheckpoint = { ckpt -> viewModel.deleteCheckpoint(ckpt) },
                            onBack = { viewModel.navigateTo(AppScreen.DASHBOARD) }
                        )
                    }
                    AppScreen.BENCHMARK -> {
                        BenchmarkScreen(
                            benchmarkResult = benchmarkResult,
                            isBenchmarking = isBenchmarking,
                            progress = benchmarkProgress,
                            onRunBenchmark = { iterations -> viewModel.runBenchmark(iterations) },
                            onBack = { viewModel.navigateTo(AppScreen.DASHBOARD) }
                        )
                    }
                    AppScreen.CHAT -> {
                        ChatScreen(
                            messages = chatMessages,
                            isGenerating = isGenerating,
                            isModelTrained = trainingState.currentStep > 0,
                            onSendPrompt = { prompt, temp -> viewModel.sendPrompt(prompt, temp) },
                            onClearChat = { viewModel.clearChat() },
                            onBack = { viewModel.navigateTo(AppScreen.DASHBOARD) }
                        )
                    }
                    AppScreen.SETTINGS -> {
                        SettingsScreen(
                            hardwareConfig = hardwareConfig,
                            onUpdateCpuThreads = { threads -> viewModel.updateCpuThreads(threads) },
                            onUpdateLearningRate = { lr -> viewModel.updateLearningRate(lr) },
                            onBack = { viewModel.navigateTo(AppScreen.DASHBOARD) }
                        )
                    }
                }
            }
        }
    }
}
