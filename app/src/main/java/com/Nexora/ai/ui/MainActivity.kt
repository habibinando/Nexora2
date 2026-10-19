package com.Nexora.ai.ui

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Chat
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.InsertDriveFile
import androidx.compose.material.icons.filled.Science
import androidx.compose.material.icons.filled.TrackChanges
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.Nexora.ai.ml.NativeOps
import com.Nexora.ai.train.TrainingController
import com.Nexora.ai.train.TrainingService
import com.Nexora.ai.ui.screens.ChatScreen
import com.Nexora.ai.ui.screens.DatasetScreen
import com.Nexora.ai.ui.screens.HomeScreen
import com.Nexora.ai.ui.screens.LabScreen
import com.Nexora.ai.ui.screens.TrainScreen
import com.Nexora.ai.ui.theme.AccentGradient
import com.Nexora.ai.ui.theme.NexoraBg
import com.Nexora.ai.ui.theme.NexoraMuted
import com.Nexora.ai.ui.theme.NexoraOnSurface
import com.Nexora.ai.ui.theme.NexoraTheme
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.collectLatest

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        installSplashScreen()
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            NexoraTheme {
                var ready by remember { mutableStateOf(false) }
                LaunchedEffect(Unit) {
                    // Real initialization: touch native lib + thread pool + checkpoint scan.
                    NativeOps.nativeSetThreads(NativeOps.nativeCpuCores())
                    delay(1200) // let the loading screen breathe
                    ready = true
                }
                Box(Modifier.fillMaxSize().background(NexoraBg)) {
                    if (ready) MainScaffold() else LoadingScreen()
                }
            }
        }
    }
}

@Composable
fun LoadingScreen() {
    val transition = rememberInfiniteTransition(label = "load")
    val rot = transition.animateFloat(0f, 360f, infiniteRepeatable(tween(1400, easing = LinearEasing), RepeatMode.Restart), label = "rot")
    val statuses = listOf("Memuat inti neural…", "Mendeteksi core CPU…", "GPU compute probe…", "Memeriksa checkpoint…")
    var idx by remember { mutableStateOf(0) }
    LaunchedEffect(Unit) { while (true) { delay(700); idx = (idx + 1) % statuses.size } }
    Column(Modifier.fillMaxSize(), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center) {
        Box(Modifier.size(120.dp), contentAlignment = Alignment.Center) {
            Canvas(Modifier.size(120.dp).rotate(rot.value)) {
                drawArc(
                    brush = AccentGradient, startAngle = 0f, sweepAngle = 270f, useCenter = false,
                    style = Stroke(width = 10f, cap = StrokeCap.Round)
                )
            }
            Text("N", fontSize = 44.sp, fontWeight = FontWeight.Black, brush = AccentGradient)
        }
        Text("NEXORA", fontSize = 28.sp, fontWeight = FontWeight.Bold, color = NexoraOnSurface, letterSpacing = 8.sp, modifier = Modifier.padding(top = 24.dp))
        Text("On-device AI language training", color = NexoraMuted, fontSize = 13.sp, modifier = Modifier.padding(top = 4.dp))
        LinearProgressIndicator(Modifier.padding(top = 32.dp, start = 48.dp, end = 48.dp).fillMaxWidth(), color = androidx.compose.material3.MaterialTheme.colorScheme.primary)
        Text(statuses[idx], color = NexoraMuted, fontSize = 12.sp, modifier = Modifier.padding(top = 16.dp))
    }
}

private data class NavItem(val route: String, val label: String, val icon: androidx.compose.ui.graphics.vector.ImageVector)

@Composable
fun MainScaffold() {
    val nav = rememberNavController()
    val context = androidx.compose.ui.platform.LocalContext.current
    val items = listOf(
        NavItem("home", "Home", Icons.Filled.Home),
        NavItem("dataset", "Data", Icons.Filled.InsertDriveFile),
        NavItem("train", "Train", Icons.Filled.TrackChanges),
        NavItem("chat", "Chat", Icons.Filled.Chat),
        NavItem("lab", "Lab", Icons.Filled.Science)
    )
    val back by nav.currentBackStackEntryAsState()
    val current = back?.destination?.route ?: "home"

    // Start foreground service whenever training is running.
    LaunchedEffect(Unit) {
        TrainingController.state.collectLatest { s ->
            if (s.phase == com.Nexora.ai.train.Phase.RUNNING) {
                context.startService(Intent(context, TrainingService::class.java))
            }
        }
    }

    Scaffold(
        containerColor = NexoraBg,
        bottomBar = {
            NavigationBar(containerColor = Color(0xFF0A1018)) {
                items.forEach { item ->
                    NavigationBarItem(
                        selected = current == item.route,
                        onClick = { nav.navigate(item.route) { popUpTo(nav.graph.startDestinationId) { saveState = true }; launchSingleTop = true; restoreState = true } },
                        icon = { Icon(item.icon, contentDescription = item.label) },
                        label = { Text(item.label, fontSize = 10.sp) }
                    )
                }
            }
        }
    ) { pad ->
        NavHost(nav, startDestination = "home", modifier = Modifier.padding(pad)) {
            composable("home") { HomeScreen(nav) }
            composable("dataset") { DatasetScreen() }
            composable("train") { TrainScreen() }
            composable("chat") { ChatScreen() }
            composable("lab") { LabScreen() }
        }
    }
}
