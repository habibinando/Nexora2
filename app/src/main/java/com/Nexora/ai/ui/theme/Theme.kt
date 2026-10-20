package com.Nexora.ai.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color

// Visual identity ported from the original Nexora project.
val CyberBackground = Color(0xFF090D16)
val CyberSurface = Color(0xFF111827)
val CyberSurfaceVariant = Color(0xFF1E293B)
val CyberSurfaceCard = Color(0xFF161F30)

val ElectricCyan = Color(0xFF00E5FF)
val ElectricCyanDark = Color(0xFF00B0FF)
val NeonPurple = Color(0xFFA855F7)
val DeepViolet = Color(0xFF7C3AED)
val NeuralGreen = Color(0xFF10B981)
val AccentOrange = Color(0xFFF97316)
val DangerRed = Color(0xFFEF4444)

val TextPrimary = Color(0xFFF8FAFC)
val TextSecondary = Color(0xFF94A3B8)
val TextTertiary = Color(0xFF64748B)
val BorderSubtle = Color(0xFF334155)
val GlowBorder = Color(0x3300E5FF)

// Compatibility aliases keep the existing Nexora2 screens compiling.
val NexoraBg = CyberBackground
val NexoraSurface = CyberSurface
val NexoraSurfaceVar = CyberSurfaceVariant
val NexoraPrimary = ElectricCyan
val NexoraCyan = ElectricCyan
val NexoraAmber = AccentOrange
val NexoraOnSurface = TextPrimary
val NexoraMuted = TextSecondary
val NexoraDanger = DangerRed

val AccentGradient: Brush = Brush.horizontalGradient(listOf(ElectricCyan, NeonPurple))
val AccentGradientVertical: Brush = Brush.verticalGradient(listOf(ElectricCyan, NeonPurple))

private val DarkScheme = darkColorScheme(
    primary = ElectricCyan,
    onPrimary = Color.Black,
    secondary = NeonPurple,
    onSecondary = Color.White,
    tertiary = NeuralGreen,
    background = CyberBackground,
    surface = CyberSurface,
    surfaceVariant = CyberSurfaceVariant,
    onBackground = TextPrimary,
    onSurface = TextPrimary,
    onSurfaceVariant = TextSecondary,
    outline = BorderSubtle,
    error = DangerRed,
    primaryContainer = DeepViolet,
    onPrimaryContainer = Color.White,
    secondaryContainer = CyberSurfaceVariant,
    onSecondaryContainer = ElectricCyan
)

@Composable
fun NexoraTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = DarkScheme, content = content)
}
