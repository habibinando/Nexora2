package com.Nexora.ai.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color

// Nexora 2026 visual identity: deep-space dark, teal-cyan accent, warm amber highlight.
val NexoraBg = Color(0xFF070B12)
val NexoraSurface = Color(0xFF0E1520)
val NexoraSurfaceVar = Color(0xFF16202E)
val NexoraPrimary = Color(0xFF2DD4BF)
val NexoraCyan = Color(0xFF22D3EE)
val NexoraAmber = Color(0xFFFBBF24)
val NexoraOnSurface = Color(0xFFE6EDF3)
val NexoraMuted = Color(0xFF64748B)
val NexoraDanger = Color(0xFFFB7185)

val AccentGradient: Brush = Brush.horizontalGradient(listOf(NexoraPrimary, NexoraCyan))
val AccentGradientVertical: Brush = Brush.verticalGradient(listOf(NexoraPrimary, NexoraCyan))

private val DarkScheme = darkColorScheme(
    primary = NexoraPrimary,
    onPrimary = Color(0xFF041410),
    secondary = NexoraAmber,
    onSecondary = Color(0xFF1A1200),
    background = NexoraBg,
    surface = NexoraSurface,
    surfaceVariant = NexoraSurfaceVar,
    onBackground = NexoraOnSurface,
    onSurface = NexoraOnSurface,
    onSurfaceVariant = NexoraMuted,
    error = NexoraDanger,
    primaryContainer = Color(0xFF0E3A33),
    onPrimaryContainer = NexoraPrimary
)

@Composable
fun NexoraTheme(content: @Composable () -> Unit) {
    // Nexora is always dark (2026 OLED-first aesthetic).
    MaterialTheme(colorScheme = DarkScheme, content = content)
}
