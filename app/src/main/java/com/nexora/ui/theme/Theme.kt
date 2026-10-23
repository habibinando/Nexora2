package com.nexora.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable

private val NexoraColorScheme = darkColorScheme(
    primary = NeonViolet,
    onPrimary = TextPrimary,
    primaryContainer = CyberCard,
    onPrimaryContainer = NeonVioletGlow,
    secondary = NeonCyan,
    onSecondary = CyberBg,
    secondaryContainer = CyberCard,
    onSecondaryContainer = NeonCyanLight,
    tertiary = NeonEmerald,
    background = CyberBg,
    onBackground = TextPrimary,
    surface = CyberSurface,
    onSurface = TextPrimary,
    surfaceVariant = CyberCard,
    onSurfaceVariant = TextSecondary,
    outline = CyberCardBorder,
    error = NeonCrimson
)

@Composable
fun NexoraTheme(
    content: @Composable () -> Unit
) {
    MaterialTheme(
        colorScheme = NexoraColorScheme,
        typography = Typography,
        content = content
    )
}
