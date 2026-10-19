package com.Nexora.ai.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.Nexora.ai.ui.theme.AccentGradient
import com.Nexora.ai.ui.theme.NexoraMuted
import com.Nexora.ai.ui.theme.NexoraOnSurface
import com.Nexora.ai.ui.theme.NexoraSurfaceVar

@Composable
fun GlassCard(modifier: Modifier = Modifier, content: @Composable () -> Unit) {
    Box(
        modifier
            .clip(RoundedCornerShape(24.dp))
            .background(NexoraSurfaceVar.copy(alpha = 0.65f))
            .border(1.dp, Color.White.copy(alpha = 0.06f), RoundedCornerShape(24.dp))
            .padding(16.dp)
    ) { content() }
}

@Composable
fun StatCard(label: String, value: String, sub: String = "", accent: Color = NexoraOnSurface) {
    GlassCard(Modifier.fillMaxWidth()) {
        Column {
            Text(label, color = NexoraMuted, fontSize = 12.sp)
            Text(value, color = accent, fontSize = 20.sp, fontWeight = FontWeight.Bold)
            if (sub.isNotEmpty()) Text(sub, color = NexoraMuted, fontSize = 11.sp)
        }
    }
}

@Composable
fun SectionTitle(text: String) {
    Text(
        text, color = NexoraOnSurface, fontSize = 18.sp, fontWeight = FontWeight.SemiBold,
        modifier = Modifier.padding(vertical = 8.dp)
    )
}

/** Real loss-history line chart drawn on Canvas. */
@Composable
fun LossChart(values: List<Float>, modifier: Modifier = Modifier) {
    val color = MaterialTheme.colorScheme.primary
    Canvas(modifier.fillMaxWidth().size(180.dp)) {
        if (values.size < 2) return@Canvas
        val min = values.min(); val max = values.max()
        val range = (max - min).coerceAtLeast(1e-4f)
        val w = size.width; val h = size.height
        val stepX = w / (values.size - 1)
        val path = Path()
        val fillPath = Path()
        values.forEachIndexed { i, v ->
            val x = i * stepX
            val y = h - ((v - min) / range) * (h - 20f) - 10f
            if (i == 0) { path.moveTo(x, y); fillPath.moveTo(x, h) }
            path.lineTo(x, y); fillPath.lineTo(x, y)
        }
        fillPath.lineTo(w, h); fillPath.close()
        // gradient fill
        drawPath(fillPath, brush = androidx.compose.ui.graphics.Brush.verticalGradient(
            listOf(color.copy(alpha = 0.35f), color.copy(alpha = 0.02f)), 0f, h
        ))
        drawPath(path, color = color, style = Stroke(width = 3f))
        // last point dot
        val lastX = (values.size - 1) * stepX
        val lastY = h - ((values.last() - min) / range) * (h - 20f) - 10f
        drawCircle(color, radius = 6f, center = Offset(lastX, lastY))
    }
}

@Composable
fun GradientLabel(text: String) {
    Text(text, fontSize = 22.sp, fontWeight = FontWeight.Bold,
        color = com.Nexora.ai.ui.theme.NexoraPrimary, modifier = Modifier.padding(vertical = 4.dp))
}

@Composable
fun KeyValueRow(k: String, v: String) {
    Row(Modifier.fillMaxWidth().padding(vertical = 3.dp), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(k, color = NexoraMuted, fontSize = 13.sp)
        Text(v, color = NexoraOnSurface, fontSize = 13.sp, fontWeight = FontWeight.Medium)
    }
}
