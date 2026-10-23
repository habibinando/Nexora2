package com.nexora.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
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
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.nexora.ui.theme.CyberCard
import com.nexora.ui.theme.CyberCardBorder
import com.nexora.ui.theme.NeonCyan
import com.nexora.ui.theme.NeonViolet
import com.nexora.ui.theme.TextMuted
import com.nexora.ui.theme.TextSecondary
import kotlin.math.max

@Composable
fun LossChart(
    lossHistory: List<Float>,
    valLossHistory: List<Float> = emptyList(),
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier
            .fillMaxWidth()
            .height(190.dp)
            .background(CyberCard, RoundedCornerShape(12.dp))
            .border(1.dp, CyberCardBorder, RoundedCornerShape(12.dp))
            .padding(12.dp)
    ) {
        if (lossHistory.isEmpty() && valLossHistory.isEmpty()) {
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = "Belum ada riwayat loss pelatihan",
                    style = MaterialTheme.typography.bodyMedium,
                    color = TextMuted
                )
            }
        } else {
            val allLosses = lossHistory + valLossHistory
            val minLoss = allLosses.minOrNull() ?: 0f
            val maxLoss = max(allLosses.maxOrNull() ?: 1f, minLoss + 0.1f)
            val latestTrainLoss = lossHistory.lastOrNull() ?: 0f
            val latestValLoss = valLossHistory.lastOrNull()

            Column(modifier = Modifier.fillMaxSize()) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    // Legend Train
                    Box(modifier = Modifier.size(8.dp).background(NeonViolet, CircleShape))
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(
                        text = "Train: %.3f".format(latestTrainLoss),
                        fontFamily = FontFamily.Monospace,
                        fontSize = 11.sp,
                        color = NeonViolet
                    )

                    Spacer(modifier = Modifier.width(12.dp))

                    // Legend Val
                    if (latestValLoss != null) {
                        Box(modifier = Modifier.size(8.dp).background(NeonCyan, CircleShape))
                        Spacer(modifier = Modifier.width(4.dp))
                        Text(
                            text = "Val: %.3f".format(latestValLoss),
                            fontFamily = FontFamily.Monospace,
                            fontSize = 11.sp,
                            color = NeonCyan
                        )
                    }

                    Spacer(modifier = Modifier.weight(1f))
                    Text(
                        text = "Real-Time Loss",
                        style = MaterialTheme.typography.labelSmall,
                        color = TextSecondary
                    )
                }

                Spacer(modifier = Modifier.height(8.dp))

                Canvas(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f)
                ) {
                    val w = size.width
                    val h = size.height

                    // Garis grid horizontal
                    val gridLines = 3
                    for (g in 0..gridLines) {
                        val y = h * (g.toFloat() / gridLines)
                        drawLine(
                            color = Color(0x1AFFFFFF),
                            start = Offset(0f, y),
                            end = Offset(w, y),
                            strokeWidth = 1.dp.toPx()
                        )
                    }

                    // 1. Gambar Train Loss Curve (Violet)
                    if (lossHistory.size >= 2) {
                        val stepX = w / (lossHistory.size - 1)
                        val strokePath = Path()
                        val fillPath = Path()

                        for (i in lossHistory.indices) {
                            val lossVal = lossHistory[i]
                            val normY = (lossVal - minLoss) / (maxLoss - minLoss)
                            val y = h - (normY * (h - 16.dp.toPx()) + 8.dp.toPx())
                            val x = i * stepX

                            if (i == 0) {
                                strokePath.moveTo(x, y)
                                fillPath.moveTo(x, h)
                                fillPath.lineTo(x, y)
                            } else {
                                strokePath.lineTo(x, y)
                                fillPath.lineTo(x, y)
                            }
                        }

                        fillPath.lineTo(w, h)
                        fillPath.close()

                        drawPath(
                            path = fillPath,
                            brush = Brush.verticalGradient(
                                colors = listOf(NeonViolet.copy(alpha = 0.25f), Color.Transparent),
                                startY = 0f,
                                endY = h
                            )
                        )

                        drawPath(
                            path = strokePath,
                            color = NeonViolet,
                            style = Stroke(width = 2.5.dp.toPx(), cap = StrokeCap.Round)
                        )
                    }

                    // 2. Gambar Validation Loss Curve (Cyan)
                    if (valLossHistory.size >= 2) {
                        val stepX = w / (valLossHistory.size - 1)
                        val valStroke = Path()

                        for (i in valLossHistory.indices) {
                            val lossVal = valLossHistory[i]
                            val normY = (lossVal - minLoss) / (maxLoss - minLoss)
                            val y = h - (normY * (h - 16.dp.toPx()) + 8.dp.toPx())
                            val x = i * stepX

                            if (i == 0) {
                                valStroke.moveTo(x, y)
                            } else {
                                valStroke.lineTo(x, y)
                            }

                            // Titik data evaluasi validasi
                            drawCircle(
                                color = NeonCyan,
                                radius = 3.dp.toPx(),
                                center = Offset(x, y)
                            )
                        }

                        drawPath(
                            path = valStroke,
                            color = NeonCyan,
                            style = Stroke(width = 2.dp.toPx(), cap = StrokeCap.Round)
                        )
                    }
                }

                Row(modifier = Modifier.fillMaxWidth()) {
                    Text(
                        text = "Min: %.3f".format(minLoss),
                        fontSize = 10.sp,
                        color = TextMuted,
                        fontFamily = FontFamily.Monospace
                    )
                    Spacer(modifier = Modifier.weight(1f))
                    Text(
                        text = "Max: %.3f".format(maxLoss),
                        fontSize = 10.sp,
                        color = TextMuted,
                        fontFamily = FontFamily.Monospace
                    )
                }
            }
        }
    }
}
