package com.Nexora.ai.train

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import com.Nexora.ai.R
import com.Nexora.ai.ui.MainActivity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch

/** Foreground service keeps the real training loop alive while screen is off. */
class TrainingService : Service() {

    companion object {
        const val CHANNEL_ID = "nexora_training"
        const val NOTIF_ID = 1001
    }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    private var job: Job? = null

    private data class NotificationSnapshot(
        val phase: Phase,
        val currentEpoch: Int,
        val totalEpochs: Int,
        val batchInEpoch: Int,
        val batchesInEpoch: Int,
        val loss: String,
        val tokensPerSec: Int,
        val operation: String,
        val operationSeconds: Long
    )

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        createChannel()
        startForeground(NOTIF_ID, buildNotification("Nexora training", "Menyiapkan…"))
        job = scope.launch {
            TrainingController.state
                .map { s ->
                    NotificationSnapshot(
                        phase = s.phase,
                        currentEpoch = s.currentEpoch.coerceAtLeast(s.epoch),
                        totalEpochs = s.totalEpochs,
                        batchInEpoch = s.batchInEpoch,
                        batchesInEpoch = s.batchesInEpoch,
                        loss = if (s.loss.isFinite()) String.format(java.util.Locale.US, "%.4f", s.loss) else "belum tersedia",
                        tokensPerSec = s.tokensPerSec.toInt(),
                        operation = s.operation,
                        operationSeconds = s.operationElapsedMs / 1000L
                    )
                }
                // Operation timer ticks every second, not every 100 ms, to avoid notification spam.
                .distinctUntilChanged()
                .collectLatest { s ->
                    val nm = getSystemService(NotificationManager::class.java)
                    val text = buildString {
                        append(s.operation.ifBlank { s.phase.name })
                        append(" · ")
                        append(s.operationSeconds)
                        append(" dtk · batch ")
                        append(s.batchInEpoch)
                        append("/")
                        append(s.batchesInEpoch)
                        append(" · loss ")
                        append(s.loss)
                        append(" · ")
                        append(s.tokensPerSec)
                        append(" tok/s")
                    }
                    nm.notify(
                        NOTIF_ID,
                        buildNotification(
                            "Nexora — epoch ${s.currentEpoch}/${s.totalEpochs}",
                            text
                        )
                    )
                    if (s.phase != Phase.RUNNING && s.phase != Phase.PAUSED) {
                        stopSelf()
                    }
                }
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int = START_STICKY

    override fun onDestroy() {
        job?.cancel(); scope.cancel()
        super.onDestroy()
    }

    private fun createChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val ch = NotificationChannel(CHANNEL_ID, "Nexora Training", NotificationManager.IMPORTANCE_LOW)
            ch.setShowBadge(false)
            getSystemService(NotificationManager::class.java).createNotificationChannel(ch)
        }
    }

    private fun buildNotification(title: String, text: String): Notification {
        val pi = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(title)
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_media_play)
            .setContentIntent(pi)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .build()
    }
}
