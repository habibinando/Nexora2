package com.nexora.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat
import com.nexora.MainActivity
import com.nexora.R

/**
 * Foreground Service untuk Menjaga Pelatihan AI Tetap Berjalan di Latar Belakang
 * Dilengkapi dengan WakeLock CPU dan Notifikasi Progres Real-Time.
 */
class TrainingForegroundService : Service() {

    companion object {
        const val CHANNEL_ID = "nexora_training_channel"
        const val NOTIFICATION_ID = 2026

        const val ACTION_START = "com.nexora.service.ACTION_START"
        const val ACTION_STOP = "com.nexora.service.ACTION_STOP"
        const val ACTION_PAUSE = "com.nexora.service.ACTION_PAUSE"

        const val EXTRA_STEP = "extra_step"
        const val EXTRA_TOTAL_STEPS = "extra_total_steps"
        const val EXTRA_EPOCH = "extra_epoch"
        const val EXTRA_TOTAL_EPOCHS = "extra_total_epochs"
        const val EXTRA_LOSS = "extra_loss"
        const val EXTRA_VAL_LOSS = "extra_val_loss"
        const val EXTRA_SPEED = "extra_speed"

        private var wakeLock: PowerManager.WakeLock? = null

        fun startService(context: Context) {
            val intent = Intent(context, TrainingForegroundService::class.java).apply {
                action = ACTION_START
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                context.startForegroundService(intent)
            } else {
                context.startService(intent)
            }
        }

        fun updateProgress(
            context: Context,
            step: Int,
            totalSteps: Int,
            epoch: Int,
            totalEpochs: Int,
            loss: Float?,
            valLoss: Float?,
            speedTokSec: Float?
        ) {
            val intent = Intent(context, TrainingForegroundService::class.java).apply {
                action = ACTION_START
                putExtra(EXTRA_STEP, step)
                putExtra(EXTRA_TOTAL_STEPS, totalSteps)
                putExtra(EXTRA_EPOCH, epoch)
                putExtra(EXTRA_TOTAL_EPOCHS, totalEpochs)
                if (loss != null) putExtra(EXTRA_LOSS, loss)
                if (valLoss != null) putExtra(EXTRA_VAL_LOSS, valLoss)
                if (speedTokSec != null) putExtra(EXTRA_SPEED, speedTokSec)
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                context.startForegroundService(intent)
            } else {
                context.startService(intent)
            }
        }

        fun stopService(context: Context) {
            val intent = Intent(context, TrainingForegroundService::class.java).apply {
                action = ACTION_STOP
            }
            context.stopService(intent)
        }
    }

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()

        // Akuisisi WakeLock agar komputasi tensor C++/NDK tidak terinterupsi saat layar mati
        val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = powerManager.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK,
            "Nexora:TrainingWakeLock"
        ).apply {
            setReferenceCounted(false)
            acquire(3 * 60 * 60 * 1000L) // 3 jam batas maksimum proteksi baterai
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
            return START_NOT_STICKY
        }

        val step = intent?.getIntExtra(EXTRA_STEP, 0) ?: 0
        val totalSteps = intent?.getIntExtra(EXTRA_TOTAL_STEPS, 100) ?: 100
        val epoch = intent?.getIntExtra(EXTRA_EPOCH, 1) ?: 1
        val totalEpochs = intent?.getIntExtra(EXTRA_TOTAL_EPOCHS, 3) ?: 3
        val loss = if (intent?.hasExtra(EXTRA_LOSS) == true) intent.getFloatExtra(EXTRA_LOSS, 0f) else null
        val valLoss = if (intent?.hasExtra(EXTRA_VAL_LOSS) == true) intent.getFloatExtra(EXTRA_VAL_LOSS, 0f) else null
        val speed = if (intent?.hasExtra(EXTRA_SPEED) == true) intent.getFloatExtra(EXTRA_SPEED, 0f) else null

        val notification = buildNotification(step, totalSteps, epoch, totalEpochs, loss, valLoss, speed)
        startForeground(NOTIFICATION_ID, notification)

        return START_NOT_STICKY
    }

    private fun buildNotification(
        step: Int,
        totalSteps: Int,
        epoch: Int,
        totalEpochs: Int,
        loss: Float?,
        valLoss: Float?,
        speed: Float?
    ): Notification {
        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )

        val lossText = loss?.let { "%.3f".format(it) } ?: "..."
        val valLossText = valLoss?.let { " | Val: %.3f".format(it) } ?: ""
        val speedText = speed?.let { " (%.1f tok/s)".format(it) } ?: ""

        val content = "Epoch $epoch/$totalEpochs • Step $step/$totalSteps • Loss: $lossText$valLossText$speedText"

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.mipmap.ic_launcher)
            .setContentTitle("Nexora AI Training Aktif • NDK r30")
            .setContentText(content)
            .setStyle(NotificationCompat.BigTextStyle().bigText("$content\nKomputasi tensor SIMD berjalan di latar belakang."))
            .setProgress(totalSteps.coerceAtLeast(1), step.coerceAtLeast(0), false)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setContentIntent(pendingIntent)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Nexora Training Service",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Notifikasi status pelatihan model bahasa Nexora on-device"
                setShowBadge(false)
            }
            val manager = getSystemService(NotificationManager::class.java)
            manager?.createNotificationChannel(channel)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        try {
            if (wakeLock?.isHeld == true) {
                wakeLock?.release()
            }
        } catch (e: Exception) {
            // Ignore safe release
        }
        wakeLock = null
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
