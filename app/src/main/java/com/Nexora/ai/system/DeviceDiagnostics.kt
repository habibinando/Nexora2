package com.Nexora.ai.system

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import android.os.PowerManager
import java.io.File
import java.util.Locale

object DeviceDiagnostics {
    data class Report(
        val device: String,
        val androidVersion: String,
        val soc: String,
        val cpuDescription: String,
        val cpuCoresRuntime: Int,
        val cpuCoresNative: Int,
        val supportedAbis: String,
        val cpuFrequency: String,
        val totalRamBytes: Long,
        val advertisedRamBytes: Long?,
        val availableRamBytes: Long,
        val lowMemory: Boolean,
        val memoryClassMb: Int,
        val largeMemoryClassMb: Int,
        val appHeapUsedBytes: Long,
        val appHeapMaxBytes: Long,
        val thermalStatus: String
    )

    fun read(context: Context): Report {
        val app = context.applicationContext
        val manager = app.getSystemService(ActivityManager::class.java)
        val memory = ActivityManager.MemoryInfo().also(manager::getMemoryInfo)
        val runtime = Runtime.getRuntime()
        val heapUsed = (runtime.totalMemory() - runtime.freeMemory()).coerceAtLeast(0L)
        val cpuCount = runtime.availableProcessors().coerceAtLeast(1)
        val nativeCpuCount = try {
            com.Nexora.ai.ml.NativeOps.nativeCpuCores().coerceAtLeast(1)
        } catch (_: Throwable) {
            cpuCount
        }
        val soc = listOf(
            Build.SOC_MANUFACTURER.takeIf { it.isNotBlank() && !it.equals("unknown", true) },
            Build.SOC_MODEL.takeIf { it.isNotBlank() && !it.equals("unknown", true) }
        ).filterNotNull().joinToString(" ").ifBlank { Build.HARDWARE }
        val thermal = try {
            val pm = app.getSystemService(PowerManager::class.java)
            when (pm.currentThermalStatus) {
                PowerManager.THERMAL_STATUS_NONE -> "Normal"
                PowerManager.THERMAL_STATUS_LIGHT -> "Sedang hangat (light)"
                PowerManager.THERMAL_STATUS_MODERATE -> "Panas (moderate)"
                PowerManager.THERMAL_STATUS_SEVERE -> "Sangat panas (severe)"
                PowerManager.THERMAL_STATUS_CRITICAL -> "Kritis (critical)"
                PowerManager.THERMAL_STATUS_EMERGENCY -> "Darurat"
                PowerManager.THERMAL_STATUS_SHUTDOWN -> "Shutdown"
                else -> "Tidak diketahui"
            }
        } catch (_: Throwable) {
            "Tidak tersedia"
        }

        return Report(
            device = listOf(Build.MANUFACTURER, Build.MODEL).filter { it.isNotBlank() }.joinToString(" "),
            androidVersion = "\${Build.VERSION.RELEASE} (API \${Build.VERSION.SDK_INT})",
            soc = soc.ifBlank { "Tidak dilaporkan oleh sistem" },
            cpuDescription = readCpuDescription(),
            cpuCoresRuntime = cpuCount,
            cpuCoresNative = nativeCpuCount,
            supportedAbis = Build.SUPPORTED_ABIS.joinToString(", "),
            cpuFrequency = readCpuFrequency(cpuCount),
            totalRamBytes = memory.totalMem,
            advertisedRamBytes = if (Build.VERSION.SDK_INT >= 34) memory.advertisedMem.takeIf { it > 0L } else null,
            availableRamBytes = memory.availMem,
            lowMemory = memory.lowMemory,
            memoryClassMb = manager.memoryClass,
            largeMemoryClassMb = manager.largeMemoryClass,
            appHeapUsedBytes = heapUsed,
            appHeapMaxBytes = runtime.maxMemory(),
            thermalStatus = thermal
        )
    }

    private fun readCpuDescription(): String {
        return try {
            val lines = File("/proc/cpuinfo").readLines()
            val keys = listOf("hardware", "model name", "processor", "cpu model")
            for (key in keys) {
                val value = lines.firstNotNullOfOrNull { line ->
                    val pair = line.split(':', limit = 2)
                    if (pair.size == 2 && pair[0].trim().equals(key, ignoreCase = true)) {
                        pair[1].trim().takeIf { it.isNotBlank() && it.any(Char::isLetter) }
                    } else null
                }
                if (!value.isNullOrBlank()) return value
            }
            Build.HARDWARE.takeIf { it.isNotBlank() } ?: "Nama CPU tidak diekspos kernel"
        } catch (_: Throwable) {
            Build.HARDWARE.takeIf { it.isNotBlank() } ?: "Nama CPU tidak diekspos kernel"
        }
    }

    private fun readCpuFrequency(coreCount: Int): String {
        val frequencies = (0 until coreCount).mapNotNull { core ->
            try {
                val value = File("/sys/devices/system/cpu/cpu$core/cpufreq/cpuinfo_max_freq")
                    .readText().trim().toLongOrNull()
                value?.takeIf { it > 0L }?.div(1000L)
            } catch (_: Throwable) {
                null
            }
        }.distinct().sorted()
        if (frequencies.isEmpty()) return "Tidak diekspos sysfs"
        return if (frequencies.size == 1) "\${frequencies.first()} MHz"
        else "\${frequencies.first()}–\${frequencies.last()} MHz (maks core yang terbaca)"
    }

    fun formatBytes(bytes: Long): String {
        if (bytes < 0L) return "—"
        val gib = bytes.toDouble() / (1024.0 * 1024.0 * 1024.0)
        return String.format(Locale.US, "%.2f GiB", gib)
    }
}
