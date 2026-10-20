package com.Nexora.ai

import android.app.Application
import com.Nexora.ai.ml.NativeOps
import com.Nexora.ai.train.TrainingController
import com.Nexora.ai.ui.AppSettings

class NexoraApp : Application() {
    override fun onCreate() {
        super.onCreate()
        // Restore saved training settings before loading an existing checkpoint.
        AppSettings.initialize(this)
        NativeOps.nativeSetThreads(NativeOps.nativeCpuCores())
        TrainingController.init(this)
    }
}
