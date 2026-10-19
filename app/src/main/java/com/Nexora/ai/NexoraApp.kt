package com.Nexora.ai

import android.app.Application
import com.Nexora.ai.ml.NativeOps
import com.Nexora.ai.train.TrainingController

class NexoraApp : Application() {
    override fun onCreate() {
        super.onCreate()
        // Eagerly load native core & size thread pool to all CPU cores (maximize CPU).
        NativeOps.nativeSetThreads(NativeOps.nativeCpuCores())
        TrainingController.init(this)
    }
}
