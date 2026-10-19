// Top-level build file for Nexora — on-device AI language training.
// Toolchain: JDK 25, Gradle 9.8.1, AGP 9.4.0+, NDK r30 (30.0.16248370)
plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.kotlin.android) apply false
    alias(libs.plugins.kotlin.compose) apply false
    alias(libs.plugins.kotlin.serialization) apply false
}
