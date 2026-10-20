// Nexora — on-device AI language training (no cloud API, no mock)
// JDK 25 build toolchain; Android bytecode target remains Java 21.
plugins {
    alias(libs.plugins.appPlugin)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.kotlin.serialization)
}

val generatedVulkanAssets = layout.buildDirectory.dir("generated/vulkanAssets")
val compileVulkanShader by tasks.registering(Exec::class) {
    val shaderSource = file("src/main/shaders/matmul.comp")
    val shaderOutput = generatedVulkanAssets.get().file("shaders/matmul.comp.spv").asFile
    inputs.file(shaderSource)
    outputs.file(shaderOutput)
    doFirst {
        if (!shaderSource.isFile) {
            throw GradleException("Vulkan shader source is missing: ${shaderSource.absolutePath}")
        }
        shaderOutput.parentFile.mkdirs()
        commandLine(
            "glslc",
            "--target-env=vulkan1.0",
            "-O",
            shaderSource.absolutePath,
            "-o",
            shaderOutput.absolutePath
        )
    }
}

android {
    namespace = "com.Nexora.ai"
    compileSdk = 37
    ndkVersion = "30.0.16248370" // NDK r30 LTS

    sourceSets.getByName("main").assets.srcDir(generatedVulkanAssets)

    defaultConfig {
        applicationId = "com.Nexora.ai"
        minSdk = 31          // Android 12+; Vulkan compute preferred, CPU fallback remains
        targetSdk = 37
        versionCode = 1
        versionName = "1.0.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        vectorDrawables { useSupportLibrary = true }

        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++20 -O3 -ffast-math -fno-exceptions -fno-rtti"
                arguments += "-DANDROID_STL=c++_static"
                abiFilters += listOf("arm64-v8a", "x86_64")
            }
        }
    }

    packaging {
        jniLibs { useLegacyPackaging = false }
        resources.excludes += setOf(
            "META-INF/{AL2.0,LGPL2.1}",
            "META-INF/*.kotlin_module"
        )
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
        debug {
            isMinifyEnabled = false
            applicationIdSuffix = ".debug"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_21
        targetCompatibility = JavaVersion.VERSION_21
        isCoreLibraryDesugaringEnabled = true
    }
    buildFeatures {
        compose = true
        buildConfig = true
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
}

// Compile the compute shader explicitly: AGP does not reliably package .comp sources
// as SPIR-V across plugin versions. The generated asset is included in debug and release APKs.
tasks.configureEach {
    if (name.startsWith("merge") && name.endsWith("Assets")) {
        dependsOn(compileVulkanShader)
    }
}

kotlin {
    jvmToolchain(25)
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_21)
        freeCompilerArgs.add("-opt-in=kotlin.RequiresOptIn")
    }
}

dependencies {
    coreLibraryDesugaring("com.android.tools:desugar_jdk_libs:2.1.4")

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.core.splashscreen)
    implementation(libs.androidx.navigation.compose)
    implementation(libs.androidx.datastore.preferences)

    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.ui)
    implementation(libs.androidx.ui.graphics)
    implementation(libs.androidx.ui.tooling.preview)
    implementation(libs.androidx.material3)
    implementation(libs.androidx.material.icons.extended)
    debugImplementation(libs.androidx.ui.tooling)

    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.kotlinx.serialization.json)
}
