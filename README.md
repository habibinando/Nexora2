# Nexora — On-Device AI Language Training (com.Nexora.ai)

Aplikasi Android **100% on-device** untuk melatih model bahasa transformer kecil
langsung di ponsel. Tanpa cloud API, tanpa mock, tanpa contoh bawaan, tanpa kata
"Gemini". Bobot model dilatih nyata (forward + backward manual), LoRA nyata,
checkpoint periodik sehingga **bobot terus bertambah** antar sesi.

## Toolchain
- JDK 25 for the Gradle/Kotlin build toolchain; Android bytecode target remains Java 21
- Gradle 9.8.1 (wrapper)
- AGP 9.4.0
- NDK r30 LTS (`ndkVersion "30.0.16248370"`)
- Native CPU kernels: ARM NEON + multithreaded GEMM
- GPU compute: Vulkan 1.0 + precompiled SPIR-V; no OpenGL ES compute backend
- CMake 3.22.1, C++20, ARM NEON + native thread pool, Vulkan loader
- compileSdk / targetSdk 37, minSdk 31, `largeHeap="true"`

## Cara build
1. Buka folder ini di Android Studio (2026.x / Iguana+). Studio akan mengambil
   Gradle 9.8.1 + dependensi secara otomatis.
2. Pastikan SDK Platform 37 (`android.jar`), NDK r30, CMake 3.22.1, dan compiler shader Vulkan `glslc` tersedia. CI memeriksa `android.jar` dan memasang `glslc` sebelum build.
3. Jalankan `./gradlew :app:assembleDebug` (atau Build > Make Project). Task shader membutuhkan executable `glslc` yang tersedia di `PATH`.
4. Instal di perangkat Android 12+ (minSdk 31). Vulkan opsional saat runtime; backend CPU tetap tersedia.

GitHub Actions hanya membangun dan mengunggah APK **debug**. Workflow tidak membangun APK release.

## Fitur nyata (tidak ada mock)
| Fitur | Lokasi kode |
|---|---|
| Transformer decoder (default dModel 128, FFN 512, 8 layer, 8 head, seq 64, vocab 16k) + backward manual | `ml/Modules.kt` |
| LoRA opsional pada Q/K/V/O & MLP; default nonaktif untuk training dari nol karena base weight awalnya acak | `Linear` di `Modules.kt`, `ui/AppSettings.kt` |
| AdamW + weight decay + gradient clipping | `ml/Optimizer.kt` |
| Cosine LR schedule dengan warmup | `CosineWarmupScheduler` |
| Dropout (inverted, mask disimpan untuk backward) | `Dropout` |
| Early stopping pada validation loss (patience dapat diatur) | `TrainingController.trainLoop` |
| Checkpoint periodik + resume bobot/optimizer/tokenizer | `data/CheckpointManager.kt` |
| Impor dataset streaming JSONL / CSV / TXT, maks 500 MB, vocab 16k dibangun dari dataset | `data/DatasetImporter.kt`, `data/Tokenizer.kt` |
| Status fase training (forward/backprop/optimizer/evaluasi/checkpoint), error terlihat, loss, val loss, LR, grad norm, token/detik, elapsed, ETA, kurva loss | `ui/screens/TrainScreen.kt` |
| Chat dengan model hasil latihan (generasi token-per-token, top-k sampling) | `ml/Inference.kt`, `ui/screens/ChatScreen.kt` |
| Benchmark: CPU GFLOPS (NEON+thread), Vulkan compute GEMM, mini-training throughput, CPU/GPU counters, info perangkat & RAM | `benchmark/BenchmarkRunner.kt`, `cpp/vulkan_compute.cpp`, `main/shaders/matmul.comp` |
| Maksimalkan CPU: thread pool sejumlah core, NEON intrinsik, `largeHeap=true` | `cpp/native.cpp`, `AndroidManifest.xml` |
| GPU compute: Vulkan compute shader tiled GEMM + tes akurasi, fallback CPU | `cpp/vulkan_compute.cpp`, `main/shaders/matmul.comp` |
| Foreground service agar training berjalan saat layar mati | `train/TrainingService.kt` |
| Loading screen animasi + SplashScreen API | `ui/MainActivity.kt` |
| UI modern 2026: gelap OLED, glassmorphism, aksen teal-cyan, Material 3, Compose | `ui/theme/Theme.kt`, `ui/components/Components.kt` |

## Paket & nama
- applicationId / package: `com.Nexora.ai`
- Nama aplikasi: `Nexora` (`@string/app_name`)

## Vulkan compute migration

Nexora no longer uses the OpenGL ES/EGL compute path. The native GPU path is Vulkan compute with a GLSL compute shader compiled to SPIR-V during the Android build. The app reads `assets/shaders/matmul.comp.spv` and sends GEMM work through a Vulkan compute pipeline. The CPU NEON/thread-pool implementation remains the fallback when Vulkan is unavailable, the buffer limits are exceeded, or the numerical checks fail.

The initial runtime test compares a 17×19 by 19×13 GPU result with the CPU reference in all four transpose combinations before GPU GEMM is accepted. This confirms shader correctness on edge tiles and transpose layouts, but it does not by itself prove GPU speedup; the benchmark reports end-to-end time including buffers, dispatch, synchronization, and result readback.

Minimum Android SDK is now API 31 (Android 12). Vulkan is an optional device feature, so sideloading remains possible on unsupported devices and the CPU backend can still run.

## Referensi teknis dan validasi GPU

- [Android NDK r30 changelog](https://github.com/android/ndk/wiki/Changelog-r30): r30 memperbarui Clang dan sysroot hingga API 37. Toolchain ini membantu membangun kode native; pemasangan NDK sendiri tidak membuat training otomatis berjalan di GPU.
- [Android NDK shader compilers](https://developer.android.com/ndk/guides/graphics/shader-compilers): halaman ini membahas shader **Vulkan** yang menggunakan SPIR-V. Nexora memakai jalur tersebut dengan shader SPIR-V yang dikompilasi pada saat build, bukan shader OpenGL ES yang dikompilasi driver saat runtime.
- [PyTorch Android model preparation recipe](https://pytorch-cn.com/tutorials/recipes/model_preparation_android.html): contoh tersebut menyiapkan model TorchScript/quantized untuk inferensi. Itu bukan pengganti loop backpropagation/optimizer Nexora, jadi tidak dianggap sebagai akselerasi training.
  Dokumentasi PyTorch yang lebih baru juga menyatakan PyTorch Mobile tidak lagi aktif didukung dan menyarankan melihat [ExecuTorch](https://docs.pytorch.org/tutorials/recipes/mobile_interpreter.html). Mengganti runtime tetap perlu jalur model yang cocok; itu bukan pengganti otomatis untuk backpropagation dan optimizer custom Nexora.
- Sebelum GEMM GPU dipakai, Nexora membandingkan hasilnya dengan GEMM CPU untuk matriks 17×19 dikali 19×13. Ukuran non-kelipatan 16 membantu menguji batas tile shader. Bila inisialisasi atau hasil tes gagal, operasi GEMM dialihkan ke CPU.
- Counter CPU/GPU di layar training direset pada awal setiap sesi training agar hasil benchmark terdahulu tidak tercampur ke angka sesi baru. Build CI membuktikan APK dapat dikompilasi; verifikasi numerik tetap perlu dijalankan pada perangkat Android target.

## Catatan
- LiteRT adalah runtime inferensi/delegates, bukan pengganti langsung untuk melatih transformer custom FloatArray ini. Integrasi membutuhkan jalur ekspor model yang kompatibel, sehingga dependency LiteRT tidak ditambahkan sekadar sebagai label GPU.
- Workflow CI menjalankan Gradle dengan JDK 25 tetapi menghasilkan bytecode Java 21. Instalasi Android API 37 diverifikasi dengan pemeriksaan `android.jar`.
- Tidak ada permission INTERNET → aplikasi sepenuhnya offline.
- Tokenizer dibangun dari dataset yang Anda impor (word-level + byte fallback).
- Untuk training dari nol, LoRA default nonaktif agar semua bobot linear belajar. Aktifkan LoRA hanya saat memakai bobot dasar terlatih yang kompatibel; saat aktif, base weights dibekukan dan jumlah parameter trainable berkurang.
- Ubah arsitektur model di tab **Lab**; diterapkan pada impor dataset berikutnya.
