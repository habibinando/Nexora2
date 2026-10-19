# Nexora — On-Device AI Language Training (com.Nexora.ai)

Aplikasi Android **100% on-device** untuk melatih model bahasa transformer kecil
langsung di ponsel. Tanpa cloud API, tanpa mock, tanpa contoh bawaan, tanpa kata
"Gemini". Bobot model dilatih nyata (forward + backward manual), LoRA nyata,
checkpoint periodik sehingga **bobot terus bertambah** antar sesi.

## Toolchain
- JDK 25 (`jvmToolchain(25)`)
- Gradle 9.8.1 (wrapper)
- AGP 9.4.0
- NDK r30 LTS (`ndkVersion "30.0.16248370"`)
- CMake 3.22.1, C++20, NEON + pthread, OpenGL ES 3.1 compute
- compileSdk / targetSdk 37, minSdk 26, `largeHeap="true"`

## Cara build
1. Buka folder ini di Android Studio (2026.x / Iguana+). Studio akan mengambil
   Gradle 9.8.1 + dependensi secara otomatis.
2. Pastikan SDK Platform 37, NDK r30, dan CMake terpasang (SDK Manager).
3. `./gradlew :app:assembleDebug` (atau Build > Make Project).
4. Instal di perangkat Android 8.0+ (minSdk 26). GPU compute opsional.

## Fitur nyata (tidak ada mock)
| Fitur | Lokasi kode |
|---|---|
| Transformer decoder (dModel 128, FFN 512, 32 layer, 8 head, seq 128, vocab 16k) + backward manual | `ml/Modules.kt` |
| LoRA (rank/alpha dapat diatur, base weight dibekukan) pada Q/K/V/O & MLP | `Linear` di `Modules.kt` |
| AdamW + weight decay + gradient clipping | `ml/Optimizer.kt` |
| Cosine LR schedule dengan warmup | `CosineWarmupScheduler` |
| Dropout (inverted, mask disimpan untuk backward) | `Dropout` |
| Early stopping pada validation loss (patience dapat diatur) | `TrainingController.trainLoop` |
| Checkpoint periodik + resume bobot/optimizer/tokenizer | `data/CheckpointManager.kt` |
| Impor dataset streaming JSONL / CSV / TXT, maks 500 MB, vocab 16k dibangun dari dataset | `data/DatasetImporter.kt`, `data/Tokenizer.kt` |
| Detail training live: loss, val loss, best, LR, grad norm, token/detik, elapsed, ETA, kurva loss | `ui/screens/TrainScreen.kt` |
| Chat dengan model hasil latihan (generasi token-per-token, top-k sampling) | `ml/Inference.kt`, `ui/screens/ChatScreen.kt` |
| Benchmark nyata: CPU GFLOPS (NEON+thread), GPU GFLOPS (GLES compute), throughput training, info perangkat & RAM | `benchmark/BenchmarkRunner.kt`, `cpp/gles_compute.cpp` |
| Maksimalkan CPU: thread pool sejumlah core, NEON intrinsik, `largeHeap=true` | `cpp/native.cpp`, `AndroidManifest.xml` |
| GPU compute: OpenGL ES 3.1 compute shader (tiled shared memory), fallback CPU | `cpp/gles_compute.cpp` |
| Foreground service agar training berjalan saat layar mati | `train/TrainingService.kt` |
| Loading screen animasi + SplashScreen API | `ui/MainActivity.kt` |
| UI modern 2026: gelap OLED, glassmorphism, aksen teal-cyan, Material 3, Compose | `ui/theme/Theme.kt`, `ui/components/Components.kt` |

## Paket & nama
- applicationId / package: `com.Nexora.ai`
- Nama aplikasi: `Nexora` (`@string/app_name`)

## Catatan
- Tidak ada permission INTERNET → aplikasi sepenuhnya offline.
- Tokenizer dibangun dari dataset yang Anda impor (word-level + byte fallback).
- Untuk kecepatan terbaik, gunakan LoRA (default aktif) — trainable params jauh
  lebih kecil sementara base weight tetap disimpan di checkpoint.
- Ubah arsitektur model di tab **Lab**; diterapkan pada impor dataset berikutnya.
