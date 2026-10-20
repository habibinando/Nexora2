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
- CMake 3.22.1, C++20, NEON + pthread, OpenGL ES 3.1 compute
- compileSdk / targetSdk 37, minSdk 26, `largeHeap="true"`

## Cara build
1. Buka folder ini di Android Studio (2026.x / Iguana+). Studio akan mengambil
   Gradle 9.8.1 + dependensi secara otomatis.
2. Pastikan SDK Platform 37 (`android.jar`), NDK r30, dan CMake terpasang. CI memeriksa `android.jar` agar instalasi SDK yang tidak lengkap tidak lolos diam-diam.
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
| Status fase training (forward/backprop/optimizer/evaluasi/checkpoint), error terlihat, loss, val loss, LR, grad norm, token/detik, elapsed, ETA, kurva loss | `ui/screens/TrainScreen.kt` |
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

## Referensi teknis dan validasi GPU

- [Android NDK r30 changelog](https://github.com/android/ndk/wiki/Changelog-r30): r30 memperbarui Clang dan sysroot hingga API 37. Toolchain ini membantu membangun kode native; pemasangan NDK sendiri tidak membuat training otomatis berjalan di GPU.
- [Android NDK shader compilers](https://developer.android.com/ndk/guides/graphics/shader-compilers): halaman ini membahas shader **Vulkan** yang menggunakan SPIR-V. Nexora saat ini memakai **OpenGL ES 3.1 compute**, yang mengirim sumber GLSL ke driver saat runtime; karena itu Shaderc/SPIR-V tidak ditambahkan tanpa migrasi backend Vulkan yang sesungguhnya.
- [PyTorch Android model preparation recipe](https://pytorch-cn.com/tutorials/recipes/model_preparation_android.html): contoh tersebut menyiapkan model TorchScript/quantized untuk inferensi. Itu bukan pengganti loop backpropagation/optimizer Nexora, jadi tidak dianggap sebagai akselerasi training.
- Sebelum GEMM GPU dipakai, Nexora membandingkan hasilnya dengan GEMM CPU untuk matriks 17×19 dikali 19×13. Ukuran non-kelipatan 16 membantu menguji batas tile shader. Bila inisialisasi atau hasil tes gagal, operasi GEMM dialihkan ke CPU.
- Counter CPU/GPU di layar training direset pada awal setiap sesi training agar hasil benchmark terdahulu tidak tercampur ke angka sesi baru. Build CI membuktikan APK dapat dikompilasi; verifikasi numerik tetap perlu dijalankan pada perangkat Android target.

## Catatan
- LiteRT adalah runtime inferensi/delegates, bukan pengganti langsung untuk melatih transformer custom FloatArray ini. Integrasi membutuhkan jalur ekspor model yang kompatibel, sehingga dependency LiteRT tidak ditambahkan sekadar sebagai label GPU.
- Workflow CI menjalankan Gradle dengan JDK 25 tetapi menghasilkan bytecode Java 21. Instalasi Android API 37 diverifikasi dengan pemeriksaan `android.jar`.
- Tidak ada permission INTERNET → aplikasi sepenuhnya offline.
- Tokenizer dibangun dari dataset yang Anda impor (word-level + byte fallback).
- Untuk kecepatan terbaik, gunakan LoRA (default aktif) — trainable params jauh
  lebih kecil sementara base weight tetap disimpan di checkpoint.
- Ubah arsitektur model di tab **Lab**; diterapkan pada impor dataset berikutnya.
