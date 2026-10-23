package com.nexora.model

/**
 * Spesifikasi Arsitektur Neural Network LM5M Nexora:
 * - Decoder-only Causal Transformer
 * - Vocabulary: 4,096 tokens (GPT-2 Byte-Level BPE)
 * - Hidden dimension (d_model): 256
 * - Layers: 5
 * - Attention Heads: 4 (head_dim: 64)
 * - RoPE (Rotary Position Embeddings, theta: 10000.0, rotate_half)
 * - SwiGLU MLP: intermediate dimension 704 (w1 gate, w3 up, w2 down)
 * - Normalization: RMSNorm (eps: 1e-05)
 * - Output Head: Weight Tying langsung ke tok_emb [4096, 256]
 * - Dropout: 0.0 (tanpa dropout)
 * - Total Parameter Eksak: 5,065,472 parameter
 * - Pre-trained steps: 747
 */
data class NexoraConfig(
    val vocabSize: Int = 4096,
    val dModel: Int = 256,
    val dFf: Int = 704,
    val contextLength: Int = 256,
    val numLayers: Int = 5,
    val numHeads: Int = 4,
    val headDim: Int = 64, // dModel / numHeads = 256 / 4 = 64
    val dropout: Float = 0.0f, // Tanpa dropout per arsitektur Colab
    val eps: Float = 1e-5f,
    val ropeTheta: Float = 10000.0f,
    val totalParams: Long = 5065472L,
    val stepsTrained: Int = 747,
    // LoRA Parameters untuk fine-tuning adaptif
    val loraEnabled: Boolean = true,
    val loraRank: Int = 4,
    val loraAlpha: Float = 8.0f
) {
    val loraScale: Float get() = if (loraRank > 0) loraAlpha / loraRank else 1.0f
}
