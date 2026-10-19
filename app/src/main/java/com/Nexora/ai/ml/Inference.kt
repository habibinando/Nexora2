package com.Nexora.ai.ml

import com.Nexora.ai.data.Tokenizer
import kotlin.math.exp
import kotlin.random.Random

/** Real autoregressive generation with the actually trained model. No mock. */
object Inference {

    data class GenConfig(
        val maxNewTokens: Int = 64,
        val temperature: Float = 0.8f,
        val topK: Int = 40,
        val seed: Long = 42
    )

    fun generate(
        model: GPT, tokenizer: Tokenizer, prompt: String,
        cfg: GenConfig = GenConfig(),
        onToken: (String) -> Unit = {}
    ): String {
        val rng = Random(cfg.seed)
        var ctx = tokenizer.encode(prompt, addBos = true, addEos = false).toMutableList()
        val outIds = mutableListOf<Int>()
        val S = model.cfg.seqLen
        repeat(cfg.maxNewTokens) {
            val window = ctx.takeLast(S).toIntArray()
            val logits = model.forward(window, 1, window.size, training = false)
            // logits: T x vocab ; take last row
            val vocab = model.cfg.vocabSize
            val rowOffset = (window.size - 1) * vocab
            // sample
            val next = sample(logits, rowOffset, vocab, cfg.temperature, cfg.topK, rng)
            if (next == tokenizer.eosId) return@repeat
            ctx += next; outIds += next
            onToken(tokenizer.decode(intArrayOf(next)))
            if (outIds.size >= cfg.maxNewTokens) return@repeat
        }
        return tokenizer.decode(outIds.toIntArray())
    }

    private fun sample(logits: FloatArray, offset: Int, vocab: Int, temp: Float, topK: Int, rng: Random): Int {
        val t = temp.coerceAtLeast(1e-4f)
        // find top-k indices
        val idx = IntArray(vocab) { it }
        // partial sort: take topK
        val sorted = idx.sortedByDescending { logits[offset + it] }.take(topK).toIntArray()
        val probs = FloatArray(topK) { exp((logits[offset + sorted[it]] - logits[offset + sorted[0]]) / t) }
        var sum = 0f; for (p in probs) sum += p
        var draw = rng.nextFloat() * sum
        for (i in probs.indices) {
            draw -= probs[i]
            if (draw <= 0f) return sorted[i]
        }
        return sorted.last()
    }
}
