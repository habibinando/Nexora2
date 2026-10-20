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
    ): String = synchronized(model) {
        // Training and inference share layer caches; keep a full generation session atomic
        // with respect to training steps, not just each individual forward pass.
        val rng = Random(cfg.seed)
        var ctx = tokenizer.encode(prompt, addBos = true, addEos = false).toMutableList()
        val outIds = mutableListOf<Int>()
        val S = model.cfg.seqLen
        for (step in 0 until cfg.maxNewTokens) {
            val window = ctx.takeLast(S).toIntArray()
            val logits = model.forward(window, 1, window.size, training = false)
            // logits: T x vocab; take the last row.
            val vocab = model.cfg.vocabSize
            val rowOffset = (window.size - 1) * vocab
            val next = sample(logits, rowOffset, vocab, cfg.temperature, cfg.topK, rng)
            if (next == tokenizer.eosId) break
            ctx += next
            outIds += next
            // Emit the whole decoded prefix so UTF-8 byte-fallback tokens remain valid.
            onToken(tokenizer.decode(outIds.toIntArray()))
        }
        tokenizer.decode(outIds.toIntArray())
    }

    private fun sample(logits: FloatArray, offset: Int, vocab: Int, temp: Float, topK: Int, rng: Random): Int {
        require(offset >= 0 && vocab > 0 && offset.toLong() + vocab <= logits.size.toLong()) {
            "Logit range is outside the output buffer."
        }
        val k = topK.coerceIn(1, vocab)
        val t = temp.takeIf { it.isFinite() }?.coerceAtLeast(1e-4f) ?: 0.8f

        // Keep only K candidates in a primitive min-heap instead of boxing and sorting
        // every vocabulary index for every generated token. Complexity becomes O(V log K),
        // allocations stay bounded by K, and the token probabilities remain exact top-K.
        val heap = IntArray(k)
        var size = 0
        fun score(token: Int): Float = logits[offset + token]

        for (candidate in 0 until vocab) {
            if (size < k) {
                var child = size
                heap[child] = candidate
                size++
                while (child > 0) {
                    val parent = (child - 1) / 2
                    if (score(heap[parent]) <= score(heap[child])) break
                    val tempIndex = heap[parent]
                    heap[parent] = heap[child]
                    heap[child] = tempIndex
                    child = parent
                }
            } else if (score(candidate) > score(heap[0])) {
                heap[0] = candidate
                var parent = 0
                while (true) {
                    val left = parent * 2 + 1
                    if (left >= size) break
                    val right = left + 1
                    var smallest = left
                    if (right < size && score(heap[right]) < score(heap[left])) smallest = right
                    if (score(heap[parent]) <= score(heap[smallest])) break
                    val tempIndex = heap[parent]
                    heap[parent] = heap[smallest]
                    heap[smallest] = tempIndex
                    parent = smallest
                }
            }
        }

        var maxLogit = Float.NEGATIVE_INFINITY
        for (token in heap) maxLogit = maxOf(maxLogit, score(token))
        val probs = FloatArray(k) { index -> exp((score(heap[index]) - maxLogit) / t) }
        var sum = 0f
        for (probability in probs) sum += probability
        var draw = rng.nextFloat() * sum
        for (index in probs.indices) {
            draw -= probs[index]
            if (draw <= 0f) return heap[index]
        }
        return heap.maxBy { score(it) }
    }
}
