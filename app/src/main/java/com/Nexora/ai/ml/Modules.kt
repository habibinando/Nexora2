package com.Nexora.ai.ml

import kotlin.math.sqrt
import kotlin.random.Random

/** Real decoder-only transformer (mini-GPT) with manual autograd. No mock. */

data class ModelConfig(
    val vocabSize: Int = 16000,
    val seqLen: Int = 128,
    val dModel: Int = 128,
    val ffnDim: Int = 512,
    val numHeads: Int = 8,
    val numLayers: Int = 32,
    val dropout: Float = 0.1f,
    val useLora: Boolean = true,
    val loraRank: Int = 8,
    val loraAlpha: Int = 16,
    val layerNormEps: Float = 1e-5f
) {
    val headDim: Int get() = dModel / numHeads
}

class Dropout(val p: Float) {
    private var mask: FloatArray? = null
    fun forward(x: FloatArray, training: Boolean, rng: Random): FloatArray {
        if (!training || p <= 0f) return x
        val scale = 1f / (1f - p)
        val m = FloatArray(x.size) { if (rng.nextFloat() < p) 0f else scale }
        mask = m
        for (i in x.indices) x[i] *= m[i]
        return x
    }
    fun backward(grad: FloatArray) {
        val m = mask ?: return
        for (i in grad.indices) grad[i] *= m[i]
    }
}

/** Linear layer y = xW + b + scale * x A B (LoRA). Base W frozen when LoRA active. */
class Linear(
    val inF: Int, val outF: Int, val useBias: Boolean = true,
    val loraRank: Int = 0, loraAlpha: Int = 16, rng: Random
) {
    val w = FloatArray(inF * outF).also { MathUtil.xavierLinear(it, inF, rng) }
    val wg = FloatArray(inF * outF)
    val bias = if (useBias) FloatArray(outF) else null
    val bg = if (useBias) FloatArray(outF) else null
    val loraA: FloatArray? = if (loraRank > 0) FloatArray(inF * loraRank).also { MathUtil.xavierLinear(it, inF, rng) } else null
    val loraB: FloatArray? = if (loraRank > 0) FloatArray(loraRank * outF) else null // init zeros
    val loraGA: FloatArray? = if (loraRank > 0) FloatArray(inF * loraRank) else null
    val loraGB: FloatArray? = if (loraRank > 0) FloatArray(loraRank * outF) else null
    val loraScale: Float = if (loraRank > 0) loraAlpha.toFloat() / loraRank else 0f
    val frozenBase: Boolean = loraRank > 0

    private var cacheX: FloatArray? = null
    private var cacheH: FloatArray? = null // x @ A, T x r

    fun forward(x: FloatArray, T: Int): FloatArray {
        cacheX = x
        val y = NativeOps.matmul(x, w, T, inF, outF) // x: T x inF, w: inF x outF
        if (bias != null) NativeOps.nativeAddBias(y, bias, T, outF)
        val A = loraA
        if (A != null) {
            val h = NativeOps.matmul(x, A, T, inF, loraRank)
            cacheH = h
            val ly = NativeOps.matmul(h, loraB!!, T, loraRank, outF)
            NativeOps.nativeAxpy(y, ly, loraScale, T * outF)
        }
        return y
    }

    fun backward(gy: FloatArray, T: Int): FloatArray {
        val x = cacheX!!
        if (!frozenBase) {
            // dW = x^T @ gy  (inF x T) @ (T x outF)
            NativeOps.nativeMatMul(x, gy, wg, inF, T, outF, true, false, 1f, 1f)
        }
        if (bg != null) {
            for (t in 0 until T) {
                val o = t * outF
                for (j in 0 until outF) bg[j] += gy[o + j]
            }
        }
        var gx = NativeOps.matmul(gy, w, T, outF, inF, transB = true) // gy @ w^T
        val A = loraA
        if (A != null) {
            val B = loraB!!
            val h = cacheH!!
            // dB = h^T @ gy * scale  (r x T) @ (T x outF)
            NativeOps.nativeMatMul(h, gy, loraGB!!, loraRank, T, outF, true, false, loraScale, 1f)
            // gyBt = gy @ B^T  (T x outF) @ (outF x r) = T x r
            val gyBt = NativeOps.matmul(gy, B, T, outF, loraRank, transB = true)
            // dA = x^T @ gyBt * scale  (inF x T) @ (T x r)
            NativeOps.nativeMatMul(x, gyBt, loraGA!!, inF, T, loraRank, true, false, loraScale, 1f)
            // gx += scale * gyBt @ A^T  (T x r) @ (r x inF)
            val lgx = NativeOps.matmul(gyBt, A, T, loraRank, inF, transB = true)
            NativeOps.nativeAxpy(gx, lgx, loraScale, T * inF)
        }
        return gx
    }

    fun zeroGrad() {
        wg.fill(0f); bg?.fill(0f); loraGA?.fill(0f); loraGB?.fill(0f)
    }

    /** Trainable tensors for the optimizer (LoRA adapters only when frozenBase). */
    fun trainableTensors(): List<FloatArray> {
        val out = mutableListOf<FloatArray>()
        if (!frozenBase) out += w
        if (bias != null) out += bias
        if (loraA != null) { out += loraA; out += loraB!! }
        return out
    }
    fun trainableGrads(): List<FloatArray> {
        val out = mutableListOf<FloatArray>()
        if (!frozenBase) out += wg
        if (bg != null) out += bg
        if (loraGA != null) { out += loraGA; out += loraGB!! }
        return out
    }
    fun allTensors(): List<FloatArray> = listOfNotNull(w, bias, loraA, loraB)
}

class Embedding(val vocab: Int, val dim: Int, rng: Random) {
    val w = FloatArray(vocab * dim).also { MathUtil.fillNormal(it, 0f, 0.02f, rng) }
    val wg = FloatArray(vocab * dim)
    private var cacheIds: IntArray? = null
    fun forward(ids: IntArray): FloatArray {
        cacheIds = ids
        val out = FloatArray(ids.size * dim)
        for (t in ids.indices) System.arraycopy(w, ids[t] * dim, out, t * dim, dim)
        return out
    }
    fun backward(gy: FloatArray) {
        val ids = cacheIds!!
        for (t in ids.indices) {
            val r = ids[t] * dim; val o = t * dim
            for (d in 0 until dim) wg[r + d] += gy[o + d]
        }
    }
    fun zeroGrad() { wg.fill(0f) }
}

class LayerNorm(val dim: Int, val eps: Float) {
    val gamma = FloatArray(dim) { 1f }
    val beta = FloatArray(dim)
    val gGamma = FloatArray(dim)
    val gBeta = FloatArray(dim)
    private var cacheX: FloatArray? = null
    private var mean: FloatArray? = null
    private var rstd: FloatArray? = null

    fun forward(x: FloatArray, T: Int): FloatArray {
        cacheX = x
        val out = FloatArray(x.size)
        val m = FloatArray(T); val rs = FloatArray(T)
        for (t in 0 until T) {
            val base = t * dim
            var mu = 0f; for (d in 0 until dim) mu += x[base + d]; mu /= dim
            var v = 0f; for (d in 0 until dim) { val dv = x[base + d] - mu; v += dv * dv }; v /= dim
            val inv = 1f / sqrt(v + eps)
            m[t] = mu; rs[t] = inv
            for (d in 0 until dim) {
                val xh = (x[base + d] - mu) * inv
                out[base + d] = gamma[d] * xh + beta[d]
            }
        }
        mean = m; rstd = rs
        return out
    }

    fun backward(gy: FloatArray, T: Int): FloatArray {
        val x = cacheX!!; val m = mean!!; val rs = rstd!!
        val gx = FloatArray(gy.size)
        val invN = 1f / dim
        for (t in 0 until T) {
            val base = t * dim
            var s1 = 0f; var s2 = 0f
            for (d in 0 until dim) {
                val xh = (x[base + d] - m[t]) * rs[t]
                val dxh = gy[base + d] * gamma[d]
                gGamma[d] += gy[base + d] * xh
                gBeta[d] += gy[base + d]
                s1 += dxh; s2 += dxh * xh
            }
            for (d in 0 until dim) {
                val xh = (x[base + d] - m[t]) * rs[t]
                val dxh = gy[base + d] * gamma[d]
                gx[base + d] = invN * rs[t] * (dim * dxh - s1 - xh * s2)
            }
        }
        return gx
    }
    fun zeroGrad() { gGamma.fill(0f); gBeta.fill(0f) }
}

class CausalSelfAttention(val cfg: ModelConfig, rng: Random) {
    private val d = cfg.dModel; private val H = cfg.numHeads; private val hd = cfg.headDim
    val qkv = Linear(d, 3 * d, useBias = true, if (cfg.useLora) cfg.loraRank else 0, cfg.loraAlpha, rng)
    val proj = Linear(d, d, useBias = true, if (cfg.useLora) cfg.loraRank else 0, cfg.loraAlpha, rng)
    private val drop = Dropout(cfg.dropout)
    private var B = 0; private var S = 0
    private var Q: FloatArray? = null // BH x S x hd
    private var K: FloatArray? = null
    private var V: FloatArray? = null
    private var attn: FloatArray? = null // BH x S x S, probabilities after dropout
    private var softmaxAttn: FloatArray? = null // BH x S x S, probabilities before dropout
    private var dropMask: FloatArray? = null // BH x S x S, one independent mask per head

    fun forward(x: FloatArray, B: Int, S: Int, training: Boolean, rng: Random): FloatArray {
        this.B = B; this.S = S
        val T = B * S
        val all = qkv.forward(x, T) // T x 3d
        val q = FloatArray(T * d); val k = FloatArray(T * d); val v = FloatArray(T * d)
        for (t in 0 until T) {
            System.arraycopy(all, t * 3 * d, q, t * d, d)
            System.arraycopy(all, t * 3 * d + d, k, t * d, d)
            System.arraycopy(all, t * 3 * d + 2 * d, v, t * d, d)
        }
        val BH = B * H
        val Qr = FloatArray(BH * S * hd); val Kr = FloatArray(BH * S * hd); val Vr = FloatArray(BH * S * hd)
        for (b in 0 until B) for (h in 0 until H) for (s in 0 until S) {
            val bh = b * H + h
            val src = (b * S + s) * d + h * hd
            val dst = (bh * S + s) * hd
            System.arraycopy(q, src, Qr, dst, hd)
            System.arraycopy(k, src, Kr, dst, hd)
            System.arraycopy(v, src, Vr, dst, hd)
        }
        Q = Qr; K = Kr; V = Vr
        val scale = 1f / sqrt(hd.toFloat())
        val At = FloatArray(BH * S * S)
        val softAt = FloatArray(BH * S * S)
        val masks = if (training && cfg.dropout > 0f) {
            FloatArray(BH * S * S) { if (rng.nextFloat() < cfg.dropout) 0f else 1f / (1f - cfg.dropout) }
        } else null
        val out = FloatArray(BH * S * hd)
        for (bh in 0 until BH) {
            val ao = bh * S * S; val qo = bh * S * hd; val ko = bh * S * hd; val vo = bh * S * hd
            for (i in 0 until S) {
                for (j in 0 until S) {
                    var s = 0f
                    val qbase = qo + i * hd; val kbase = ko + j * hd
                    for (e in 0 until hd) s += Qr[qbase + e] * Kr[kbase + e]
                    At[ao + i * S + j] = if (j > i) -1e9f else s * scale
                }
                MathUtil.softmaxRow(At, ao + i * S, S)
            }
            System.arraycopy(At, ao, softAt, ao, S * S)
            if (masks != null) for (i in 0 until S * S) At[ao + i] *= masks[ao + i]
            // out = At @ V  (S x S) @ (S x hd)
            for (i in 0 until S) {
                for (e in 0 until hd) {
                    var s = 0f
                    for (j in 0 until S) s += At[ao + i * S + j] * Vr[vo + j * hd + e]
                    out[qo + i * hd + e] = s
                }
            }
        }
        attn = At
        softmaxAttn = softAt
        dropMask = masks
        // rearrange out back to T x d
        val y = FloatArray(T * d)
        for (b in 0 until B) for (h in 0 until H) for (s in 0 until S) {
            val bh = b * H + h
            val src = (bh * S + s) * hd
            val dst = (b * S + s) * d + h * hd
            System.arraycopy(out, src, y, dst, hd)
        }
        return proj.forward(y, T)
    }

    fun backward(gy: FloatArray): FloatArray {
        val B = this.B; val S = this.S; val BH = B * H; val T = B * S
        val d = this.d; val hd = this.hd; val H = this.H
        val gYproj = proj.backward(gy, T) // T x d
        // rearrange gYproj to BH x S x hd
        val gOut = FloatArray(BH * S * hd)
        for (b in 0 until B) for (h in 0 until H) for (s in 0 until S) {
            val bh = b * H + h
            val src = (b * S + s) * d + h * hd
            val dst = (bh * S + s) * hd
            System.arraycopy(gYproj, src, gOut, dst, hd)
        }
        val At = attn!!
        val probabilities = softmaxAttn!!
        val Vr = V!!; val Qr = Q!!; val Kr = K!!
        val gAt = FloatArray(BH * S * S)
        val gV = FloatArray(BH * S * hd)
        val gQ = FloatArray(BH * S * hd)
        val gK = FloatArray(BH * S * hd)
        val mask = dropMask
        val scoreScale = 1f / sqrt(hd.toFloat())
        for (bh in 0 until BH) {
            val ao = bh * S * S; val qo = bh * S * hd; val ko = bh * S * hd; val vo = bh * S * hd
            // gV = At^T @ gOut ; gAt = gOut @ V^T
            for (i in 0 until S) for (e in 0 until hd) {
                var s = 0f
                for (j in 0 until S) s += At[ao + j * S + i] * gOut[qo + j * hd + e]
                gV[vo + i * hd + e] = s
            }
            for (i in 0 until S) for (j in 0 until S) {
                var s = 0f
                for (e in 0 until hd) s += gOut[qo + i * hd + e] * Vr[vo + j * hd + e]
                gAt[ao + i * S + j] = s
            }
            // Dropout backward, then softmax backward using the pre-dropout probabilities.
            if (mask != null) for (i in 0 until S * S) gAt[ao + i] *= mask[ao + i]
            for (i in 0 until S) {
                var rowSum = 0f
                for (j in 0 until S) {
                    val idx = ao + i * S + j
                    rowSum += gAt[idx] * probabilities[idx]
                }
                for (j in 0 until S) {
                    val idx = ao + i * S + j
                    gAt[idx] = probabilities[idx] * (gAt[idx] - rowSum)
                }
            }
            // causal: j>i grad should be zero (masked) — already softmaxed to 0, ds ~0, but enforce
            for (i in 0 until S) for (j in i + 1 until S) gAt[ao + i * S + j] = 0f
            // gQ = gAt @ K ; gK = gAt^T @ Q
            for (i in 0 until S) for (e in 0 until hd) {
                var sq = 0f; var sk = 0f
                for (j in 0 until S) {
                    sq += gAt[ao + i * S + j] * Kr[ko + j * hd + e]
                    sk += gAt[ao + j * S + i] * Qr[qo + j * hd + e]
                }
                gQ[qo + i * hd + e] = sq * scoreScale
                gK[ko + i * hd + e] = sk * scoreScale
            }
        }
        // rearrange gQ,gK,gV back to T x d and concat
        val gQKV = FloatArray(T * 3 * d)
        for (b in 0 until B) for (h in 0 until H) for (s in 0 until S) {
            val bh = b * H + h
            val src = (bh * S + s) * hd
            val dstQ = (b * S + s) * d + h * hd
            System.arraycopy(gQ, src, gQKV, (b * S + s) * 3 * d + 0 * d + h * hd, hd)
            System.arraycopy(gK, src, gQKV, (b * S + s) * 3 * d + 1 * d + h * hd, hd)
            System.arraycopy(gV, src, gQKV, (b * S + s) * 3 * d + 2 * d + h * hd, hd)
        }
        return qkv.backward(gQKV, T)
    }

    fun zeroGrad() { qkv.zeroGrad(); proj.zeroGrad() }
    fun linears() = listOf(qkv, proj)
}

class MLP(val cfg: ModelConfig, rng: Random) {
    val fc1 = Linear(cfg.dModel, cfg.ffnDim, useBias = true, if (cfg.useLora) cfg.loraRank else 0, cfg.loraAlpha, rng)
    val fc2 = Linear(cfg.ffnDim, cfg.dModel, useBias = true, if (cfg.useLora) cfg.loraRank else 0, cfg.loraAlpha, rng)
    private val drop = Dropout(cfg.dropout)
    private var cachePreGelu: FloatArray? = null
    fun forward(x: FloatArray, T: Int, training: Boolean, rng: Random): FloatArray {
        var h = fc1.forward(x, T)
        cachePreGelu = h.copyOf()
        NativeOps.nativeGeluInplace(h, h.size)
        h = drop.forward(h, training, rng)
        return fc2.forward(h, T)
    }
    fun backward(gy: FloatArray, T: Int): FloatArray {
        var gh = fc2.backward(gy, T)
        drop.backward(gh)
        NativeOps.nativeGeluBackwardInplace(gh, cachePreGelu!!, gh.size)
        return fc1.backward(gh, T)
    }
    fun zeroGrad() { fc1.zeroGrad(); fc2.zeroGrad() }
    fun linears() = listOf(fc1, fc2)
}

class Block(val cfg: ModelConfig, rng: Random) {
    val ln1 = LayerNorm(cfg.dModel, cfg.layerNormEps)
    val attn = CausalSelfAttention(cfg, rng)
    val ln2 = LayerNorm(cfg.dModel, cfg.layerNormEps)
    val mlp = MLP(cfg, rng)
    fun forward(x: FloatArray, B: Int, S: Int, training: Boolean, rng: Random): FloatArray {
        val a = x.let {
            val h = ln1.forward(it, B * S)
            val att = attn.forward(h, B, S, training, rng)
            val out = it.clone()
            MathUtil.add(out, att); out
        }
        val h2 = ln2.forward(a, B * S)
        val m = mlp.forward(h2, B * S, training, rng)
        val out = a.clone()
        MathUtil.add(out, m)
        return out
    }
    fun backward(gy: FloatArray, T: Int): FloatArray {
        val gm = mlp.backward(gy, T)
        val gLn2 = ln2.backward(gm, T)
        val gA = gy.clone(); MathUtil.add(gA, gLn2)
        val gAtt = attn.backward(gA)
        val gLn1 = ln1.backward(gAtt, T)
        val gx = gA.clone(); MathUtil.add(gx, gLn1)
        return gx
    }
    fun zeroGrad() { ln1.zeroGrad(); attn.zeroGrad(); ln2.zeroGrad(); mlp.zeroGrad() }
    fun linears() = attn.linears() + mlp.linears()
    fun norms() = listOf(ln1, ln2)
}

class GPT(val cfg: ModelConfig, val rng: Random = Random(1337)) {
    val tokEmb = Embedding(cfg.vocabSize, cfg.dModel, rng)
    val posEmb = FloatArray(cfg.seqLen * cfg.dModel).also { MathUtil.fillNormal(it, 0f, 0.02f, rng) }
    val posG = FloatArray(cfg.seqLen * cfg.dModel)
    val blocks = List(cfg.numLayers) { Block(cfg, rng) }
    val lnF = LayerNorm(cfg.dModel, cfg.layerNormEps)
    private val drop = Dropout(cfg.dropout)
    private var cacheX: FloatArray? = null
    private var cacheTokens: IntArray? = null
    private var B = 0; private var S = 0

    val numParams: Long
        get() {
            var n = tokEmb.w.size.toLong() + posEmb.size
            for (b in blocks) for (l in b.linears()) n += l.allTensors().sumOf { it.size.toLong() }
            n += lnF.gamma.size + lnF.beta.size
            for (b in blocks) for (ln in b.norms()) n += ln.gamma.size + ln.beta.size
            return n
        }
    val trainableParams: Long
        get() {
            var n = tokEmb.w.size.toLong() + posEmb.size + lnF.gamma.size + lnF.beta.size
            for (b in blocks) {
                for (l in b.linears()) n += l.trainableTensors().sumOf { it.size.toLong() }
                for (ln in b.norms()) n += ln.gamma.size + ln.beta.size
            }
            return n
        }

    fun forward(tokens: IntArray, B: Int, S: Int, training: Boolean): FloatArray {
        this.B = B; this.S = S; cacheTokens = tokens
        val T = B * S
        var x = tokEmb.forward(tokens)
        for (t in 0 until T) {
            val pos = t % S
            val o = t * cfg.dModel; val po = pos * cfg.dModel
            for (d in 0 until cfg.dModel) x[o + d] += posEmb[po + d]
        }
        x = drop.forward(x, training, rng)
        for (block in blocks) x = block.forward(x, B, S, training, rng)
        x = lnF.forward(x, T)
        cacheX = x
        // logits = x @ tokEmb.w^T (weight tying): x T x d, w vocab x d -> need w^T d x vocab
        return NativeOps.matmul(x, tokEmb.w, T, cfg.dModel, cfg.vocabSize, transB = true)
    }

    /** gradLogits: T x vocab (already softmax-onehot scaled). Returns loss. */
    fun backward(gradLogits: FloatArray): FloatArray {
        val T = B * S; val d = cfg.dModel
        val x = cacheX!!
        // dX = gradLogits @ tokEmb.w  (T x vocab) @ (vocab x d)
        var dX = NativeOps.matmul(gradLogits, tokEmb.w, T, cfg.vocabSize, d)
        // tied weight grad: dW = gradLogits^T @ x  (vocab x T) @ (T x d)
        NativeOps.nativeMatMul(gradLogits, x, tokEmb.wg, cfg.vocabSize, T, d, true, false, 1f, 1f)
        dX = lnF.backward(dX, T)
        for (i in blocks.indices.reversed()) dX = blocks[i].backward(dX, T)
        drop.backward(dX)
        // pos emb grad
        val tokens = cacheTokens!!
        for (t in 0 until T) {
            val pos = t % S; val o = t * d; val po = pos * d
            for (dd in 0 until d) posG[po + dd] += dX[o + dd]
        }
        tokEmb.backward(dX)
        return dX
    }

    fun zeroGrad() {
        tokEmb.zeroGrad(); posG.fill(0f); lnF.zeroGrad()
        for (b in blocks) b.zeroGrad()
    }

    /** All trainable (tensor, grad) pairs for optimizer + checkpoint. */
    fun trainablePairs(): List<Pair<FloatArray, FloatArray>> {
        val out = mutableListOf<Pair<FloatArray, FloatArray>>()
        out += tokEmb.w to tokEmb.wg
        out += posEmb to posG
        out += lnF.gamma to lnF.gGamma
        out += lnF.beta to lnF.gBeta
        for (b in blocks) {
            for (ln in b.norms()) { out += ln.gamma to ln.gGamma; out += ln.beta to ln.gBeta }
            for (l in b.linears()) {
                val ts = l.trainableTensors(); val gs = l.trainableGrads()
                for (i in ts.indices) out += ts[i] to gs[i]
            }
        }
        return out
    }

    /** All tensors (for full checkpoint save/load, including frozen base weights). */
    fun allTensors(): List<FloatArray> {
        val out = mutableListOf(tokEmb.w, posEmb, lnF.gamma, lnF.beta)
        for (b in blocks) {
            for (ln in b.norms()) { out += ln.gamma; out += ln.beta }
            for (l in b.linears()) out += l.allTensors()
        }
        return out
    }
}

/** Cross-entropy over next-token prediction. Returns loss; fills gradLogits. */
fun crossEntropy(logits: FloatArray, targets: IntArray, vocab: Int, gradLogits: FloatArray): Float {
    val T = targets.size
    var loss = 0f
    for (t in 0 until T) {
        val o = t * vocab
        MathUtil.softmaxRow(logits, o, vocab)
        val p = logits[o + targets[t]].coerceAtLeast(1e-9f)
        loss += -kotlin.math.log(p)
        for (v in 0 until vocab) gradLogits[o + v] = (logits[o + v] - if (v == targets[t]) 1f else 0f) / T
    }
    return loss / T
}
