+++
title = "Canonicalization and the E8M0 Power-of-Two Trick"
description = "MX block scales are powers of two, which would make scale-folding an exact exponent-field add. Here's why v1 sets that trick up and never uses it."
slug = "canonicalization-e8m0-power-of-two"
date = 2026-08-17
weight = 2
draft = true
[taxonomies]
categories = ["MX-Quantization Dialect"]
tags = ["mlir", "compilers", "quantization", "canonicalization"]
+++

<!-- KaTeX includes (inline; requires markdown.render_unsafe = true) -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/contrib/auto-render.min.js"></script>
<script>
 document.addEventListener("DOMContentLoaded", function() {
 renderMathInElement(document.body, {
 delimiters: [
 {left: "$$", right: "$$", display: true},
 {left: "\\[", right: "\\]", display: true},
 {left: "$", right: "$", display: false},
 {left: "\\(", right: "\\)", display: false}
 ],
 throwOnError: false
 });
 });
</script>

<!-- SECTION 1 — the op named after a trick it doesn't perform -->
## The fold in the name

The dialect has an operation called `mx.fold_scale`. The name describes a specific, elegant rewrite: a scale-only multiply sitting in front of a block-scaled value can be folded *into* the shared scale instead of touching the data. For a block-scaled tensor, the value model is $X_i = S \cdot M_i$ : one shared scale $S$ per block, one mantissa $M_i$ per element ([Post 1](@/posts/designing-mx-dialect.md) covers the type and the four ops). Multiplying the whole block by a scalar $\alpha$ is then just scalar associativity:

$$\alpha \cdot (S \cdot M_i) = (\alpha \cdot S) \cdot M_i$$

Push $\alpha$ into $S$ and you rescale an entire block by rewriting one scale value,
leaving every mantissa untouched. That is what "fold scale" means.

v1 never does it.

The scale $S$ is stored as `f8E8M0FNU`: an 8-bit exponent, zero mantissa bits, representing powers of two and nothing else (Post 1 leans on this; it's the whole reason the format is cheap). That structure is what would make the fold exact: if $\alpha$ is itself a power of two, then $\alpha \cdot S$ is a power of two too, and folding becomes a single integer add on the exponent field. No rounding, no new storage, type-safe. It is one of the cleanest rewrites the format allows. And v1 ships three canonicalization patterns, none of which perform it.

This post is about that gap. The exponent-field add is the intuition the op is named for; it is not the code v1 runs. What v1 runs is three rewrites that *clear* `fold_scale` out of the IR before it reaches lowering: by merging constants, dropping identities, and relocating the scale into the f32 domain, never once writing an E8M0 scale. Why v1 clears the op instead of lowering it is the point. First, what makes the fold exact at all.

<!-- SECTION 2 — the exponent-field add: the intuition you're about to not use -->
## Why powers of two make it exact

Start with what `f8E8M0FNU` actually stores. Eight exponent bits, zero mantissa
bits. The value it represents is $2^e$ for the stored exponent $e$, nothing else.
There is no fraction to carry, no mantissa to round. The representable set is
exactly the powers of two, and the encoding is just the integer $e$.

Now multiply two of them. A scale $S = 2^e$, a scalar $\alpha$ that also happens to be a
power of two, $\alpha = 2^a$:

$$\alpha \cdot S = 2^{a} \cdot 2^{e} = 2^{a+e}$$

The product is a power of two, so it lands exactly on a representable E8M0 value, assuming $a+e$ stays within the exponent's range; the add is exact, the encoding can still overflow at the extremes. And computing it doesn't require a floating-point multiply at all: the stored exponents add. $a + e$ is an integer add on the exponent field. No rounding step exists in that path, so there is nothing to lose. That is the exponent-field add, and it is why folding a power-of-two scalar into an E8M0 scale is exact by construction, not exact by luck.

The alternative shows what "exact" is buying you. Take $\alpha = 4.5$. Now:

$$\alpha \cdot S = 4.5 \cdot 2^{e}$$

That is not a power of two. It sits between $2^{e+2} = 4 \cdot 2^{e}$ and $2^{e+3} = 8 \cdot 2^{e}$, and `f8E8M0FNU` has no encoding for anything in between. To store it you would have to round to one neighbor or the other, and neither is the value you wanted. So folding a non-power-of-two scalar into the scale is not a cheaper version of the same operation. It is a different operation that silently corrupts the scale, or a verifier failure if you check. "Exact" means the result stays inside E8M0's representable set with no rounding; only a power-of-two $\alpha$ does that.

This is the trick the operation is named for. It is genuinely elegant: a scale
multiply reduced to an integer add, exact, touching one scale value instead of a
whole block. Everything in the setup points at implementing it.

<!-- SECTION 3 — why v1 doesn't absorb into S: the causal chain -->
## Why v1 doesn't do it

The reason isn't that the exponent-field add is hard. It's one integer add. The
reason is what implementing it would drag in.

Folding $\alpha$ into $S$ means writing a *new scale value*. The add $a + e$ produces an exponent that has to be stored back into the scale tensor. That is not a rewrite of the IR's structure: it's a computation that produces data and puts it somewhere. And producing-and-storing a value is what lowering does, not what canonicalization does. Canonicalization rewrites IR into simpler IR at the same level; it never emits the runtime arithmetic that computes a scale and writes it
down. So the elegant one-line trick, taken seriously, is a lowering.

That reclassification is the whole cost. A `fold_scale` with a real lowering path is an op that survives canonicalization and arrives at the conversion stage needing a conversion pattern, a bufferized result, and — the part that matters — a guard. Because once $\alpha$ reaches an E8M0 scale, a non-power-of-two $\alpha$ is a live correctness bug. $4.5 \cdot S$ has to round to a wrong power of two, silently. The lowering can't be shipped without a check that $\alpha$ is a power of two, rejecting or specially handling everything else. The guard isn't optional decoration on the lowering; it's the lowering's admission that the trick only works on the values it was designed for.

v1's thesis doesn't need any of that. The point of v1 is the dialect design and
the canonicalization contract: that `fold_scale` is well-formed, that it composes,
and that it can be cleared from the IR before lowering. A runtime scale-rewrite
proves none of that better than clearing the op does, and it costs a whole
lowering path plus a mandatory guard. So v1 makes the opposite call. `fold_scale`
stays canonicalization-only: no lowering, no guard, no scale ever rewritten. The
op is not lowered: it is *eliminated*, before the conversion stage ever sees it.

That single decision is what the three patterns exist to carry out. If `fold_scale` will never be lowered, then every `fold_scale` in the IR has to be gone by the time conversion runs, because conversion has no pattern for it and a surviving op is a hard error. Clearing it is not an optimization. It is the thing that makes the canonicalization-only decision safe.

## Three patterns, three shapes

If `fold_scale` is never lowered, every one that appears has to be gone before
conversion runs. v1 does that with three canonicalization patterns. None of them
touches $S$. Each matches a different shape of IR and clears it a different way.

**Merge — `FoldScalePow2`.** Two `fold_scale` ops stacked on the same value
collapse into one. Given

```mlir
%inner = mx.fold_scale %t, %alpha1 : !mx.tensor<...>, f32 -> !mx.tensor<...>
%outer = mx.fold_scale %inner, %alpha2 : !mx.tensor<...>, f32 -> !mx.tensor<...>
```

with `%alpha1 = 4.0` and `%alpha2 = 8.0`, the pattern replaces both with a single `fold_scale` carrying the product:

```mlir
%c = arith.constant 3.200000e+01 : f32   // 4.0 * 8.0, folded at compile time
%r = mx.fold_scale %t, %c : !mx.tensor<...>, f32 -> !mx.tensor<...>
```

It fires only when both scalars are compile-time constants, because it computes
the product right there with an `APFloat` multiply and bakes the result in. The
greedy driver reapplies it, so a three-deep stack ($2.0$, $4.0$, $8.0$) collapses
to one op holding $64.0$. Two honest notes. There is no power-of-two check on the
constants despite the name: it folds $4.0 \cdot 8.0$ the same as it would fold any
two floats, because the product is never written into an E8M0 scale, so a
non-power-of-two result harms nothing. And the fold is an f32 `APFloat` multiply,
not the exponent-field add from the intuition. Same result for constant inputs,
different mechanism: the exponent-field add is *why folding would be exact*; the
`APFloat` multiply is *what the code does*.

**Eliminate — `FoldScaleIdentity`.** Scaling by one is a no-op. The pattern checks
the scalar is exactly $1.0$ and rewires every use of the result straight to the
input:

```mlir
%r = mx.fold_scale %t, %one : ...   // %one = 1.0
// disappears; uses of %r become %t
```

No new op. This is elimination, not folding.

**Relocate — `DequantizeFoldScaleFusion`.** A cross-op rewrite, and the only one
that clears a `fold_scale` by moving its scalar somewhere else. Given a
`dequantize_block` fed by a `fold_scale`:

```mlir
%s = mx.fold_scale %t, %alpha : !mx.tensor<...>, f32 -> !mx.tensor<...>
%d = mx.dequantize_block %s : !mx.tensor<...> -> tensor<32x64xf32>
```

it dequantizes the original tensor first, then applies $\alpha$ as an ordinary f32
multiply on the result:

```mlir
%d = mx.dequantize_block %t : !mx.tensor<...> -> tensor<32x64xf32>
%sp = tensor.splat %alpha : tensor<32x64xf32>
%r  = arith.mulf %d, %sp : tensor<32x64xf32>
```

The pattern matches on the *consumer* (`dequantize_block`) and walks backward to its producer with `getDefiningOp<FoldScaleOp>()`. If the producer is a `fold_scale`, it rewrites the pair. The `fold_scale` is gone; the scaling survives as a plain float multiply on the dequantized tensor. Fusion never evaluates $\alpha$, it just hands it to a runtime `arith.mulf`, so a runtime $\alpha$ works as well as a constant: nothing here asks whether $\alpha$ is a power of two, or even whether it's known at compile time. When $\alpha$ *is* a constant, the greedy driver folds the splat into a `dense` attribute (`arith.constant dense<4.000000e+00> : tensor<32x64xf32>`); when it's a function argument, the `tensor.splat %arg` stays. Same pattern, two shapes of output.

Where each pattern is registered matters. `FoldScalePow2` and `FoldScaleIdentity`
hang off `FoldScaleOp`: they match a `fold_scale` directly. `DequantizeFoldScaleFusion`
hangs off `DeQuantizeBlockOp`, because a canonicalization pattern is registered on
the op it *matches*, and this one matches the dequantize consumer, not the
`fold_scale` it removes. You register on the op you match, not the op you delete.

Three shapes: a stack, an identity, a dequantize-consumer. Three clearings: merge, eliminate, relocate. All three are `OpRewritePattern` — greedy-driver canonicalizations, opportunistic and level-preserving. Not one writes a scale.

## What the patterns don't cover

Three patterns clear three shapes: a constant chain, an identity, a dequantize-consumer. Their coverage is the union of those three, and the union has holes.

Take a single `fold_scale` whose result feeds something other than a
`dequantize_block`, a function return or a `block_matmul`:
```mlir
%s = mx.fold_scale %t, %c : !mx.tensor<...>, f32 -> !mx.tensor<...>
return %s : !mx.tensor<...>
```
No pattern fires. `FoldScalePow2` needs a *stack* of two `fold_scale` ops; there's
only one. `FoldScaleIdentity` needs $\alpha$ to be exactly $1.0$; this $\alpha$ isn't.
`DequantizeFoldScaleFusion` needs the consumer to be a `dequantize_block`; a
`return` isn't. So the op survives canonicalization untouched, and here's the part that matters: whether `%c` is a clean power of two like $2.0$ is irrelevant. No pattern even inspects it, because no pattern absorbs $\alpha$ into a scale. The power-of-two question, the whole exactness story from the first half of this post, never comes up.

What happens to the survivor is the point. It flows into the conversion pass, which
marks the `mx` dialect illegal and must translate every `mx` op to `linalg`.
`fold_scale` has no conversion pattern, by design; it was only ever meant to be
canonicalized away. So the conversion driver hits an illegal op it cannot rewrite
and fails the whole compilation:
```mlir
error: failed to legalize operation 'mx.fold_scale' that was explicitly marked illegal
```

That is not a bug. It's the boundary v1 drew, enforcing itself. The two drivers behave oppositely on an unmatched op: canonicalization is opportunistic, so no match is a no-op; conversion is mandatory, so no match is a hard error. A `fold_scale` the three patterns don't clear sails through the first driver and dies in the second. The failure is loud and it happens at compile time, before any wrong number is produced. Compare the alternative: a lowering that quietly rounds $4.5 \cdot S$ to the wrong power of two and ships a silently corrupt scale. Between a clear compile-time failure on an unsupported input and a silent runtime corruption, v1 takes the failure.

Making the coverage total is possible, but it takes a real `fold_scale` lowering: one that applies $\alpha$ without leaning on a downstream `dequantize_block` to absorb it. That's a cleaner design on the coverage axis, and more than v1 needs. Its job is the dialect and the canonicalization contract, and for the shapes v1 supports, the three patterns clear every `fold_scale` before conversion.

## What it would take to make it exact

The real lowering is where the power-of-two restriction has to be enforced. The exponent-field add from the top of this post, the exact one-integer-add fold the op is named for, is what it would run.

Writing a new scale means computing $\alpha \cdot S$ and storing it into the `f8E8M0FNU` scale tensor. In v1, $\alpha$ is harmless no matter what it is: `FoldScalePow2` multiplies $4.5$ into another f32 constant and leaves it as an f32 value on the op. f32 holds $4.5$ exactly, and the value never reaches a scale. The moment a lowering tries to write $\alpha \cdot S$ into the scale, that changes. If $\alpha = 4.5$, then $\alpha \cdot S$ is not a power of two, and `f8E8M0FNU` cannot store it: a verifier failure, or a silent round to the wrong power of two, the exact corruption v1 avoided by never writing the scale at all.

So the guard `FoldScalePow2` conspicuously lacks becomes mandatory the instant absorption is real. The lowering checks that $\alpha$ is a power of two, takes the exponent-field add when it is, and widens to a larger scale type (carrying $\alpha$ separately in f32) when it isn't. That check isn't defensive boilerplate. It's the lowering admitting what the E8M0 format demanded all along: the fold is exact only on the values the scale type can represent, and the exponent-field add is what "exact" looks like when it can.

v1 keeps `fold_scale` on the canonicalization side of the line, where $\alpha$ never touches a scale and the power-of-two question never has to be asked. v1.5 moves it across, at which point the question is unavoidable and the guard is the answer. The op has carried the name of the exact fold the whole time. v1 just never had to make it exact.