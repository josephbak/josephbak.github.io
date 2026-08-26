+++
title = "Lowering MX Block-Matmul to linalg.generic with Block-Scale Affine Maps"
description = "Lowering a block-scaled matmul to linalg.generic: four indexing maps, one floordiv on the scale operand, and why that single choice makes the lowering clean and later blocks vectorization."
slug = "lowering-mx-block-matmul"
date = 2026-08-24
weight = 3
draft = false
[taxonomies]
categories = ["MX-Quantization Dialect"]
tags = ["mlir", "compilers", "quantization"]
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

## Why the matmul can't be a matmul

A matrix multiply contracts two operands along a shared dimension. Both are uniform: every element is one value of one type, addressed the same way. `linalg.matmul` is built for exactly that, and it's the obvious target for lowering a matmul.

Block-scaled matmul breaks the uniformity on one side. In v1, only the `A` operand is quantized; `B` stays a dense `f32` tensor. The quantized operand `A` is not one tensor. It's a mantissa tensor at full resolution and a scale tensor at one value per 32-element block along the contraction axis. For a `32×64` `A`, that is a `32×64` mantissa in `f8E4M3FN` and a `32×2` scale in `f8E8M0FNU`: the scale is 32 times coarser along `K` than the mantissa it belongs to. `A`'s true value at each position is `mantissa · scale_for_its_block`, and the two tensors are addressed on different strides.

`linalg.matmul` cannot express that. It wants two uniform, identically-addressed operands. The only way to hand it a block-scaled `A` is to reconstruct the full-resolution values first: multiply every mantissa by its block's scale into a dense `32×64` `f32` tensor, then multiply that by `B`. There is no ready-made op for it upstream either. The `linalg` structured ops are `generic`, `matmul`, `contract`, `batch_matmul`, and a handful of elementwise and reduction ops; none of them carries a block scale. The scale-aware primitives that do exist, `arith.scaling_extf` and `arith.scaling_truncf`, are elementwise conversions, not a contraction.

Reconstructing `A` is the one thing worth avoiding. The mantissa lives in `f8E4M3FN`, one byte per element. Expanding it to an `f32` reconstruction quadruples that to four bytes per element, and the resulting `32×64` tensor exists only to be read once by the multiply and thrown away. In a workload whose entire premise is cutting memory traffic, materializing a fat intermediate purely to feed a rigid op is the loss the lowering exists to prevent. Quantizing `A` and then immediately reconstructing it in full precision defeats the reason for quantizing it.

With the named ops ruled out, and reconstruction the only way to use one, the lowering target is `linalg.generic`, which lets each operand carry its own indexing map. That single choice, a generic with a custom map on the scale operand, is what the rest of this post is about. Four indexing maps, and one of them does all the work.

## Where the block scaling lives

The op going in is a single `mx.block_matmul` with three operands:

```mlir
module {
  func.func @block_matmul(%arg0: !mx.tensor<32x64xf8E4M3FN, block_size = 32, scale_type = f8E8M0FNU>, %arg1: tensor<64x64xf32>, %arg2: tensor<32x64xf32>) -> tensor<32x64xf32> {
    %0 = mx.block_matmul %arg0, %arg1, %arg2 : <32x64xf8E4M3FN, block_size = 32, scale_type = f8E8M0FNU>, tensor<64x64xf32>, tensor<32x64xf32> -> tensor<32x64xf32>
    return %0 : tensor<32x64xf32>
  }
}
```

The block-scaled operand `%arg0` is one `!mx.tensor` value. The mantissa and its block scale are both inside that type; nothing about the split is visible yet. `%arg1` (`B`) and `%arg2` (the accumulator) are ordinary `f32` tensors. [Post 1](@/posts/designing-mx-dialect.md) covers how the type carries the block metadata.

Here is what `mx.block_matmul` lowers to:

```mlir
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 32)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @block_matmul(%arg0: tensor<32x64xf8E4M3FN>, %arg1: tensor<32x2xf8E8M0FNU>, %arg2: tensor<64x64xf32>, %arg3: tensor<32x64xf32>) -> tensor<32x64xf32> {
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1, %arg2 : tensor<32x64xf8E4M3FN>, tensor<32x2xf8E8M0FNU>, tensor<64x64xf32>) outs(%arg3 : tensor<32x64xf32>) {
    ^bb0(%in: f8E4M3FN, %in_0: f8E8M0FNU, %in_1: f32, %out: f32):
      %1 = arith.extf %in : f8E4M3FN to f32
      %2 = arith.extf %in_0 : f8E8M0FNU to f32
      %3 = arith.mulf %1, %2 : f32
      %4 = arith.mulf %3, %in_1 : f32
      %5 = arith.addf %out, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<32x64xf32>
    return %0 : tensor<32x64xf32>
  }
}
```

Set the payload region aside for the next section and look at the four indexing maps. The iteration space is `(d0, d1, d2)`, which is `(m, n, k)`: rows of the output, columns of the output, and the contraction axis. Three of the four maps are the ordinary matmul projections. `#map`, the mantissa, reads `(m, k)`. `#map2`, `B`, reads `(k, n)`. `#map3`, the accumulator, reads `(m, n)`. Each is a bare pair of iteration dimensions, passed straight through.

`#map1`, the scale, is the one that isn't ordinary:

```mlir
#map1 = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 32)>
```

As the contraction sweeps `k` from 0 to 63, the mantissa is read at every `k`, but the scale is read at `k floordiv 32`. For `k` in 0 through 31 that index is 0; at `k = 32` it becomes 1. One scale element is shared across a whole block of 32 mantissa elements, which is exactly what a block-scaled format means. The `32×2` shape of the scale operand is the same fact seen from the other side: 64 contraction positions divided into blocks of 32 gives 2 scale columns, one per block, `K / 32`.

That single `floordiv` is the entire block-scaled semantics. Strip it out and this is a plain matmul; put it in and the scale operand is addressed at block resolution while everything else stays at element resolution. That asymmetry, the thing `linalg.matmul` could not express, is one arithmetic expression in one of four maps.

The maps say which element each iteration reads. The next section is what the iteration does with them.

## The payload reconstructs `A` one element at a time

Return to the `^bb0` region of that lowered op, on its own (the comments are added here; `mx-opt` prints the region without them):

```mlir
^bb0(%in: f8E4M3FN, %in_0: f8E8M0FNU, %in_1: f32, %out: f32):
  %1 = arith.extf %in   : f8E4M3FN  to f32   // mantissa -> f32
  %2 = arith.extf %in_0 : f8E8M0FNU to f32   // scale    -> f32
  %3 = arith.mulf %1, %2 : f32               // A_real = mantissa * scale
  %4 = arith.mulf %3, %in_1 : f32            // A_real * B
  %5 = arith.addf %out, %4 : f32             // accumulate into C
  linalg.yield %5 : f32
```

Four block arguments, one per operand, in the order the maps were listed: the mantissa scalar, the scale scalar, the `B` scalar, and the running accumulator. Each is a single value, already selected by that operand's indexing map for this `(m, n, k)`. The payload never sees a tensor. It sees four numbers and produces one.

The work is a fused dequantize-then-multiply-accumulate. Widen the mantissa to `f32`, widen the scale to `f32`, multiply them to reconstruct `A`'s true value at this position, multiply that by `B`, add to the accumulator. The reconstruction `mantissa * scale` and the contraction `* B` happen in the same region, on the same scalar, in registers. `A`'s full-precision value exists for the duration of one multiply-accumulate and is gone.

That is the point of lowering to a fused generic rather than a dequantize op followed by a matmul. Dequantize-then-matmul would compute every `A_real`, write all of them into a dense `32×64` `f32` tensor, and then read that tensor back in the matmul. The fused form computes each `A_real` exactly where it is consumed and never writes it anywhere. The reconstructed `A` is never a value in the program: `%3` is a scalar in a register, produced and consumed inside one loop iteration, not a tensor-typed result that has to live in memory. There is no `32×64` `f32` intermediate to write and reread. Keeping `A` in its one-byte form instead of expanding it to a four-byte reconstruction is the memory-traffic win the dialect exists for.

The grouping is deliberate. Floating-point multiply is not associative: each product rounds to the nearest representable value, and regrouping changes which intermediate gets rounded. The payload computes `(mantissa * scale) * B`, not `(mantissa * B) * scale`. Because the E8M0 scale is a power of two, `mantissa * scale` is exact whenever the result stays in the format's normal range: multiplying by a power of two only shifts the exponent and leaves the mantissa bits untouched, so nothing rounds. (At the extremes, a product that overflows to infinity or underflows into the subnormal range can still lose precision; the v1 lowering assumes in-range blocks.) Reconstructing `A`'s true value first and doing the one lossy multiply against `B` last is faithful to what the op means: dequantize `A`, then contract. The other grouping would round a meaningless `mantissa * B` intermediate and then scale the rounding error.

One thing the IR does not do: exploit that the scale is a power of two. At the IR level `arith.mulf %1, %2` is a general `f32` multiply, and nothing records that `%2` is constrained to powers of two. A backend emits an ordinary floating-point multiply, not the exponent-add the power-of-two case would in principle allow. The exactness is a property to reason about for correctness, not an optimization the v1 lowering expresses.

## One floordiv, two bars

`#map1`, the scale map, puts a `floordiv` inside an indexing map: `(d0, d2 floordiv 32)`. The `linalg.generic` verifies with it; a named matmul op would not.

`linalg.matmul` and `linalg.contract` require every indexing map to be a projected permutation: each result is a bare iteration dimension, passed through with no arithmetic. `d2 floordiv 32` is arithmetic on a dimension, so it is not a projected permutation, and a named op's verifier refuses it outright with "provided affine_map is not a projected permutation."

`linalg.generic` does not impose that requirement. Its verifier asks a weaker, structural question: taken together, do the operands' indexing maps let you recover the iteration space? Concatenate all four maps into one map from loops to operand coordinates, then try to invert it. If the concatenated map inverts, the op is well-formed; if it doesn't, the op is rejected. The check is on the whole system, not on each map in isolation.

The scale map, on its own, does not invert: we cannot recover `k` from `k floordiv 32`, since 32 different values of `k` collapse to the same block index. But it does not have to invert on its own. The mantissa map `(m, k)` exposes both `m` and `k` directly, and the `B` map `(k, n)` adds `n`. All three of `m`, `n`, and `k` are recoverable from operands other than the scale, so the scale map rides along on coordinates the system has already pinned down. The concatenated system inverts, and the generic accepts it.

Whether a non-projective map is fatal depends entirely on that whole-system question. The `quantize` op in this same dialect shows the other outcome. Its per-block reduction wants to address a block index `b` and a within-block index `j` at once, and written naively its input map reads `(i, b, j) -> (i, b * 32 + j)`, two iteration dimensions fused into one coordinate. There the fusion is fatal, because nothing recovers them: no other operand exposes `b` and `j` separately, so `b * 32 + j = 32` could be `(b=1, j=0)` or `(b=0, j=32)` and the system cannot tell which. The map does not invert, the op is rejected, and `quantize` has to reshape its input with `tensor.expand_shape` so `b` and `j` land on separate axes first. The matmul's `floordiv` never hits this. It transforms one dimension that another operand already pins down, so there is nothing left to recover.

This is why the lowering targets a generic in the first place. The op set was designed to be expressible as `linalg.generic`, not abstracted away from it, and the block-scale access is exactly the kind of map a generic permits and a named matmul op does not. The escape hatch is deliberate: `linalg.generic` trades the per-map guarantee for a whole-system one, and the block scale needs precisely that trade.

## The second bar comes later

The generic cleared the `floordiv`, but it did not make it a projected permutation. It only postponed the moment that matters.

The per-map bar that a named matmul op enforces at definition does not disappear when we lower to a generic instead. It reappears in the structured vectorizer. To vectorize a `linalg.generic`, every indexing map must be a projected permutation, the same requirement `linalg.matmul` would have imposed up front. The vectorizer will look at `d2 floordiv 32`, see arithmetic on a dimension, and refuse it, exactly as the named op would have.

Nothing about the map changed between lowering and vectorization. What changed is which check it faces. The generic waived the per-map bar and asked only that the whole system invert, so the `floordiv` passed. The vectorizer does not waive it. The same access that read cleanly one pass ago is the one it cannot take as written.

So the choice that made the lowering clean is the choice that blocks vectorization. One `floordiv`, legal in the op it lowered to and rejected by the transform that has to run on it next. That collision, and what it takes past it, is a problem for later in the series.

## The map and what follows

Post 4 takes a different thread down the pipeline: how one `!mx.tensor` value becomes two memory buffers, and how the accumulator this op carries turns into an in-place write. Post 5 returns to the `floordiv` and drives the dialect to running code, where the vectorizer's refusal has to be answered.

The thesis was that block-scaled matmul reduces to one `floordiv` in one affine map. It is the whole block structure, it is why the op lowers to a `linalg.generic`, and it is the reason the clean lowering carries a cost into the next pass. One choice in one map, right for the stage it was built for and pointed straight at the next problem. The map is the design; everything else in the lowering follows it.