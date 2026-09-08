+++
title = "5 - Two Gaps and a Roofline: Executing an MX Dialect End-to-End"
description = "Driving a block-scaled dialect to running code: a vectorizer that reads syntax instead of values, an f8 conversion with no CPU path, and a roofline number that rises for two different reasons."
slug = "two-gaps-and-a-roofline"
date = 2026-09-07
weight = 5
draft = true
[taxonomies]
categories = ["MX-Quantization Dialect"]
tags = ["mlir", "compilers", "quantization", "vectorization"]
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

## The bar the generic postponed

[Post 3](@/posts/lowering-mx-block-matmul.md) ended owing an answer. The scale operand's indexing map, `(d0, d2 floordiv 32)`, cleared `linalg.generic`'s verifier, which asks only whether the operands' maps taken together recover the iteration space. The structured vectorizer asks a stricter question of every map on its own, and the same `floordiv` fails it. Choosing a generic did not clear that bar. It postponed it, and this is where it comes due.

Everything below is written against `llvm-project` at revision `6f92180` (2026-05-19), the commit this dialect builds on. Where upstream has since changed behavior this post depends on, the change is noted where it comes up.

Three things sit between a lowering that verifies and a number worth reporting: a vectorizer, a path to executable code, and a roofline. On this pipeline each of them reported something other than what it appeared to.

The schedule that tiles is short. It is Transform-dialect IR, data rather than compiled code, loaded into the pipeline and interpreted:

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.generic"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %tiled, %loop_n, %loop_k =
      transform.structured.tile_using_for %matmul tile_sizes [0, 32, 32]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
```

The tile sizes are `[0, 32, 32]` over the iteration space `(m, n, k)`. `M` is not tiled: it is already 32, and tiling it would produce a single-trip loop. `N` and `K` are tiled by 32, and the size on `K` is the one that matters. A `K` tile equal to the block size keeps each tile inside one MX block, so the block index is fixed for the whole tile.

The schedule is loaded by `mlir-opt` rather than `mx-opt`, which registers neither the Transform dialect nor its Linalg extension. No `mx` ops survive `--mx-to-linalg`, so the second tool sees only upstream dialects.

```
mx-opt test/MX/lower-matmul.mlir --mx-to-linalg \
  | mlir-opt --transform-preload-library='transform-library-paths=test/MX/schedule.mlir' \
             --transform-interpreter
```

That produces the loop nest (payload elided; tiling leaves it unchanged):

```mlir
#map = affine_map<(d0) -> (d0 floordiv 32)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 32)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
...
      %1 = scf.for %arg6 = %c0_0 to %c64_1 step %c32_2 iter_args(%arg7 = %arg5) -> (tensor<32x64xf32>) {
        %2 = affine.apply #map(%arg6)
        %extracted_slice = tensor.extract_slice %arg0[0, %arg6] [32, 32] [1, 1] : tensor<32x64xf8E4M3FN> to tensor<32x32xf8E4M3FN>
        %extracted_slice_3 = tensor.extract_slice %arg1[0, %2] [32, 1] [1, 1] : tensor<32x2xf8E8M0FNU> to tensor<32x1xf8E8M0FNU>
        ...
        %3 = linalg.generic {indexing_maps = [#map1, #map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice, %extracted_slice_3, %extracted_slice_4 : tensor<32x32xf8E4M3FN>, tensor<32x1xf8E8M0FNU>, tensor<32x32xf32>) outs(%extracted_slice_5 : tensor<32x32xf32>) {
```

Two lines in that nest are why the `K` tile is the block size:

```mlir
%2 = affine.apply #map(%arg6)
%extracted_slice_3 = tensor.extract_slice %arg1[0, %2] [32, 1] [1, 1] : tensor<32x2xf8E8M0FNU> to tensor<32x1xf8E8M0FNU>
```

`#map` is `(d0) -> (d0 floordiv 32)`, so `%2` is the block index, computed once per tile from the tile's induction variable `%arg6` and hoisted above the generic. `%extracted_slice_3` uses it to take a `[32, 1]` window of the scale, one block-column, and hands the generic a `tensor<32x1xf8E8M0FNU>`. Inside the tile the reduction runs over 32 contraction positions that all belong to one block, so the scale each row needs is a single value, constant for the entire reduction. That is precisely the condition a vector broadcast wants.

Adding `transform.structured.vectorize %tiled` to the schedule, so that vectorization runs on the handle tiling returned rather than on a fresh match, gives:

```
error: Attempted to vectorize, but failed
```

The message names no cause, and its location is misleading: it points at line 7 of the input, the untiled generic, because the tiled op inherits the location of the op it was built from. The vectorizer's own debug output says what happened:

```
[linalg-vectorization Vectorization.cpp:2537] Attempting to vectorize: %3 = linalg.generic
  {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                    affine_map<(d0, d1, d2) -> (d0, d2 floordiv 32)>, ...]}
  ins(%extracted_slice, %extracted_slice_3, %extracted_slice_4
      : tensor<32x32xf8E4M3FN>, tensor<32x1xf8E8M0FNU>, tensor<32x32xf32>)
[linalg-vectorization Vectorization.cpp:2268] precondition failed: not projected permutations
[linalg-vectorization Vectorization.cpp:2545] Vectorization pre-conditions failed
```

The shapes in the first line are `32x32` and `32x1`, so the op under consideration is the tiled one, not the original. The gate is `allIndexingsAreProjectedPermutation`, called from `vectorizeLinalgOpPrecondition` before any vectorization work begins. It requires every indexing map on the op to be a projected permutation: each result a bare iteration dimension, carried through untouched. Three of the four maps qualify. `(d0, d2 floordiv 32)` performs a division on `d2`, and one failing map fails the op.

What tiling changed and what it did not is the point. It changed the scale's behavior: the value is now fixed across the tile's reduction, and the block index has been lifted out into an `affine.apply` above the loop body. It did not change a character of the map, which still reads `d2 floordiv 32`. The precondition inspects the map's expression, so it sees the division and stops. Nothing in an affine expression records that `d2` now ranges over 32 values instead of 64, and no check that reads the expression could discover it.

The requirement is a stand-in for the question that matters, which is whether each operand's access can become a uniform vector load or broadcast. Untiled, the two agree: the scale genuinely changes partway through the reduction, there is no single value to broadcast, and refusing is right. Tiled, they come apart. The access is exactly a broadcast, and the map is exactly as non-projective as before.

So the access is not unvectorizable. It is unvectorizable as written.

## What the map would have to become

Written differently, the same access vectorizes. Take the tiled nest exactly as the schedule emitted it and change the scale operand's map and its type:

```diff
-#map2 = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 32)>
+#map2 = affine_map<(d0, d1, d2) -> (d0)>

-%extracted_slice_3 = tensor.extract_slice %arg1[0, %2] [32, 1] [1, 1] : tensor<32x2xf8E8M0FNU> to tensor<32x1xf8E8M0FNU>
+%extracted_slice_3 = tensor.extract_slice %arg1[0, %2] [32, 1] [1, 1] : tensor<32x2xf8E8M0FNU> to tensor<32xf8E8M0FNU>

-%3 = linalg.generic { ... } ins(%extracted_slice, %extracted_slice_3, %extracted_slice_4 : tensor<32x32xf8E4M3FN>, tensor<32x1xf8E8M0FNU>, tensor<32x32xf32>) ...
+%3 = linalg.generic { ... } ins(%extracted_slice, %extracted_slice_3, %extracted_slice_4 : tensor<32x32xf8E4M3FN>, tensor<32xf8E8M0FNU>, tensor<32x32xf32>) ...
```

The map loses its `floordiv`; the operand loses its trailing unit axis. The type appears twice, once where the slice produces it and once where the generic lists its inputs, so both occurrences update together. Nothing else moves: same loops, same `affine.apply`, same payload, same other three operands.

Vectorizing the edited nest needs a schedule that only vectorizes, since the tiling has already been done and baked in:

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.generic"]} in %root
      : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %matmul : !transform.any_op
    transform.yield
  }
}
```

Run against the edited file, it succeeds:

```mlir
#map = affine_map<(d0) -> (d0 floordiv 32)>
#map1 = affine_map<(d0, d1) -> (d0, 0, d1)>
#map2 = affine_map<(d0) -> (d0, 0, 0)>
#map3 = affine_map<(d0, d1) -> (0, d1, d0)>
...
        %2 = affine.apply #map(%arg6)
        ...
        %4 = vector.transfer_read %extracted_slice[%c0_6, %c0_6], %3 {permutation_map = #map1} : tensor<32x32xf8E4M3FN>, vector<32x32x32xf8E4M3FN>
        %6 = vector.transfer_read %extracted_slice_0[%c0_6], %5 {permutation_map = #map2} : tensor<32xf8E8M0FNU>, vector<32x32x32xf8E8M0FNU>
        %8 = vector.transfer_read %extracted_slice_1[%c0_6, %c0_6], %7 {permutation_map = #map3} : tensor<32x32xf32>, vector<32x32x32xf32>
        %10 = vector.transfer_read %extracted_slice_2[%c0_6, %c0_6], %9 : tensor<32x32xf32>, vector<32x32xf32>
        ...
        %15 = vector.multi_reduction <add>, %14, %10 [2] : vector<32x32x32xf32> to vector<32x32xf32>
```

The `floordiv` did not leave the program. `#map` on the `affine.apply` still computes the block index once per tile. What changed is that the generic no longer computes it a second time.

Each input is read into a `vector<32x32x32>` spanning the tile's `(m, n, k)` space, and each `permutation_map` says where that operand's own axes land among those three positions. A `0` marks a position the operand does not vary along, so one value covers the whole extent. The mantissa map `(d0, 0, d1)` puts rows at `m` and columns at `k` with a `0` at `n`. The `B` map `(0, d1, d0)` carries its `0` at `m`. The scale map `(d0, 0, 0)` is the only one with two zeros: it varies along `m` alone, so a single read per row serves every position in the tile. That is the access the block format was built for, now stated in the IR.

The last operation reads against it. `vector.multi_reduction <add>, %14, %10 [2]` collapses axis 2, the contraction axis, summing 32 products into each output element. The same axis is broadcast along for the scale and reduced along for the accumulator, which is not a contradiction: the broadcast says where a value comes from, the reduction says where results go.

The `floordiv` could be dropped because tiling had already made it redundant. `%2 = affine.apply #map(%arg6)` evaluates `k floordiv 32` once per tile on the loop counter, and `tensor.extract_slice %arg1[0, %2] [32, 1] [1, 1]` cuts out that block's column. By the time the generic runs, one column is all it has, and inside the tile `d2` ranges over 0 to 31, so `d2 floordiv 32` returns 0 on every iteration and indexes the only column present.

Dropping it is not enough on its own, and the operand has to come along, for reasons that are separate even though both concern the same map. The map's second result, `d2 floordiv 32`, is an arithmetic expression rather than a bare dimension, which is exactly what the vectorizer's precondition rejects. The map also has two results, and `linalg.generic` requires an operand's rank to equal its map's result count, so a one-result map forces a rank-one operand. One constraint is about what a result contains, enforced by the vectorizer; the other is about how many results there are, enforced by the op's own verifier. Change either alone and the op is malformed.

Replacing the arithmetic with a constant does not help either. A map of `(d0, 0)` is no more a projected permutation than `(d0, d2 floordiv 32)`, since a literal is not a bare dimension. While the operand keeps a second axis the map must index it with something, and everything it could index it with fails. Dropping the axis is what lets the map stop mentioning it.

Untiled, the same edit would be a miscompile. There the scale operand is the full `32x2` and `k` sweeps 0 to 63 in one loop. At `k = 5` the map selects scale column 0; at `k = 40` it selects column 1. Those are different numbers, and the result depends on getting the right one. Replace the map with `(d0)` and every iteration reads column 0, so all 32 products in the second half of the reduction are scaled by the first block's factor.

What the pipeline does not have is anything that performs this rewrite on its own.