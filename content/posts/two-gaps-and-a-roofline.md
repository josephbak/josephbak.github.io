+++
title = "5 - Two Gaps and a Roofline: Executing an MX Dialect End-to-End"
description = "Driving a block-scaled dialect to running code: a vectorizer that reads syntax instead of values, an f8 conversion with no CPU path, and a roofline number that rises for two different reasons."
slug = "two-gaps-and-a-roofline"
date = 2026-09-14
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

That is the first of two gaps between a lowering that verifies and code that runs. The second stops the pipeline before it reaches LLVM IR. Neither says what is wrong. The number at the end rises for two reasons, and only one of them is a win.

Everything below is written against `llvm-project` at revision `6f92180` (2026-05-19), the commit this dialect builds on. Where upstream has since changed something this post relies on, the change is noted at that point.

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

The schedule is loaded by `mlir-opt` rather than `mx-opt`. Both are pass drivers, but `mx-opt` is this project's own, and it registers only the dialects the project needs, which does not include the Transform dialect or its Linalg extension. No `mx` ops survive `--mx-to-linalg`, so the second tool sees only upstream dialects.

```
$ mx-opt lower-matmul.mlir --mx-to-linalg \
    | mlir-opt --transform-preload-library='transform-library-paths=schedule.mlir' \
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

`#map` is `(d0) -> (d0 floordiv 32)`, so `%2` is the block index, computed once per tile from `%arg6`, the loop counter of the enclosing `scf.for`, and hoisted above the generic. `%extracted_slice_3` uses it to take a `[32, 1]` window of the scale, one block-column, and hands the generic a `tensor<32x1xf8E8M0FNU>`. Inside the tile the reduction runs over 32 contraction positions that all belong to one block, so the scale each row needs is a single value, constant for the entire reduction.

This is the point at which the tiled form should vectorize. Adding `transform.structured.vectorize %tiled` to the schedule, so that vectorization runs on the handle tiling returned rather than on a fresh match, gives:

```
<stdin>:7:10: error: Attempted to vectorize, but failed
```

The message names no cause, and its location is misleading: it points at line 7 of the input, the untiled generic, because the tiled op inherits the location of the op it was built from. Re-running with `--debug-only=linalg-vectorization` says what happened:

```
[linalg-vectorization Vectorization.cpp:2537] Attempting to vectorize: %3 = linalg.generic
  {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                    affine_map<(d0, d1, d2) -> (d0, d2 floordiv 32)>, ...]}
  ins(%extracted_slice, %extracted_slice_3, %extracted_slice_4
      : tensor<32x32xf8E4M3FN>, tensor<32x1xf8E8M0FNU>, tensor<32x32xf32>)
[linalg-vectorization Vectorization.cpp:2268] precondition failed: not projected permutations
[linalg-vectorization Vectorization.cpp:2545] Vectorization pre-conditions failed
```

The operand types in that dump are `32x32` and `32x1`, so the op under consideration is the tiled one, not the original. The gate is `allIndexingsAreProjectedPermutation`, called from `vectorizeLinalgOpPrecondition` before any vectorization work begins. It requires every indexing map on the op to be a projected permutation, which rules out arithmetic on an iteration dimension. Three of the four maps qualify. `(d0, d2 floordiv 32)` performs a division on `d2`, and one map is enough to reject the op.

What tiling changed and what it did not is the point. It changed the scale's behavior: the value is now fixed across the tile's reduction, and the block index has been lifted into an `affine.apply` above the loop body. It did not change a character of the map, which still reads `d2 floordiv 32`. The precondition inspects the map's expression, so it sees the division and stops. An affine map stores an expression, not the range its dimensions take, so nothing in it records that `d2` now runs to 32 instead of 64.

Untiled, the same refusal is the right call. The scale changes partway through the reduction, lanes covering `k = 31` and `k = 32` need different values, and there is nothing to broadcast. Tiling is what makes the access uniform, and the map is the same either way.

The access is not unvectorizable. It is unvectorizable as written.

## What the map would have to become

Written differently, the same access vectorizes. Take the tiled nest exactly as the schedule emitted it and change one line:

```diff
-#map2 = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 32)>
+#map2 = affine_map<(d0, d1, d2) -> (d0, 0)>
```

The map's second result stops being a division on `d2` and becomes a literal. Nothing else moves: the `extract_slice` still cuts a `[32, 1]` window, the scale operand is still `tensor<32x1xf8E8M0FNU>`, and the loops, the `affine.apply` and the payload are untouched.

Vectorizing the edited nest needs a schedule that only vectorizes, since the file already holds the tiled loops:

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %g = transform.structured.match ops{["linalg.generic"]} in %root
      : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %g : !transform.any_op
    transform.yield
  }
}
```

Run against the edited file, it succeeds, with the map aliases reassigned in the output (loops and constants elided):

```mlir
#map = affine_map<(d0) -> (d0 floordiv 32)>
#map1 = affine_map<(d0, d1) -> (d0, 0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, 0, 0)>
#map3 = affine_map<(d0, d1) -> (0, d1, d0)>
...
        %2 = affine.apply #map(%arg6)
        ...
        %4 = vector.transfer_read %extracted_slice[%c0_9, %c0_9], %3 {permutation_map = #map1} : tensor<32x32xf8E4M3FN>, vector<32x32x32xf8E4M3FN>
        %6 = vector.transfer_read %extracted_slice_3[%c0_9, %c0_9], %5 {permutation_map = #map2} : tensor<32x1xf8E8M0FNU>, vector<32x32x32xf8E8M0FNU>
        %8 = vector.transfer_read %extracted_slice_4[%c0_9, %c0_9], %7 {permutation_map = #map3} : tensor<32x32xf32>, vector<32x32x32xf32>
        %10 = vector.transfer_read %extracted_slice_5[%c0_9, %c0_9], %9 : tensor<32x32xf32>, vector<32x32xf32>
        ...
        %15 = vector.multi_reduction <add>, %14, %10 [2] : vector<32x32x32xf32> to vector<32x32xf32>
```

The `floordiv` did not leave the program. `#map` on the `affine.apply` still computes the block index once per tile. What changed is that the scale operand's map no longer recomputes it on every iteration.

Each input is read into a `vector<32x32x32>` spanning the tile's `(m, n, k)` space, and each `permutation_map` says where that operand's own axes land among those three positions. A `0` marks a position the operand does not vary along, so one value is broadcast across the whole extent. The mantissa map `(d0, 0, d1)` puts rows at `m` and columns at `k` with a `0` at `n`. The `B` map `(0, d1, d0)` carries its `0` at `m`. The scale map `(d0, 0, 0)` is the only one with two zeros: it varies along `m` alone, so a single read per row serves every position in the tile. That is the access the block format was built for, now stated in the IR.

The last operation reads against that broadcast. `vector.multi_reduction <add>, %14, %10 [2]` collapses axis 2, the contraction axis, summing 32 products into each output element. Axis 2 is also the axis the scale's map broadcasts along, and the two are properties of different operands: the scale does not vary along it, the accumulator sums along it.

The literal is correct because tiling had already made the division redundant. `tensor.extract_slice %arg1[0, %2] [32, 1] [1, 1]` uses the block index in `%2` to cut out one column. By the time the generic runs, one column is all it has, and inside the tile `d2` ranges over 0 to 31, so `d2 floordiv 32` returns 0 on every iteration. Writing `0` states what the expression evaluates to.

That the constant is admitted at all comes down to a parameter. `allIndexingsAreProjectedPermutation` calls `isProjectedPermutation` with `allowZeroInResults` set to true, while the method's own default in `AffineMap.h` is false. Constant-zero results are let through by an allowance the header marks for removal once they are supported more broadly. Arithmetic on a dimension is refused either way. Dropping the result entirely, for a map of `(d0)` over an operand with its unit axis removed, gives the same broadcast without relying on the allowance.

Untiled, either version would be a miscompile. The scale operand is the full `32x2`, `k` sweeps 0 to 63 in one loop, and the two halves of that sweep need different columns; a map that always reads the first block would use its factor for all 64 positions. The rewrite is valid only where tiling has already confined `k` to one block, which is why it has to be recognized rather than simply applied.

What the pipeline does not have is anything that performs this rewrite on its own.

## The range the verifier already knows

The one-line edit was made by hand. Whether anything upstream makes it is a question for the tiled nest as the schedule emitted it, with `d2 floordiv 32` still in the scale map. Every pass registered in `mlir-opt` at this revision was tried on that nest, one at a time. Of those that accept it, none rewrites the scale map. The two transform ops that expose the same folding patterns leave it unchanged. No test under `mlir/test/Dialect/Linalg` starts from a `linalg` map containing a `floordiv`.

The candidate that should have caught it is unit-dim folding. The `1` in `tensor<32x1xf8E8M0FNU>` is why the column index can only ever be 0, and `--linalg-fold-unit-extent-dims` exists to remove size-1 axes from `linalg` ops. It is not inert on this operand: it rank-reduces the scale slice to `tensor<32>` and then expands it straight back to `tensor<32x1>`, because a map needs one result per axis and the generic's scale map still has two. What it never edits is the map.

The reason is how it decides an axis can go. Its `isUnitDim` check clears a size-1 axis for removal only when the index into it is the constant 0: either written as `0`, or reducing to `0` after the loops being dropped, those with a single iteration, are replaced by zero. Every loop in the tile runs 32 times, so no loop is dropped and nothing is substituted; the test comes down to whether the index is written as `0`. `d2 floordiv 32` is not, and the check has no way to see a zero that holds only because `d2` stops at 31. Hand the same pass the edited map, with its literal `0`, and it removes the axis and rewrites the map to `(d0)` over `tensor<32xf8E8M0FNU>`.

The fact the edit relies on is already in the op, and the verifier computes it. The map says which column to read for a given `d2`, not how far `d2` goes; the operands do. The mantissa slice is 32 columns wide and read at column `d2`, so `d2` stops at 31. The verifier behind `IndexingMapOpInterface` inverts the indexing maps taken together to recover how many times each loop runs, the same whole-system inversion that let the `floordiv` through in [Post 3](@/posts/lowering-mx-block-matmul.md). It then evaluates every map result at the first and last iteration and checks it against the operand's size. For `d2 floordiv 32` both ends come out 0, and a division that never decreases as `d2` grows cannot be anything else in between. Change the divisor to 16, so the last iteration asks for a second column, and the op no longer verifies:

```
probe-oob.mlir:21:14: error: 'linalg.generic' op inferred input/output operand #1 has shape's dimension #1 to be greater than or equal to 2, but found 1
```

The verifier works out that the index is always 0 and uses it to accept the op. None of the passes tried uses it to simplify the map.

What is missing is a rewrite that asks the verifier's question and keeps the answer. It would evaluate each map result that is not a bare dimension at the first and last iteration of the loops the operand shapes imply, and replace the result with that constant wherever both ends agree and the expression moves in one direction as its dimensions grow. On the untiled op the ends are 0 and 1, so it leaves the map alone, which is the recognition the edit required. The rewrite is not specific to block scales: any map result pinned to a constant by the loop it sits in qualifies. Building it belongs to a later version of this dialect.

What v1 ships is the tiled form. The next question is whether any of it runs.

## Where the f8 conversion stops

Measuring anything means running it, and running it means getting the module down to LLVM IR. Two things stood between the bufferized dialect and a running binary: one pass ordering that is not arbitrary, and one conversion that had no implementation.

The ordering first. Most of the chain is mechanical, but two steps have to go in a fixed relative order, and the reason is the block scale:

~~~
  ...
  --convert-linalg-to-loops
  --lower-affine
  ...
~~~

In the bufferized `linalg.generic`, the expression `d2 floordiv 32` sits inside the indexing map. A map is an attribute on the op, a piece of compile-time data describing how operands are read, and nothing in the module computes it. The generic has no explicit loops and no explicit index arithmetic; the indices are implied by the maps. `--convert-linalg-to-loops` changes that. Expanding the generic into `scf.for` means every index now has to be computed by some operation, and the scale's block index appears as an `affine.apply`, an op that evaluates an affine map on index values and yields an index. The bufferized module contains no `affine.apply`. After the loop conversion there is one.

`--lower-affine` is the pass that turns `affine.apply` into ordinary arithmetic, so it can only run once one exists. Placed earlier it would walk a module where the block index is still a subexpression inside an attribute, find nothing to lower, and report success. The failure would surface much later, as an unlowered affine op arriving at a stage that cannot handle it. The constraint looks like pass-ordering trivia and comes directly from a decision made several stages earlier: put the block index in an affine map, and the pass that lowers affine ops has to wait until something materializes it.

With that settled the chain runs to completion. What comes out is LLVM dialect except for one operation.

A small probe isolates it. `probe-f8-extf.mlir` holds two functions, each taking an eight-bit float as an argument (which avoids compile-time folding), each widening it to `f32`:

~~~mlir
func.func @extf_e4m3(%x: f8E4M3FN) -> f32 {
  %0 = arith.extf %x : f8E4M3FN to f32
  return %0 : f32
}
func.func @extf_e8m0(%y: f8E8M0FNU) -> f32 {
  %0 = arith.extf %y : f8E8M0FNU to f32
  return %0 : f32
}
~~~

The obvious thing to reach for does not exist here:

~~~
$ mx-opt probe-f8-extf.mlir --arith-expand="include-f8e4m3fn=true"
error: <Pass-Options-Parser>: no such option include-f8e4m3fn
~~~

Its counterpart for the scale type does:

~~~
$ mx-opt probe-f8-extf.mlir \
    --arith-expand="include-f8e8m0=true" \
    --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts
~~~

~~~mlir
llvm.func @extf_e4m3(%arg0: i8) -> f32 {
  %0 = builtin.unrealized_conversion_cast %arg0 : i8 to f8E4M3FN
  %1 = arith.extf %0 : f8E4M3FN to f32
  llvm.return %1 : f32
}
llvm.func @extf_e8m0(%arg0: i8) -> f32 {
  %0 = llvm.mlir.constant(23 : i32) : i32
  %1 = llvm.zext %arg0 : i8 to i32
  %2 = llvm.shl %1, %0 : i32
  ...
  %7 = llvm.bitcast %6 : i32 to f32
  llvm.return %7 : f32
}
~~~

The scale conversion is gone, rewritten as integer work on the byte. The mantissa conversion is still there, spelled exactly as it was written, inside a function that is otherwise LLVM dialect. Above it sits a cast turning the incoming `i8` back into `f8E4M3FN`, left behind because the operation it feeds never converted.

Both signatures became `i8`, so both types were handled. Only one of the two operations was. There is no eight-bit float arithmetic for the conversion to lower to on the CPU path, so the conversion has to be emulated with integer operations, and some pass has to supply that emulation. For `f8E8M0FNU`, `--arith-expand` does. For `f8E4M3FN` on this build, nothing did.

Handing the result to `mlir-translate` ends the attempt:

~~~
error: Dialect `arith' not found for custom op 'arith.extf'
~~~

The translator does not register `arith`, so a module still carrying an `arith` operation cannot be parsed at all. No LLVM IR comes out, and the backend never sees the conversion. What the backend would have done with it is a question this attempt never reaches.

Upstream has since closed the gap. Commit `c3c6e286e7eb9`, landed 1 September 2026, adds the missing expansion for `f8E4M3FN` in both directions and exposes it as `include-f8e4m3fn`. That is a little over three months after the build this post is pinned to.

So the two gaps are not the same kind of thing. The vectorizer refused: a check was in place, doing what it was written to do, and getting past it needs a rewrite that none of the passes tried performs. It is still open. This one was not a refusal. No check stood in the way; the implementation was absent, and it has since landed.

That does not restore the measurement. An emulated conversion is a sequence of integer operations standing in for hardware that is not present, and timing it measures the stand-in. What the format actually changes is how many bytes move, and that can be counted without running anything.

## What the intensity number is made of

The roofline for this workload is computed, not measured. Every byte in it comes from the bufferized memref types: shape times element width, each buffer counted once. 

Three numbers describe a workload on a machine. Operational intensity, written `I`, is how many floating-point operations the workload performs per byte it moves, and it belongs to the workload. Bandwidth, `BW`, is how fast the machine moves bytes, and peak rate, `P_peak`, is how fast it does arithmetic; both belong to the machine. Attainable performance is whichever ceiling binds first:

$$P = \min(P_{\text{peak}}, \text{BW} \times I)$$

The two ceilings meet at an intensity called the ridge, `P_peak / BW`, a fixed property of the hardware. Quantization does not move it. Quantization changes `I`, sliding the workload along a roof that stays where it is. The machine here is a base M4, whose 120 GB/s of LPDDR5X bandwidth Apple publishes; its peak arithmetic rate is not published, and the figure used below is reverse-engineered from core count, FMA units, lane width and clock.

For `M=32, N=64, K=64` with a block size of 32, against an all-`f32` baseline:

|  | bytes | FLOPs | I |
|---|---|---|---|
| f32 | 32,768 | 262,144 | 8.00 |
| mx | 26,688 | 393,216 | 14.73 |

Intensity rises by 1.84×, and that is the least useful number in the table, because it is a product of two effects that mean opposite things:

$$1.84 = 1.23 \times 1.50$$

The 1.23× is the byte ratio and it is the data-movement result. The 1.50× is FLOP inflation: the mx payload performs three floating-point operations per point where the baseline performs two, the extra one being the multiply that reconstructs `A` from its mantissa and scale. Intensity is FLOPs over bytes, so moving fewer bytes raises it and doing more arithmetic raises it too. Only the first is a win.

The 1.23× byte ratio itself needs unpacking, because larger and more flattering numbers are available. Narrowing the mantissa from `f32` to `f8E4M3FN` is exactly 4× on those bytes, 8,192 down to 2,048. Add the block scale back and `A` as a whole goes from 8,192 to 2,112, or 3.88×. But `B` and the accumulator are still `f32` in v1:

```
f32   AAAAAAAABBBBBBBBBBBBBBBBCCCCCCCC  32,768 bytes
mx    AABBBBBBBBBBBBBBBBCCCCCCCC        26,688 bytes

A = A operand (mantissa + scale)   B = B operand   C = accumulator
1 character = 1 KiB, widths rounded
```

`A` shrinks to a quarter of its width while the 24,576 bytes of `B` and the accumulator do not move at all. A 3.88× reduction on one operand is a 1.23× reduction on the working set. The 4× is a true statement about a format; the 1.23× is a true statement about this workload.

Both points also sit in the wrong region to show what quantization is for. With this machine's bandwidth and peak the ridge falls near 4.7 FLOPs per byte, and 8.00 and 14.73 are both above it, so both are compute-bound. The movement is inside the compute-bound region, not a crossing out of the memory-bound one. That is not because the problem is small. It follows from the reuse a matmul of this shape has: 262,144 operations over 32,768 bytes, because every element of `A` and `B` participates in many products.

The regime where the memory result would appear is decode. In language-model inference, prefill processes the whole prompt at once and decode generates one token at a time, which means the activation matrix has a single row: `M = 1`. FLOP count is `2·M·N·K` and scales with `M`, so it collapses. The `B` operand is `K×N` and has to be read in full regardless of how many rows it multiplies. Little arithmetic over the same weight bytes puts intensity far below the ridge, and the workload becomes memory-bound.

That regime also shows what v1 does not buy. At `M=1` the `A` operand is one row of 64 values, 256 bytes against `B`'s 16,384, so quantizing it moves the total from 16,896 bytes to 16,706. The same model at `M=1` gives an intensity shift of 1.52× against a byte ratio of 1.01×, while the FLOP inflation is 1.50×. Almost the entire apparent gain is the dequantization multiply, and the memory win is a rounding error.

<img src="/img/roofline.svg"
     alt="Analytical roofline for mx.block_matmul against an f32 baseline at two shapes"
     style="filter: none; background: #fff;">

Both shapes move rightward, which is what an optimization is supposed to look like, and at neither shape is the movement mostly a memory result. At `M=32` the byte ratio is 1.23× against a 1.50× FLOP inflation; at `M=1` it is 1.01× against the same 1.50×. The plot cannot separate the byte saving from the added arithmetic, which is why each intensity carries its decomposition rather than standing as a single number.

## Reading the same model as a duration

The roofline's axes are intensity and rate. Neither is time, and time is what quantization is supposed to improve.

The same two ceilings give it. Moving the data and doing the arithmetic happen at once, so the longer of the two sets the duration: bytes over bandwidth, or FLOPs over peak rate, whichever is larger. That is the `min` above seen from the other side, since dividing the work by the lower of two rates gives the longer of two times. Both inputs are ceilings, so the durations that come out are floors: the best case, not a prediction.

<img src="/img/roofline-time.svg"
     alt="Modelled time per call for the f32 baseline and mx block matmul at both shapes"
     style="filter: none; background: #fff;">

At `M=32` the arithmetic ceiling binds, so the mx path performs 1.5× the operations at the same peak rate and takes 1.5× as long. Both points sat on the roofline's ceiling at the same height, and the one further right is the slower one.

At `M=1` the bandwidth ceiling binds, and 16,706 bytes against 16,896 is a 1% difference, so the two durations are level at about 140 ns. The roofline nevertheless puts the mx point at 88 GFLOP/s against the baseline's 58. GFLOP/s counts operations per second, and the mx path performs 1.5× the operations in the same time, so the rate rises. The operations it added are dequantization: work that unpacks the format rather than producing any part of the answer. The axis cannot tell a multiply that computes the result from one that undoes the compression, and it credits both.

The ratios are worth more than the absolute nanoseconds. Real code reaches some fraction of peak, so both bars would stretch, and if both stretch by the same factor the 1.5× between them is unchanged. That assumes the two variants sit a similar distance from peak, which is not guaranteed when one of them loads one-byte values, widens them, and carries an extra multiply. What does not depend on the model is the direction: more arithmetic against the same ceiling can only take longer. The size of the gap does.

Where that extra arithmetic comes from is worth naming, because it is not inherent to the format. [Post 3](@/posts/lowering-mx-block-matmul.md) lowers `mx.block_matmul` to a single fused `linalg.generic`, reconstructing each `A` value inside the loop nest rather than materializing a dequantized `f32` tensor first. The reconstruction therefore sits inside the `n` loop and runs once per output column: `M·N·K` multiplies where dequantizing `A` once would have cost `M·K`. The entire 1.50× is the price of not writing that intermediate. Fusion optimizes for bytes. This shape is limited by FLOPs. The decision is sound against the objective it was chosen for and costs time against the one that binds here.

So neither shape comes out faster. At `M=32` the mx path takes 1.5× as long; at `M=1` the two are level. That is the honest state of `A`-only quantization with `B` and the accumulator still in `f32`: the format cost is paid on every point and the byte saving is too small to repay it. What v1 establishes is the lowering and the byte accounting, neither of which cares which operand carries the format. Quantizing `B` is what moves the decode point. Under the same byte arithmetic, quantizing both operands at `M=1` takes the working set from 16,896 bytes to 4,546 and the modelled time from 141 ns to 38 ns, still memory-bound, so the time ratio and the byte ratio are the same 3.7×. v1 does not quantize `B`, so that is a projection from the type information rather than a result.

One caveat about the roof. Because the peak arithmetic rate is reverse-engineered rather than published, the ridge is drawn as a band rather than a line, and the modelled durations inherit that uncertainty. The intensity values do not depend on it at all, since they come from the type information alone.

## What the diagnosis bought

The dialect lowered, bufferized, and tiled without incident. Then the vectorizer refused the scale map, the folding pass that looked like the fix turned out to be solving a different problem, the f8 conversion had no implementation on the CPU path, and the intensity number at the end folded a saving and an overhead into one figure.

Both gaps were real on `llvm-project` at `6f92180`. One has since been filled: `arith-expand` gained the `f8E4M3FN` expansion on 1 September 2026, so a build from today lowers what this one could not. The other is open at HEAD. The pattern that would close it recognizes a `floordiv` confined to a single block and rewrites the operand and the map together, and it has not been written. It is general rather than specific to this dialect: any block-scaled format lowered through an affine map meets the same wall.

What v1 has is the lowering and the byte accounting. Four ops, a parameterized type, three canonicalizations, a conversion to `linalg` on tensors, a 1:N split that bufferizes copy-free, and a tiled schedule whose block-aligned `K` makes the broadcast form reachable. The byte model built on that works from types and shapes, so it can price a change the code does not implement, which is how a quantized `B` gets a number here without existing.

What v1 does not have is a speedup. At `M=32` the mx path takes 1.5× as long, and at `M=1` the two are level. Quantizing `A` pays at neither shape, because at `M=32` arithmetic binds and at `M=1` `A` is a rounding error against `B`. A second version would quantize `B`, build the vectorization pattern, and measure on a build with the f8 expansion.

The series worked down one pipeline, a stage per post. [Post 1](@/posts/designing-mx-dialect.md) put a block scale into a type. [Post 2](@/posts/canonicalization-e8m0-power-of-two.md) found the power-of-two structure in the scale format and what it makes exact. [Post 3](@/posts/lowering-mx-block-matmul.md) reduced block-scaled matmul to one `floordiv` in one affine map. [Post 4](@/posts/bufferizing-block-scaled-types.md) split one value into two buffers and accumulated in place. This post took all of it to the last stage, where two upstream gaps stopped the pipeline and the number at the end was computed rather than measured.