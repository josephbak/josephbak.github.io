+++
title = "4 - Bufferizing Block-Scaled Types: The 1:N Split"
description = "A block-scaled type has no single buffer: mantissa and scale differ in both element type and shape. The 1:N split makes two buffers where the type system wants one, and that split is what lets the matmul accumulator write in place, copy-free."
slug = "bufferizing-block-scaled-types"
date = 2026-08-29
weight = 4
draft = true
[taxonomies]
categories = ["MX-Quantization Dialect"]
tags = ["mlir", "compilers", "quantization", "bufferization"]
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

## A tensor is a value, a buffer is a place

Post 3 lowered `mx.block_matmul` to a single `linalg.generic` on tensors. By the end of it, the block-scaled operand was no longer one value: the lowering consumed a `!mx.tensor` and threaded through two tensors in its place, a mantissa tensor and a scale tensor, each with its own element type and its own shape. The affine maps read from both, the fused payload multiplied them back together per element, and the whole thing verified.

That post stopped at a deliberate line: one `!mx.tensor` value had become two tensors, and the question of how one value becomes two buffers was left for here.

The gap between those two is where the work is. A tensor is a value. It has no memory location, no allocation, no aliasing. Somewhere between the linalg form Post 3 produced and a program that runs, every tensor has to become a `memref` (memory reference): a concrete buffer with an address. For an ordinary tensor that step is close to mechanical. For a block-scaled type it is the whole story of this post, because a block-scaled type does not have a single buffer it can bufferize into, and the reason it doesn't is baked into what the type is.

So this post is about one transition and its consequence. The transition: a value-semantic block-scaled type has no 1:1 image in memory, so it splits into two buffers rather than one. The consequence, once the split is in place: the matmul's accumulator can be written in place, with no defensive copy, which is what keeps the memory-traffic benchmark honest. The split is the spine. The copy-free accumulation is what the spine pays for.

The split and the bufferization are not the same step, and conflating them is the fastest way to lose the thread. The two-tensor form already existed at the end of [Post 3](@/posts/lowering-mx-block-matmul.md), on tensors, before any buffer was allocated. Bufferization takes that pair the rest of the way to memory. Counting and memory are separate axes, and this post treats them separately.

## Why a block-scaled type has no single buffer

Bufferization gives every tensor a `memref`: a concrete buffer at a concrete address. For an ordinary `tensor<32x64xf32>` the mapping is direct. One tensor, one buffer, 32 by 64 four-byte floats laid out in memory. The element type fixes the byte width, the shape fixes the extent, and the buffer is those two facts made physical.

A block-scaled value has no such buffer. `A`'s mantissa is `f8E4M3FN`, one byte per element, at full `32×64` resolution. Its scale is `f8E8M0FNU`, also one byte, but at `32×2`: one value per 32-element block along the contraction axis. [Post 3](@/posts/lowering-mx-block-matmul.md) established that shape arithmetic. What matters here is that the two components disagree on both axes that define a buffer: different element types, `f8E4M3FN` against `f8E8M0FNU`, and different shapes, `32×64` against `32×2`. No single element type and no single extent describes both, so no single `memref` holds both.

Consider what a single buffer would have to be. It would need one element type, but the mantissa and scale are different types. It would need one shape, but they are different shapes. The only way to force them into one buffer is to abandon the type that describes each and drop to raw bytes: a flat `i8` buffer with the mantissa and scale interleaved by hand, and offset arithmetic in every access to say which bytes are mantissa and which are scale for a given block. That is not a `memref` of a meaningful type any more. It is a byte array with a layout convention living outside the type system, and every op that touches it has to know the convention.

The multi-buffer choice was locked before any of this lowering was written, for exactly this reason. A block-scaled type is two things with different types and different shapes wearing one type name, and memory has no slot for "two things wearing one name." Bufferization has to make that plurality explicit: the type presents as one value, and in memory it becomes two buffers.

## Split at lowering, bufferize later

The path from one `!mx.tensor` to two buffers is two separate transitions. The first changes how many values there are. The second changes what those values are. They happen in different passes, mine and upstream's, and they move along different axes.

| stage | `!mx.tensor` | tensor pair | memref pair |
|---|---|---|---|
| count | 1 value | 2 values | 2 buffers |
| semantics | value | value | memory |
| produced by | — | my `TypeConverter` (at lowering) | upstream one-shot-bufferize |
| transition | — | count: 1 → 2 | semantics: value → memory |

The split is the first transition, and it already happened in [Post 3](@/posts/lowering-mx-block-matmul.md). It is not part of bufferization at all. When `mx.block_matmul` lowered to `linalg.generic`, the `TypeConverter` registered for the pass rewrote the one `!mx.tensor` operand into two tensor operands, a mantissa tensor and a scale tensor. That is a 1:N type conversion: one source type maps to N replacement types, here N = 2. It runs at lowering, on tensors, and it changes only the count. Both replacements are still value-semantic tensors with no memory attached.

The visible proof is the function signature. Going in, one block-scaled argument:

```mlir
func.func @block_matmul(%arg0: !mx.tensor<32x64xf8E4M3FN, block_size = 32, scale_type = f8E8M0FNU>, %arg1: tensor<64x64xf32>, %arg2: tensor<32x64xf32>) -> tensor<32x64xf32>
```

Coming out of `--mx-to-linalg`, that one argument is two:

```mlir
func.func @block_matmul(%arg0: tensor<32x64xf8E4M3FN>, %arg1: tensor<32x2xf8E8M0FNU>, %arg2: tensor<64x64xf32>, %arg3: tensor<32x64xf32>) -> tensor<32x64xf32>
```

`%arg0` became `%arg0` (mantissa) and `%arg1` (scale). `B` and the accumulator shifted down to `%arg2` and `%arg3`. The arity went from three arguments to four, and the extra one is the split made concrete: the block-scaled type is gone and its two components stand on their own. Everything is still a `tensor`. No buffer exists yet.

The second transition is bufferization proper, and it is upstream's job. One-shot-bufferize takes the tensor pair and gives each tensor a `memref`, turning value semantics into memory semantics. It does not change the count: two tensors become two buffers, with `B` and the accumulator becoming buffers alongside them. The mantissa and scale are already separate by the time it runs, so bufferization never sees a block-scaled type and never reasons about the split. It bufferizes four dense, standard tensors, each the way any tensor bufferizes.

That ordering is the point. The split is done early, in the `--mx-to-linalg` lowering, so by the time bufferization runs there is nothing block-scaled left to handle. Why the accumulator in particular bufferizes without a defensive copy is where the post goes next.

## Why split and not pack

The split is one design choice out of two. The other was to keep a single buffer and pack both components into it.

Packing is physically possible. The mantissa is `32×64` one-byte elements and the scale is `32×2` one-byte elements, so both are just bytes, and bytes fit end to end in one flat buffer:

```
memref<2112xi8>

byte 0                              2047 2048        2111
  |                                    |   |            |
  +------------------------------------+   +------------+
  |          mantissa bytes            |   | scale bytes|
  |          32 x 64 = 2048            |   | 32 x 2 = 64|
  +------------------------------------+   +------------+
```

```
memref<2112xi8>   (one flat buffer, 2112 bytes)

[ mantissa: bytes 0..2047 ]  [ scale: bytes 2048..2111 ]
   32 x 64 = 2048               32 x 2 = 64
```

It fits. So the objection is not that packing is impossible. The objection is what packing costs at every point that reads the buffer.

The point of a `memref` type is that it describes its own buffer. `memref<32x2xf8E8M0FNU>` carries its element type and its shape, so an access is a logical coordinate, `load %scale[i, b]`, and the compiler derives the address and hands back a typed value. The type does the bookkeeping. The packed buffer throws that away. Its type is `memref<2112xi8>`, which says only "2112 bytes." It does not say that the first 2048 are mantissa and the rest are scale, that the mantissa is `f8E4M3FN` in a `32×64` shape, or that the scale is `f8E8M0FNU` in a `32×2` shape. None of that survives in the type, so all of it moves into the code.

The type carries nothing, so the code carries everything. Every load returns `i8`, which is not what the bytes are, so each access has to `bitcast` back to `f8E4M3FN` or `f8E8M0FNU` by hand. The boundary at 2048 and the two strides are not in the type either, so they are hand-written into every op that touches the buffer. Change the block size and the boundary moves and the strides change, and every access site has to be edited by hand, because none of them learned it from a type.

A custom layout cannot buy this back. `memref` does support custom layouts, but a layout is an address function: it varies where each element lives while holding the element type fixed at one type for the whole buffer. Column-major, strided, tiled: all of them remap the same kind of element to different addresses. The packed buffer needs the opposite, two different element types in one buffer, which is the one thing a layout keeps fixed. Element type sits a level above layout and is uniform across the buffer, so no layout attribute recovers the typed access, and the packed buffer stays raw `i8` with hand arithmetic.

Packing does not give one clean buffer. It gives a byte array the type system can no longer reason about, with the split we tried to avoid reappearing by hand at every reader. The split into two typed memrefs keeps both buffers self-describing: each carries its own element type and shape, each bufferizes as an ordinary dense memref, and every pass downstream still sees two real tensors instead of an opaque blob.

