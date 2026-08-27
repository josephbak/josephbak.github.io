+++
title = "4 - Bufferizing Block-Scaled Types: The 1:N Split"
description = "A block-scaled type has no single buffer: mantissa and scale differ in both element type and shape. The 1:N split makes two buffers where the type system wants one, and that split is what lets the matmul accumulator write in place, copy-free."
slug = "bufferizing-block-scaled-types"
date = 2026-08-DD
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