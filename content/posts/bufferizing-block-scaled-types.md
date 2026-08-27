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

The split and the bufferization are not the same step, and conflating them is the fastest way to lose the thread. The two-tensor form already existed at the end of Post 3, on tensors, before any buffer was allocated. Bufferization takes that pair the rest of the way to memory. Counting and memory are separate axes, and this post treats them separately.