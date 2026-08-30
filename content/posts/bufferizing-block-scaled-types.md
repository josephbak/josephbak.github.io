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

[Post 3](@/posts/lowering-mx-block-matmul.md) lowered `mx.block_matmul` to a single `linalg.generic` on tensors. By the end of it, the block-scaled operand was no longer one value: the lowering consumed a `!mx.tensor` and threaded through two tensors in its place, a mantissa tensor and a scale tensor, each with its own element type and its own shape.

Post 3 stopped at tensors: one `!mx.tensor` value had become two tensors, and how those become two buffers was left for here.

The gap between those two is where the work is. A tensor is a value. It has no memory location, no allocation, no aliasing. Somewhere between the linalg form Post 3 produced and a program that runs, every tensor has to become a `memref` (memory reference): a concrete buffer with an address. For an ordinary tensor that step is close to mechanical. For a block-scaled type it is the problem this post is about, because a block-scaled type does not have a single buffer it can bufferize into, and the reason it doesn't is baked into what the type is.

This post is about one transition and one payoff. The transition: a value-semantic block-scaled type has no 1:1 image in memory, so it splits into two buffers rather than one. The payoff, once the split is done and the ops reach bufferization as ordinary tensors: the matmul's accumulator bufferizes in place, with no defensive copy, which is what keeps the memory-traffic benchmark honest. The split is the spine. The copy-free accumulation is what it sets up.

The split and the bufferization are not the same step, and it is worth keeping them apart. The two-tensor form already existed at the end of Post 3, on tensors, before any buffer was allocated. Bufferization takes that pair the rest of the way to memory. Counting and memory are separate axes.

## Why a block-scaled type needs two buffers

For an ordinary `tensor<32x64xf32>`, bufferization is direct. One tensor, one buffer, 32 by 64 four-byte floats laid out in memory. The element type fixes the byte width, the shape fixes the extent, and the buffer is those two facts made physical.

A block-scaled value has no such buffer. `A`'s mantissa is `f8E4M3FN`, one byte per element, at full `32×64` resolution. Its scale is `f8E8M0FNU`, also one byte, but at `32×2`: one value per 32-element block along the contraction axis ([Post 3](@/posts/lowering-mx-block-matmul.md) established that shape arithmetic). The two disagree on both axes that define a buffer, element type and shape, so no single element type and no single extent describes both. There is no one `memref` that holds both.

The obvious rescue is to drop the type. If no *typed* buffer holds both, use one untyped one: a flat `memref<...xi8>` of raw bytes, big enough for the mantissa and the scale together, with the structure managed by hand. Both components are just bytes, and bytes fit end to end:

```
memref<2112xi8>   (one flat buffer, 2112 bytes)

[ mantissa: bytes 0..2047 ]  [ scale: bytes 2048..2111 ]
   32 x 64 = 2048               32 x 2 = 64
```

It fits. The objection is not that packing is impossible; it is what packing costs at every point that reads the buffer.

The point of a `memref` type is that it describes its own buffer. `memref<32x2xf8E8M0FNU>` carries its element type and its shape, so an access is a logical coordinate, `load %scale[i, b]`, and the compiler derives the address and hands back a typed value. The type does the bookkeeping. The packed `i8` buffer throws that away. Its type says only "2112 bytes." It does not record which bytes are mantissa and which are scale, or that the two differ in element type and shape at all. The single `i8` type erases exactly the distinction that made them two things.

The code then carries what the type no longer does. Every load returns `i8`, which is not what the bytes are, so each access has to `bitcast` back to `f8E4M3FN` or `f8E8M0FNU` by hand. The boundary at 2048 and the two strides are not in the type either, so they are hand-written into every op that touches the buffer. Change the block size and the boundary moves and the strides change, and every access site has to be edited by hand, because none of them learned it from a type.

A custom layout cannot buy this back. A `memref` layout is an address function: it says where each element lives, while the element type is fixed above it, one type for the whole buffer. Column-major, strided, tiled all remap the same kind of element to different addresses. What the packed buffer needs is the opposite, two different element types in one buffer, and that is exactly what a layout cannot vary. The element type is decided before the layout is consulted, so no layout attribute recovers the typed access, and the packed buffer stays raw `i8` with hand arithmetic.

So the single buffer is a false economy. It gives a byte array the type system can no longer reason about, with the split it was meant to avoid reappearing by hand at every reader. The honest form is two typed memrefs, each carrying its own element type and shape, each an ordinary dense buffer, each still legible to every pass downstream. A block-scaled type is two components wearing one type name, and memory has no slot for "two things wearing one name." The type presents as one value; in memory it becomes two buffers.

## Split at lowering, bufferize later

The path from one `!mx.tensor` to two buffers is two separate transitions. The first changes how many values there are. The second changes what those values are. They happen in different passes, one in the `mx-to-linalg` lowering and one upstream, and they move along different axes.

| stage | `!mx.tensor` | tensor pair | memref pair |
|---|---|---|---|
| count | 1 value | 2 values | 2 buffers |
| semantics | value | value | memory |
| produced by | — | `TypeConverter` | `one-shot-bufferize` |
| transition | — | count: 1 → 2 | semantics: value → memory |

The split is the first transition, and it already happened in Post 3. It is not part of bufferization at all. When `mx.block_matmul` lowered to `linalg.generic`, the custom `TypeConverter` registered for the pass rewrote the one `!mx.tensor` operand into two tensor operands, a mantissa tensor and a scale tensor. That is a 1:N type conversion: one source type maps to N replacement types, here N = 2. It runs at lowering, on tensors, and it changes only the count. Both replacements are still value-semantic tensors with no memory attached.

The proof is the function signature. Going in, one block-scaled argument:

```mlir
func.func @block_matmul(%arg0: !mx.tensor<32x64xf8E4M3FN, block_size = 32, scale_type = f8E8M0FNU>, %arg1: tensor<64x64xf32>, %arg2: tensor<32x64xf32>) -> tensor<32x64xf32>
```

Coming out of `mx-to-linalg`, that one argument is two:

```mlir
func.func @block_matmul(%arg0: tensor<32x64xf8E4M3FN>, %arg1: tensor<32x2xf8E8M0FNU>, %arg2: tensor<64x64xf32>, %arg3: tensor<32x64xf32>) -> tensor<32x64xf32>
```

The original `%arg0` became two `%arg0` (mantissa) and `%arg1` (scale). `B` and the accumulator shifted down to `%arg2` and `%arg3`. The arity went from three arguments to four, and the extra one is the split made concrete. Everything is still a `tensor`; no buffer exists yet.

The second transition is bufferization proper. The upstream `one-shot-bufferize` pass takes the tensor pair and gives each tensor a `memref`, turning value semantics into memory semantics. It does not change the count: two tensors become two buffers, with `B` and the accumulator becoming buffers alongside them. Because the mantissa and scale are already separate by the time it runs, bufferization never sees a block-scaled type. It bufferizes four dense, standard tensors.

That ordering is the point. The block-scaled structure is resolved at lowering, one pass before bufferization runs, so the hard part is handled by the type conversion, not the bufferizer. Bufferization is left with an ordinary problem.

## Teaching an old pass about new ops

The `one-shot-bufferize` pass is upstream code, and it was written before this dialect existed. Yet it bufferizes the ops that the `mx-to-linalg` lowering produced, ops that did not exist when it was compiled. A pass that predates these ops still knows how to bufferize them, and how it manages that is this section.

There are two ways a pass can know how to transform an op, and they differ in where the knowledge lives.

The first is a pattern, which is how the `mx-to-linalg` lowering works. The pass carries the knowledge: a pattern per op it handles, an enumerated set of ops it knows, and nothing outside that set gets touched. That works because the lowering deals with a closed set, four mx ops, all known when the pass was written. The pass can hold the whole list.

Bufferization cannot work that way, because its op set is open. It has to bufferize ops that did not exist when it was compiled, including out-of-tree ops like the ones this dialect lowers to. It cannot carry a pattern for every op that will ever need bufferizing. So the knowledge cannot live in the pass; it has to live on the op.

That is the second mechanism: an interface. `BufferizableOpInterface` is a contract an op implements, saying "here is how to bufferize me." The pass calls `op.bufferize()` without knowing the concrete op type, and each op supplies its own logic, so the pass handles ops it has never seen. The deciding axis between the two mechanisms is not whether the transformation is a lowering. Both `mx-to-linalg` and bufferization are lowerings. The axis is whether the consumer's op set is closed, which allows a pattern, or open, which forces an interface.

This is the [expression problem](https://en.wikipedia.org/wiki/Expression_problem): letting new ops and new behaviors be added independently, without editing either side. A new op can become bufferizable without the bufferization pass changing, and the pass can gain a new op without that op's dialect changing. The interface plus an external model is MLIR's answer to it.

An external model implements an interface for a dialect's ops, but lives outside that dialect rather than inside it. The bufferization logic for `linalg` ops could have gone inside the `linalg` dialect, but that would make `linalg` depend on the bufferization infrastructure, and a dialect should not drag in a transformation framework just to define its ops. So the implementation lives outside the op-owning dialect, which keeps `linalg` free of any bufferization dependency at the cost of one step: something has to attach the model before the pass runs.

Upstream implements `BufferizableOpInterface` for its own ops, `linalg`, `arith`, `tensor`, `func`, the ones the lowering produces. Those implementations exist; attaching them to the dialect's context does not happen automatically. The models are upstream; the registration that attaches them is not. Until it runs, nothing bufferizes, and the pass says so precisely: an unattached model makes `one-shot-bufferize` fail with an interface "promised but not implemented" error that names the dialect whose op it could not bufferize.

Four registrations, one per dialect: `arith`, `linalg`, `tensor`, `func`. The count tracks upstream dialects in the lowered IR, not mx ops, of which there are none left by the time bufferization runs. Add an op from a fifth dialect and there would be a fifth registration.

Three of the four are filed under their own dialect's `Transforms` directory. `func` is not: its model lives under the bufferization dialect, in the `func_ext` namespace. Bufferizing a function boundary is a calling-convention decision about how tensors cross a signature, which is bufferization policy, not a fact about what `func.func` is, so the model is filed with the pass that needs it rather than the dialect it operates on. Looking for it under the func dialect turns up nothing; the model is filed by what consumes it.

The promise mechanism is what makes the missing registrations tractable. Each unattached model produces one precise error naming one dialect, so the registrations surface one at a time with an exact pointer, rather than as a single opaque failure to untangle.

## Getting the accumulation copy-free

The split is done and everything is in memory. What remains is one question about the accumulator, and it is where the memory-traffic thesis is either kept or quietly broken.

`mx.block_matmul` accumulates: its result is `acc + A·B`, and the `acc` operand is the running sum. When it bufferizes, `acc` becomes a buffer and the result becomes a buffer too. The safe, dumb thing for a bufferizer to do is allocate a fresh buffer for the result, copy `acc` into it, and accumulate there, leaving the original `acc` untouched. That defensive copy is a full pass over the accumulator, and for a matmul whose whole purpose is to move as few bytes as possible, an unasked-for copy of the output on every call is exactly the cost the dialect exists to avoid.

The copy is unnecessary here, and the bufferizer proves it. Running the `mx-to-linalg` lowering and `one-shot-bufferize` with `bufferize-function-boundaries=1` gives (verbatim, comments added):

```mlir
func.func @block_matmul(
    %arg0: memref<32x64xf8E4M3FN, strided<[?, ?], offset: ?>>,   // mantissa
    %arg1: memref<32x2xf8E8M0FNU, strided<[?, ?], offset: ?>>,   // scale
    %arg2: memref<64x64xf32, strided<[?, ?], offset: ?>>,        // B
    %arg3: memref<32x64xf32, strided<[?, ?], offset: ?>>)        // acc
    -> memref<32x64xf32, strided<[?, ?], offset: ?>> {
  linalg.generic {...}
    ins(%arg0, %arg1, %arg2 : ...)
    outs(%arg3 : memref<32x64xf32, strided<[?, ?], offset: ?>>) {   // accumulates into acc
    ^bb0(%in: f8E4M3FN, %in_0: f8E8M0FNU, %in_1: f32, %out: f32):
      ...
      %4 = arith.addf %out, %3 : f32
      linalg.yield %4 : f32
  }
  return %arg3 : memref<32x64xf32, strided<[?, ?], offset: ?>>   // returns the same buffer
}
```

The accumulator arrives as `%arg3`, the `linalg.generic` writes into it directly through `outs`, and the function returns that same `%arg3`. No fresh allocation, no copy. The bytes written are the essential ones, the accumulator's own. The bufferization test for this op asserts exactly that, checking with `CHECK-NOT` that no `memref.copy` survives.

Two things had to hold for that. The first is that the op names its destination. `acc` is the `outs` operand of the generic, so the bufferizer has a concrete buffer to write into rather than an anonymous result it must find storage for. That destination-passing shape is the reason the accumulator is a real operand and not a fresh output, and it is settled in [Post 1](@/posts/designing-mx-dialect.md); here it is what makes the copy-free result reachable at all.

The second is the analysis that decides the copy can be skipped. The bufferizer asks, for each buffer it wants to write in place: is there any read of this buffer's original value that happens after the write? If there is, the write would clobber a value something still needs, so a defensive copy goes in. If there is not, the write is safe and the copy is skipped. For the accumulator, the only read of its value is the accumulate itself, `acc + A·B`, and that read is part of the same operation as the write, not a later use of the original. No later reader means no conflict, so the copy is skipped.

That distinction is the whole mechanism, and it is easy to state slightly wrong. It is not that the accumulator is never read: it is read, once, inside the accumulation. It is that nothing reads the *original* accumulator *after* the write lands. The read that exists and the read that would cause a conflict are different reads, one during the operation and one that would have to come after it, and only the second forces a copy. There is no second read here.

One flag makes the difference between the copy-free form above and a version with a copy: `bufferize-function-boundaries=1`. Without it, the bufferizer handles the body but leaves the function signature tensor-typed, and it cannot see whether a caller will read the original `acc` after the call. Blind past the boundary, it assumes the worst and inserts the defensive copy. With the flag, the signature itself becomes memref-typed under a defined calling convention, and the question the intraprocedural analysis could not answer, whether a later reader exists, is answered by the convention: the caller is responsible for preserving anything it still needs, so this function is free to write `acc` in place. The `?` marks on the arguments, `strided<[?, ?], offset: ?>`, are that convention in the signature: the strides and offset are left dynamic so the boundary accepts a buffer with any layout the caller brings.

The flag is experimental upstream. Its analysis is intraprocedural and leans on the boundary convention. For a single-function benchmark it holds; a multi-function call graph, where callers and callees have to agree on who copies what, is where its edges would need testing. That is the honest boundary of the v1 result.

The reason any of this matters beyond tidiness is the measurement. The benchmark counts bytes moved, and a stray per-call `memref.copy` of a `32×64` `f32` accumulator is 8 KB of traffic the workload never asked for, on every matmul. One unnecessary copy would not make the result wrong; it would make the number misreport what the design actually moves. The copy-free form is what keeps the number the design's number.

## What the split bought

The whole post turned on one move: a block-scaled type has no single buffer, so it becomes two. That split is not a bufferization trick. It happens at lowering, on tensors, and by the time bufferization runs there is nothing block-scaled left, only four ordinary tensors that bufferize the way any tensor does. Splitting early is what let the hard part of bufferization be no part at all.

The split also paid for the accumulator. Because `acc` crosses the boundary as its own operand and nothing reads its original value after the write, the bufferizer writes it in place, and the matmul returns the buffer it accumulated into with no defensive copy. That is what keeps the benchmark honest: the bytes it counts are the bytes the design moves, not bytes an unnecessary copy added.

The `floordiv` in the scale map came through untouched. That expression, the one that made the lowering clean in [Post 3](@/posts/lowering-mx-block-matmul.md), is still sitting in the `linalg.generic` in the bufferized output above, still non-projective. It came through because this pipeline lowered and bufferized with nothing in between: no tiling, no vectorization, so nothing challenged the map. Bufferizing on its own is what let the split and the copy-free accumulator show up in isolation. The schedule that tiles and vectorizes is a separate step, and it runs on the tensor form before any of these buffers exist. That is where the `floordiv` stops being free, and where Post 5 begins.