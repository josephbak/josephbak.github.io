+++
title = "A Dialect Is a Set of Co-Designed Decisions: Designing for MX Quantization in MLIR"
description = "Designing an out-of-tree MLIR dialect for OCP Microscaling quantization: one parameterized type, four ops, and the decisions that constrain each other across passes."
slug = "designing-mx-dialect"
date = 2026-08-12
weight = 1
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

Modern hardware can do arithmetic far faster than it can fetch the numbers to compute on. For large-model inference the bottleneck is rarely the multiply-accumulate units. It's memory bandwidth, the cost of moving weights and activations from DRAM to the compute. Quantization attacks that bottleneck directly: store each number in fewer bits, move fewer bytes, and spend the machine's idle arithmetic capacity reconstructing full-precision values on the fly. It trades FLOPs, which are cheap, for memory traffic, which is not.

`mlir-mx` is an out-of-tree MLIR dialect implementing OCP Microscaling (MX) block-scaled quantization, the format behind NVIDIA Blackwell and AMD's MI300-series accelerators. The design goal is clear: represent block-scaled data in the IR, and lower it to code that moves those smaller bytes and reconstructs full precision on the fly. Four operations cover the core: `quantize_block` (pack f32 into a mantissa plus a shared scale), `dequantize_block` (the inverse), `block_matmul` (multiply while still block-scaled), and `fold_scale` (a canonicalization for scale-only multiplies, the subject of Post 2). Plus one parameterized type, `!mx.tensor`, carrying the block-scale metadata.

That list, a type and four ops, makes it look like a dialect is a parts catalog: pick a type, pick some operations, done. It isn't. **A dialect is a set of co-designed decisions, and the type, ops, and attributes are just where those decisions become visible.** A dialect gets built in stages: first you define the types, then the ops, then the lowerings that turn those ops into simpler IR, then bufferization that turns values into memory. These stages are not independent. The type's parameters are shaped by what the ops need; the ops' signatures by how they'll lower; the lowering by how it'll schedule and bufferize. A choice made at the type stage can be forced by something that doesn't happen until the lowering stage, several passes later. Pull one decision and others move. The rest of this post walks the type and the four ops. Two forces get the full treatment: the **block axis** of the type, forced by how the matmul reduces, and the **accumulator operand** of `block_matmul`, forced by how it bufferizes.

## What block scaling is

Ordinary quantization picks one scale factor for a whole tensor: every value is stored as a small integer or low-bit float, times that one shared number. The problem is outliers: one large value forces a coarse scale, and all the small values lose precision under it.

Block scaling fixes this by using many scales instead of one. Split the data into fixed-size blocks of 32 contiguous elements, in MX, and give each block its own scale. A value is reconstructed as `mantissa × block_scale`:

```
values:  [ v0  v1  v2  ...  v31 ][ v32  v33  ...  v63 ]
         └──────── block 0 ──────┘└─────── block 1 ────┘
scale:            S0                        S1
                  (one shared scale per 32 elements)
```

Each element stores a small mantissa; each block of 32 stores one shared scale. OCP MX (Open Compute Project Microscaling) is the cross-vendor standard for this: block size 32, an FP8 mantissa, and a power-of-two scale. It's what NVIDIA Blackwell and AMD's MI300-series accelerators run in hardware.

## The type: `!mx.tensor`

Every op in the dialect operates on one type, so it gets defined first. An `!mx.tensor` is a block-scaled tensor: low-bit mantissa data plus the metadata needed to interpret it. Written out:

```mlir
!mx.tensor<32x64xf8E4M3FN, block_size = 32, scale_type = f8E8M0FNU>
```

`MX_TensorType` is a custom type defined from scratch in the `mx` dialect (a TableGen `TypeDef`, mnemonic `tensor`, so it prints as `!mx.tensor`). It is **not** a builtin `RankedTensorType` and does not inherit from one. It carries four parameters: a `shape` (an array of dimension sizes, the same `int64_t`-array representation MLIR's own shaped types use), an `elementType` for the mantissa, a `blockSize`, and a `scaleType`. The first two mirror what a ranked tensor holds; the last two are what make it *block-scaled* rather than merely low-bit.

- **`shape`, `elementType`** (`32x64`, `f8E4M3FN`): the mantissa tensor. `f8E4M3FN` is an 8-bit float, 4 exponent bits and 3 mantissa bits, holding each quantized value.
- **`block_size`** (`32`): how many contiguous elements share one scale. 32 is the OCP MX block size.
- **`scale_type`** (`f8E8M0FNU`): the shared scale's element type. E8M0 is 8 exponent bits and zero mantissa bits, so it represents powers of two and nothing else. Why that constraint is useful is Post 2's subject.

Making it an independent type, rather than reusing `RankedTensorType` or implementing MLIR's `ShapedTypeInterface`, is a deliberate choice. A block-scaled tensor is not a single shaped value: at bufferization it decomposes into a mantissa tensor and a separate scale tensor. Implementing the shaped-type interface would promise the 1:1 tensor semantics that decomposition deliberately breaks. That story is Post 4; the point here is that even the *type* was defined with the lowering in mind.

Two flags on the definition matter later. `hasCustomAssemblyFormat = 1` means the `NxNx...` syntax is parsed by hand rather than by a declarative format; the mechanics of that (and the rule to use `getChecked` over `get` in a parser) are in a linked note. `genVerifyDecl = 1` means the type has a verifier: it checks that `blockSize > 0` and that the block size evenly divides the last dimension. Structural checks only — no scale math, no value semantics. The verifier's entire job is well-formedness.

The type stores the scale's *type* and *how many elements it covers*, never the scale values themselves. A helper, `getNumScaleBlocks()`, computes how many scales a tensor needs (one per block along the last axis), but that's a count, not the data. This is just what a type is in MLIR: a compile-time label, not runtime storage. Where the scale values actually live is a bufferization question with a non-obvious answer, and it gets its own post (Post 4).

One parameter is doing more work than it looks. The type says block size 32, but not *which axis* is blocked. So it fixes a convention: **blocking always runs along the last axis.** That looks like an arbitrary tidiness rule. It isn't. It's the first of the two co-designed decisions, and the thing that forces it doesn't appear until the matmul lowers, well into the pipeline. We'll come back to it once the ops are on the table.

## The four ops

A note on scope: this is v1, a deliberately minimal first version with four ops and one type, and some features held back to later versions (marked v1.5 or v2 as they come up).

With the type defined, the four operations follow. Each takes or produces an `!mx.tensor` and does one job.

**`quantize_block`** turns a normal `f32` tensor into a block-scaled one. It reads a `tensor<...xf32>` and produces an `!mx.tensor`: for each block of 32 elements it picks a shared scale, then rounds each element against that scale into the low-bit mantissa type. This is the lossy step. The whole round-trip's precision loss happens here, when 32 full-precision values are forced to share one power-of-two scale and each is rounded to three mantissa bits.

**`dequantize_block`** is the inverse: `!mx.tensor` back to `tensor<...xf32>`. For each element it multiplies the mantissa by its block's scale. Because the scale is a power of two, that multiply just shifts the exponent and adds no error of its own, so dequantize faithfully reconstructs whatever the quantize step produced, no more and no less.

**`block_matmul`** is the point of the whole dialect: multiply while the data is still block-scaled, so the smaller bytes are what moves. It takes a block-scaled `A` of shape `M×K`, an ordinary `f32` matrix `B` of shape `K×N`, and an accumulator `C` of shape `M×N`, and computes `C += A · B`. The reduction runs along `K`, the shared dimension the two matrices multiply over. As the sum sweeps `K`, `A`'s mantissas are read per element while `A`'s scales are read once per block of 32. The scale indexing is coarser than the mantissa indexing by exactly the block size. That coarseness is what settles the deferred question of which axis gets blocked.

**`fold_scale`** doesn't quantize, dequantize, or multiply. It's a canonicalization, an algebraic cleanup the compiler applies before code generation, for the case where a block-scaled value gets multiplied by a scalar. Rather than touch every element, the scalar can be handled more cheaply by rewriting the surrounding ops. It carries no lowering of its own and must be gone before the conversion pass runs. (Post 2 covers which forms v1 folds and the power-of-two property in play.)

One detail of `block_matmul`'s signature will matter later: it carries the accumulator `C` as an operand, not just a result. That's the second co-designed decision, and like the first, its justification shows up further down the pipeline.

## The block axis

Back to the deferred question: the type blocks along the last axis. Why fix that as a rule?

Start from what `block_matmul` computes. Each output element is a sum over `K`:

```
C[m,n] = Σ  A[m,k] · B[k,n]
     k=0..K-1
```

`A` is block-scaled, so `A[m,k]` isn't stored directly. It's a mantissa times its block's scale. The whole question is: **what does that scale depend on?** And that depends entirely on which axis is blocked.

Block along `K`, the last axis, which is what the type fixes. Then the scale for `A[m,k]` is `S[m, k/32]`: one scale per row, shared across each run of 32 consecutive `k`. The sharing runs along the reduction.

```
row m of A, K = 64, two blocks along k:

  k:  0  1  2 ...  31     32  33 ...  63
      └── S[m,0] ──┘      └── S[m,1] ──┘
      one scale for       one scale for
      this run of k       this run of k

  reduction sweeps k  ──────────────▶
  scale is constant within each run it sums over
```

Because the scale is constant across each block of the sum, it factors out of that block's partial sum: 32 mantissa terms share one scale. The scale aligns with the direction being summed.

Now block along `M`, the rows, instead. The scale for `A[m,k]` becomes `S[m/32, k]`: shared across 32 rows, but varying with `k`.

```
reduction sweeps k  ──────────────▶

  k =    0       1       2     ...
       S[·,0]  S[·,1]  S[·,2]  ...
       └─ a different scale at every k ─┘

  scale changes on every step of the sum it's inside
```

Now the scale sits inside the reduction and changes on every step. It can't factor out; every term of the sum carries its own scale. The block sharing (across rows) is orthogonal to the reduction (along `k`), so the one structural advantage of block scaling, a shared scale amortized over a run, is unavailable to the matmul.

Here's the part worth being precise about, because it's easy to overstate. Blocking along `M` doesn't cost more storage. The same mantissas and the same number of scales get stored; the count is `M·K/32` either way, the same ~4× compression over `f32`. What's lost is exploitability. The scale can't align with the reduction, so the matmul can't use the block structure: the format keeps its storage cost and loses its computational benefit. That is what "no traffic win" means. The bytes don't grow; the one operation the dialect exists for just can't take advantage of them.

So the last-axis rule isn't tidiness. It's forced: **the block axis must be the reduction axis, because a shared scale only aligns with a matmul if it's constant along the axis being summed.** For `A` of shape `M×K`, the reduction axis is `K`, which is the last axis. The type fixes last-axis blocking, and `block_matmul`'s reduction is the reason.

That's the first cross-stage force, stated plainly: a choice at the type stage (which axis carries the blocks), justified only by a fact about the lowering stage (how the matmul reduces), which runs passes later. Define the type without the matmul in mind and last-axis looks arbitrary; you might just as easily block the rows. Design the two stages together and it's the only choice that works.

One honest edge, since it shows where the rule's generality stops. "Last axis" is really shorthand for "the reduction axis," and those coincide only because `A` is the left operand, whose reduction axis is last. A quantized `B`, deferred to v1.5, reduces along its first axis, so the same principle would put its blocks there. The convention is last-axis for `A`; the law underneath is reduction-axis. When `B` gets quantized, the type will need an explicit block-axis parameter rather than a fixed last-axis rule.

The block axis was a choice at the type stage, forced by the lowering stage. The next decision is the same shape, one stage further down: a choice in an op's signature, forced by bufferization.

## The accumulator operand

This is the second of the two forces. The first was a type-stage choice forced by the lowering stage. This one is an op-signature choice forced by bufferization, near the end of the pipeline.

`block_matmul` takes three operands: the block-scaled `A`, the matrix `B`, and an accumulator `C`. It computes `C += A · B`. That third operand is the one to look at. `C` is both an input and the result: the op reads the incoming `C`, adds `A · B` to it, and produces the updated `C`. It could have been written the other way, as a pure `C = A · B` with `C` as a result only, no accumulator input. Why carry `C` in?

The answer doesn't live in the op definition or the lowering. It lives at bufferization, where tensors become memory.

A quick frame for what bufferization does. Up through the lowering, everything is *value-semantic*: a tensor is a mathematical value, not a location in memory, and every op conceptually produces a fresh one. That's clean to reason about but not how hardware works, since hardware writes to buffers. Bufferization is the stage that assigns each tensor value a concrete memory buffer, and its central question for every op is whether the op can write its result into an existing buffer, in place, or needs a fresh allocation and a copy.

In-place is what you want. A matmul that allocates a new `M×N` buffer on every call and then copies is moving bytes the algorithm doesn't require, which is exactly the memory traffic this dialect exists to cut. For the benchmark to measure only essential traffic, the accumulator has to be updated in place.

That is what the `C` operand makes possible. Because `C` is passed in as an operand, the op's result can reuse `C`'s own buffer: the GEMM (General Matrix Multiply) form `C += A · B` names a destination, "write the result back into `C`." This is destination-passing style (DPS), the pattern MLIR's own `linalg` ops use for the same reason. Write the op the other way, `C = A · B` with no accumulator input, and there's no incoming buffer for the result to reuse at all.

The distinction is exactly the right size to be careful about. The accumulator operand doesn't *guarantee* in-place accumulation; it *enables* it. Whether the write actually happens in place is decided later, by bufferization's own conflict analysis and its function-boundary handling, which is Post 4's subject. What the operand settles, at definition time, is whether in-place is available at all. The pure `C = A · B` form forecloses it: no incoming buffer, nothing for any later analysis to reuse, a fresh allocation guaranteed. The GEMM form keeps the option open for bufferization to take.

So the accumulator operand is not a modeling nicety. It's the op signature being shaped, at definition time, to preserve an option that only a much later stage can act on. That's the same force as the block axis, one stage over: an earlier choice made to serve a need further down the pipeline.

Two decisions, the same shape. The block axis was the type stage constrained by the lowering stage. The accumulator was the op stage constrained by the bufferization stage. Neither makes sense from inside its own stage: last-axis blocking looks arbitrary next to the type, the accumulator operand looks like a stylistic quirk next to the op. Each is justified only by a stage further down. That's what "co-designed" means in practice, and it's why a dialect is not a parts catalog. The parts are easy. The constraints between them, running across stages from type definition to the end of the pipeline, are the actual design.

## What's deferred, and what's next

v1 is four ops and one type, and the boundaries are deliberate. The things left out are the ones that would have grown the scope past what one version should carry: a quantized `B` operand (v1 leaves `B` in `f32`), per-channel scales, group quantization across dimension boundaries, and a real lowering for `fold_scale` (v1 only cleans it up, never lowers it). Further out, past v1.5 and v2, the natural direction is lowering these ops to accelerator backends where MX is a native hardware format rather than something emulated on CPU. That's the long arc; v1 is the IR foundation it would stand on.

The rest of the series follows the pipeline down, roughly one stage per post:

- **Post 2** takes `fold_scale` and the power-of-two scale type, and shows why folding a block scale is exact for some scalars and impossible for others.
- **Post 3** is the lowering: `block_matmul` becomes a `linalg.generic`, and the block scale becomes a single `floordiv` in an affine map. It's the technical center of the project.
- **Post 4** is bufferization: how one `!mx.tensor` value becomes two memory buffers, and how the in-place accumulation promised above actually gets decided.
- **Post 5** drives the whole thing to running code, runs into two upstream limits in MLIR, and quantifies the memory traffic the dialect was built to cut.

The thesis was that a dialect is a set of co-designed decisions, not a parts catalog. The two moments in this post were the evidence: a type that blocks its last axis because a matmul later in the pipeline reduces along it, and an op that carries an accumulator because bufferization near the end of the pipeline needs a destination to write into. The ops are the easy part. What makes it a dialect is that the decisions behind them are wired to each other, across the whole pipeline.