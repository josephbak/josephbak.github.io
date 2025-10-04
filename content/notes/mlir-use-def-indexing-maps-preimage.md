+++
title = "From Use–Def to Preimage: Fusing Structured Ops in MLIR"
description = "A math-first primer on use-def, indexing maps, and preimages for producer/consumer fusion with linalg + tensors."
slug = "mlir-use-def-indexing-maps-preimage-inline-katex"
date = 2025-10-01
updated = 2025-10-01
draft = false
[taxonomies]
tags = ["mlir", "compilers", "affine", "linalg", "transform-dialect"]
# series = ["MLIR Transform Notes"]
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

_A short, practical note for future-me about the core ideas behind producer/consumer fusion in MLIR’s structured world._

---

## TL;DR

- **Use-def:** follow SSA edges from a consumer tile back to the op that defines its input.
- **Indexing map:** for each operand, an affine map $f:\mathbb{Z}^n\to\mathbb{Z}^r$ from loop indices to subscripts.
- **Invert vs preimage:** only invertible if $f$ is unimodular/permutation; in general compute the **preimage** $ f^{-1}(S) = \lbrace \mathbf{i}\mid f(\mathbf{i})\in S \rbrace $.
- **Fusion across mismatched spaces:** pull back the consumer tile through the producer’s output map to compute **exactly** the producer iterations you need _in place_.
- **Rematerialization:** deliberate recomputation of cheap producers per tile to kill memory traffic.

---

## 1) Use-Def in one picture

```
%Y = linalg.matmul ... // def(%Y) = matmul
%Z = linalg.generic ins(%Y) ... // %Y is used here → consumer

matmul ──defines──▶ generic // def→use
generic ──uses──────▶ matmul // use→def (reverse walk for fusion)
```

**Quick defs (SSA/MLIR):**
- **SSA value**: defined once, used many times.
- **def(v)**: the op (or block arg) that defines `v`.
- **uses(v)**: the ops that consume `v` as an operand.
- **def→use**: from a value’s def to all its consumers (“who reads what I produce?”).
- **use→def**: from a consumer operand back to the op that defined it (what we walk for fusion).
- **producer/consumer**: if `P` defines `%x` and `C` takes `%x`, then `P` is `%x`’s **producer**, `C` its **consumer**.

We start from a **consumer tile** (a 2‑D slice of `%Z`) and walk **use→def** to find the **producer** (`%Y` from matmul) to potentially fuse.

---

## 2) Indexing maps (formal and concrete)

For a structured op with iteration indices $\mathbf{i}=(i_0,\ldots,i_{n-1})$, each operand gets an affine map
$$
f_{\text{opnd}}:\mathbb{Z}^n\to\mathbb{Z}^r,\qquad \mathbf{i}\mapsto \text{subscripts}
$$

**Notation.** The subscript in $f_{\text{opnd}}$ means *per-operand* indexing map. For a given op you have one map per operand:
- Inputs: $f_A, f_B, \ldots$ (often referred to generically as $f_{\text{in}}$ for an input).
- Output/result: $f_{\text{out}}$.
These maps live over the op's iterator space and tell **which element** of each operand is touched at a point.

that answers: _“At this loop point, which element of this tensor/buffer do I touch?”_

**Matmul** (iterators $(i,j,k)$; $i,j$ parallel, $k$ reduction):
- $f_A(i,j,k)=(i,k)$ 
- $f_B(i,j,k)=(k,j)$ 
- $f_C(i,j,k)=(i,j)$ ← output map

**Elementwise same-shape** (iterators $(i,j)$):
- for each input, $f_{\text{in}}(i,j)=(i,j)$; for the output, $f_{\text{out}}(i,j)=(i,j)$

Broadcasts, transposes, striding, dilations are all just different affine maps.

---

## 3) “Invert” vs **preimage** (the actual thing you need)

Given a consumer **tile** $S\subseteq\mathbb{Z}^r$ in output index space and the producer’s output map $f_{\text{out}}:\mathbb{Z}^n\to\mathbb{Z}^r$, what you want is the **preimage**:

$$
T = f_{\text{out}}^{-1}(S) = \lbrace \mathbf{i}\in\mathbb{Z}^n \mid f_{\text{out}}(\mathbf{i}) \in S \rbrace.
$$

- If $f_{\text{out}}$ is **unimodular/permutation** → there is a true algebraic inverse (nice!).
- If it **drops** dims (projection/reduction), **strides**, or **dilates** → no inverse; **preimage** still exists and is what we use.

**Examples**

- **Permutation:** $f(i,j)=(j,i)$. For a rectangular tile $S=I\times J$, $f^{-1}(S)=J\times I$.
- **Reduction:** matmul $f(i,j,k)=(i,j)$. For tile $S=I\times J$, $f^{-1}(S)=I\times J\times [0..K)$.
- **Stride:** $f(i,j)=(2i,j)$. Preimage adds congruence: $ \lbrace (i,j)\mid 2i\in I,\ j\in J \rbrace $ (appears as strided slices).

Once you have $T$ (the subset of producer iterations), the **input slices** you need are just the **images** $g_A(T), g_B(T),\ldots$ under each input map $g$.

---

## 4) Fusion across mismatched iteration spaces (recipe)

1. **Tile the consumer** over its parallel dims: choose $S\subseteq\mathbb{Z}^r$.
2. **Use→def** to find the producer of the consumer’s operand.
3. **Pull back** the tile: $T=f_{\text{out}}^{-1}(S)$ in the producer’s loop space.
4. **Extract input slices** using each input map’s image of $T$.
5. **Run the producer on $T$ _inside_ the consumer tile**; feed the partial result directly; no giant temporaries.

**Mini numeric sanity check** 
Let $M=4,\ N=6,\ K=3$. Tile size $(T_m,T_n)=(2,2)$ at offset $(i_0,j_0)=(2,4)$.

- Consumer tile $S=\lbrace i\in\lbrace 2,3 \rbrace,\ j\in\lbrace 4,5 \rbrace \rbrace$.
- Matmul output map $f(i,j,k)=(i,j)$ ⇒
 $T=\lbrace (i,j,k)\mid i\in\lbrace 2,3 \rbrace,\ j\in\lbrace 4,5 \rbrace,\ k\in\lbrace 0,1,2 \rbrace \rbrace$.
- Needed slices:
 - $A(i,k)$ ⇒ rows $\lbrace 2,3 \rbrace$ and all $k$ ⇒ $A_{\text{slice}} : 2 \times 3$.
 - $B(k,j)$ ⇒ all $k$ and cols $\lbrace 4,5 \rbrace$ ⇒ $B_{\text{slice}} : 3 \times 2$.
- Compute a $2\times 2$ partial, then apply the elementwise op and insert.

---

## 5) Rematerialization (when/why)

We call it **rematerialization** when we recompute values per tile instead of storing a big temporary in memory.

**Typical cases**

- Upstream elementwise producers: $L=\mathrm{relu}(L_0)$, $R=\exp(R_0)$. 
 If you fuse them into each tile, elements like $L[i,k]$ are recomputed **per $j$-tile**; duplication factor $\approx N/T_n$. Similarly $R[k,j]$ across $i$-tiles; factor $\approx M/T_m$.

- Same producer result used by multiple consumers → recompute the same producer tile in each consumer’s loop.

**Why it can win:** modern kernels are memory-bound. Extra flops are cheap; avoiding DRAM traffic and improving locality is often a net gain.

**When to avoid:** producer is heavy and reused many times; or fusion hides `vector.contract`/MMA patterns and spikes register pressure.

---

## 6) Practical heuristics

- **Tile parallel dims first**; shape inner kernels for cache/SMEM; keep unit-stride for vector lanes.
- **Fuse cheap, pure producers/epilogues**; prefer rematerialization over big temporaries.
- **Keep contractions recognizable** so they lower to `vector.contract`/tensor cores before heavy epilogues.
- **Measure pressure**: spills on CPU; regs/thread & occupancy on GPU.

---

## 7) Glossary (fast)

- **Use-def:** SSA graph edges from values to their users/definers.
- **Indexing map:** affine map from loop indices to subscripts ($f:\mathbb{Z}^n\to\mathbb{Z}^r$).
- **Preimage:** $ f^{-1}(S) = \lbrace \mathbf{i} \mid f(\mathbf{i})\in S \rbrace $; what you compute to fuse a producer tile.
- **Rematerialization:** recomputing values per tile to avoid storing/reloading them from memory.
- **Materialize loops:** turn implicit iteration into `scf.for`/`scf.forall` + (tensor) slices.

---

## 8) Tiny MLIR snippet to anchor the math

**Matmul tile inside consumer tile (conceptual):**
```mlir
%lhs_slice = tensor.extract_slice %L[%i0, 0] [Tm, K] [1,1]
%rhs_slice = tensor.extract_slice %R[0, %j0] [K, Tn] [1,1]
%res_slice = tensor.extract_slice %Res[%i0, %j0][Tm, Tn] [1,1]

%partial = linalg.generic {
 indexing_maps = [ (i,j,k)->(i,k), (i,j,k)->(k,j), (i,j,k)->(i,j) ],
 iterator_types = ["parallel","parallel","reduction"]
} ins(%lhs_slice, %rhs_slice : tensor<Tm x Kxf32>, tensor<K x Tnxf32>)
 outs(%res_slice : tensor<Tm x Tnxf32>) {
^bb0(%a: f32, %b: f32, %acc: f32):
 %p = arith.mulf %a, %b : f32
 %s = arith.addf %acc, %p : f32
 linalg.yield %s : f32
} -> tensor<Tm x Tnxf32>
```
