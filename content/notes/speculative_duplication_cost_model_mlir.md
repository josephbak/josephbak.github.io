+++
title = "Speculative Duplication and Cost Modeling in MLIR"
description = "Explains speculative duplication in MLIR with cost-model reasoning, latency-path analysis, and visual examples of critical-path shortening using FMA contraction."
slug = "mlir-speculative-duplication-cost-model"
date = 2025-10-06
updated = 2025-10-07
draft = false
[taxonomies]
tags = ["mlir", "compilers"]
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

## 1. Overview

**Speculative duplication** duplicates part of a computation—intentionally—to unlock a more profitable transformation (e.g., fusion or vectorization) on a *hot path*, guided by a **cost model**. In MLIR (Multi-Level Intermediate Representation), a canonical example is enabling **FMA (Fused Multiply-Add)** contraction when a producer operation has multiple consumers.

---

## 2. Running Example (Floating-Point)

### Baseline IR (Intermediate Representation)
```mlir
%y   = arith.mulf %x, %c1
%tmp = arith.subf %y, %c3
%z   = arith.addf %y, %c2
```
We would like to contract `%y = x*c1` and the `addf` into a single `arith.fma %x, %c1, %c2` for `%z`.  
However, `%y` has multiple uses (`%tmp` and `%z`), so replacing `%y` would change `%tmp`’s semantics—**illegal under SSA (Static Single Assignment)**.

### Speculative Duplication Plan
```mlir
%y   = arith.mulf %x, %c1        // kept for %tmp
%z   = arith.fma  %x, %c1, %c2   // duplicate multiply inside fma
%tmp = arith.subf %y, %c3
```
- Correctness is preserved (each consumer gets the right value).
- `%z`’s path benefits from FMA’s shorter dependency chain.

---

## 3. Legality Preconditions (Floating-Point)

- **Contraction allowed:** `arith.fastmath` must include `contract`; otherwise `mulf+addf → fma` is not IEEE (Institute of Electrical and Electronics Engineers)-legal (two roundings vs one).
- **No side effects** between producer and consumer.
- **Target must support real FMA**; otherwise the rewrite regresses to `mul+add`.

Examples:
```mlir
%z = arith.addf %y, %c2 {fastmath = [contract]}

func.func @f() attributes { arith.fastmath = [contract, reassoc] } { ... }
```

---

## 4. Why the Naive “Total Work” View Fails

A naive comparison sums operation costs:

$$
C_{add} > C_{fma} + P_{dup},
$$

which seems impossible because usually $C_{fma} \ge C_{add}$.  
That view assumes all work is serialized—counting every instruction as if they block each other.

### What Actually Matters
Only the **critical-path latency** to produce `%z` affects performance.  
After duplication, the extra multiply for `%tmp` is **independent** of `%z`’s computation—it runs on another path or execution unit. Its cost should not be counted serially.

---

## 5. Path-Aware Model

Let $L_{op}$ denote latency (or an effective throughput proxy) for an operation on the `%z` path.

Baseline (no duplication): critical path to `%z` is `mulf → addf`,
$$
L_{baseline} = L_{mul} + L_{add}.
$$

With speculative duplication: critical path to `%z` is `fma` plus any path penalty,
$$
L_{dup} = L_{fma} + P_{dup, path}.
$$

**Profitability condition (latency-bound):**
$$
\boxed{L_{mul} + L_{add} > L_{fma} + P_{dup, path}}
$$

- $P_{dup, path}$ = penalty that actually affects `%z`’s schedule (e.g., spill caused by higher pressure).  
- The multiply for `%tmp` exists in both cases and is off `%z`’s path; it should not be added to `%z`’s serial latency.

---

## 6. Visual Intuition for Path Latency

### (a) Baseline — sequential dependency chain
```
        c1       c2
         │        │
x ───▶ mulf ───▶ addf ───▶ z
```
- `%z` depends on both `mulf` and `addf` sequentially.
- Latency to produce `%z`: **L_mul + L_add** (fully serial).

### (b) After speculative duplication — independent paths
```
            ┌──────────▶ subi ───▶ tmp
x ──▶ mulf ─┘
 │
 └──────────▶ fma  ───▶ z
```
- `%z` now depends only on `fma`.
- `%tmp` depends on the old `mulf` (runs in parallel).
- Latency to produce `%z`: **L_fma + P_dup,path**, which is much shorter.

### (c) Timing timeline
```
Baseline:
Time →  [ mulf ][ addf ] → z ready at cycle 8

With duplication:
Time →  [ mulf ][ subi ]         → tmp ready at 8
         [ fma ]                 → z ready at 4
```
- More total instructions executed overall.
- But `%z` finishes earlier, shortening the critical path.

### (d) Concept summary
| View | Measures | `%z` ready time | Misleading? |
|------|-----------|----------------|--------------|
| Total work | Counts all instructions | Same or higher | Yes — ignores parallelism |
| Path latency | Longest dependency chain to `%z` | Lower | Correct |

---

## 7. Typical Numbers (FP32)

- x86 AVX-512: $L_{mul} \approx 3\text{–}4$, $L_{add} \approx 3\text{–}4$, $L_{fma} \approx 4$.  
  Inequality holds if $P_{dup, path} < 2$ cycles.

- NVIDIA GPU (FP32 core): $L_{mul} \approx 4\text{–}6$, $L_{add} \approx 4\text{–}6$, $L_{fma} \approx 4\text{–}6$.  
  Often equal; still beneficial when shortening dependency chains aids scheduling.

- ARM/Apple: $L_{fma} \approx L_{mul}$ and both are small; inequality often holds for modest $P_{dup, path}$.

---

## 8. Implementation Sketch

### Matching
1. Match `mulf` feeding `addf` producing `%z`.
2. `%y` has multiple uses.
3. `fastmath` includes `contract`.
4. Target supports FMA (Fused Multiply-Add).

### Rewriting
```cpp
struct SpeculativeFMA : OpRewritePattern<arith::AddFOp> {
  LogicalResult matchAndRewrite(arith::AddFOp add,
                                PatternRewriter &rewriter) const override {
    auto mul = add.getLhs().getDefiningOp<arith::MulFOp>();
    if (!mul) return failure();
    if (!allowsContract(add)) return failure();
    if (mul.getResult().hasOneUse()) return failure(); // handled elsewhere
    if (!targetHasFMA(add.getType())) return failure();

    double Lmul = 4.0, Ladd = 4.0, Lfma = 4.0, Pdup = 1.0;
    if (Lmul + Ladd <= Lfma + Pdup) return failure();

    auto fma = rewriter.create<arith::FMAOp>(
        add.getLoc(), mul.getLhs(), mul.getRhs(), add.getRhs());
    rewriter.replaceOp(add, fma.getResult());
    return success();
  }
};
```

---

## 9. When It’s Worth It

- **Latency-bound regions:** critical-path shortening dominates extra work.  
- **Hardware FMA available:** FMA throughput ≈ MUL throughput.  
- **Moderate register pressure:** duplication penalty small.  
- **Fast-math with contraction:** legally allowed transformation.

### When It Isn’t
- Throughput-bound code with wide ILP (Instruction Level Parallelism).  
- High register pressure or spilling (large $P_{dup, path}$).  
- Targets that lower FMA (Fused Multiply-Add) back into `mul+add`.  
- Strict IEEE (Institute of Electrical and Electronics Engineers)-754 without `contract`.

---

## 10. Summary

- **Goal:** Improve latency on a hot path by fusing multiply and add via duplication.  
- **Mechanism:** Duplicate the multiply so one consumer can form an `fma`.  
- **Decision rule:** apply if
  $$
  L_{mul} + L_{add} > L_{fma} + P_{dup, path}.
  $$
- **Key idea:** judge by **critical-path latency**, not **total instruction count**.  
- **Legality:** requires `fastmath=contract` and hardware FMA.

---

## 11. Takeaway

Speculative duplication in MLIR (Multi-Level Intermediate Representation) is not about doing *less* work—it is about **restructuring dependency graphs** to shorten *critical latency paths*. The compiler duplicates an operation only when a **path-aware cost model** predicts that the shorter dependency chain outweighs any overhead from added instructions, register pressure, or scheduling noise.
