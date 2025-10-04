+++
title = "MLIR Generic DAG Rewriter"
date = 2025-10-02
page_template = "post.html"
insert_anchor_links = "right"
generate_feeds = true
+++

<!-- Notes on MLIR (Multi-Level Intermediate Representation) Generic DAG (Directed Acyclic Graph) Rewriter -->

A compact, publish‑ready summary of what I learned from **https://mlir.llvm.org/docs/Rationale/RationaleGenericDAGRewriter/** and the discussion.

---

## TL;DR
- MLIR (Multi-Level Intermediate Representation) needs a **general DAG (Directed Acyclic Graph)-to-DAG (Directed Acyclic Graph)** rewrite engine to handle many optimizations and lowerings across abstraction levels.
- This is different from **CSE (Common Subexpression Elimination)**: DAG (Directed Acyclic Graph) rewriting **changes** the shape (e.g., fuse `mul+add` → **FMA (Fused Multiply Add)**); CSE (Common Subexpression Elimination) only **deduplicates**.
- The infrastructure aims for **declarative, reusable patterns**, robust legality checks, good diagnostics, and flexible algorithms because global DAG tiling is **NP (Nondeterministic Polynomial time)-complete**.

---

## Core idea: “Match one DAG (Directed Acyclic Graph), replace with another”
- IR (Intermediate Representation) is a graph: nodes = **operations**, edges = **SSA (Static Single Assignment) values**.
- Pattern rewriter finds a **subgraph** (e.g., `add` feeding `mul`) and replaces it with an **equivalent** but better subgraph (e.g., a single **FMA (Fused Multiply Add)**).

**ASCII**
```
Before:            After:
 a   b   c          a   b   c
  \ /               \ | /
  add                fma         # (a + b) * c
   |                               
   v
  mul
```

---

## Canonicalization vs Constant Folding vs Clients
- **Canonicalization**: normalize equivalent forms (e.g., `x+0 → x`, reorder commutative ops) so later passes see a **standard** shape.
- **Constant folding**: if operands are constants, compute the result constant. In MLIR (Multi-Level Intermediate Representation), `fold()` **returns** constants; **clients** (passes like canonicalizer) update IR (Intermediate Representation). This avoids iterator invalidation.
- **Clients** = the users of these APIs (Application Programming Interfaces): canonicalization, simplifiers, etc.—they call `op.fold()` and perform the IR (Intermediate Representation) edits safely.

---

## AST (Abstract Syntax Tree) vs DAG (Directed Acyclic Graph)
- **AST (Abstract Syntax Tree)** is a **tree** → no natural sharing; `(x+y)` duplicated in `(x+y)*(x+y)`.
- **DAG (Directed Acyclic Graph)** permits sharing; one node for `(x+y)` with two uses.
- DAG (Directed Acyclic Graph) matching is more powerful than tree matching, but must guard against **duplicating** shared work.

---

## How DAG (Directed Acyclic Graph) sharing happens (construction)
- **Hash‑consing / value numbering**: map `(op, operands, attributes, types)` → existing node; reuse it instead of re‑creating.
- **SSA (Static Single Assignment)** naturally encodes sharing: one definition, many uses.

---

## CSE (Common Subexpression Elimination) vs Peephole “Combiners”
- **CSE (Common Subexpression Elimination)**: remove **duplicate** equivalent ops when the earlier one **dominates** the later.
- **Combiners** (peepholes): local **algebraic rewrites** within a small window (e.g., fuse, fold, strength‑reduce). Order and local cost matter.

**Strength reduction examples**
- `x*2 → x<<1` (bit‑shift), `x%2^k → x&(2^k−1)` for **unsigned** or non‑negative `x`.
- For non‑power‑of‑two divisors, use “magic” multiply+shift sequences, not masking.

---

## Dominance, Hoisting, LICM (Loop‑Invariant Code Motion), GVN (Global Value Numbering), CFG (Control‑Flow Graph)
- **Dominance**: A dominates B if A is executed **on every path** to B.
- **Hoisting**: move an op to a **dominating** block so it runs once and is reused. Trade‑off: longer live ranges → higher **register pressure**.
- **LICM (Loop‑Invariant Code Motion)**: move loop‑invariant work to the loop preheader (or sink it) when safe/profitable.
- **GVN (Global Value Numbering)**: global equivalence + redundancy elimination across the **CFG (Control‑Flow Graph)** (MLIR (Multi-Level Intermediate Representation) core relies more on CSE (Common Subexpression Elimination), canonicalization, SCCP (Sparse Conditional Constant Propagation), etc.; classic GVN (Global Value Numbering) is on the LLVM (Low Level Virtual Machine) side).
- **CFG (Control‑Flow Graph)**: nodes = basic blocks; edges = possible control transfer.

---

## MLIR (Multi-Level Intermediate Representation) rewrite infrastructure
- **RewritePattern + PatternRewriter** with greedy drivers (`applyPatternsAndFoldGreedily`).
- **DRR (Declarative Rewrite Rules)** (TableGen) and **PDL/PDLL (Pattern Description Language / Pattern Description Language frontend)** for declarative pattern authoring.
- **Canonicalizer** + op‑level `fold()` unify local cleanups.
- **Dialect Conversion**: legality, 1→N/N→M rewrites, type conversion; not just peepholes.
- **Transform dialect**: orchestrate how/where to apply transformations using IR (Intermediate Representation) itself.

---

## LLVM (Low Level Virtual Machine) SelectionDAG (Selection Directed Acyclic Graph) instruction selection
- Declarative `Pat<>` patterns (TableGen) match target‑independent DAGs (Directed Acyclic Graphs) and rewrite to machine instructions.
- Pros: declarative, identity‑aware, compact state machine, type‑checked, extensible with custom C++.
- Cons: narrowly scoped to **instruction selection**, requires rebuild to extend, limited diagnostics, SelectionDAG (Selection Directed Acyclic Graph) constraints (e.g., multi‑result pain), accumulated tech debt.

---

## Equality saturation (e‑graphs) — why it matters
- Solves the “optimizing too early” problem: keep **all** equivalent forms; pick best later via a cost model.
- Trade‑off: search‑space growth; prototypes exist around MLIR (Multi-Level Intermediate Representation), but not core yet.

---

## Goals of MLIR (Multi-Level Intermediate Representation) generic DAG (Directed Acyclic Graph) rewriter
- Support **1→N**, **M→1**, **M→N** with **benefits**; infra picks best local match.
- Separate (1) best pattern at a root, (2) whole‑graph rewrite strategy, (3) pattern definitions.
- Enable iterative rewrites with different client trade‑offs.
- Make patterns easy, safe, and resilient: simple APIs (Application Programming Interfaces), clean legality separation, strong provenance and diagnostics.

## Non‑goals / limits
- Not for global data‑flow problems (e.g., CSE (Common Subexpression Elimination), SCCP (Sparse Conditional Constant Propagation)).
- Limited to DAGs (Directed Acyclic Graphs); won’t see through cycles / across block arguments.
- Pattern “benefits” are magic numbers; each application interprets them.

---

## Minimal, runnable MLIR (Multi-Level Intermediate Representation) snippets

**A) CSE (Common Subexpression Elimination)**
```mlir
module {
  func.func @cse(%a: i32, %b: i32) -> i32 {
    %0 = arith.addi %a, %b : i32 loc("A")
    %1 = arith.addi %a, %b : i32 loc("B")
    %2 = arith.muli %0, %1 : i32
    return %2 : i32
  }
}
```
Run: `mlir-opt input.mlir -cse`
Result: `%1` removed; `%2 = arith.muli %0, %0`.

**B) LICM (Loop‑Invariant Code Motion)**
```mlir
module {
  func.func @licm(%A: i32, %n: index) -> i32 {
    %c0  = arith.constant 0 : index
    %c1  = arith.constant 1 : index
    %z0  = arith.constant 0 : i32
    %res = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %z0) -> (i32) {
      %two   = arith.constant 2 : i32
      %scale = arith.muli %A, %two : i32
      %acc1  = arith.addi %acc, %scale : i32
      scf.yield %acc1 : i32
    }
    return %res : i32
  }
}
```
Run: `mlir-opt input.mlir -loop-invariant-code-motion`
Effect: `%two` and `%scale` are hoisted to the preheader; loop body shrinks.

---

## Practical takeaways
- Use the **canonicalizer** + **CSE (Common Subexpression Elimination)** early to normalize IR (Intermediate Representation); then apply richer **RewritePattern**s.
- Prefer **declarative patterns** (DRR (Declarative Rewrite Rules), PDL/PDLL (Pattern Description Language / Pattern Description Language frontend)) over ad‑hoc C++ when possible.
- Be mindful of **dominance**, **legality**, and **register pressure** when hoisting/fusing.
- Consider **equality saturation** for domains where early canonicalization can kill opportunities.
- Remember limits: this infra isn’t a substitute for **CFG (Control‑Flow Graph)**-wide analyses like **GVN (Global Value Numbering)** or **SCCP (Sparse Conditional Constant Propagation)**.

---

*Prepared as a quick post‑able note; feel free to copy/paste to your site as is.*

