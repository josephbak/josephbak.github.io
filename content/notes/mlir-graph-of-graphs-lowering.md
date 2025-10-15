+++
title = "Lowering as Graph-of-Graphs: A Structural View of MLIR and LLVM Pipelines"
description = "A formal note describing how compiler lowering pipelines can be represented as graph transformations over computation graphs, forming a hierarchical graph-of-graphs structure."
slug = "mlir-graph-of-graphs-lowering"
date = 2025-10-14
updated = 2025-10-14
draft = false
[taxonomies]
tags = ["mlir", "llvm", "compilers", "graph", "category-theory"]
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
 {left: "\[", right: "\]", display: true},
 {left: "$", right: "$", display: false},
 {left: "\(", right: "\)", display: false}
 ],
 throwOnError: false
 });
 });
</script>

_A structural note on how the MLIR/LLVM lowering process itself forms a higher-order computation graph — a graph of graph transformations._

---

## TL;DR

- Every program in MLIR or LLVM IR can be represented as a **computation graph** $G = (V,E)$.
- Each **compiler pass** acts as a **graph transformation** $P_i: G_{i-1} \to G_i$.
- The entire **pipeline** is therefore a **meta-graph** (a directed graph whose nodes are IR graphs and edges are passes).
- Conceptually, compiler lowering is a **graph of transformations over graphs of computation**.
- This interpretation connects modern compiler design with **graph rewriting systems** and **category theory**.

---

## 1) Level-0 — The IR Graph: Computation

At the base level, a program is represented as a directed graph

$$
G_0 = (V_0, E_0)
$$

where:
- $V_0$: operations (e.g., `mhlo.add`, `mhlo.dot`, `linalg.matmul`)
- $E_0$: data dependencies (SSA edges)

Example:

```
   %x
    │
    ▼
 mhlo.add
    │
    ▼
 mhlo.relu
    │
    ▼
 mhlo.return
```

This is the **program graph**—it captures what is computed, not how it is lowered.

---

## 2) Level-1 — The Pass Graph: Local Rewrite Structure

Each compiler pass applies a set of rewrite rules to subgraphs of the IR.

Formally:
$$
P = (V_P, E_P)
$$

where:
- $V_P$: rewrite rules (pattern + replacement)
- $E_P$: dependencies among rules (e.g., one rewrite must precede another)

Example (MHLO → Linalg lowering):

```
 Pass: mhlo → linalg
 ───────────────────
 mhlo.add  ───→  linalg.add
 mhlo.relu ───→  linalg.max
```

This graph represents the **local transformation flow** within a single pass.

---

## 3) Level-2 — The Pipeline Meta-Graph: Graphs of Graphs

Each compiler pass maps one complete IR graph to another:

$$
P_i : G_{i-1} \rightarrow G_i
$$

Hence the overall compiler pipeline can be viewed as a **meta-graph**:

$$
\mathcal{G} = (\mathcal{V}, \mathcal{E}), \quad
\mathcal{V} = \{G_0, G_1, \ldots, G_n\}, \quad
\mathcal{E} = \{(G_{i-1}, G_i) \mid G_i = P_i(G_{i-1})\}
$$

Example pipeline:

```
[MHLO Graph] --(P1)--> [Linalg Graph] --(P2)--> [LLVM Graph]
```

- Nodes ($\mathcal{V}$): IR snapshots (entire program graphs)
- Edges ($\mathcal{E}$): passes (transformations)

The meta-graph is **directed and acyclic**, since lowering moves from high-level to low-level representations.

---

## 4) Hierarchy Summary

| Level | Nodes | Edges | Meaning |
|:------|:------|:------|:--------|
| 0: IR Graph | Operations | Dataflow | The computation itself |
| 1: Pass Graph | Rewrite rules | Rule dependencies | Local transformations |
| 2: Pipeline Meta-Graph | IR snapshots | Passes between IRs | The full lowering pipeline |

Each higher level acts **on** the level beneath it.

---

## 5) Category-Theoretic View

Define a category $\mathcal{C}$ whose objects are IR graphs and whose morphisms are passes:

$$
\text{Obj}(\mathcal{C}) = \{G_i\}, \quad
\text{Mor}(\mathcal{C}) = \{P_i: G_{i-1} \to G_i\}
$$

Then a lowering pipeline is a **composition of morphisms**:

$$
G_0 \xrightarrow{P_1} G_1 \xrightarrow{P_2} G_2 \xrightarrow{P_3} \cdots \xrightarrow{P_n} G_n
$$

A compiler can thus be interpreted as a **functor** between categories of graphs, preserving semantic structure through successive transformations.

---

## 6) ASCII Visualization

```
LEVEL 2: Meta-Graph (Pipeline)
──────────────────────────────
   [G0: MHLO] ──P1──> [G1: Linalg] ──P2──> [G2: LLVM]
          │                  │                    │
          ▼                  ▼                    ▼
LEVEL 1: Pass Graphs
──────────────────────────────
   mhlo.add→linalg.add   linalg.matmul→llvm.call
   mhlo.relu→linalg.max  ...
          │                  │
          ▼                  ▼
LEVEL 0: IR Graphs
──────────────────────────────
   mhlo.add ─→ mhlo.relu → return
   linalg.add ─→ linalg.max → return
   llvm.add ─→ llvm.ret
```

---

## 7) Relation to Established Frameworks

The “graph-of-graphs” viewpoint exists under several names across different research areas:

| Domain | Equivalent Concept | Reference |
|:--------|:-------------------|:-----------|
| Compiler theory | Pass pipelines as graph transformations | LLVM / MLIR design docs |
| Graph rewriting systems | Double-pushout (DPO) formalism | Ehrig et al., *Graph Grammars*, 1999 |
| Category theory | Functors between graph categories | Carette et al., *Compiling to Categories*, 2010 |
| Equality saturation | DAG of equivalent graphs | Willsey et al., *egg*, 2021 |
| Verified compilation | IR equivalence proofs | Lopes et al., *Alive2*, 2020 |
| Meta-scheduling | Compiler DAG search | Chen et al., *TVM Unity*, 2023 |

---

## 8) Selected References

1. **C. Lattner et al.**, *MLIR: A Compiler Infrastructure for the End of Moore’s Law*, arXiv:2002.11054 (2021).  
   Defines the dialect + pass architecture; each dialect transition corresponds to a graph morphism.

2. **H. Ehrig et al.**, *Handbook of Graph Grammars and Computing by Graph Transformation*, World Scientific (1999).  
   Establishes the double-pushout formalism used to describe structured graph rewriting.

3. **J. Willsey et al.**, *egg: Fast and Extensible Equality Saturation*, PLDI (2021).  
   Models optimization as the saturation of a DAG of equivalent program graphs.

4. **N. Lopes et al.**, *Alive2: Bounded Translation Validation for LLVM*, arXiv:2004.04344.  
   Proves semantic equivalence between IR transformations, effectively linking graphs $G_i$ and $G_{i+1}$.

5. **O. Kiselyov, J. Carette, C. Shan**, *Compiling to Categories*, ICFP (2010).  
   Provides categorical semantics for compilation as functor composition.

6. **T. Chen et al.**, *TVM: End-to-End Optimization Stack for Deep Learning*, OSDI (2018).  
   Represents compiler pass pipelines as meta-graphs for automated search and scheduling.

---

## 9) Takeaway

Lowering in MLIR/LLVM is not a simple sequence of textual transformations.  
It is a **hierarchical system of graph transformations**, where:

$$
\mathcal{G} : G_0 \xrightarrow{P_1} G_1 \xrightarrow{P_2} \cdots \xrightarrow{P_n} G_n
$$

Each $G_i$ is an IR graph (computation), and each $P_i$ is a morphism (pass).  
This structure unifies compiler pipelines, graph rewriting systems, and categorical reasoning into a single framework for understanding how complex program transformations can be represented, optimized, and verified.

---

**Author:** Joseph Bak  
**Date:** 2025-10-14  
**Keywords:** MLIR, LLVM, Compiler Passes, Graph Transformation, Category Theory
