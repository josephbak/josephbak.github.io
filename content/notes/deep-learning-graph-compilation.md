+++
title = "Dynamic, Hybrid, and Static Graphs in Deep Learning Compilers"
description = "A math-first explanation of how static, execution-based hybrid, and source-transformation hybrid graph compilation differ during training and inference."
slug = "static-dynamic-hybrid-graph-compilation"
date = 2025-10-08
updated = 2025-10-14
draft = false
[taxonomies]
tags = ["compilers", "mlir", "jax", "pytorch", "training-graphs", "static", "aot-autograd"]
+++

<!-- KaTeX includes -->
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

_A detailed note comparing static, dynamic, and hybrid (execution-based and source-transformation) graph construction in modern deep learning compilers._

---

## 1. Computation Graphs: Formal Basis

Every deep learning model can be expressed as a **computational graph**

$$
G = (V, E)
$$

where:
- $V$: operations (e.g., matmul, relu, add)
- $E$: data dependencies

During training, two subgraphs exist:
1. Forward graph: $G_f$, computing $y = f_\theta(x)$
2. Backward graph: $G_b$, computing gradients $\nabla_\theta L = \frac{\partial L(f_\theta(x))}{\partial \theta}$

Combined:
$$
G_{train} = G_f \cup G_b
$$

---

## 2. Static Graph Compilation

### Definition

A **static graph** is built and optimized **before any execution**.  
It represents *all* possible computations and data flows as a fixed IR.

Formally:
$$
\text{Compile}(f) = G_{static} \xrightarrow[]{\text{optimize}} \text{binary}
$$

The graph is immutable at runtime—no retracing or rebuilding.

### Workflow Example (ONNX, TensorFlow Graph Mode, IREE)

```
Python model
   ↓ (convert to static IR)
GraphDef / MLIR / ONNX IR
   ↓ optimization passes (fusion, CSE, folding)
LLVM / GPU code
   ↓
Executable binary
```

### Advantages

- Full ahead-of-time (AOT) optimization
- Deterministic performance and memory planning
- Ideal for **inference**

### Disadvantages

- Inflexible to input-dependent control flow
- Harder debugging (no eager execution)
- Not compatible with arbitrary Python dynamism

---

## 3. Execution-Based Hybrid (Trace-Then-Compile)

This is the approach used by **PyTorch 2.0 (Dynamo + AOTAutograd + Inductor)**.

### Process

1. **Execution + Tracing**
   - Run model once with tracer tensors.
   - Record each operation to build a forward graph $G_f$.
2. **Symbolic AD (AOTAutograd)**
   - Apply differentiation rules to $G_f$ symbolically to generate $G_b$.
3. **Fusion + Lowering**
   - Fuse $G_f$ and $G_b$ → $G_{train}$.
   - Lower to MLIR/LLVM → compile.
4. **Caching**
   - Cache compiled binaries per guard set (shape, dtype, branch).

```
Python Code
   ↓ run once with tracer
FX IR
   ↓ AOTAutograd
Forward + Backward Graph
   ↓ Inductor / MLIR
Optimized machine code
   ↓
Reuse until guard fails
```

### Key Features

- Dynamic graph building, static optimization afterward.
- Guarded execution: retrace when shape or branch changes.
- Caching of compiled variants.

### Backward Graph Construction

Backward IR built symbolically:
$$
G_b = AD(G_f)
$$
No numerical gradients computed until execution.

### Optimizer State

Optimizer represented as a state transition:
$$
(\theta', s') = Update(\theta, s, \nabla_\theta L)
$$
Mutated in-place in PyTorch’s imperative semantics.

---

## 4. Source-Transformation Hybrid (Compile-From-Code)

Used in **JAX/XLA, TensorFlow XLA, MLIR AOTAutograd**.

### Process

1. **Symbolic Compilation**
   - Analyze pure function `train_step(params, x, y)` symbolically.
2. **Automatic Differentiation**
   - Compute gradients at IR level.
3. **Compilation**
   - Optimize + lower to machine code once.

```
Pure Function
   ↓
HLO / MLIR IR (forward + backward fused)
   ↓
XLA / LLVM backend
   ↓
Optimized executable
```

### Properties

- Optimizers are pure functions:
  $$
  (\theta', s') = O(\theta, s, g)
  $$
- One compiled binary handles all iterations.
- Symbolic ops (`cond`, `while`) capture dynamic control flow.

### Advantages

- Compile once, run many times.
- Shape polymorphism through symbolic dimensions.
- Maximal optimization potential.

### Disadvantages

- Requires pure, side-effect-free code.
- Longer initial compile time.

---

## 5. Dynamic Behavior Handling

| Dynamic Feature | Execution-Based | Source-Transformation | Static |
|------------------|----------------|------------------------|---------|
| **Branching** | Records one path, guards + retrace | Encodes all paths (`cond`) | Unsupported |
| **Randomness** | Retrace or seed input | RNG state as input | Unsupported |
| **Shape polymorphism** | Shape guards + retrace | Symbolic dimensions | Static shapes only |
| **Control flow** | Runtime guards | Symbolic ops (`while`, `cond`) | Fixed |
| **Python dynamism** | Allowed | Disallowed | Disallowed |

---

## 6. Compilation and Caching Costs

| Property | Static | Execution-Based | Source-Transformation |
|-----------|---------|----------------|------------------------|
| **Number of compiles** | 1 | Variable (depends on guards) | 1 |
| **When compiled** | Before execution | On first execution of each variant | Before execution |
| **Reuse** | Full reuse | Cached by (shape, dtype, branch) | Cached by signature |
| **Flexibility** | Low | High | Medium |
| **Compile time** | Long, once | Medium, possibly many | Long, once |
| **Runtime cost** | Minimal | Low after warm-up | Minimal |

Execution-based systems may retrace multiple times, producing several compiled binaries:
```
Cache:
 ├─ Key(shape=32x32, branch=A) → compiled_graph_1
 ├─ Key(shape=64x64, branch=B) → compiled_graph_2
```
Source-transform systems produce exactly one.

---

## 7. Training Timeline Visualization

```
Iteration →
|----------------------------------------------→ time

Static:
   [Compile once]  [Execute binary many times]

Execution-Based:
   [Trace + Compile once]  [Run compiled binary]
   [Retrace if guard fails → recompile]

Source-Transform:
   [Symbolic compile once]  [Run binary many times]
```

---

## 8. Conceptual Summary

| Category | Static | Execution-Based | Source-Transformation |
|-----------|---------|----------------|------------------------|
| **Graph construction** | Predefined IR | Runtime tracing | Symbolic compilation |
| **Backward creation** | Predefined or AOT | Symbolic AD of traced IR | Symbolic AD at compile time |
| **Dynamic support** | None | Guards + retrace | Symbolic ops |
| **Optimizer state** | Static updates | Mutable buffers | Functional updates |
| **Fidelity to runtime** | Exact but rigid | Faithful to executed path | Semantically equivalent |
| **Recompilation** | Never | When guards fail | Rare |
| **Best use** | Inference | Research/training | Production training |

---

## 9. ASCII Conceptual Diagram

```
                 ┌───────────────────────┐
                 │ Static Compilation    │
                 │  (ONNX, IREE, TF)     │
                 └──────────┬────────────┘
                            │
      ┌─────────────────────┼──────────────────────┐
      │                     │                      │
┌───────────────┐    ┌────────────────┐     ┌────────────────────┐
│ Execution-     │    │ Source-        │     │ Fully Dynamic      │
│ Based Hybrid   │    │ Transform      │     │ (Eager, No Compile)│
│ (PyTorch 2.0)  │    │ (JAX/XLA)      │     │ (Old PyTorch)      │
└──────┬─────────┘    └────────┬───────┘     └──────────┬────────┘
       │                       │                        │
Run Once → Trace → Compile → Cache      Symbolic Compile Once → Run
       │                       │                        │
Multiple variants         One static binary        No global optimization
```

---

## 10. References

1. PyTorch 2.0 Compiler Internals — https://pytorch.org/get-started/pytorch-2.0/  
2. JAX and XLA Compilation — https://jax.readthedocs.io  
3. TensorFlow XLA — https://www.tensorflow.org/xla  
4. MLIR AOT Autodiff RFC — https://mlir.llvm.org  
5. Inductor + AOTAutograd Architecture Notes, PyTorch Core (2023)  
6. IREE Compiler Documentation — https://iree.dev

---

**Summary:**  
Static compilation predefines all computation; execution-based hybrid tracing compiles what actually happens dynamically; source-transformation statically compiles what could happen symbolically. Training workloads balance flexibility and performance through hybridization: dynamic in definition, static in optimization.
