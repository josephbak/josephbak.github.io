+++
title = "Compiler Strategies for Cache- and Memory-Friendly Design Across CPUs and GPUs"
description = "A systems-level overview of general compiler rules and hybrid design principles for optimizing cache and memory performance on modern CPUs and GPUs."
slug = "compiler-cache-memory-cpu-gpu-hybrid"
date = 2025-10-18
updated = 2025-10-18
draft = false
[taxonomies]
tags = ["compilers", "cpu", "gpu", "cache-optimization", "memory-hierarchy", "systems-design"]
+++

<!-- KaTeX includes (inline; requires markdown.render_unsafe = true) -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.js"></script>

---

# Overview

Modern compilers are increasingly **memory- and cache-aware**, as performance is no longer limited by raw FLOPs (floating-point operations per second) but by **data movement** and **cache locality**. CPUs and GPUs expose radically different memory hierarchies, and compiler strategies must adapt to each while still leveraging shared principles of *locality, reuse, and predictability*.

This note outlines:

1. General cache and memory friendliness rules for **modern CPUs**.
2. Equivalent performance rules for **modern GPUs**.
3. A **hybrid compiler design** that targets both architectures from a single intermediate representation (IR).

---

# 1. Cache- and Memory-Friendliness Rules for CPUs

Modern CPUs (Central Processing Units) are latency-oriented and hierarchical. Typical cache layers include:

| Level | Typical Size | Access Time | Notes |
|--------|---------------|-------------|-------|
| **L1** | 32–64 KB | ~1 ns | Split into instruction (I) and data (D) caches |
| **L2** | 256–2048 KB | ~4 ns | Unified per core |
| **L3** | 4–64 MB | ~10–20 ns | Shared across cores |
| **DRAM** | GBs | ~80–120 ns | Main memory |

### 1.1 Universal Compiler Principles for CPUs

| Principle | Explanation | Benefit |
|------------|-------------|----------|
| **Spatial Locality** | Access contiguous memory (arrays, SoA layouts) | Fewer cache-line fills |
| **Temporal Locality** | Reuse recently accessed data | Higher cache hit rate |
| **Hot Code Compactness** | Keep hot loops < 8–16 KB to fit in L1I (Instruction Cache) | Fewer ICACHE misses |
| **Predictable Branching** | Favor likely branches as fallthrough | Improves branch predictor accuracy |
| **Prefetch-Friendly Access** | Stride-1 loops and predictable indexing | Enables hardware prefetchers |
| **Loop Tiling / Blocking** | Partition loops so working sets fit in cache | Reduces cache thrash |
| **Data Structure Layout** | Use Structure of Arrays (SoA) for vectorized ops | Improves SIMD (Single Instruction Multiple Data) efficiency |
| **Quantization and Packing** | Compress frequently accessed values (e.g., thresholds) | Better cache density |

**Compiler Techniques:**
- Polyhedral loop optimization and cache tiling (e.g., MLIR affine dialect)
- Function inlining guided by instruction cache budget
- Profile-guided reordering of hot functions
- Prefetch insertion (software prefetch or hardware hinting)

**References:**
- Drepper, *What Every Programmer Should Know About Memory* (2007)
- Fog, *Instruction Tables and Cache Optimizations* (2018)
- Intel Optimization Manual, vol. 1–3 (2024)

---

# 2. Memory Hierarchies and Rules for GPUs

GPUs (Graphics Processing Units) are throughput-oriented, designed for **massive parallelism** and **bandwidth utilization** rather than latency minimization. A simplified hierarchy:

| Memory Type | Latency | Scope | Access | Notes |
|--------------|----------|--------|--------|-------|
| **Registers** | ~1 cycle | Per-thread | Fastest | Holds scalars and temporaries |
| **Shared Memory (L1)** | 20–30 ns | Per-block | Programmable SRAM | Manually managed cache |
| **L2 Cache** | ~100 ns | All SMs | Automatic | Shared across chip |
| **Global Memory (VRAM/HBM)** | 300–800 ns | Device | Coalesced access needed |
| **Host Memory (CPU RAM)** | µs | Across PCIe/NVLink | Very high latency |

### 2.1 Universal GPU Compiler Rules

| Principle | Explanation | Benefit |
|------------|-------------|----------|
| **Memory Coalescing** | Adjacent threads in a warp access consecutive addresses | Full memory transaction utilization |
| **Shared Memory Reuse** | Tile reusable data into shared memory | Reduces global memory traffic |
| **Warp Uniformity** | All threads in a warp follow same control path | Prevents divergence serialization |
| **Predication over Branching** | Replace if/else with arithmetic masks | Keeps SIMD lanes active |
| **High Occupancy** | Use few registers & shared memory per thread | Hides latency by overlapping warps |
| **Compute–Memory Balance** | Increase arithmetic intensity (FLOPs/byte) | Hides memory latency |
| **Explicit Tiling** | Use kernel-level loop tiling & block synchronization | Improves data reuse and scheduling |

**Compiler Techniques:**
- Kernel fusion and loop tiling (e.g., MLIR’s GPU dialect)
- Thread-block scheduling with register pressure control
- Warp-level predication lowering
- Autotuning for launch geometry `(grid, block, warp)`

**References:**
- NVIDIA CUDA Best Practices Guide (2025)
- AMD ROCm Optimization Guide (2024)
- Volkov, *Understanding Latency Hiding on GPUs* (2016)

---

# 3. Hybrid Compiler Design: Unified IR for CPUs and GPUs

The goal: **a single intermediate representation (IR)** that expresses control flow, memory access, and parallel structure so it can be lowered to both CPU and GPU backends efficiently.

### 3.1 Core Design

A unified IR should model computation as **structured regions with explicit data dependencies**:

```
forest.apply %X {
  tree.begin %t0
    %n0 = node.split.num  fid=3, thr=0.72
    br_if %n0, label %L1, %L2
  %L1: node.leaf  val=0.13
  %L2: node.leaf  val=-0.02
  tree.end
}
```

This high-level control flow can be specialized differently per backend.

### 3.2 Lowering Strategies

| Phase | CPU Lowering | GPU Lowering |
|--------|---------------|---------------|
| **Control Flow** | Emit branchy code; flatten hot paths | Use predicated arithmetic; avoid divergence |
| **Parallelism** | Vectorize (AVX512/SVE) across samples | Map threads to (sample, tree) grid |
| **Memory Layout** | SoA (Structure of Arrays) | Coalesced global layout + shared tiles |
| **Quantization** | i16 thresholds for cache density | Same quantization reused for bandwidth saving |
| **Runtime API (Application Programming Interface)** | `void predict_cpu(X, Y)` | `__global__ void predict_gpu(X, Y)` |

### 3.3 Scheduling Knobs

| Parameter | Description | Target Effect |
|------------|-------------|----------------|
| `tile_size` | Loop tile size | Cache fit (CPU) / Shared mem fit (GPU) |
| `packet_width` | SIMD (Single Instruction Multiple Data) width | Vector utilization |
| `grid/block` | Launch geometry | Occupancy, latency hiding |
| `quant_bits` | Quantization level (8/12/16) | Cache or bandwidth tradeoff |

### 3.4 Autotuning Layer

An autotuner can search for optimal `(tile, packet, block, quant)` given hardware profiles:

\[
S^* = \arg\max_{S \in \text{ScheduleSpace}} \text{Throughput}(S, \text{arch})
\]

Similar frameworks exist in **TVM**, **IREE**, and **Halide**.

---

# 4. Converging Trends

1. **Hybrid IRs (Intermediate Representations)** — MLIR’s multi-dialect model enables CPU/GPU lowering from a shared graph.
2. **Hardware-intent inference** — compilers infer data reuse and control-flow divergence statically.
3. **Autotuning as compilation** — schedule search is now part of the compiler pipeline.
4. **Quantization-unified optimization** — same quantized model runs efficiently across heterogeneous targets.

---

# References

1. Ulrich Drepper, *What Every Programmer Should Know About Memory*, Red Hat, 2007.
2. Intel Corporation, *Optimization Reference Manual*, 2024.
3. Agner Fog, *Instruction Tables: Lists of Instruction Latencies, Throughputs and Micro-operation Counts*, 2018.
4. Vasily Volkov, *Understanding Latency Hiding on GPUs*, UC Berkeley, 2016.
5. NVIDIA Corporation, *CUDA Best Practices Guide*, 2025.
6. AMD ROCm Developer Tools, *ROCm Optimization Guide*, 2024.
7. MLIR Project, *Affine and GPU Dialects*, LLVM.org, 2025.
8. Chen et al., *TVM: End-to-End Optimization Stack for Deep Learning*, OSDI 2018.
9. Ragan-Kelley et al., *Halide: Decoupling Algorithms from Schedules*, CACM 2017.
10. Google IREE Team, *Unified Compiler Infrastructure for ML on CPUs, GPUs, and Accelerators*, 2023.

