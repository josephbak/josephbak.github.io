+++
title = "Toy Compiler (MLIR Tutorial)"
weight = 1
draft = false
+++

## Overview
The **Toy Compiler** is built by following the MLIR tutorial, which provides a hands-on path into the design of compilers for domain-specific languages. The Toy language itself is deliberately simple and tensor-oriented, but the project’s real value lies in exposing the principles of working with MLIR: defining dialects, building operations, applying passes, and lowering programs into other IRs such as LLVM IR.

## Progress So Far
Several important steps have already been completed:
- A **parser and AST** were written to represent Toy programs in memory.  
- A custom **dialect** was created to express Toy semantics directly within MLIR.  
- **StructType** was added to the dialect, enabling the representation of composite data structures.  
- Support for **constant folding and inlining** has been implemented, demonstrating how MLIR passes can simplify programs automatically.  
- Toy programs can now be lowered through MLIR to LLVM IR, which opens the door to execution through LLVM’s JIT or native compilation.

## Lessons Learned
The most significant takeaways so far are:
- **Dialect design:** MLIR’s dialect mechanism provides a flexible way to encode domain-specific operations at the IR level. This is very different from monolithic IRs, where one is constrained to pre-defined semantics.  
- **Optimization infrastructure:** Constant folding and inlining were not implemented as ad hoc hacks, but through reusable pass infrastructure. This shows how MLIR encourages composability of transformations.  
- **Lowering:** Understanding the process of lowering from one IR to another is crucial, as it clarifies how high-level semantics eventually map to machine code. The separation of concerns across levels of IR mirrors how modern compiler stacks are layered.  
- **Debugging passes:** Stepping through MLIR output at different phases reveals how each transformation contributes to the final program shape.

## Next Steps
Future work will focus on deepening the compiler’s capabilities:
1. Implement additional optimizations such as common subexpression elimination (CSE) and dead code elimination.  
2. Extend the Toy language slightly, for example by adding new operations or data types, to explore how dialect evolution affects the IR.  
3. Refine the lowering pipeline so that more complex Toy programs can be compiled and executed using LLVM’s JIT infrastructure.  
4. Investigate integration with external toolchains, such as lowering all the way to the RISC-V backend to run Toy programs in emulation.  
5. Document the full compiler pipeline with diagrams, showing each IR stage from AST to LLVM IR.

This project is ongoing, and while Toy itself is not meant for production use, the compiler acts as a small but meaningful laboratory for learning MLIR, dialect design, and IR transformation pipelines.

