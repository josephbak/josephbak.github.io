+++
title = "RISC-V Toolchain Experiment"
weight = 2
draft = false
+++

## Overview
This project is an attempt to build a minimal **RISC-V cross-compilation toolchain**, starting from a high-level language frontend and ending with execution on an emulator. The primary goal is to understand each stage of the compilation process, the interaction between tools, and the subtleties of targeting a modern ISA like RISC-V. While prepackaged RISC-V toolchains exist, the exercise of assembling one from its components is intended to expose the moving parts that are usually abstracted away.

## Current Status
At the moment, the pipeline is functional at a basic level:
- The **Clang frontend** is successfully generating LLVM IR for simple C programs.  
- The **LLVM RISC-V backend** lowers this IR into target assembly.  
- The **LLD linker** resolves symbols and produces an ELF binary.  
- The resulting binary can be executed using **QEMU RISC-V** emulation.

With this setup, a small "Hello, world" program has been compiled and executed, demonstrating end-to-end functionality of the toolchain.

## Challenges
Several areas present difficulties that require further investigation:
- **C runtime support:** Bare-metal targets lack libc, so writing a minimal runtime is necessary to bootstrap execution.  
- **Linker scripts:** Understanding and customizing linker scripts is critical when targeting non-standard environments, such as embedded boards or simulators.  
- **ISA variants:** Deciding between RV32I and RV64I changes assumptions about memory layout and register usage, and complicates the toolchain configuration.  
- **Debugging and inspection:** QEMU provides useful insight, but debugging tool integration is still somewhat brittle.

## Next Steps
The immediate objectives include:
1. Developing a minimal C runtime to enable bare-metal execution without relying on a full libc.  
2. Running experiments with optimization flags (`-O1`, `-O2`, `-O3`) and comparing the generated assembly to see how LLVM optimizations are reflected in RISC-V code.  
3. Expanding the set of programs tested beyond trivial cases, including simple I/O and recursion.  
4. Investigating toolchain support for RISC-V extensions, particularly the vector extension, and determining what changes are necessary to enable them.  
5. Laying groundwork for targeting physical RISC-V hardware instead of just emulation.

This project is intentionally exploratory, serving as both a learning exercise and a stepping stone toward more serious compiler and systems work.
