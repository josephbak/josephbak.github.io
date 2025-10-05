+++
title = "MLIR as a Rewritable Graph Database"
date = 2025-10-04
page_template = "post.html"
insert_anchor_links = "right"
generate_feeds = true
+++

## 1. Concept Overview

**MLIR (Multi-Level Intermediate Representation)** can be understood not just as a compiler IR framework, but as a *rewritable, typed graph database*.

At its core, MLIR stores structured, typed, and interconnected data (operations and values) that can be **queried, transformed, and restructured** — just like a database manages records with schema and queries.

| Database Concept | MLIR Equivalent | Description |
|------------------|----------------|-------------|
| **Schema** | Operation definitions (via TableGen) | Define what nodes (ops) can exist and what fields (operands, attributes) they have. |
| **Records** | Operation instances in IR | Concrete nodes representing program fragments. |
| **Query / Update** | Rewrite patterns (DRR or C++) | Declarative or imperative transformations on subsets of the IR graph. |
| **Query Engine** | Pass manager / pattern driver | Executes rewrites efficiently, ensuring consistency and termination. |
| **Materialized View** | `.mlir` textual snapshot | Human-readable dump of current IR database state. |

So MLIR ≈ *a database of program operations*, where **dialects** define schemas, **rewrites** define queries/updates, and **passes** execute those transformations.

---

## 2. Why "Database", Not Just "Graph"

MLIR’s IR is a **typed, modular, and queryable graph**, not a raw untyped DAG (Directed Acyclic Graph). Here’s the key distinction:

| Aspect | Plain Graph | MLIR as Graph Database |
|---------|--------------|------------------------|
| **Structure** | Unstructured nodes/edges | Nodes (ops) and edges (values) with rich typing. |
| **Schema** | None | Defined in ODS/TableGen dialects. |
| **Queries** | Manual graph traversal | Pattern-matching and rewrite rules. |
| **Transactions** | Not defined | PatternRewriter guarantees consistent state after rewrite. |
| **Namespaces** | Flat | Modular dialects (isolated schemas). |
| **Persistence** | Optional | `.mlir` provides textual serialization. |

Thus, the term *database* highlights the presence of:
- **Schema-driven structure** (dialects + ops)
- **Rule-driven updates** (rewrites)
- **Transaction-like consistency** (rewrites maintain SSA, type, dominance)
- **Query semantics** (find + replace via patterns)

---

## 3. The Core Mapping

### Ops = Schema
Define what data (ops) can exist in the IR:
```tablegen
def TFL_LeakyReluOp : TFL_Op<TFL_Dialect, "leaky_relu", [NoMemoryEffect, SameValueType]> {
  let arguments = (ins F32Tensor:$x, F32Attr:$alpha);
  let results   = (outs F32Tensor:$y);
}
```
This is analogous to:
```sql
CREATE TABLE LeakyRelu (x FLOAT, alpha FLOAT, y FLOAT);
```
Each operation instance is a *record* conforming to this schema.

### Patterns = Rewrite Rules
Declarative DRR or imperative C++ patterns are like database *update queries*:
```tablegen
def : Pat<(TF_LeakyReluOp $x, F32Attr:$a), (TFL_LeakyReluOp $x, $a)>;
```
Equivalent to:
```sql
UPDATE ops SET dialect = 'tfl' WHERE op = 'tf.LeakyRelu';
```
They match subgraphs and replace them with new equivalent forms.

### Passes = Query Engine
Passes orchestrate which rewrites run and in what order:
- Apply rewrites until fixpoint (no matches).
- Maintain consistency (SSA, type legality).
- Optimize or lower IR progressively.

Example:
```cpp
struct LegalizeTFToTFLPass : public PassWrapper<LegalizeTFToTFLPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateWithGenerated(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
```

---

## 4. Related Work and Analogues

### a. Compilers as Databases
- **Martin Odersky (Scala creator)**: described compilers as *in-memory databases* that store program facts and derive new ones.
- **Reddit/HackerNews discussions**: reinforce the view of compilers as structured data systems that can be queried and transformed.

### b. MLIR Research
- **MLIR LangRef:** describes the IR as a *graph of operations and values*.
- **MLIR for Graph Algorithms (LLVM Doc):** formalizes MLIR as a graph manipulation substrate.
- **Relational MLIR Dialects:** academic work modeling *relational algebra* as MLIR dialects for query optimization (e.g., PVLDB 2022, Jungmair et al.).
- **Equality Saturation (Egg/MLIR integration):** uses e-graphs (equivalence graphs) as rewrite databases for exploring all possible transformations simultaneously.

### c. Database-style Compilers
- **Soufflé Datalog Compiler:** represents program facts as relations; transformations and analyses as queries.
- **Flix, QL, CodeQL:** treat program analysis as querying a database of IR facts.
- **Egglog / E-graphs:** treat rewrites as fact insertions and equivalence updates.

---

## 5. Where the Analogy Fails

While the *conceptual mapping* is strong, MLIR isn’t a literal database.

| Database Property | MLIR Reality | Why it breaks |
|--------------------|--------------|----------------|
| **Persistence & durability (ACID)** | In-memory IR, no transaction logs | MLIR rewrites are ephemeral and non-rollbackable. |
| **Relational query language** | No general SQL/Datalog layer | Rewrites are specialized, not general-purpose. |
| **Concurrency & isolation** | Single-threaded mutation | No multi-user or concurrent transactions. |
| **Indexing and query optimization** | Manual traversal | Pattern matching is structural, not cost-based. |
| **General data model** | Program IR only | Can’t represent arbitrary user data. |

So the analogy fails when expecting full DBMS features. MLIR is *database-like* in structure and semantics, not in engineering.

---

## 6. Why the Analogy Still Matters

Despite its limits, the framing of MLIR as a *graph database of program operations* is:
- **Pedagogically powerful:** clarifies dialects as schemas, rewrites as queries, and passes as engines.
- **Architecturally descriptive:** explains why MLIR scales — modular schemas, composable transformations, structured consistency.
- **Future-oriented:** MLIR could evolve toward persistent or incremental IR stores, especially for IDEs or incremental compilers.

In short:
> MLIR bridges compiler theory and data systems. Thinking of it as a *rewritable graph database* highlights how its modular, queryable, schema-driven architecture goes beyond a traditional IR graph.

---

## 7. Key References

- [MLIR LangRef – Structure of IR](https://mlir.llvm.org/docs/LangRef/)
- [MLIR for Graph Algorithms (LLVM Rationale Doc)](https://mlir.llvm.org/docs/Rationale/MLIRForGraphAlgorithms/)
- [PVLDB 2022 – "An MLIR Dialect for Relational Algebra"](https://www.vldb.org/pvldb/vol15/p2389-jungmair.pdf)
- [Egg and Equality Saturation](https://egraphs-good.github.io/)
- [Soufflé Datalog Compiler](https://souffle-lang.github.io/)
- [Martin Odersky – Compilers as Databases (Talk summary)](https://news.ycombinator.com/item?id=10037971)

---

**TL;DR:**
> MLIR is not just a graph of operations — it’s a typed, schema-driven, queryable *graph database* for program transformations. Dialects define the schema, rewrites define the queries, and passes execute them to evolve the IR toward hardware or optimized form.

