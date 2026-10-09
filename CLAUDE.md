# CLAUDE.md — kernel-tv

> **Scope.** Guidance for working on this repository: the kernel-tv translation
> validator.
> **Language.** Chat/explanations in Chinese; keep **all code, comments, commit
> messages, and docs (including this file) in English**.
> **History.** kernel-tv lived as `tv/` inside a Triton checkout until it was
> split out into its own repository (2026-09-04); paths in the archival
> documents below still carry that `tv/` prefix. The old, Triton-only version of
> this file is `CLAUDE.legacy.md` (SUPERSEDED — do not follow it).

---

## 1. What this is (current state)

A working **SMT translation validator**: `triton-tv a.ttir b.ttir` proves two
MLIR functions semantically equivalent with Z3.
Exit codes: `0` = EQUIVALENT (UNSAT), `1` = NOT EQUIVALENT (SAT, prints a
counterexample), `2` = UNKNOWN. Today it validates **Triton TTIR** (add_kernel,
softmax) at realistic sizes.

**M0 is DONE.** The code is split into an MLIR-free core + per-language builders
(§6). Verified: core has zero MLIR, `libkernel-smt.a` links only Z3, all unit tests
and the eval suite green.

**Current design facts (accurate — trust these over any older doc):**
- **Memory = Option B**: one byte-addressable Z3 array `Array(BV64,BV8)` per
  pointer argument (Triton args don't alias). `Memory::store` builds a **single
  `z3::lambda`** heap update; `checkEquivalence` compares at a **symbolic witness
  address** (`select(m1,w) != select(m2,w)`), NOT array extensionality — the
  latter returns `unknown`. (`semantics/Memory.*`, `semantics/Equivalence.*`)
- **Validator FP = Abstract mode**: each FP value is an opaque `BitVec` id;
  ops are uninterpreted Z3 functions with **only 5 axioms** (5 reserved consts
  distinct; add/mul/max commutative; neg involutive). Deliberately no
  associativity, NaN propagation, or zero identities. `lt`/`le` throw.
  (`semantics/AbstractFp.*`) The MLIR-free core now has partial FPA support
  for constants, add/sub/mul/div, and fused multiply-add; the binary does not
  select it yet. Z3 FPA does not retain NaN payload bits.
- **Ops modeled**: `arith` constant/addi/subi/muli/andi/extsi/cmpi(10 preds)/
  addf/subf/mulf/divf/maxnumf; `math.exp`; `tt.` get_program_id/make_range/splat/
  addptr(scalar **and** tile)/load/store (both **require a mask**)/reduce
  (combine-region fold, **1-D single-operand only**).
- **Not modeled**: control flow (`scf.if/for/while` are `llvm_unreachable`),
  `tt.call` (rely on `-inline`), `tt.dot`, multi-dim reduce, `arith.cmpf`,
  `arith.select`, most casts, all `math.*` except `exp`. An unmodeled op is a
  **hard crash**, not a graceful "unsupported" verdict (to fix in M1).
- **Practical note**: proving EQUIVALENCE (UNSAT) is fast even at 1024 elems;
  proving NON-equivalence (SAT) is much harder — use **small tiles** for
  inequality cases.

## 2. Macro goal — abstract tensor semantics, mapped per language

**The whole pipeline = abstract tensor semantics (core) + per-language mapping
(builder).** Those two together are what turns a GPU kernel into SMT semantics:

- The **core** owns a *reasonably complete but abstract* set of **tensor
  operations** — elementwise map, reduce/scan, contraction, structural
  (iota/splat/broadcast/reshape/transpose), gather/scatter, masked windowed
  memory access, loop combinators, program identity. They are defined by their
  **semantics**, not by any language's op names, and depend on **no language**.
- Each **builder** has exactly one job: **map its own language's tensor
  operations onto that abstract set**. The mapping is not 1:1 — one core op
  serves many surface syntaxes (Triton `tt.load(ptrs,mask,other)`, TileLang
  `T.copy`, Pallas `pl.load(ref,idx)` are all "masked windowed read"), and one
  language op may expand into several core ops.
- **Completeness test:** adding a new language should need **no new core ops**.

⚠️ **Known gap:** today's `Context` API was lifted from the Triton handlers, so
parts of it are still Triton-shaped — e.g. `addPtr`/`splatPtr` are *pointer-level*
addressing, not abstract tensor ops (memref/TPU languages don't address by
pointer). Generalizing the op set along the lines above is M1 work. See
`doc/kernel-smt-design.md` §"Abstract tensor operation set".

**Near-term north star:** ship a complete implementation on **Triton** and use it
to **find & reproduce a real Triton compilation bug**.

The library split:
- **`kernel-smt`** — hardware-neutral core (the abstract tensor semantics above).
- **`kernel-gpu-smt`** — GPU/SIMT layer (M2): layouts/`convert_layout`, shared
  memory, warp/lane, async TMA/mbarrier, warp specialization.
- **(future) `kernel-accel-smt`** — TPU/Mosaic + Trainium/NKI; **one layer covers
  both**.

Full goals + success criteria: `doc/kernel-smt-goals.md`. Interfaces:
`doc/kernel-smt-design.md`. Numbered plan: `doc/roadmap.md`.

**Locked decisions:** Builder API (adapter calls the lib; no neutral IR);
incremental access model (linear byte-heap + pointer, abstract enough to swap for
memref/TPU later); `program_id` in the core; core holds no `mlir::Value` —
pointer provenance is an opaque `MemId`; **FP axiom profiles** (exact bit-to-bit
vs reassoc-allowed); **loop model** `--loop-model=unroll|summarize|auto` (§7).

## 3. Development rules

- 🔴 **Never write into the Triton checkout.** `$TRITON_ROOT` names a build we
  only consume; treat it as read-only. Everything kernel-tv needs lives in this
  repository — if something looks like it needs a Triton change, that is a
  design problem here.
- **How the Triton dependency works.** Triton's core is CMake OBJECT libraries
  with no exported package, so there is nothing to `find_package()`. What an
  executable links is the raw `.o` files out of the build tree (~305 of them),
  so `cmake/PrebuiltTriton.cmake` harvests exactly the set `bin/triton-opt`
  links — that set is, by construction, the one `triton-tv` needs. A second
  language backend follows the same shape: one env var naming a built checkout.
- ⚠️ **`builder/mlir` is not language-neutral yet.** `DTypeOf`/`Env`/`State`
  include `triton/Dialect/Triton/IR/Types.h` for `tt.ptr`, and `State`'s
  dispatch calls the `tt.*` handlers directly. Decoupling it is a prerequisite
  for a TileLang (or any second) backend — see §2 and `doc/roadmap.md`.
- Any change to a design component **must update** the tests.
- **Chinese** for chat; **English** for code/comments/commits/docs.
- Commit policy: may **auto-commit new changes** (short one-line msg, no Claude
  author line); **ask before** history rewrites (squash/amend/rebase).

## 4. Build

C++ changes require a rebuild. Full instructions and the Z3 requirement
(**>= 4.8.12**, for `z3::sgt`/`sge`) are in `README.md`.

```bash
# core only — fast, no MLIR
cmake -S . -B build -G Ninja -DZ3_ROOT=<z3>
ninja -C build kernel-smt kernel-smt-tests

# full — needs an already-built Triton checkout; the triton-tv link is slow
TRITON_ROOT=<triton> cmake -S . -B build -G Ninja -DZ3_ROOT=<z3>
ninja -C build triton-tv tv-validator-tests
```

## 5. Testing

**Unit tests.** Two groups:
- **core (Z3-only, no MLIR)** — `ctest --test-dir build -R KernelSmt` (5 tests:
  Types, AbstractFp, Memory, Context, Equivalence).
- **builder (needs MLIR)** — `ctest --test-dir build -R TestTritonTV` (Env, State).

**Routine testing — validator gates + optimization-permutation campaign:**

```bash
python eval/run_eval.py all       # gates: pairs + inequal + compile-options, then timing
python eval/permute_passes.py     # bug hunt on add_kernel (all 1/2/3-pass TTIR perms)
python eval/permute_passes.py \
  --baseline eval/compile-options/softmax_kernel/standard.ttir \
  --out eval/compile-options/softmax_kernel     # bug hunt on softmax
```

- **pairs** — curated equivalent/non-equivalent pairs (verdict must match tag).
- **inequal** — genuinely non-equivalent pairs; must ALL be caught as NEQ
  (soundness). Use small tiles.
- **compile-options** — pass variants vs the unoptimized standard; must be EQUIV.
- **permute_passes.py** — every 1/2/3-pass TTIR-optimization permutation vs the
  reference. Any `NOT EQUIVALENT` = a potential miscompile (saves `BUG_neq.ttir`
  and stops — re-check soundness before filing).

Last runs: pairs 4/4, inequal 6/6, compile-options 9/9; add & softmax each 259
permutations all EQUIVALENT; no miscompile found yet.

⚠️ `eval/compile-options/generate.py` picks the target from the **live GPU**
if one exists. On a different machine that silently changes which architecture's
IR you are validating — pin it explicitly when it matters.

## 6. Where things are

**Naming:** `kernel-tv` is the project (repo, `triton-tv` binary, CMake project
and build helpers); `kernel-smt` is the SMT semantic modelling under it (the
core library, the `kernel_smt` namespace, `kernel-smt-builder-*`, and the
planned `kernel-gpu-smt` / `kernel-accel-smt`). Keep new names on the right side
of that line.


```
semantics/     CORE `kernel-smt` — MLIR-free, links ONLY z3 (namespace kernel_smt)
               Types · Value · AbstractFp · Memory · Context · Equivalence
  test/        Z3-only unit tests (SimpleTest.h harness)
builder/       per-language builders — the ONLY place that includes MLIR
  mlir/        shared by all MLIR languages: DTypeOf · Env · State (walk +
               dispatch + control-flow stubs) · ArithOps
  triton/      Triton-specific: TritonOps (tt.*)
bin/           triton-tv.cpp — validator main
cmake/         PrebuiltMLIRCompiler.cmake · PrebuiltTriton.cmake — how a
               language backend attaches to an already-built compiler
test/validator/  builder-level C++ tests (need MLIR)
eval/          pairs/ inequal/ compile-options/ solver-cost/ run_eval.py
               permute_passes.py
eq_fuzzing/    equivalence fuzzer driving triton-opt
benchmark/     benchmark_kernels.py — 420+ collected @triton.jit kernels
doc/           roadmap.md · kernel-smt-goals.md · kernel-smt-design.md ·
               m0-plan.md · code-navigation.md · tensor-languages-survey.md
  kb/          knowledge base on external tools — REFERENCE, NOT plans
               (alive2-loops.md). Nothing in kb/ is an adopted decision.
paper/         noticable.md — scaling-issue log (e.g. the store-blowup fix)
```

**Start here to read the code:** `doc/code-navigation.md` (layer map,
recommended reading order, key invariants, the 3-edit recipe for adding an op).

## 7. Known limitations / next work

**M1 is next** — extend the core to (almost) all of TTIR. Coverage today:
7 of ~48 `tt.*` ops, ~12 of ~30 `arith.*`, 1 of ~14 `math.*`, **0 of 7**
control-flow ops.

**Decided:**
- **Loops** — `--loop-model=unroll | summarize | auto`. `summarize` lifts a loop
  to a combinator over `k ∈ [0,N)`; its enabling analysis is **affine recurrence
  recognition** (turn `ptr += stride` into `base + k*stride`), decomposed by
  **SCC of the carried-value dependence graph**. `scf.while` is out of scope
  (measured: 2.4% of kernels, all irregular).
- **Generalize the core op set** away from its Triton shape (§2).
- **Atomics are out of scope** (`tt.atomic_rmw`/`cas`, `tl.atomic_add`) — not
  because they are hard, but because the question is ill-posed: they are
  non-deterministic across program instances, and FP add is not associative, so
  the result is not deterministic and "bit-exact equivalence" has no meaning.
  Must report UNSUPPORTED, never a verdict.
- **M1 sub-phases are defined by measured coverage**, not a hand-picked kernel
  list — see `doc/roadmap.md` §M1. Baseline: **15.7%** (100/636) of the
  hand-written corpus is modelable today; the top blockers are dtype casts, for
  loops, `tl.where`, and data-dependent `if` — **`tt.dot` ranks only 9th**.
  MVP target ≥40% hand-written / ≥80% inductor; complete ≥60% + flash attention.

**Measured fact (not a decision):** solver blowup is `trip_count × tile_width`,
and today **both** are statically unfolded — `Context::reduce` folds
element-wise, `Memory::store` unfolds lanes.

**Open — proposals only, nothing decided:** how to report a bounded/truncated
result; unmodeled-op behavior (hard crash today vs an `UNSUPPORTED` verdict); a
solver `--timeout` (none inside the binary today); pointer aliasing (may-alias
vs today's no-alias assumption); UB policy; `program_id` range constraint;
`tt.dot` fidelity; reduce ordering; M1's target-kernel list.

**Long-term awareness:** modeling is **single program-instance**;
whole-**kernel-launch** semantics + functional correctness (does GEMM really
compute GEMM) is a long-term goal — don't design anything that blocks it.
Equivalence assumes **race-free**; a possible tie-in is the Triton Sanitizer for
cross-instance data-race detection (⚠️ unverified).
