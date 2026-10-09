# TTIR modeling — design specification

An SMT equivalence checker for Triton TTIR. Given two TTIR functions, it proves
them semantically equivalent or produces an input on which they differ. It runs
before layouts exist, so the whole layout question belongs to a separate
checker at the TTGIR level.

Section 4 is open and will be filled after review. The rest is settled.

---

## 1. User interface

One command, two files:

```
triton-tv a.ttir b.ttir
```

| Exit | Printed | Meaning |
|---|---|---|
| 0 | `EQUIVALENT` | UNSAT: the two functions agree on every input |
| 1 | `NOT EQUIVALENT` + a model | SAT: the model is a counterexample |
| 2 | `UNKNOWN` | The solver gave up, or the program uses something unmodeled |

Both files are parsed with MLIR; the first `tt.func` in each is the function
under comparison. Kernel arguments pair up by position. Pointer arguments start
with equal, symbolic memory and are compared after the walk; scalar and tensor
arguments start equal.

The binary also prints the time spent interpreting and the time spent in the
solver, since the two scale differently and the split decides where to look
when a case is slow.

**Required flags.** A solver timeout, and a switch selecting the FP axiom
profile and the loop model once section 4 fixes them.

**Internal structure is the implementer's choice.** This repository splits an
MLIR-free core (`semantics/`) from per-language builders (`builder/`) so that
one core can serve several tile languages. A TTIR checker has one language to
serve, and a working interface reached directly is worth more here than a shape
inherited from a different constraint. The contract above is what the rest of
the project depends on.

---

## 2. Milestones: supported structures

Four stages, each defined by the program structures it admits and by a corpus
that measures it. A stage passes when every kernel in its target set either
validates against itself under a semantics-preserving pass permutation or
reports `UNKNOWN` for a reason the stage explicitly excludes.

### Stage 1 — elementwise maps and single-axis reductions

The tutorial kernels: `add_kernel`, `softmax`.

Structures: program id, `tt.make_range`, `tt.splat`, `tt.broadcast`,
`tt.expand_dims`, `tt.addptr`, masked `tt.load` and `tt.store`, 1-D
`tt.reduce` over one operand, and the `arith` integer and float operations
those kernels reach. `doc/ttir.md` lists the minimal set for `add_kernel`.

This stage is where the memory model earns its keep: one symbolic byte-addressed
array per pointer argument, and an equivalence check at a symbolic witness
address.

### Stage 2 — blocked matrix multiplication

A standard tiled GEMM.

New structures: `tt.dot`, a `scf.for` loop over the contracted axis, and a
loop-carried accumulator. The loop is the harder half. Its trip count is a
compile-time constant in the kernels that matter, which makes unrolling viable
and makes the choice of loop model a cost question rather than a feasibility
one. Section 4 fixes it.

### Stage 3a — hand-written library kernels

`benchmark/benchmark_kernels.py`: 636 kernels collected from FlagGems, FLA,
TritonBench, torchao and triton_kernels. Attention, linear attention, quantized
GEMM, nested loops.

These carry the hard features. An AST scan of the corpus in 2026-08 found 100 of
the 636 modelable under the ops available then, with the blockers ranking: dtype
casts 266 kernels, `for` loops 197, `tl.where` and select 185, data-dependent
`if` 121, `maximum` 100, `minimum` 60, `abs` 49, `tt.dot` 37. The order in which
those are unlocked decides how fast coverage moves, and the scan that produced
the ranking is worth rebuilding as a committed tool, since the stage criterion
is a percentage of this corpus.

### Stage 3b — inductor-generated kernels

`benchmark/inductor_kernels.py`: kernels emitted by `torch.compile`.

Fused pointwise and reduction, template-produced, available in unlimited
quantity. They are regular where 3a is irregular, so they grade breadth rather
than depth, and they are the highest-value target for finding real bugs: this is
the most-executed Triton code there is.

3a and 3b are scored separately.

---

## 3. Milestones: bug catching

A checker that validates transformations without ever contradicting one has
proved only that it agrees with the compiler. These three steps close that gap
against bugs that are already known to exist.

### 3.1 Mechanism study

`bug-report/` holds the output of an AI-assisted differential fuzzing campaign
against Triton and TileLang: 983 reports written, 41 confirmed, 738 rejected, 20
parked for want of an owner ruling. Each confirmed report carries its IR, its
repro, and its own analysis of what an SMT model would need to see the fault.

Four of them touch TTIR. `bug-report/README.md` names one, `ttir-broad/HIT-0007`,
where `--triton-combine` turns a NaN into an Inf. The other three are found by
the `Level:` field in the reports themselves, all in the parked set:
`r2-corpus/HIT-0071`, `HIT-0142` and `HIT-0164`, with
`--loop-invariant-code-motion`, `--triton-licm` and `--cse` as their culprit
passes. Everything else in the campaign sits at TTGIR or below, which places it
beyond a TTIR checker by construction.

For each of the four, establish two things: the mechanism by which the two
compilations come apart, and whether it is a real bug that reproduces. Both are
open questions. The parked three were parked because their difference has zero
value magnitude, a NaN payload or the sign of a zero, with `max_abs_diff 0.0`
and `max_ulp 0`, and whether that counts as a defect depends on which FP axiom
profile section 4 selects. Reproduction is its own question: 13 of the 41 pin
the compiler commit, and the repro commands reference an `env.sh` that travels
outside this repository.

The output is a written mapping from bug to required structure, plus a verdict
on each bug's reality. It decides what sections 2 and 4 have to deliver, so it
comes first.

The first evidence review is in [`ttir-bug-mechanisms.md`](ttir-bug-mechanisms.md).
It separates reported GPU bit differences from TTIR semantic differences and
records which reproduction steps remain open.

### 3.2 Current reach

The checker validates semantics-preserving pass permutations today. The gates in
`eval/` stand at 4/4 on curated equivalent and non-equivalent pairs, 6/6 on
genuinely non-equivalent pairs, and 9/9 on compile-option variants;
`eval/permute_passes.py` reports all 259 one-, two- and three-pass TTIR
permutations of `add_kernel` and of `softmax` as EQUIVALENT.

That is agreement on programs whose structures are already modeled. Every gate
above is a syntactic transformation of a Stage 1 kernel.

### 3.3 Catch one

Model what 3.1 identified, and reproduce the verdict on the bug's own IR pair:
the checker must report `NOT EQUIVALENT` where the fuzzer found a bit
difference, and `EQUIVALENT` on the sibling pairs the same passes produce
without one. One bug caught end to end is the milestone.

### 3.4 (ultimate work) Fuzzing with the validator as oracle

Once the semantics of section 2 are in place, run a campaign of our own.
`eval/permute_passes.py` is the seed: it walks the one-, two- and three-pass
TTIR permutations of a kernel and puts each pair to the checker. Scaling it
means the Stage 3 corpora in place of two kernels, and the full droppable pass
set in place of three.

The oracle is what changes. The earlier campaign drew 20 random inputs per
compiled pair, so it could see a fault only on the inputs it happened to draw. A
validator answers over the whole input space at once, which brings bugs with
rare triggers into reach.

Whether the TTIR pass set still holds one is open. A campaign that finishes
clean is a legitimate result and is reported as one, and it bounds the pass set
only as far as the section 2 coverage numbers say how much of the corpus the
checker could speak about at all.

---

## 4. Key decisions

Two choices shape everything else: how a floating-point value is encoded, and
how a loop is discharged. 
"Abstract" is put at priority only because the existing previous mlir-tv work did so, 
not indicating this is best for triton-tv. But let's assume it is good and start from it

### 4.1 Floating-point encoding

`FPMode` in `semantics/Types.h` already names the four candidates. Build against
the first; keep the encoding behind that enum so a second can be added without
disturbing the operation handlers.

**Abstract, the priority.** Each FP value is an opaque `BitVec(width)` tag whose
bits carry no IEEE meaning, and each operation is an uninterpreted Z3 function
declared per type and per operation. Properties the proofs need are asserted as
axioms. Five are emitted today: the five reserved constants `+0`, `-0`, `+inf`,
`-inf`, `NaN` are pairwise distinct; `add`, `mul` and `max` are commutative; and
`neg` is involutive. Associativity, NaN propagation and the zero identity are
all withheld. The cost of an operation is one uninterpreted function
application, so a proof stays in the quantifier-free fragment and scales with
the program rather than with the arithmetic. What this buys is every fault that
moves data to the wrong place, reorders a computation, or drops one. What it
gives up is every fault that depends on what a number actually is.

Four optional encodings sit beyond it.

*FPA, the IEEE-754 theory.* Z3 models the format exactly, with rounding modes
and the real behaviour of subnormals, infinities and NaN. `Types.cpp` already
maps each float `DType` to its `fpa_sort(expBits, sigBits)` and `Memory.cpp`
already converts between the raw byte representation and that sort, so the
plumbing exists. This encoding can distinguish the NaN-versus-Inf result in
`ttir-broad/HIT-0007`. It cannot decide the three parked payload-only reports:
Z3 FPA does not retain individual NaN payloads. The solver pays for FPA:
`bin/triton-tv.cpp` carries two FPA commutativity checks that print their own
solve time, which is the cheapest available measurement of the gap.

Implementation note (2026-10-09): the core now supports FPA constants,
add/sub/mul/div, and fused multiply-add. A CPU-only regression asks Z3 to find
an accumulator-order counterexample. The `triton-tv` binary still selects
Abstract; `tt.dot` and `scf.for` are not yet interpreted. The payload-only
reports need a payload-aware bitvector encoding or an explicit `UNKNOWN`
result.

*Axiom profiles on top of Abstract.* Adding associativity as an axiom makes
reassociating transformations provably equivalent; withholding it can make the
abstract formulas differ. A different expression tree can be SAT in this
underconstrained model without a concrete floating-point witness. Two profiles
are wanted, exact bit-to-bit and reassociation-allowed, selected by the flag in
section 1. The exact-bit profile also needs a payload-aware representation of
NaNs; an associativity switch alone does not provide one. Under the chosen
exact-bit policy, payload-only differences count as mismatches once modeled.

*Real.* An FP value becomes a rational and the operations become exact
arithmetic over it. Reassociation and the algebraic identities hold by
construction, so a transformation that only rearranges arithmetic is proved in
one step. Bit-exactness goes with the rounding: two programs the solver calls
equal here can still produce different bits.

*IntegerRange.* A value is tracked as an interval rather than as a number. The
questions this answers are about magnitude: whether a sum can overflow, whether
an index stays in bounds, whether a value can reach infinity. It decides those
for every input at once and gives up on equality.

Both are directions rather than work items. Each answers a question that differs
from equivalence, and each is in the enum so that the choice stays open.

### 4.2 Loop model

A `tt.func` with an `scf.for` is where the modeling stops being local. Four
approaches are available, and which of them wins here is open. Pick one, put it
behind a `--loop-model` flag, and keep the state-threading interface narrow
enough that a second can be added later.

*Static unrolling.* Require `lb`, `ub` and `step` to be compile-time constants
and thread the state through `(ub - lb) / step` copies of the body. Exact, and
the trip counts in the Stage 2 GEMM kernels are constants, so it is available
immediately. Its cost is the product of trip count and tile width, and both
are already the measured driver of solver blowup in this checker.

*Bounded unrolling.* Unroll to a fixed budget and report `UNKNOWN` past it. This
turns a blowup into an honest verdict and gives the campaign in section 3.4 a
way to finish.

*Summarization.* Lift the loop to a combinator over `k` in `[0, N)` and reason
about the closed form. The enabling analysis is affine recurrence recognition,
which turns `ptr += stride` into `base + k * stride`, decomposed by the strongly
connected components of the carried-value dependence graph. It removes the trip
count from the cost entirely, and it pays for that with an analysis that fails
on any carried value it cannot put in closed form.

*Structural correspondence.* Both programs under comparison usually carry the
same loop, since a TTIR pass rarely changes its shape. Proving the bodies
equivalent under an arbitrary carried state, plus equal trip counts, then gives
the whole loop by induction, with no unrolling at all. This is available to a
translation validator and closed to a general verifier, and it lapses the moment
a pass does restructure the loop.

`scf.while` stays out: it appears in 2.4% of the corpus, and every occurrence
measured there is irregular.

---

## 5. TTIR operations in scope

`doc/ttir.md` is the authoritative list: every op in
`include/triton/Dialect/Triton/IR/TritonOps.td`, roughly 40, split into about 25
with a direct bitvector encoding and about 15 that need array theory, a fold
encoding, or an axiomatized combiner. It also gives the minimal subset for
`add_kernel`, which is Stage 1's floor.

**Out of scope: atomics.** `tt.atomic_rmw` and `tt.atomic_cas` report
`UNSUPPORTED` and receive no verdict. Their result depends on the order program
instances reach them, and floating-point addition is not associative, so the
value itself is undetermined and bit-exact equivalence has no meaning to check.
They block 24 kernels in the Stage 3a corpus, which is the cost of excluding
them.
