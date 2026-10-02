# Known TTIR reports: mechanism and checker reach

This is the first pass over the four cases named in
[`ttir-modeling.md`](ttir-modeling.md), section 3.1. The source is the archived
IR and reports in `bug-report/` at repository revision
`67d0cc632982aee8c2381b2ced3ed4463488bca7`. The CPU witness below was run
locally; the original Triton compiler and GPU runs were not independently
reproduced in this checkout. A report's recorded runtime mismatch and a proved
TTIR semantic mismatch are different claims.

The checking policy for this work is **exact output bits**. A NaN payload or
signed-zero difference therefore counts as a mismatch. That policy does not by
itself establish that a Triton pass violated its documented contract.

## HIT-0007: dot accumulator placement

**Evidence.** The saved [reference](../bug-report/triton/ttir-broad/HIT-0007/reference.ttir)
and [candidate](../bug-report/triton/ttir-broad/HIT-0007/candidate.ttir)
move a loop-carried accumulator from `arith.addf(dot(A, B, 0), acc)` into the
`C` operand of `tt.dot(A, B, acc)`. The [report](../bug-report/triton/ttir-broad/HIT-0007.md)
records repeatable NaN-versus-infinity GPU results and attributes the rewrite
to `--triton-combine`. It also records that this dot lowers through an FMA chain
with `C` as its initial value. On that lowering, moving `acc` changes the
order of rounded operations.

[`eval/dot_accumulator_witness.py`](../eval/dot_accumulator_witness.py) supplies
a small independent numerical witness for the mechanism. It uses two finite
binary32 products and `acc = -Inf`: the chain starting at zero overflows to
`+Inf` and its final addition with `-Inf` yields NaN; the chain starting at
`-Inf` remains `-Inf`. The witness is not a compiler or GPU reproduction.

**Checker obligation.** Model `tt.dot` as an ordered, rounded contraction with
an explicit accumulator and precision/lowering policy. The checker must also
handle the surrounding `scf.for` and `arith.addf`. A generic uninterpreted dot
or real-number sum cannot produce a trustworthy counterexample for this
mechanism. The existing Abstract FP mode can distinguish different expression
trees, but its unconstrained values need not correspond to any binary32 input;
a SAT result from that mode alone is not evidence of a compiler wrong-code case.
A bit-exact witness needs an IEEE-aware mode and must compare output bit
patterns, including NaN and signed zero.

**Current verdict.** The archived GPU divergence is well supported by the
report, and the numeric mechanism has an independent CPU witness. This checkout
has not repeated the compiler pass or GPU execution. Whether `tt.dot` promises
the summation order needed to call this a pass violation remains an upstream
contract question. It is the strongest candidate for the section 3.3
end-to-end checker milestone.

## HIT-0071, HIT-0142, HIT-0164: one reduction family

These three reports use the same `triton_red_fused_amin_1_6afabf07` kernel.
Their culprit passes are, respectively,
`--loop-invariant-code-motion`, `--triton-licm`, and `--cse`. The archived
[reports](../bug-report/triton/r2-corpus/HIT-0071.md) record 28, 33, and 25
output words with different NaN payload bits, with no differing numeric value
or NaN classification. The [parked-case review](../bug-report/PARKED.md) says
the compiled TTGIR reduction input layouts were identical. The reports record
four identical launches per arm for each case, but use an external `env.sh` and
un-pinned compiler setup, so they have not been independently rerun here.

The saved TTIR pairs all contain one `scf.for`, one `tt.reduce`, ordered float
comparisons (`arith.cmpf olt` and `une`), and `arith.select`. None contains a
`tt.dot`, despite a generic sentence in the report summaries saying the kernel
contains a dot. The transformed bodies visibly hoist loop-invariant address
and mask expressions or merge duplicate expressions. The loop-carried minimum
selection and reduction combiner remain in the same apparent order. This is a
static reading, not an equivalence proof. The large text diffs also include
renamed SSA values and removed source locations; line counts are poor evidence
of a semantic change.

**Checker obligation.** A TTIR checker needs `scf.for`, 2-D broadcast and
expand-dims, masked load/store, signed integer division/remainder,
`arith.cmpf` with NaN behavior, `arith.ori`, `arith.select`, and a one-axis
`tt.reduce` whose combiner selects an operand. Exact payload analysis also
requires preserving concrete NaN bit patterns through selects and memory. An
IEEE theory that canonicalizes all NaNs cannot prove payload equality. If the
TTIR pairs validate as bit-equivalent under a sound model, the recorded GPU
mismatch belongs to lowering, layout execution, or another downstream effect;
a TTIR translation validator cannot catch it by comparing those TTIR pairs.

**Current verdict.** The archived runs report bit differences, which count
under this project's policy. Their cause is unresolved. Treat these as one
mechanism investigation, not three established TTIR semantic bugs. Do not use
them as section 3.3 success cases until a bit-exact TTIR check or a smaller
lowering-level probe locates the first semantic divergence.

## Next checks

1. Re-run the four archived pass commands with a pinned Triton build and verify
   that generated TTIR matches the saved candidate IR. Record the compiler SHA,
   toolchain, command, and artifacts.
2. For HIT-0007, reduce the saved pair to a small `tt.dot` plus accumulator
   case and run an IEEE-aware checker on the IR pair. Keep the exact dot
   precision and accumulation order explicit; report `UNKNOWN` if that contract
   is unavailable.
3. For the reduction family, first prove or refute bit-exact equivalence of
   each saved TTIR pair. Then apply the one-arm lane-uniformity probe proposed
   in `bug-report/PARKED.md` to locate a downstream split if TTIR agrees.
4. Repeat the original GPU runs only in a captured environment with saved
   inputs, output bits, compiled IR, and cubins. A CPU witness or IR text diff
   is not a substitute for that reproduction.
