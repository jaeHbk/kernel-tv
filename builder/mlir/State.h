#ifndef TV_BUILDER_MLIR_STATE_H
#define TV_BUILDER_MLIR_STATE_H

#include "builder/mlir/Env.h"
#include "semantics/Context.h"
#include "semantics/Memory.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"

#include <map>
#include <z3++.h>

namespace Semantics {

using kernel_smt::Context;
using kernel_smt::Memory;
using kernel_smt::MemState;

// Complete symbolic program state at a point during execution.
//
// Memory model: one independent symbolic Memory per kernel pointer argument
// (Option B — separate-arrays model). This is sound because Triton kernels
// are required to pass non-aliasing pointer arguments; the compiler freely
// reorders accesses across distinct arguments. Each argument's Memory is an
// Array(BitVec(64), BitVec(8)) whose index is the absolute byte address.
//
// Per-argument memories live in `memState`, keyed by an opaque kernel_smt::MemId.
// The adapter maps each pointer-argument mlir::Value to its MemId via
// `ptrArgToMem`; the core Memory / MemState never see mlir::Value.
//
// Non-pointer arguments (scalars, tensors passed by value) live only in Env.
//
// The State is a builder-side DRIVER: it holds a reference to the core Context
// (which owns the z3::context, the FP model, and the FPMode) plus the
// MLIR-keyed Env / ptrArgToMem, and walks a tt.func calling the core builder
// API. All derived states share the same Context by reference, so FP function
// names and axiom emission stay consistent.
//
// All interpret* methods are pure: they return an updated State and do not
// mutate *this. State is copy-constructible and move-constructible, but NOT
// copy/move-assignable (it holds references — the Context and z3::context).
class State {
public:
  Env env;

  // One Memory per !tt.ptr<T> kernel argument, keyed by MemId.
  MemState memState;

  // Maps each pointer-argument mlir::Value to the MemId of its Memory.
  // The adapter owns this map (core never sees mlir::Value); handlers use it
  // only at function entry (initFromFunc) and equivalence pairing.
  std::map<mlir::Value, MemId, ValuePtrLess> ptrArgToMem;

  // Future: Memory sharedMem;  // for TTGIR ttg.local_alloc / ttg.local_store

  // The core builder for this interpretation run. Owns the z3::context, the FP
  // model (AbstractFpRegistry), and the FPMode. Shared by reference across all
  // derived states.
  Context &context;

  // Convenience references into `context` so handlers can keep using s.ctx /
  // s.fpMode. They alias context.z3() / context.mode().
  z3::context &ctx;
  FPMode fpMode;

  State(Env env, MemState memState,
        std::map<mlir::Value, MemId, ValuePtrLess> ptrArgToMem,
        Context &context);

  // Factory: build an initial State from a function's argument list.
  //
  // For each argument:
  //   - !tt.ptr<T>  → fresh Ptr in env (with base MemId set) +
  //                   fresh Memory in memState named "<prefix>_mem_argN"
  //   - tensor<...> → fresh Tensor in env
  //   - scalar      → fresh Scalar in env
  //
  // Call this once for each of the two programs being compared, using
  // distinct prefixes ("src", "tgt") so their Z3 symbols are distinguishable.
  // MemIds are minted in argument order, so the two programs' pointer arguments
  // pair up by position (src arg i ↔ tgt arg i get the same MemId).
  static State initFromFunc(mlir::ValueRange args, Context &context,
                            const std::string &prefix = "arg");

  // Interpret a single MLIR operation and return the updated state.
  // Dispatches on dialect/op name. Unimplemented ops call llvm_unreachable.
  State interpretOp(mlir::Operation *op) const;

  // Interpret all ops in `block` sequentially.
  State interpretBlock(mlir::Block &block) const;

private:
  // Control flow handlers. Only bounded, static scf.for is implemented.

  // scf.if: execute both branches from *this; merge with z3::ite on condition.
  State interpretIf(mlir::Operation *op) const;

  // scf.for — static unrolling, including loop-carried values.
  State interpretFor(mlir::Operation *op) const;

  // scf.while — deferred.
  State interpretWhile(mlir::Operation *op) const;
};

// Final-memory equivalence lives in the MLIR-free core
// (semantics/Equivalence.h). A builder pairs the two programs' memories by
// MemId (e.g. positionally by kernel argument) and calls
// kernel_smt::checkEquivalence with the resulting pairing + solver.

} // namespace Semantics

#endif // TV_BUILDER_MLIR_STATE_H
