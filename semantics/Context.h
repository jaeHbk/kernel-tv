#ifndef KERNEL_TV_CONTEXT_H
#define KERNEL_TV_CONTEXT_H

// kernel-tv core — the builder API.
//
// A Context is the single object a builder (per-language adapter) talks to when
// it models a program onto the core. It owns the Z3 context, the chosen FP
// encoding (AbstractFp through a registry, or selected FPA operations), and
// the FPMode. Every op-EXPRESSION body lives here as a pure Z3 method; a builder
// handler only reads its own IR operands, maps types to DType, calls the
// matching Context method, and binds the result. The core never sees MLIR.
//
// The Z3 encoding (constant names inside the built lambdas, op formulas) is
// preserved byte-for-byte from the pre-M0 handlers, so verdicts/timing do not
// change.

#include "semantics/AbstractFp.h"
#include "semantics/Types.h"
#include "semantics/Value.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include <z3++.h>

namespace kernel_smt {

class Context {
public:
  Context(z3::context &z, FPMode mode);

  // Non-copyable, non-assignable: holds a z3::context by reference and owns the
  // FP registry (which also holds the context by reference). A State driver
  // holds a Context by reference and shares it across all derived states, so
  // axiom emission stays idempotent and function names stay consistent.
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;

  z3::context &z3() { return z_; }
  FPMode mode() const { return mode_; }
  // Abstract FP registry. Builders call this only to emit accumulated axioms
  // before solver.check(); float ops route through Context methods below.
  AbstractFpRegistry &fp() { return fpReg_; }

  //--- symbolic inputs -----------------------------------------------------//

  // Fresh shared program-ID variable (BV32). The name is fixed per axis
  // (program_id_x/y/z) so both programs being compared share the same symbol.
  //   axis: 0 = x, 1 = y, 2 = z
  Scalar programId(int axis);

  // Fresh unconstrained symbolic input.
  //   empty shape -> Scalar of `ty`
  //   non-empty   -> Tensor with elem `ty` (Array(BitVec(32), elem_sort))
  Value freshInput(DType ty, const Shape &shape, const std::string &name);

  // Fresh unconstrained symbolic scalar pointer (BitVec(64)) with provenance.
  Ptr freshPtr(DType pointee, MemId base, const std::string &name);

  //--- constants -----------------------------------------------------------//

  // Scalar integer constant. i1 is a Bool; other widths are BitVec(width).
  Scalar constInt(int64_t v, DType ty);

  // Scalar float constant from IEEE bits. FPA converts the bits to a Z3 FP
  // value; Abstract keeps them as an opaque BitVec(width) tag.
  Scalar constFloatBits(uint64_t ieeeBits, DType ty);

  // Splat a scalar into a tensor of `shape`.
  Tensor splatConst(const Scalar &s, const Shape &shape);

  // Dense (non-splat) integer tensor from element values in row-major order.
  // i1 elements become Bool; other widths BitVec(width). `values` are the
  // sign-extended element values.
  Tensor denseIntTile(const std::vector<int64_t> &values, DType elem,
                      const Shape &shape);

  // Dense (non-splat) float tensor from element IEEE bit patterns (row-major).
  Tensor denseFloatTile(const std::vector<uint64_t> &ieeeBits, DType elem,
                        const Shape &shape);

  // iota tile: result[i] = start + i, as BitVec(32). (tt.make_range)
  Tensor iota(int64_t start, const Shape &shape, DType elem);

  //--- elementwise ---------------------------------------------------------//
  // Scalar or tensor; polymorphic int-vs-float on the operand DType.

  Value add(const Value &a, const Value &b);    // addi / addf
  Value sub(const Value &a, const Value &b);    // subi / subf
  Value mul(const Value &a, const Value &b);    // muli / mulf
  Value div(const Value &a, const Value &b);    // divf (float only)
  // One rounded multiply-add. Currently available only in FPA mode.
  Scalar fma(const Scalar &a, const Scalar &b, const Scalar &c);
  Value maxnum(const Value &a, const Value &b); // maxnumf (float only)
  Value exp(const Value &a);                    // math.exp (float only)

  // Logical/bitwise AND: i1 -> Bool &&, other integer widths -> bitwise &.
  Value andOp(const Value &a, const Value &b);

  // Sign-extend an integer value to `dst`.
  Value extsi(const Value &v, DType dst);

  enum class IPred { eq, ne, slt, sle, sgt, sge, ult, ule, ugt, uge };
  // Integer comparison; result element type is always I1 (Bool).
  Value cmpInt(IPred pred, const Value &a, const Value &b);

  //--- structural ----------------------------------------------------------//

  // Splat a scalar into a tensor (data tile).
  Tensor splat(const Scalar &s, const Shape &shape);

  // Splat a scalar pointer into a pointer tile, carrying its MemId provenance.
  Tensor splatPtr(const Ptr &p, const Shape &shape);

  // Scalar pointer + scalar offset: newPtr = ptr + sizeof(pointee) * offset.
  Ptr addPtr(const Ptr &ptr, const Scalar &off);

  // Pointer tile + integer offset tile: newAddr[i] = ptr[i] + stride * off[i],
  // stride = sizeof(pointee). `pointee` is supplied by the builder because a
  // pointer tile carries only DType::Ptr as its element (not the pointee).
  Tensor addPtr(const Tensor &ptrs, const Tensor &off, DType pointee);

  //--- reduce --------------------------------------------------------------//

  // 1-D reduction of a tile to a scalar by folding `combine` over the elements
  // in ascending index order. The builder supplies `combine` (e.g. turning a
  // tt.reduce combine region into [&](a,b){ return ctx.maxnum(a,b); }); the
  // core has no notion of a region. `axis` is accepted for the API but only
  // 1-D reductions are supported today.
  Scalar
  reduce(const Tensor &in, int axis,
         const std::function<Value(const Value &, const Value &)> &combine);

private:
  z3::context &z_;
  FPMode mode_;
  AbstractFpRegistry fpReg_;
};

} // namespace kernel_smt

#endif // KERNEL_TV_CONTEXT_H
