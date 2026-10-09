#include "semantics/Context.h"

#include <optional>
#include <stdexcept>

// Core (kernel-tv) — MLIR-free: use std exceptions instead of llvm_unreachable.
//
// All op-expression bodies were lifted VERBATIM from the MLIR handlers in
// builder/{mlir,triton}/{ArithOps,TritonOps}.cpp. The Z3 constant names inside
// the built lambdas (__ci/__bi/__ei/__ri/__si/__ai/__ue/__dense_base/...) are
// kept unchanged so the SMT encoding is byte-for-byte identical.

using namespace kernel_smt;

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

namespace {

// Returns true for the float DTypes.
bool isFloat(DType ty) {
  switch (ty) {
  case DType::F16:
  case DType::BF16:
  case DType::F32:
  case DType::F64:
    return true;
  default:
    return false;
  }
}

// Element DType of a Value (Scalar or Tensor). Used to pick int-vs-float.
DType elemTypeOf(const Value &v) {
  if (auto *s = std::get_if<Scalar>(&v))
    return s->ty;
  if (auto *t = std::get_if<Tensor>(&v))
    return t->elem;
  throw std::logic_error("elemTypeOf: expected Scalar or Tensor");
}

} // namespace

//===----------------------------------------------------------------------===//
// Context — construction
//===----------------------------------------------------------------------===//

Context::Context(z3::context &z, FPMode mode) : z_(z), mode_(mode), fpReg_(z) {}

//===----------------------------------------------------------------------===//
// Shared elementwise driver (was ArithOps.cpp applyBinaryOp / splatConstTile)
//===----------------------------------------------------------------------===//

// Build a Tensor from a uniform (splat) constant value. (was splatConstTile)
Tensor Context::splatConst(const Scalar &s, const Shape &shape) {
  z3::expr i = z_.bv_const("__ci", 32);
  return Tensor{z3::lambda(i, s.e), shape, s.ty, std::nullopt};
}

namespace {

// Apply a binary operation element-wise to two Values (scalar or tile).
// op_fn takes (lhs_elem, rhs_elem) and returns the result element.
// resultElem describes the result element type (may differ from input, e.g.
// cmpi returns i1 from integer inputs).
Value applyBinaryOp(z3::context &ctx, const Value &lhs, const Value &rhs,
                    DType resultElem,
                    const std::function<z3::expr(z3::expr, z3::expr)> &op_fn) {
  if (auto *l = std::get_if<Scalar>(&lhs)) {
    z3::expr res = op_fn(l->e, std::get<Scalar>(rhs).e);
    return Scalar{res, resultElem};
  }
  if (auto *l = std::get_if<Tensor>(&lhs)) {
    auto *r = &std::get<Tensor>(rhs);
    z3::expr i = ctx.bv_const("__bi", 32);
    z3::expr res = op_fn(z3::select(l->e, i), z3::select(r->e, i));
    return Tensor{z3::lambda(i, res), l->shape, resultElem, std::nullopt};
  }
  throw std::logic_error("applyBinaryOp: unexpected Value variant");
}

// Overload that preserves the input element type.
Value applyBinaryOp(z3::context &ctx, const Value &lhs, const Value &rhs,
                    const std::function<z3::expr(z3::expr, z3::expr)> &op_fn) {
  DType ty = std::holds_alternative<Scalar>(lhs) ? std::get<Scalar>(lhs).ty
                                                 : std::get<Tensor>(lhs).elem;
  return applyBinaryOp(ctx, lhs, rhs, ty, op_fn);
}

} // namespace

//===----------------------------------------------------------------------===//
// Symbolic inputs
//===----------------------------------------------------------------------===//

Scalar Context::programId(int axis) {
  // Fixed Z3 symbol names so both programs (src and tgt) share the same
  // program-ID variable — they run at the same block position.
  const char *pidName = "program_id_x";
  switch (axis) {
  case 0:
    pidName = "program_id_x";
    break;
  case 1:
    pidName = "program_id_y";
    break;
  case 2:
    pidName = "program_id_z";
    break;
  default:
    throw std::logic_error("programId: axis out of range");
  }
  return Scalar{z_.bv_const(pidName, 32), DType::I32};
}

Value Context::freshInput(DType ty, const Shape &shape,
                          const std::string &name) {
  if (!shape.empty()) {
    z3::sort elemSort = getElemSort(z_, ty, mode_);
    z3::sort arrSort = z_.array_sort(z_.bv_sort(32), elemSort);
    z3::expr arr = z_.constant(name.c_str(), arrSort);
    return Tensor{arr, shape, ty, std::nullopt};
  }
  z3::sort sort = getElemSort(z_, ty, mode_);
  z3::expr val = z_.constant(name.c_str(), sort);
  return Scalar{val, ty};
}

Ptr Context::freshPtr(DType pointee, MemId base, const std::string &name) {
  z3::expr addr = z_.bv_const(name.c_str(), 64);
  return Ptr{addr, pointee, base};
}

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

Scalar Context::constInt(int64_t v, DType ty) {
  if (ty == DType::I1)
    return Scalar{z_.bool_val(v != 0), ty};
  unsigned bw = getByteWidth(ty) * 8;
  return Scalar{z_.bv_val(v, bw), ty};
}

Scalar Context::constFloatBits(uint64_t ieeeBits, DType ty) {
  unsigned bw = getByteWidth(ty) * 8;
  z3::expr bits = z_.bv_val(ieeeBits, bw);
  if (mode_ == FPMode::FPA)
    return Scalar{bits.mk_from_ieee_bv(getElemSort(z_, ty, mode_)), ty};
  return Scalar{bits, ty};
}

Tensor Context::denseIntTile(const std::vector<int64_t> &values, DType elem,
                             const Shape &shape) {
  unsigned bw = getByteWidth(elem) * 8;
  z3::sort elemSort = getElemSort(z_, elem, mode_);
  z3::expr arr =
      z_.constant("__dense_base", z_.array_sort(z_.bv_sort(32), elemSort));
  unsigned idx = 0;
  for (int64_t v : values) {
    z3::expr e = (elem == DType::I1) ? z_.bool_val(v != 0) : z_.bv_val(v, bw);
    arr = z3::store(arr, z_.bv_val(idx, 32), e);
    ++idx;
  }
  return Tensor{arr, shape, elem, std::nullopt};
}

Tensor Context::denseFloatTile(const std::vector<uint64_t> &ieeeBits,
                               DType elem, const Shape &shape) {
  unsigned bw = getByteWidth(elem) * 8;
  z3::sort elemSort = getElemSort(z_, elem, mode_);
  z3::expr arr =
      z_.constant("__dense_fp_base", z_.array_sort(z_.bv_sort(32), elemSort));
  unsigned idx = 0;
  for (uint64_t bits : ieeeBits) {
    z3::expr value = z_.bv_val(bits, bw);
    if (mode_ == FPMode::FPA)
      value = value.mk_from_ieee_bv(elemSort);
    arr = z3::store(arr, z_.bv_val(idx, 32), value);
    ++idx;
  }
  return Tensor{arr, shape, elem, std::nullopt};
}

Tensor Context::iota(int64_t start, const Shape &shape, DType elem) {
  // range[i] = start + i (element at position i has value start + i)
  z3::expr i = z_.bv_const("__ri", 32);
  z3::expr elemExpr = z_.bv_val(start, 32) + i;
  return Tensor{z3::lambda(i, elemExpr), shape, elem, std::nullopt};
}

//===----------------------------------------------------------------------===//
// Elementwise
//===----------------------------------------------------------------------===//

Value Context::add(const Value &a, const Value &b) {
  DType elem = elemTypeOf(a);
  if (isFloat(elem)) {
    if (mode_ == FPMode::FPA)
      return applyBinaryOp(z_, a, b,
                           [](z3::expr x, z3::expr y) { return x + y; });
    if (mode_ != FPMode::Abstract)
      throw std::logic_error("add: unsupported FP mode");
    AbstractFp &afp = fpReg_.get(elem);
    return applyBinaryOp(z_, a, b,
                         [&](z3::expr x, z3::expr y) { return afp.add(x, y); });
  }
  return applyBinaryOp(z_, a, b, [](z3::expr x, z3::expr y) { return x + y; });
}

Value Context::sub(const Value &a, const Value &b) {
  DType elem = elemTypeOf(a);
  if (isFloat(elem)) {
    if (mode_ == FPMode::FPA)
      return applyBinaryOp(z_, a, b,
                           [](z3::expr x, z3::expr y) { return x - y; });
    if (mode_ != FPMode::Abstract)
      throw std::logic_error("sub: unsupported FP mode");
    AbstractFp &afp = fpReg_.get(elem);
    return applyBinaryOp(z_, a, b,
                         [&](z3::expr x, z3::expr y) { return afp.sub(x, y); });
  }
  return applyBinaryOp(z_, a, b, [](z3::expr x, z3::expr y) { return x - y; });
}

Value Context::mul(const Value &a, const Value &b) {
  DType elem = elemTypeOf(a);
  if (isFloat(elem)) {
    if (mode_ == FPMode::FPA)
      return applyBinaryOp(z_, a, b,
                           [](z3::expr x, z3::expr y) { return x * y; });
    if (mode_ != FPMode::Abstract)
      throw std::logic_error("mul: unsupported FP mode");
    AbstractFp &afp = fpReg_.get(elem);
    return applyBinaryOp(z_, a, b,
                         [&](z3::expr x, z3::expr y) { return afp.mul(x, y); });
  }
  return applyBinaryOp(z_, a, b, [](z3::expr x, z3::expr y) { return x * y; });
}

Value Context::div(const Value &a, const Value &b) {
  DType elem = elemTypeOf(a);
  if (!isFloat(elem))
    throw std::logic_error("div: only float division is modeled");
  if (mode_ == FPMode::FPA)
    return applyBinaryOp(z_, a, b,
                         [](z3::expr x, z3::expr y) { return x / y; });
  if (mode_ != FPMode::Abstract)
    throw std::logic_error("div: unsupported FP mode");
  AbstractFp &afp = fpReg_.get(elem);
  return applyBinaryOp(z_, a, b,
                       [&](z3::expr x, z3::expr y) { return afp.div(x, y); });
}

Scalar Context::fma(const Scalar &a, const Scalar &b, const Scalar &c) {
  if (mode_ != FPMode::FPA || !isFloat(a.ty) || a.ty != b.ty || a.ty != c.ty)
    throw std::logic_error("fma: requires matching float types in FPA mode");
  return Scalar{z3::fma(a.e, b.e, c.e, z_.fpa_rounding_mode()), a.ty};
}

Value Context::maxnum(const Value &a, const Value &b) {
  DType elem = elemTypeOf(a);
  if (!isFloat(elem))
    throw std::logic_error("maxnum: only float maxnum is modeled");
  if (mode_ != FPMode::Abstract)
    throw std::logic_error("maxnum: unsupported FP mode");
  AbstractFp &afp = fpReg_.get(elem);
  return applyBinaryOp(z_, a, b,
                       [&](z3::expr x, z3::expr y) { return afp.max(x, y); });
}

Value Context::andOp(const Value &a, const Value &b) {
  DType elem = elemTypeOf(a);
  // i1 -> Bool in Z3: use logical &&. Other widths: use bitwise &.
  bool isBool = (elem == DType::I1);
  return applyBinaryOp(z_, a, b, [isBool](z3::expr x, z3::expr y) -> z3::expr {
    return isBool ? (x && y) : (x & y);
  });
}

Value Context::cmpInt(IPred pred, const Value &a, const Value &b) {
  auto cmpFn = [pred](z3::expr x, z3::expr y) -> z3::expr {
    switch (pred) {
    case IPred::eq:
      return x == y;
    case IPred::ne:
      return x != y;
    case IPred::slt:
      return z3::slt(x, y);
    case IPred::sle:
      return z3::sle(x, y);
    case IPred::sgt:
      return z3::sgt(x, y);
    case IPred::sge:
      return z3::sge(x, y);
    case IPred::ult:
      return z3::ult(x, y);
    case IPred::ule:
      return z3::ule(x, y);
    case IPred::ugt:
      return z3::ugt(x, y);
    case IPred::uge:
      return z3::uge(x, y);
    }
    throw std::logic_error("cmpInt: unhandled IPred");
  };
  // Result element type is always i1 (Bool in Z3).
  return applyBinaryOp(z_, a, b, DType::I1, cmpFn);
}

Value Context::extsi(const Value &v, DType dst) {
  unsigned dstBw = getByteWidth(dst) * 8;
  if (auto *sc = std::get_if<Scalar>(&v)) {
    unsigned srcBw = sc->e.get_sort().bv_size();
    return Scalar{z3::sext(sc->e, dstBw - srcBw), dst};
  }
  if (auto *ti = std::get_if<Tensor>(&v)) {
    unsigned srcBw = ti->e.get_sort().array_range().bv_size();
    unsigned extra = dstBw - srcBw;
    z3::expr i = z_.bv_const("__ei", 32);
    z3::expr elem = z3::sext(z3::select(ti->e, i), extra);
    return Tensor{z3::lambda(i, elem), ti->shape, dst, std::nullopt};
  }
  throw std::logic_error("extsi: unexpected Value variant");
}

Value Context::exp(const Value &a) {
  DType elem = elemTypeOf(a);
  if (!isFloat(elem))
    throw std::logic_error("exp: only float exp is modeled");
  if (mode_ != FPMode::Abstract)
    throw std::logic_error("exp: unsupported FP mode");
  AbstractFp &afp = fpReg_.get(elem);
  if (auto *sc = std::get_if<Scalar>(&a))
    return Scalar{afp.exp(sc->e), elem};
  auto &t = std::get<Tensor>(a);
  z3::expr i = z_.bv_const("__ue", 32);
  z3::expr e = afp.exp(z3::select(t.e, i));
  return Tensor{z3::lambda(i, e), t.shape, elem, std::nullopt};
}

//===----------------------------------------------------------------------===//
// Structural
//===----------------------------------------------------------------------===//

Tensor Context::splat(const Scalar &s, const Shape &shape) {
  z3::expr i = z_.bv_const("__si", 32);
  return Tensor{z3::lambda(i, s.e), shape, s.ty, std::nullopt};
}

Tensor Context::splatPtr(const Ptr &p, const Shape &shape) {
  z3::expr i = z_.bv_const("__si", 32);
  return Tensor{z3::lambda(i, p.e), shape, DType::Ptr, p.base};
}

Ptr Context::addPtr(const Ptr &ptr, const Scalar &off) {
  // newPtr = ptr + sizeof(pointee) * offset.
  unsigned pteSize = getByteWidth(ptr.pointee);
  z3::expr o = off.e;
  unsigned offBw = o.get_sort().bv_size();
  z3::expr off64 = (offBw == 64) ? o : z3::sext(o, 64 - offBw);
  z3::expr addr = ptr.e + z_.bv_val((uint64_t)pteSize, 64) * off64;
  return Ptr{addr, ptr.pointee, ptr.base};
}

Tensor Context::addPtr(const Tensor &ptrs, const Tensor &off, DType pointee) {
  unsigned pteSize = getByteWidth(pointee);
  z3::expr i = z_.bv_const("__ai", 32);
  z3::expr ptr_i = z3::select(ptrs.e, i); // BV(64) address
  z3::expr off_i = z3::select(off.e, i);  // BV(32) element offset
  // Sign-extend the element offset to 64 bits and scale by pointee size.
  z3::expr off64 = z3::sext(off_i, 32); // BV(32) -> BV(64)
  z3::expr stride = z_.bv_val((uint64_t)pteSize, 64);
  z3::expr addr_i = ptr_i + stride * off64;
  return Tensor{z3::lambda(i, addr_i), ptrs.shape, ptrs.elem, ptrs.ptrBase};
}

//===----------------------------------------------------------------------===//
// Reduce
//===----------------------------------------------------------------------===//

Scalar Context::reduce(
    const Tensor &in, int axis,
    const std::function<Value(const Value &, const Value &)> &combine) {
  (void)axis; // only 1-D reductions today
  unsigned n = 1;
  for (int64_t d : in.shape)
    n *= static_cast<unsigned>(d);
  if (n < 1)
    throw std::logic_error("reduce: empty tile");

  DType elem = in.elem;
  auto elemAt = [&](unsigned i) -> Value {
    return Scalar{z3::select(in.e, z_.bv_val(i, 32)), elem};
  };

  Value acc = elemAt(0);
  for (unsigned i = 1; i < n; i++)
    acc = combine(acc, elemAt(i));

  return std::get<Scalar>(acc);
}
