// Z3-only unit tests for the kernel-tv core Context (the builder API). NO MLIR:
// every op is driven through neutral kernel_smt types, proving Context builds and
// runs without any MLIR present. Op-expression bodies were lifted verbatim from
// the pre-M0 handlers; these tests check the resulting Z3 semantics.

#include "semantics/Context.h"
#include "semantics/Types.h"
#include "semantics/Value.h"

#include "SimpleTest.h"
#include <array>
#include <z3++.h>

using namespace kernel_smt;

//===----------------------------------------------------------------------===//
// Small helpers
//===----------------------------------------------------------------------===//

// Read element i of a tensor's Z3 array.
static z3::expr at(const Tensor &t, unsigned i) {
  return z3::select(t.e, t.e.ctx().bv_val(i, 32));
}

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

TEST(Context, ConstIntScalar) {
  z3::context z;
  Context ctx(z, FPMode::Abstract);

  Scalar s = ctx.constInt(7, DType::I32);
  EXPECT_EQ(s.ty, DType::I32);

  z3::solver solver(z);
  solver.add(s.e != z.bv_val(7, 32));
  EXPECT_EQ(solver.check(), z3::unsat);
}

TEST(Context, ConstIntBoolIsBool) {
  z3::context z;
  Context ctx(z, FPMode::Abstract);

  Scalar t = ctx.constInt(1, DType::I1);
  EXPECT_TRUE(t.e.is_bool());
  z3::solver solver(z);
  solver.add(!t.e);
  EXPECT_EQ(solver.check(), z3::unsat);
}

//===----------------------------------------------------------------------===//
// iota / splat
//===----------------------------------------------------------------------===//

TEST(Context, IotaValues) {
  z3::context z;
  Context ctx(z, FPMode::Abstract);

  Tensor r = ctx.iota(0, Shape{4}, DType::I32);
  EXPECT_EQ(r.elem, DType::I32);
  EXPECT_EQ(r.shape.size(), 1u);

  z3::solver solver(z);
  // r[3] must be 3.
  solver.add(at(r, 3) != z.bv_val(3, 32));
  EXPECT_EQ(solver.check(), z3::unsat);
}

TEST(Context, IotaWithStart) {
  z3::context z;
  Context ctx(z, FPMode::Abstract);

  Tensor r = ctx.iota(10, Shape{8}, DType::I32);
  z3::solver solver(z);
  solver.add(at(r, 5) != z.bv_val(15, 32));
  EXPECT_EQ(solver.check(), z3::unsat);
}

TEST(Context, SplatScalar) {
  z3::context z;
  Context ctx(z, FPMode::Abstract);

  Scalar s = ctx.constInt(99, DType::I32);
  Tensor t = ctx.splat(s, Shape{16});
  z3::solver solver(z);
  solver.add(at(t, 0) != z.bv_val(99, 32) || at(t, 10) != z.bv_val(99, 32));
  EXPECT_EQ(solver.check(), z3::unsat);
}

//===----------------------------------------------------------------------===//
// Integer elementwise
//===----------------------------------------------------------------------===//

TEST(Context, AddIntScalar) {
  z3::context z;
  Context ctx(z, FPMode::Abstract);

  Scalar a = ctx.constInt(3, DType::I32);
  Scalar b = ctx.constInt(4, DType::I32);
  Value r = ctx.add(a, b);
  auto &rs = std::get<Scalar>(r);

  z3::solver solver(z);
  solver.add(rs.e != z.bv_val(7, 32));
  EXPECT_EQ(solver.check(), z3::unsat);
}

TEST(Context, AddIntTensorElementwise) {
  z3::context z;
  Context ctx(z, FPMode::Abstract);

  Tensor x = ctx.iota(0, Shape{4}, DType::I32);  // 0,1,2,3
  Tensor y = ctx.iota(10, Shape{4}, DType::I32); // 10,11,12,13
  Value r = ctx.add(x, y);
  auto &rt = std::get<Tensor>(r);

  z3::solver solver(z);
  solver.add(at(rt, 2) != z.bv_val(14, 32)); // 2 + 12
  EXPECT_EQ(solver.check(), z3::unsat);
}

TEST(Context, MulAndSubInt) {
  z3::context z;
  Context ctx(z, FPMode::Abstract);

  Scalar a = ctx.constInt(6, DType::I32);
  Scalar b = ctx.constInt(4, DType::I32);
  auto m = std::get<Scalar>(ctx.mul(a, b));
  auto s = std::get<Scalar>(ctx.sub(a, b));

  z3::solver solver(z);
  solver.add(m.e != z.bv_val(24, 32) || s.e != z.bv_val(2, 32));
  EXPECT_EQ(solver.check(), z3::unsat);
}

TEST(Context, AndBoolVsBitwise) {
  z3::context z;
  Context ctx(z, FPMode::Abstract);

  // i1 -> logical &&
  Scalar t = ctx.constInt(1, DType::I1);
  Scalar f = ctx.constInt(0, DType::I1);
  auto r1 = std::get<Scalar>(ctx.andOp(t, f));
  EXPECT_TRUE(r1.e.is_bool());

  // i32 -> bitwise &
  Scalar a = ctx.constInt(0b1100, DType::I32);
  Scalar b = ctx.constInt(0b1010, DType::I32);
  auto r2 = std::get<Scalar>(ctx.andOp(a, b));

  z3::solver solver(z);
  solver.add(r1.e); // false && x is false
  EXPECT_EQ(solver.check(), z3::unsat);
  solver.reset();
  solver.add(r2.e != z.bv_val(0b1000, 32));
  EXPECT_EQ(solver.check(), z3::unsat);
}

TEST(Context, ExtsiSignExtends) {
  z3::context z;
  Context ctx(z, FPMode::Abstract);

  // -1 as i32 -> i64 stays -1 (sign extend).
  Scalar a = ctx.constInt(-1, DType::I32);
  auto r = std::get<Scalar>(ctx.extsi(a, DType::I64));
  EXPECT_EQ(r.ty, DType::I64);

  z3::solver solver(z);
  solver.add(r.e != z.bv_val((int64_t)-1, 64));
  EXPECT_EQ(solver.check(), z3::unsat);
}

TEST(Context, CmpIntResultIsBool) {
  z3::context z;
  Context ctx(z, FPMode::Abstract);

  Scalar a = ctx.constInt(3, DType::I32);
  Scalar b = ctx.constInt(5, DType::I32);
  auto lt = std::get<Scalar>(ctx.cmpInt(Context::IPred::slt, a, b));
  EXPECT_EQ(lt.ty, DType::I1);
  EXPECT_TRUE(lt.e.is_bool());

  z3::solver solver(z);
  solver.add(!lt.e); // 3 < 5 is true
  EXPECT_EQ(solver.check(), z3::unsat);
}

//===----------------------------------------------------------------------===//
// Float elementwise (Abstract mode: uninterpreted, but commutativity axiom)
//===----------------------------------------------------------------------===//

TEST(Context, AddFloatCommutativeUnderAxioms) {
  z3::context z;
  Context ctx(z, FPMode::Abstract);

  // Two symbolic f32 scalars carried as BV32 ids.
  z3::expr xa = z.bv_const("xa", 32);
  z3::expr xb = z.bv_const("xb", 32);
  Scalar a{xa, DType::F32};
  Scalar b{xb, DType::F32};

  auto ab = std::get<Scalar>(ctx.add(a, b));
  auto ba = std::get<Scalar>(ctx.add(b, a));

  z3::solver solver(z);
  ctx.fp().addAxioms(solver); // add-commutativity is emitted here
  solver.add(ab.e != ba.e);
  EXPECT_EQ(solver.check(), z3::unsat)
      << "add on floats must be commutative under the emitted axiom";
}

TEST(Context, SubFloatNotCommutative) {
  z3::context z;
  Context ctx(z, FPMode::Abstract);

  z3::expr xa = z.bv_const("xa", 32);
  z3::expr xb = z.bv_const("xb", 32);
  Scalar a{xa, DType::F32};
  Scalar b{xb, DType::F32};

  auto ab = std::get<Scalar>(ctx.sub(a, b));
  auto ba = std::get<Scalar>(ctx.sub(b, a));

  z3::solver solver(z);
  ctx.fp().addAxioms(solver);
  solver.add(ab.e != ba.e); // sub is uninterpreted, no commutativity axiom
  EXPECT_EQ(solver.check(), z3::sat)
      << "sub on floats has no commutativity axiom, so a-b may differ from b-a";
}

//===----------------------------------------------------------------------===//
// addPtr
//===----------------------------------------------------------------------===//

TEST(Context, AddPtrScalarScalesByPointee) {
  z3::context z;
  Context ctx(z, FPMode::Abstract);

  // base pointer to f32 (4 bytes); offset 3 -> base + 12.
  Ptr base = ctx.freshPtr(DType::F32, MemId{0}, "p");
  Scalar off = ctx.constInt(3, DType::I32);
  Ptr r = ctx.addPtr(base, off);
  EXPECT_EQ(r.pointee, DType::F32);
  EXPECT_EQ(r.base, MemId{0});

  z3::solver solver(z);
  solver.add(r.e != base.e + z.bv_val(12, 64));
  EXPECT_EQ(solver.check(), z3::unsat);
}

TEST(Context, AddPtrTileScalesByPointeeAndKeepsProvenance) {
  z3::context z;
  Context ctx(z, FPMode::Abstract);

  // ptr tile all equal to a base address; offset tile = iota.
  Ptr base = ctx.freshPtr(DType::I32, MemId{7}, "p");
  Tensor ptrs = ctx.splatPtr(base, Shape{4});
  Tensor off = ctx.iota(0, Shape{4}, DType::I32); // 0,1,2,3
  Tensor r = ctx.addPtr(ptrs, off, DType::I32);   // i32 -> 4-byte stride
  EXPECT_TRUE(r.ptrBase.has_value());
  EXPECT_EQ(*r.ptrBase, MemId{7});

  z3::solver solver(z);
  // r[2] = base + 2*4 = base + 8
  solver.add(at(r, 2) != base.e + z.bv_val(8, 64));
  EXPECT_EQ(solver.check(), z3::unsat);
}

//===----------------------------------------------------------------------===//
// reduce (combine callback)
//===----------------------------------------------------------------------===//

TEST(Context, ReduceSumCallback) {
  z3::context z;
  Context ctx(z, FPMode::Abstract);

  Tensor x = ctx.iota(1, Shape{4}, DType::I32); // 1,2,3,4
  // Fold with integer add via the Context.
  Scalar r = ctx.reduce(
      x, 0, [&](const Value &a, const Value &b) { return ctx.add(a, b); });
  EXPECT_EQ(r.ty, DType::I32);

  z3::solver solver(z);
  solver.add(r.e != z.bv_val(1 + 2 + 3 + 4, 32));
  EXPECT_EQ(solver.check(), z3::unsat);
}

TEST(Context, ReduceSingleElement) {
  z3::context z;
  Context ctx(z, FPMode::Abstract);

  Tensor x = ctx.iota(42, Shape{1}, DType::I32); // just [42]
  Scalar r = ctx.reduce(
      x, 0, [&](const Value &a, const Value &b) { return ctx.add(a, b); });
  z3::solver solver(z);
  solver.add(r.e != z.bv_val(42, 32));
  EXPECT_EQ(solver.check(), z3::unsat);
}

//===----------------------------------------------------------------------===//
// programId — shared symbol across two Contexts on the same z3::context
//===----------------------------------------------------------------------===//

TEST(Context, ProgramIdSharedSymbol) {
  z3::context z;
  Context src(z, FPMode::Abstract);
  Context tgt(z, FPMode::Abstract);

  Scalar px = src.programId(0);
  Scalar py = tgt.programId(0);
  EXPECT_EQ(px.ty, DType::I32);

  // Same name -> same Z3 constant -> always equal.
  z3::solver solver(z);
  solver.add(px.e != py.e);
  EXPECT_EQ(solver.check(), z3::unsat);
}

//===----------------------------------------------------------------------===//
// IEEE FP operations needed for the dot accumulator-order witness
//===----------------------------------------------------------------------===//

TEST(Context, FpaConstantsPreserveInfinityAndSignedZero) {
  z3::context z;
  Context ctx(z, FPMode::FPA);
  Scalar inf = ctx.constFloatBits(0x7f800000u, DType::F32);
  Scalar negZero = ctx.constFloatBits(0x80000000u, DType::F32);
  EXPECT_TRUE(inf.e.is_fpa());

  z3::solver solver(z);
  solver.add(!inf.e.mk_is_inf() ||
             negZero.e.mk_to_ieee_bv() != z.bv_val(0x80000000u, 32));
  EXPECT_EQ(solver.check(), z3::unsat);
}

TEST(Context, FpaDotAccumulatorOrderChangesNanClass) {
  z3::context z;
  Context ctx(z, FPMode::FPA);
  Scalar product = ctx.constFloatBits(0x7f400000u, DType::F32);
  Scalar one = ctx.constFloatBits(0x3f800000u, DType::F32);
  Scalar zero = ctx.constFloatBits(0, DType::F32);
  Scalar negInf = ctx.constFloatBits(0xff800000u, DType::F32);

  Scalar dot = ctx.fma(product, one, ctx.fma(product, one, zero));
  Scalar reference = std::get<Scalar>(ctx.add(dot, negInf));
  Scalar candidate = ctx.fma(product, one,
                             ctx.fma(product, one, negInf));

  z3::solver solver(z);
  solver.add(!reference.e.mk_is_nan() ||
             !candidate.e.mk_is_inf() ||
             reference.e.mk_to_ieee_bv() == candidate.e.mk_to_ieee_bv());
  EXPECT_EQ(solver.check(), z3::unsat);
}

TEST(Context, FpaFindsAccumulatorOrderCounterexample) {
  z3::context z;
  Context ctx(z, FPMode::FPA);
  Scalar product = std::get<Scalar>(ctx.freshInput(DType::F32, {}, "product"));
  Scalar one = ctx.constFloatBits(0x3f800000u, DType::F32);
  Scalar zero = ctx.constFloatBits(0, DType::F32);
  Scalar negInf = ctx.constFloatBits(0xff800000u, DType::F32);

  Scalar dot = ctx.fma(product, one, ctx.fma(product, one, zero));
  Scalar reference = std::get<Scalar>(ctx.add(dot, negInf));
  Scalar candidate = ctx.fma(product, one,
                             ctx.fma(product, one, negInf));

  z3::solver solver(z);
  solver.add(!product.e.mk_is_nan() && !product.e.mk_is_inf());
  solver.add(reference.e.mk_is_nan() && candidate.e.mk_is_inf());
  EXPECT_EQ(solver.check(), z3::sat);
}

int main() { return simpletest::runAll(); }
