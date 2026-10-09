// Z3-only unit tests for the kernel-tv core Memory (byte-addressable heap +
// masked windowed load/store). NO MLIR: element types are the neutral
// kernel_smt::DType, proving Memory builds/runs without any MLIR present.
//
// Ported from the pre-M0 MLIR-based MemoryModelTest.cpp. getElemSort /
// getByteWidth / fpExpSigBits are covered by TypesTest, so this file focuses on
// Memory behavior (constructor, masked load/store, roundtrip, equivalence, the
// Abstract-FP byte roundtrip, and the large-tile witness comparison).

#include "semantics/Memory.h"
#include "semantics/Value.h"

#include "SimpleTest.h"
#include <z3++.h>

using namespace kernel_smt;

//===----------------------------------------------------------------------===//
// Test helpers
//===----------------------------------------------------------------------===//

// Pointer tile: ptr[i] = baseAddr + i * stride  (Array(BV32, BV64))
static Tensor makePtrTile(z3::context &ctx, DType elem, uint64_t baseAddr,
                          uint64_t stride, Shape shape) {
  z3::expr i    = ctx.bv_const("i", 32);
  z3::expr base = ctx.bv_val((uint64_t)baseAddr, 64);
  z3::expr st   = ctx.bv_val((uint64_t)stride, 64);
  return Tensor{z3::lambda(i, base + st * z3::zext(i, 32)), shape, elem,
                std::nullopt};
}

// Mask tile: mask[i] = val for all i  (Array(BV32, Bool))
static Tensor makeMaskTile(z3::context &ctx, bool val, Shape shape) {
  z3::expr i = ctx.bv_const("i", 32);
  return Tensor{z3::lambda(i, ctx.bool_val(val)), shape, DType::I1,
                std::nullopt};
}

// Value tile: val[i] = constant  (Array(BV32, BV32))
static Tensor makeConstTile(z3::context &ctx, DType elem, uint32_t val,
                            Shape shape) {
  z3::expr i = ctx.bv_const("i", 32);
  return Tensor{z3::lambda(i, ctx.bv_val(val, 32)), shape, elem, std::nullopt};
}

// Value tile: val[i] = i  (identity)
static Tensor makeIdentityTile(z3::context &ctx, DType elem, Shape shape) {
  z3::expr i = ctx.bv_const("i", 32);
  return Tensor{z3::lambda(i, i), shape, elem, std::nullopt};
}

//===----------------------------------------------------------------------===//
// Memory constructor
//===----------------------------------------------------------------------===//

TEST(Memory, ConstructorArraySort) {
  z3::context ctx;
  Memory m(ctx, FPMode::IntegerRange, "test_mem");

  auto sort = m.array.get_sort();
  EXPECT_EQ(sort.sort_kind(),        Z3_ARRAY_SORT);
  EXPECT_EQ(sort.array_domain().bv_size(), 64u);
  EXPECT_EQ(sort.array_range().bv_size(),   8u);
}

TEST(Memory, TwoMemoriesAreDistinct) {
  // Two distinct symbolic memories must be distinguishable (SAT for ≠).
  z3::context ctx;
  Memory m1(ctx, FPMode::IntegerRange, "mem_a");
  Memory m2(ctx, FPMode::IntegerRange, "mem_b");

  z3::solver solver(ctx);
  solver.add(m1.array != m2.array);
  EXPECT_EQ(solver.check(), z3::sat);
}

//===----------------------------------------------------------------------===//
// load
//===----------------------------------------------------------------------===//

TEST(Memory, MaskedLoadAllFalseReturnsOther) {
  // mask[i] = false for all i → loaded[i] must equal other[i] = 99
  z3::context ctx;
  Memory m(ctx, FPMode::IntegerRange, "mem_false_mask");

  auto ptrTile   = makePtrTile(ctx, DType::I32, 0x1000, 4, {4});
  auto falseMask = makeMaskTile(ctx, false, {4});
  auto other99   = makeConstTile(ctx, DType::I32, 99, {4});

  Tensor loaded = m.load(ptrTile, falseMask, other99);

  z3::solver solver(ctx);
  z3::expr idx = ctx.bv_const("idx", 32);
  // loaded[idx] != 99 should be UNSAT
  solver.add(z3::select(loaded.e, idx) != ctx.bv_val(99u, 32));
  EXPECT_EQ(solver.check(), z3::unsat);
}

//===----------------------------------------------------------------------===//
// store + load roundtrip
//===----------------------------------------------------------------------===//

TEST(Memory, StoreLoadRoundtrip) {
  // Store val[i]=i at ptr[i]=0x2000+i*4 with all-true mask, then load back.
  // Loaded values must equal stored values (UNSAT for ≠).
  z3::context ctx;
  Memory m(ctx, FPMode::IntegerRange, "mem_roundtrip");

  auto ptrTile   = makePtrTile(ctx, DType::I32, 0x2000, 4, {4});
  auto valTile   = makeIdentityTile(ctx, DType::I32, {4}); // val[i] = i
  auto trueMask  = makeMaskTile(ctx, true, {4});
  auto zeroOther = makeConstTile(ctx, DType::I32, 0, {4});

  m.store(ptrTile, valTile, trueMask);
  Tensor loaded = m.load(ptrTile, trueMask, zeroOther);

  z3::solver solver(ctx);
  z3::expr idx = ctx.bv_const("idx", 32);
  // Constrain idx to the tile range [0, 3] — addresses outside this range
  // were never written, so the loaded value is unconstrained (from the
  // initial symbolic memory) and the query would be trivially SAT.
  solver.add(z3::ult(idx, ctx.bv_val(4u, 32)));
  // loaded[idx] != val[idx] = idx  should be UNSAT within bounds
  solver.add(z3::select(loaded.e, idx) != idx);
  EXPECT_EQ(solver.check(), z3::unsat);
}

TEST(Memory, MaskedStorePreservesUnmasked) {
  // Step 1: store 0 everywhere (all-true mask).
  // Step 2: store 42 only at index 0 (partial mask).
  // Result: loaded[0] == 42, loaded[1] == 0.
  z3::context ctx;
  Memory m(ctx, FPMode::IntegerRange, "mem_masked");

  auto ptrTile  = makePtrTile(ctx, DType::I32, 0x3000, 4, {2});
  auto trueMask = makeMaskTile(ctx, true, {2});
  auto zeroTile = makeConstTile(ctx, DType::I32, 0, {2});

  // Step 1: fill with zeros
  m.store(ptrTile, zeroTile, trueMask);

  // Step 2: partial mask — only index 0
  z3::expr i      = ctx.bv_const("i", 32);
  Tensor partMask{z3::lambda(i, i == ctx.bv_val(0u, 32)), {2}, DType::I1,
                  std::nullopt};
  Tensor val42    = makeConstTile(ctx, DType::I32, 42, {2});
  m.store(ptrTile, val42, partMask);

  Tensor loaded = m.load(ptrTile, trueMask, zeroTile);

  z3::solver solver(ctx);

  // loaded[0] must be 42
  z3::expr l0 = z3::select(loaded.e, ctx.bv_val(0u, 32));
  solver.add(l0 != ctx.bv_val(42u, 32));
  EXPECT_EQ(solver.check(), z3::unsat) << "loaded[0] should be 42";
  solver.reset();

  // loaded[1] must be 0
  z3::expr l1 = z3::select(loaded.e, ctx.bv_val(1u, 32));
  solver.add(l1 != ctx.bv_val(0u, 32));
  EXPECT_EQ(solver.check(), z3::unsat) << "loaded[1] should be 0";
}

//===----------------------------------------------------------------------===//
// Equivalence checking (the validator's top-level query)
//===----------------------------------------------------------------------===//

TEST(Memory, EquivalentProgramsUNSAT) {
  // Two programs starting from the same symbolic memory and making identical
  // stores produce equivalent final states — UNSAT for (m1 ≠ m2) at any addr.
  z3::context ctx;

  // Shared initial memory — both programs start here (copy-construct).
  Memory init(ctx, FPMode::IntegerRange, "mem_init");
  Memory m1 = init;
  Memory m2 = init;

  auto ptrTile  = makePtrTile(ctx, DType::I32, 0x4000, 4, {4});
  auto valTile  = makeConstTile(ctx, DType::I32, 7, {4});
  auto trueMask = makeMaskTile(ctx, true, {4});

  m1.store(ptrTile, valTile, trueMask);
  m2.store(ptrTile, valTile, trueMask); // identical

  // Assert the written bytes differ at address 0x4000 (first element, byte 0).
  z3::solver solver(ctx);
  z3::expr addr = ctx.bv_val((uint64_t)0x4000, 64);
  solver.add(z3::select(m1.array, addr) != z3::select(m2.array, addr));
  EXPECT_EQ(solver.check(), z3::unsat);
}

TEST(Memory, NonEquivalentProgramsSAT) {
  // Program 1 stores 42, program 2 stores 43 at the same address.
  // The solver must find a counterexample — SAT.
  z3::context ctx;

  Memory init(ctx, FPMode::IntegerRange, "mem_init2");
  Memory m1 = init;
  Memory m2 = init;

  auto ptrTile  = makePtrTile(ctx, DType::I32, 0x5000, 4, {1});
  auto trueMask = makeMaskTile(ctx, true, {1});

  m1.store(ptrTile, makeConstTile(ctx, DType::I32, 42, {1}), trueMask);
  m2.store(ptrTile, makeConstTile(ctx, DType::I32, 43, {1}), trueMask);

  z3::solver solver(ctx);
  z3::expr addr = ctx.bv_val((uint64_t)0x5000, 64);
  solver.add(z3::select(m1.array, addr) != z3::select(m2.array, addr));
  EXPECT_EQ(solver.check(), z3::sat);
}

//===----------------------------------------------------------------------===//
// Abstract FP: byte-heap roundtrip
//
// Under FPMode::Abstract, FP values are BitVec(width) ids. They should
// store and load through the byte heap just like integers do.
//===----------------------------------------------------------------------===//

TEST(Memory, AbstractFpStoreLoadRoundtrip) {
  z3::context ctx;

  // Store an arbitrary symbolic FP id, then load it back; it must match.
  Memory init(ctx, FPMode::Abstract, "mem_fp_abs");
  Memory m = init;

  uint64_t baseAddr = 0x9000;
  auto ptrTile  = makePtrTile(ctx, DType::F32, baseAddr, 4, {1});
  auto trueMask = makeMaskTile(ctx, true, {1});

  // Build a value tile whose single element is a fresh symbolic BV(32).
  z3::expr sym  = ctx.bv_const("fp_sym", 32);
  z3::expr i    = ctx.bv_const("i_v", 32);
  Tensor valTile{z3::lambda(i, sym), {1}, DType::F32, std::nullopt};

  m.store(ptrTile, valTile, trueMask);

  // Load and compare against the original symbolic value.
  z3::expr idx0    = ctx.bv_val(0u, 32);
  z3::expr loaded0 = z3::select(
      m.load(ptrTile, trueMask, valTile).e, idx0);

  z3::solver solver(ctx);
  solver.add(loaded0 != sym);
  EXPECT_EQ(solver.check(), z3::unsat);
}

//===----------------------------------------------------------------------===//
// MemState — keyed by MemId, distinct memories are distinguishable.
//===----------------------------------------------------------------------===//

TEST(Memory, MemStateKeyedByMemId) {
  z3::context ctx;
  MemState ms;
  ms.mems.emplace(std::piecewise_construct,
                  std::forward_as_tuple(static_cast<MemId>(0)),
                  std::forward_as_tuple(ctx, FPMode::IntegerRange, "mem0"));
  ms.mems.emplace(std::piecewise_construct,
                  std::forward_as_tuple(static_cast<MemId>(1)),
                  std::forward_as_tuple(ctx, FPMode::IntegerRange, "mem1"));

  EXPECT_EQ(ms.mems.size(), 2u);
  z3::solver solver(ctx);
  solver.add(ms.mems.at(static_cast<MemId>(0)).array !=
             ms.mems.at(static_cast<MemId>(1)).array);
  EXPECT_EQ(solver.check(), z3::sat);
}

//===----------------------------------------------------------------------===//
// Large-tile equivalence via the lambda store + pointwise witness comparison.
// Scaled stand-in for the 1024-element add_kernel that previously blew up:
// identical big stores must agree at a SYMBOLIC witness address (UNSAT),
// differing big stores must be detected (SAT). Exercises the lambda-based
// store at a non-trivial tile size without array extensionality.
//===----------------------------------------------------------------------===//

TEST(Memory, LargeTileWitnessEquivalence) {
  z3::context ctx;

  const int N = 256;
  Memory init(ctx, FPMode::IntegerRange, "mem_big");

  auto ptrTile  = makePtrTile(ctx, DType::I32, 0x10000, 4, {N});
  auto trueMask = makeMaskTile(ctx, true, {N});

  // Two memories given the SAME store must agree everywhere.
  Memory m1 = init;
  Memory m2 = init;
  m1.store(ptrTile, makeConstTile(ctx, DType::I32, 7, {N}), trueMask);
  m2.store(ptrTile, makeConstTile(ctx, DType::I32, 7, {N}), trueMask);

  z3::expr w = ctx.bv_const("__w", 64);
  z3::solver solver(ctx);
  solver.add(z3::select(m1.array, w) != z3::select(m2.array, w));
  EXPECT_EQ(solver.check(), z3::unsat) << "identical big stores must agree";

  // A memory storing a different value must be detected.
  Memory m3 = init;
  m3.store(ptrTile, makeConstTile(ctx, DType::I32, 8, {N}), trueMask);
  solver.reset();
  solver.add(z3::select(m1.array, w) != z3::select(m3.array, w));
  EXPECT_EQ(solver.check(), z3::sat) << "differing big stores must be detected";
}

TEST(Memory, FpaSignedZeroByteRoundtrip) {
  z3::context ctx;
  Memory m(ctx, FPMode::FPA, "mem_fpa_zero");
  auto ptrs = makePtrTile(ctx, DType::Ptr, 0x5000, 4, {1});
  auto mask = makeMaskTile(ctx, true, {1});
  z3::expr i = ctx.bv_const("fpa_i", 32);
  z3::expr bits = ctx.bv_val(0x80000000u, 32);
  z3::expr negZero = bits.mk_from_ieee_bv(ctx.fpa_sort(8, 24));
  Tensor vals{z3::lambda(i, negZero), {1}, DType::F32, std::nullopt};

  m.store(ptrs, vals, mask);
  Tensor loaded = m.load(ptrs, mask, vals);
  z3::solver solver(ctx);
  solver.add(z3::select(loaded.e, ctx.bv_val(0, 32)).mk_to_ieee_bv() !=
             bits);
  EXPECT_EQ(solver.check(), z3::unsat);
}

int main() { return simpletest::runAll(); }
