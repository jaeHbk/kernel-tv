#include "semantics/Memory.h"

#include <stdexcept>

// Core must stay free of MLIR/LLVM: switch on the neutral DType and use std
// exceptions instead of llvm_unreachable.

using namespace kernel_smt;

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

// Returns true for the float DTypes.
static bool isFloat(DType ty) {
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

// Convert a Z3 value to a bitvector for writing into the byte-addressable heap.
//   Abstract / IntegerRange FP -> pass through (already BitVec)
//   FPA FP                     -> ieee bitvector reinterpretation
//   integer (i>=8)             -> pass through
//   i1                         -> BitVec(1)
//   pointer                    -> pass through (already BitVec(64))
static z3::expr toBV(z3::context &ctx, z3::expr val, DType ty, FPMode mode) {
  if (isFloat(ty)) {
    if (mode == FPMode::FPA) {
      // Z3's FP sort does not retain the payload of a NaN. This conversion is
      // suitable for class-level witnesses (NaN vs Inf), not a proof that two
      // programs write identical NaN payload bits.
      return z3::expr(ctx, Z3_mk_fpa_to_ieee_bv(ctx, val));
    }
    return val; // Abstract / IntegerRange: already BitVec(width)
  }
  if (ty == DType::I1)
    return z3::ite(val, ctx.bv_val(1, 1), ctx.bv_val(0, 1));
  return val; // other integers / pointers: already BitVec
}

//===----------------------------------------------------------------------===//
// Memory
//===----------------------------------------------------------------------===//

Memory::Memory(z3::context &ctx, FPMode fpMode, const std::string &name)
    : ctx(ctx), fpMode(fpMode),
      array(ctx.constant(name.c_str(),
            ctx.array_sort(ctx.bv_sort(64), ctx.bv_sort(8)))) {}

z3::expr Memory::readBytes(z3::expr addr, unsigned byteWidth) const {
  // Little-endian: addr+0 → LSB, addr+(byteWidth-1) → MSB.
  // concat(a, b) in Z3 puts `a` as the more-significant bits.
  z3::expr result = z3::select(array, addr);
  for (unsigned i = 1; i < byteWidth; i++) {
    z3::expr byte_i = z3::select(array, addr + ctx.bv_val(i, 64));
    result = z3::concat(byte_i, result);
  }
  return result; // BitVec(byteWidth * 8)
}

Tensor Memory::load(const Tensor &ptrTile,
                    const Tensor &maskTile,
                    const Tensor &otherTile) const {
  // Real mode cannot ride the byte-addressable heap (no fixed bit pattern).
  // Abstract mode uses a BV(width) carrier, so it falls through like an int.
  if (fpMode == FPMode::Real)
    throw std::logic_error("load: Real FP mode not yet supported");

  unsigned byteWidth = getByteWidth(otherTile.elem);

  // Result tile: λi:BitVec(32). mask[i] ? loadElem(ptr[i]) : other[i]
  z3::expr i       = ctx.bv_const("__load_i", 32);
  z3::expr ptr_i   = z3::select(ptrTile.e,   i);
  z3::expr mask_i  = z3::select(maskTile.e,  i);
  z3::expr other_i = z3::select(otherTile.e, i);

  // Load raw bytes from heap, then convert to the element sort.
  z3::expr rawBV = readBytes(ptr_i, byteWidth); // BitVec(byteWidth * 8)

  z3::expr loaded_i = [&]() -> z3::expr {
    if (isFloat(otherTile.elem)) {
      if (fpMode == FPMode::FPA) {
        auto [expBits, sigBits] = fpExpSigBits(otherTile.elem);
        z3::sort fps = ctx.fpa_sort(expBits, sigBits);
        return z3::expr(ctx, Z3_mk_fpa_to_fp_bv(ctx, rawBV, fps));
      }
      // Abstract: opaque BV id, no conversion.
      // IntegerRange: bitvector significand, no conversion.
      return rawBV;
    }
    if (otherTile.elem == DType::I1) {
      // i1 is stored as a byte; convert back to Bool.
      return rawBV != ctx.bv_val(0, 8);
    }
    return rawBV; // other integers / pointers
  }();

  z3::expr resultExpr = z3::lambda(i, z3::ite(mask_i, loaded_i, other_i));
  // A load always yields data values, never pointers → ptrBase is empty
  // (matches the old 4-arg Z3Tile ctor which left ptrBase null).
  return Tensor{resultExpr, otherTile.shape, otherTile.elem, std::nullopt};
}

Memory &Memory::store(const Tensor &ptrTile,
                      const Tensor &valTile,
                      const Tensor &maskTile) {
  // Real mode is not byte-storable; Abstract uses BV(width) so it just works.
  if (fpMode == FPMode::Real)
    throw std::logic_error("store: Real FP mode not yet supported");

  unsigned byteWidth = getByteWidth(valTile.elem);

  // Compute total number of tile elements from shape.
  unsigned n = 1;
  for (int64_t dim : valTile.shape)
    n *= static_cast<unsigned>(dim);

  // Represent the whole masked tile store as ONE lambda update of the heap,
  // instead of an n×byteWidth chain of z3::store nodes. The old chain made the
  // formula explode (1024 elems × 4 bytes = 4096 store nodes per program) and
  // forced checkEquivalence into array extensionality. The new heap is a single
  // function of the byte address:
  //
  //   array' = λ addr. (the matching byte if some masked lane covers addr,
  //                     else the old byte at addr)
  //
  // Tile shapes are compile-time constants in TTIR, so we statically unfold the
  // disjunction over lane index i and byte offset j. Lanes are applied in
  // ascending order, so the highest lane is the outermost ite — last-writer-wins
  // on aliasing, matching the previous store-chain order.
  //
  // This MUST be paired with the pointwise witness-address comparison in
  // checkEquivalence: select(array', witness) β-reduces to a quantifier-free
  // byte formula, so the solver never has to decide array extensionality over
  // two lambdas (which returns `unknown`).
  z3::expr oldArray = array;
  z3::expr addr     = ctx.bv_const("__st_addr", 64);
  z3::expr body     = z3::select(oldArray, addr); // default: keep old byte

  for (unsigned idx = 0; idx < n; idx++) {
    z3::expr idxExpr = ctx.bv_val(idx, 32);
    z3::expr ptr_i   = z3::select(ptrTile.e,  idxExpr);
    z3::expr mask_i  = z3::select(maskTile.e, idxExpr);
    z3::expr val_i   = z3::select(valTile.e,  idxExpr);
    z3::expr bv_i    = toBV(ctx, val_i, valTile.elem, fpMode);

    for (unsigned j = 0; j < byteWidth; j++) {
      z3::expr byteAddr = ptr_i + ctx.bv_val(j, 64);
      z3::expr byte_j   = bv_i.extract(j * 8 + 7, j * 8);
      body = z3::ite(mask_i && (addr == byteAddr), byte_j, body);
    }
  }

  array = z3::lambda(addr, body);
  return *this;
}
