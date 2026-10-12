#include "builder/triton/TritonOps.h"

#include "builder/mlir/DTypeOf.h"
#include "semantics/Context.h"
#include "semantics/Memory.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Region.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

#include <cassert>
#include <optional>

// Builder-side Triton (tt.*) handlers. Each handler reads its own IR operands /
// attributes (MLIR side), maps types via dtypeOf, calls the core Context
// builder (which owns every op-expression body) or the core Memory, and binds
// the result. The tt.reduce combine-region walk stays here (adapter); the core
// only gets a fold callback.

using namespace Semantics;

using kernel_smt::getElemSort;

//===----------------------------------------------------------------------===//
// tt.get_program_id
//===----------------------------------------------------------------------===//

State Semantics::handleTtGetProgramId(const State &s, mlir::Operation *op) {
  auto pidOp = llvm::cast<mlir::triton::GetProgramIdOp>(op);
  int axis = 0;
  switch (pidOp.getAxis()) {
  case mlir::triton::ProgramIDDim::X:
    axis = 0;
    break;
  case mlir::triton::ProgramIDDim::Y:
    axis = 1;
    break;
  case mlir::triton::ProgramIDDim::Z:
    axis = 2;
    break;
  }
  State next = s;
  next.env.bind(pidOp.getResult(), s.context.programId(axis));
  return next;
}

//===----------------------------------------------------------------------===//
// tt.make_range
//===----------------------------------------------------------------------===//

State Semantics::handleTtMakeRange(const State &s, mlir::Operation *op) {
  auto rangeOp = llvm::cast<mlir::triton::MakeRangeOp>(op);
  uint32_t start = rangeOp.getStart();

  auto resultTy =
      llvm::cast<mlir::RankedTensorType>(rangeOp.getResult().getType());
  DType elem = dtypeOf(resultTy.getElementType());
  kernel_smt::Shape shape(resultTy.getShape().begin(), resultTy.getShape().end());

  State next = s;
  next.env.bind(rangeOp.getResult(), s.context.iota(start, shape, elem));
  return next;
}

//===----------------------------------------------------------------------===//
// tt.splat
//===----------------------------------------------------------------------===//

State Semantics::handleTtSplat(const State &s, mlir::Operation *op) {
  auto splatOp = llvm::cast<mlir::triton::SplatOp>(op);
  Value src = s.env.lookup(splatOp.getSrc());

  auto resultTy =
      llvm::cast<mlir::RankedTensorType>(splatOp.getResult().getType());
  kernel_smt::Shape shape(resultTy.getShape().begin(), resultTy.getShape().end());

  auto makeTile = [&]() -> Tensor {
    if (auto *sc = std::get_if<Scalar>(&src))
      return s.context.splat(*sc, shape);
    if (auto *ptr = std::get_if<Ptr>(&src))
      return s.context.splatPtr(*ptr, shape);
    llvm_unreachable("tt.splat: source must be scalar or pointer");
  };

  State next = s;
  next.env.bind(splatOp.getResult(), makeTile());
  return next;
}

State Semantics::handleTtExpandDims(const State &s, mlir::Operation *op) {
  auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("axis");
  assert(axisAttr && "tt.expand_dims: missing axis");
  auto &src = std::get<Tensor>(s.env.lookup(op->getOperand(0)));
  State next = s;
  next.env.bind(op->getResult(0),
                s.context.expandDims(src, axisAttr.getInt()));
  return next;
}

State Semantics::handleTtBroadcast(const State &s, mlir::Operation *op) {
  auto &src = std::get<Tensor>(s.env.lookup(op->getOperand(0)));
  auto resultTy = llvm::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  kernel_smt::Shape shape(resultTy.getShape().begin(), resultTy.getShape().end());
  State next = s;
  next.env.bind(op->getResult(0), s.context.broadcast(src, shape));
  return next;
}

State Semantics::handleTtDot(const State &s, mlir::Operation *op) {
  // The command line must opt into the ordered-FMA lowering assumption before
  // reaching this handler. TTIR's inputPrecision alone does not fix an exact
  // summation order, so this is a conditional check of that lowering policy.
  auto &a = std::get<Tensor>(s.env.lookup(op->getOperand(0)));
  auto &b = std::get<Tensor>(s.env.lookup(op->getOperand(1)));
  auto &c = std::get<Tensor>(s.env.lookup(op->getOperand(2)));
  State next = s;
  next.env.bind(op->getResult(0), s.context.dotFma(a, b, c));
  return next;
}

//===----------------------------------------------------------------------===//
// tt.addptr
//===----------------------------------------------------------------------===//

State Semantics::handleTtAddPtr(const State &s, mlir::Operation *op) {
  auto addptrOp = llvm::cast<mlir::triton::AddPtrOp>(op);
  Value ptrV = s.env.lookup(addptrOp.getPtr());
  Value offV = s.env.lookup(addptrOp.getOffset());

  // Scalar pointer + scalar offset:  newPtr = ptr + sizeof(pointee) * offset.
  // (Softmax computes a per-row base pointer this way before splatting it.)
  if (auto *ptr = std::get_if<Ptr>(&ptrV)) {
    State next = s;
    next.env.bind(addptrOp.getResult(),
                  s.context.addPtr(*ptr, std::get<Scalar>(offV)));
    return next;
  }

  auto &ptrTile = std::get<Tensor>(ptrV);
  auto &offTile = std::get<Tensor>(offV);

  // Determine pointee byte size from the ptr tile's element type. Tensor.elem
  // is just DType::Ptr (the pointee is not carried on the tile), so read the
  // pointee from the op's pointer-tile type via MLIR.
  auto ptrTensorTy =
      llvm::cast<mlir::RankedTensorType>(addptrOp.getPtr().getType());
  auto ptrTy =
      llvm::cast<mlir::triton::PointerType>(ptrTensorTy.getElementType());
  DType pointee = dtypeOf(ptrTy.getPointeeType());

  State next = s;
  next.env.bind(addptrOp.getResult(),
                s.context.addPtr(ptrTile, offTile, pointee));
  return next;
}

//===----------------------------------------------------------------------===//
// tt.load
//===----------------------------------------------------------------------===//

State Semantics::handleTtLoad(const State &s, mlir::Operation *op) {
  auto loadOp = llvm::cast<mlir::triton::LoadOp>(op);
  auto &ptrTile = std::get<Tensor>(s.env.lookup(loadOp.getPtr()));

  // Mask operand (required by add_kernel; guard for completeness).
  mlir::Value maskVal = loadOp.getMask();
  assert(maskVal && "tt.load: unmasked loads not yet supported");
  auto &maskTile = std::get<Tensor>(s.env.lookup(maskVal));

  // Other (passthrough) tile — synthesize a zero tile if absent.
  auto resultTy =
      llvm::cast<mlir::RankedTensorType>(loadOp.getResult().getType());
  DType elem = dtypeOf(resultTy.getElementType());
  kernel_smt::Shape shape(resultTy.getShape().begin(), resultTy.getShape().end());

  mlir::Value otherVal = loadOp.getOther();
  Tensor otherTile = [&]() -> Tensor {
    if (otherVal)
      return std::get<Tensor>(s.env.lookup(otherVal));
    // Default other: zero (or false for i1).
    z3::sort elemSort = getElemSort(s.ctx, elem, s.fpMode);
    z3::expr zero = elemSort.is_bool()
                        ? s.ctx.bool_val(false)
                        : elemSort.is_fpa()
                              ? s.context.constFloatBits(0, elem).e
                              : s.ctx.bv_val(0, elemSort.bv_size());
    z3::expr i = s.ctx.bv_const("__lo_i", 32);
    return Tensor{z3::lambda(i, zero), shape, elem, std::nullopt};
  }();

  // Find the Memory for this pointer's base memory.
  assert(ptrTile.ptrBase &&
         "tt.load: pointer tile has no ptrBase — provenance lost");
  const Memory &mem = s.memState.mems.at(*ptrTile.ptrBase);

  Tensor loaded = mem.load(ptrTile, maskTile, otherTile);

  State next = s;
  next.env.bind(loadOp.getResult(), std::move(loaded));
  return next;
}

//===----------------------------------------------------------------------===//
// tt.store
//===----------------------------------------------------------------------===//

State Semantics::handleTtStore(const State &s, mlir::Operation *op) {
  auto storeOp = llvm::cast<mlir::triton::StoreOp>(op);
  auto &ptrTile = std::get<Tensor>(s.env.lookup(storeOp.getPtr()));
  auto &valTile = std::get<Tensor>(s.env.lookup(storeOp.getValue()));

  // Mask operand (required by add_kernel; guard for completeness).
  mlir::Value maskVal = storeOp.getMask();
  assert(maskVal && "tt.store: unmasked stores not yet supported");
  auto &maskTile = std::get<Tensor>(s.env.lookup(maskVal));

  // Find the Memory for this pointer's base memory.
  assert(ptrTile.ptrBase &&
         "tt.store: pointer tile has no ptrBase — provenance lost");

  // Copy state and mutate the target Memory in the copy.
  State next = s;
  next.memState.mems.at(*ptrTile.ptrBase).store(ptrTile, valTile, maskTile);
  return next;
}

//===----------------------------------------------------------------------===//
// tt.reduce
//
// Single-operand reduction of a 1-D tile to a scalar (the softmax case: max
// over a row, sum over a row). The combine region — e.g.
//   ^bb0(%a, %b): %r = arith.maxnumf %a, %b; tt.reduce.return %r
// — is interpreted here (adapter) by walking it; the core Context::reduce folds
// the callback over the tile's elements in index order. Tile size is a
// compile-time constant, so the fold is statically unrolled into a nested
// combine application (under Abstract FP, the combine is an uninterpreted
// function, so this is exact and order-faithful).
//===----------------------------------------------------------------------===//

State Semantics::handleTtReduce(const State &s, mlir::Operation *op) {
  assert(op->getNumOperands() == 1 &&
         "tt.reduce: only single-operand reductions are supported");
  auto &inTile = std::get<Tensor>(s.env.lookup(op->getOperand(0)));

  mlir::Block &combine = op->getRegion(0).front();
  mlir::Value argA = combine.getArgument(0);
  mlir::Value argB = combine.getArgument(1);

  // Interpret the combine region with its two block args bound to (a, b) and
  // return the value yielded by tt.reduce.return. This region walk stays in the
  // adapter; the core only sees this as an opaque combine callback.
  auto applyCombine = [&](const Value &a, const Value &b) -> Value {
    State seed = s; // shares context; env additionally binds the two args
    seed.env.bind(argA, a);
    seed.env.bind(argB, b);
    const State *cur = &seed;
    std::optional<State> buf;
    for (mlir::Operation &cop : combine) {
      if (cop.getName().getStringRef() == "tt.reduce.return")
        return cur->env.lookup(cop.getOperand(0));
      buf.emplace(cur->interpretOp(&cop));
      cur = &*buf;
    }
    llvm_unreachable("tt.reduce: combine region has no tt.reduce.return");
  };

  // 1-D reduction → scalar result.
  Scalar acc = s.context.reduce(inTile, /*axis=*/0, applyCombine);

  State next = s;
  next.env.bind(op->getResult(0), acc);
  return next;
}
