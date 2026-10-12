#include "builder/mlir/ArithOps.h"

#include "builder/mlir/DTypeOf.h"
#include "semantics/Context.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstdint>
#include <vector>

// Builder-side arith / math handlers. Each handler only reads its own IR
// operands / attributes (MLIR side), maps types via dtypeOf, calls the core
// Context builder (which owns every op-expression body), and binds the result.

using namespace Semantics;

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

// Element DType of an op result (scalar or tensor).
static DType resultElemType(mlir::Operation *op) {
  mlir::Type t = op->getResult(0).getType();
  if (auto tt = llvm::dyn_cast<mlir::RankedTensorType>(t))
    return dtypeOf(tt.getElementType());
  return dtypeOf(t);
}

//===----------------------------------------------------------------------===//
// arith.constant — decode the MLIR attribute (adapter), build via Context
//===----------------------------------------------------------------------===//

State Semantics::handleArithConstant(const State &s, mlir::Operation *op) {
  auto cop = llvm::cast<mlir::arith::ConstantOp>(op);
  mlir::Type resType = op->getResult(0).getType();
  mlir::Attribute value = cop.getValue();
  Context &ctx = s.context;

  // Compute the Value through a lambda so all branches return directly,
  // avoiding any need to default-construct Value (z3::expr has no default
  // ctor).
  auto makeVal = [&]() -> Value {
    // --- scalar integer ---
    if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(value))
      return ctx.constInt(intAttr.getInt(), dtypeOf(resType));

    // --- dense integer tensor ---
    if (auto denseAttr = llvm::dyn_cast<mlir::DenseIntElementsAttr>(value)) {
      auto tensorTy = llvm::cast<mlir::RankedTensorType>(resType);
      DType elem = dtypeOf(tensorTy.getElementType());
      kernel_smt::Shape shape(tensorTy.getShape().begin(),
                            tensorTy.getShape().end());
      if (denseAttr.isSplat()) {
        int64_t v = denseAttr.getSplatValue<llvm::APInt>().getSExtValue();
        return ctx.splatConst(ctx.constInt(v, elem), shape);
      }
      std::vector<int64_t> values;
      for (const llvm::APInt &apv : denseAttr)
        values.push_back(apv.getSExtValue());
      return ctx.denseIntTile(values, elem, shape);
    }

    // --- dense float tensor ---
    if (auto denseFP = llvm::dyn_cast<mlir::DenseFPElementsAttr>(value)) {
      auto tensorTy = llvm::cast<mlir::RankedTensorType>(resType);
      DType elem = dtypeOf(tensorTy.getElementType());
      kernel_smt::Shape shape(tensorTy.getShape().begin(),
                            tensorTy.getShape().end());
      // Abstract/IntegerRange carry FP as BitVec(width); use the IEEE bit
      // pattern as the opaque id (same convention as the scalar float case).
      if (denseFP.isSplat()) {
        uint64_t bits = denseFP.getSplatValue<llvm::APFloat>()
                            .bitcastToAPInt()
                            .getZExtValue();
        return ctx.splatConst(ctx.constFloatBits(bits, elem), shape);
      }
      std::vector<uint64_t> bits;
      for (const llvm::APFloat &apf : denseFP.getValues<llvm::APFloat>())
        bits.push_back(apf.bitcastToAPInt().getZExtValue());
      return ctx.denseFloatTile(bits, elem, shape);
    }

    // --- scalar float ---
    if (auto floatAttr = llvm::dyn_cast<mlir::FloatAttr>(value)) {
      uint64_t bits = floatAttr.getValue().bitcastToAPInt().getZExtValue();
      return ctx.constFloatBits(bits, dtypeOf(resType));
    }
    llvm_unreachable("arith.constant: unsupported attribute kind");
  };

  State next = s;
  next.env.bind(op->getResult(0), makeVal());
  return next;
}

//===----------------------------------------------------------------------===//
// Integer binary ops — arith.addi, subi, muli, andi
//===----------------------------------------------------------------------===//

State Semantics::handleArithAddi(const State &s, mlir::Operation *op) {
  Value lhs = s.env.lookup(op->getOperand(0));
  Value rhs = s.env.lookup(op->getOperand(1));
  State next = s;
  next.env.bind(op->getResult(0), s.context.add(lhs, rhs));
  return next;
}

State Semantics::handleArithSubi(const State &s, mlir::Operation *op) {
  Value lhs = s.env.lookup(op->getOperand(0));
  Value rhs = s.env.lookup(op->getOperand(1));
  State next = s;
  next.env.bind(op->getResult(0), s.context.sub(lhs, rhs));
  return next;
}

State Semantics::handleArithMuli(const State &s, mlir::Operation *op) {
  Value lhs = s.env.lookup(op->getOperand(0));
  Value rhs = s.env.lookup(op->getOperand(1));
  State next = s;
  next.env.bind(op->getResult(0), s.context.mul(lhs, rhs));
  return next;
}

State Semantics::handleArithDivsi(const State &s, mlir::Operation *op) {
  Value lhs = s.env.lookup(op->getOperand(0));
  Value rhs = s.env.lookup(op->getOperand(1));
  State next = s;
  next.env.bind(op->getResult(0), s.context.divSigned(lhs, rhs));
  return next;
}

State Semantics::handleArithRemsi(const State &s, mlir::Operation *op) {
  Value lhs = s.env.lookup(op->getOperand(0));
  Value rhs = s.env.lookup(op->getOperand(1));
  State next = s;
  next.env.bind(op->getResult(0), s.context.remSigned(lhs, rhs));
  return next;
}

State Semantics::handleArithAndi(const State &s, mlir::Operation *op) {
  Value lhs = s.env.lookup(op->getOperand(0));
  Value rhs = s.env.lookup(op->getOperand(1));
  State next = s;
  next.env.bind(op->getResult(0), s.context.andOp(lhs, rhs));
  return next;
}

//===----------------------------------------------------------------------===//
// arith.extsi — sign-extend
//===----------------------------------------------------------------------===//

State Semantics::handleArithExtsi(const State &s, mlir::Operation *op) {
  auto extOp = llvm::cast<mlir::arith::ExtSIOp>(op);
  Value src = s.env.lookup(extOp.getIn());
  DType dstElem = resultElemType(op);
  State next = s;
  next.env.bind(op->getResult(0), s.context.extsi(src, dstElem));
  return next;
}

//===----------------------------------------------------------------------===//
// arith.cmpi — integer comparison
//===----------------------------------------------------------------------===//

State Semantics::handleArithCmpi(const State &s, mlir::Operation *op) {
  auto cmpiOp = llvm::cast<mlir::arith::CmpIOp>(op);
  Value lhs = s.env.lookup(cmpiOp.getLhs());
  Value rhs = s.env.lookup(cmpiOp.getRhs());

  Context::IPred pred;
  switch (cmpiOp.getPredicate()) {
  case mlir::arith::CmpIPredicate::eq:
    pred = Context::IPred::eq;
    break;
  case mlir::arith::CmpIPredicate::ne:
    pred = Context::IPred::ne;
    break;
  case mlir::arith::CmpIPredicate::slt:
    pred = Context::IPred::slt;
    break;
  case mlir::arith::CmpIPredicate::sle:
    pred = Context::IPred::sle;
    break;
  case mlir::arith::CmpIPredicate::sgt:
    pred = Context::IPred::sgt;
    break;
  case mlir::arith::CmpIPredicate::sge:
    pred = Context::IPred::sge;
    break;
  case mlir::arith::CmpIPredicate::ult:
    pred = Context::IPred::ult;
    break;
  case mlir::arith::CmpIPredicate::ule:
    pred = Context::IPred::ule;
    break;
  case mlir::arith::CmpIPredicate::ugt:
    pred = Context::IPred::ugt;
    break;
  case mlir::arith::CmpIPredicate::uge:
    pred = Context::IPred::uge;
    break;
  default:
    llvm_unreachable("unhandled CmpIPredicate");
  }

  State next = s;
  next.env.bind(op->getResult(0), s.context.cmpInt(pred, lhs, rhs));
  return next;
}

//===----------------------------------------------------------------------===//
// FP binary ops — arith.addf, subf, mulf, divf, maxnumf
//===----------------------------------------------------------------------===//

State Semantics::handleArithAddf(const State &s, mlir::Operation *op) {
  Value lhs = s.env.lookup(op->getOperand(0));
  Value rhs = s.env.lookup(op->getOperand(1));
  State next = s;
  next.env.bind(op->getResult(0), s.context.add(lhs, rhs));
  return next;
}

State Semantics::handleArithSubf(const State &s, mlir::Operation *op) {
  Value lhs = s.env.lookup(op->getOperand(0));
  Value rhs = s.env.lookup(op->getOperand(1));
  State next = s;
  next.env.bind(op->getResult(0), s.context.sub(lhs, rhs));
  return next;
}

State Semantics::handleArithMulf(const State &s, mlir::Operation *op) {
  Value lhs = s.env.lookup(op->getOperand(0));
  Value rhs = s.env.lookup(op->getOperand(1));
  State next = s;
  next.env.bind(op->getResult(0), s.context.mul(lhs, rhs));
  return next;
}

State Semantics::handleArithDivf(const State &s, mlir::Operation *op) {
  Value lhs = s.env.lookup(op->getOperand(0));
  Value rhs = s.env.lookup(op->getOperand(1));
  State next = s;
  next.env.bind(op->getResult(0), s.context.div(lhs, rhs));
  return next;
}

State Semantics::handleArithMaxnumf(const State &s, mlir::Operation *op) {
  Value lhs = s.env.lookup(op->getOperand(0));
  Value rhs = s.env.lookup(op->getOperand(1));
  State next = s;
  next.env.bind(op->getResult(0), s.context.maxnum(lhs, rhs));
  return next;
}

//===----------------------------------------------------------------------===//
// math.exp — element-wise FP unary
//===----------------------------------------------------------------------===//

State Semantics::handleMathExp(const State &s, mlir::Operation *op) {
  Value in = s.env.lookup(op->getOperand(0));
  State next = s;
  next.env.bind(op->getResult(0), s.context.exp(in));
  return next;
}
