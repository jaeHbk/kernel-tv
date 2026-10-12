#include "builder/mlir/State.h"

#include "builder/mlir/ArithOps.h"
#include "builder/mlir/DTypeOf.h"
#include "builder/triton/TritonOps.h"

#include "mlir/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

#include <optional>
#include <vector>

using namespace Semantics;

//===----------------------------------------------------------------------===//
// State
//===----------------------------------------------------------------------===//

State::State(Env env, MemState memState,
             std::map<mlir::Value, MemId, ValuePtrLess> ptrArgToMem,
             Context &context)
    : env(std::move(env)), memState(std::move(memState)),
      ptrArgToMem(std::move(ptrArgToMem)), context(context), ctx(context.z3()),
      fpMode(context.mode()) {}

//===----------------------------------------------------------------------===//
// initFromFunc
//===----------------------------------------------------------------------===//

State State::initFromFunc(mlir::ValueRange args, Context &context,
                          const std::string &prefix) {
  Env env;
  MemState memState;
  std::map<mlir::Value, MemId, ValuePtrLess> ptrArgToMem;

  // MemIds are minted in argument order, so the two programs' pointer
  // arguments pair up by position (src arg i ↔ tgt arg i get the same MemId).
  uint32_t nextMemId = 0;

  for (auto [idx, arg] : llvm::enumerate(args)) {
    std::string symName = prefix + "_arg" + std::to_string(idx);

    if (auto ptrTy = llvm::dyn_cast<mlir::triton::PointerType>(arg.getType())) {
      // Pointer argument: mint a MemId, bind a symbolic address in env, and
      // create an independent Memory keyed by that MemId.
      MemId id = static_cast<MemId>(nextMemId++);
      env.bind(arg,
               context.freshPtr(dtypeOf(ptrTy.getPointeeType()), id, symName));
      ptrArgToMem[arg] = id;

      std::string memName = prefix + "_mem_arg" + std::to_string(idx);
      memState.mems.emplace(
          std::piecewise_construct, std::forward_as_tuple(id),
          std::forward_as_tuple(context.z3(), context.mode(), memName));
    } else {
      env.bind(arg, makeSymbolicValue(arg.getType(), context, symName));
    }
  }

  return State(std::move(env), std::move(memState), std::move(ptrArgToMem),
               context);
}

//===----------------------------------------------------------------------===//
// interpretBlock
//===----------------------------------------------------------------------===//

State State::interpretBlock(mlir::Block &block) const {
  // State is not copy/move-assignable (it holds references). Use
  // optional::emplace to in-place construct each successor via State's move
  // constructor, avoiding any assignment operator.
  const State *current = this;
  std::optional<State> buf;
  for (mlir::Operation &op : block) {
    buf.emplace(current->interpretOp(&op));
    current = &*buf;
  }
  return buf ? std::move(*buf) : *this;
}

//===----------------------------------------------------------------------===//
// interpretOp dispatch
//===----------------------------------------------------------------------===//

State State::interpretOp(mlir::Operation *op) const {
  llvm::StringRef opName = op->getName().getStringRef();

  // --- Structured control flow ---
  if (opName == "scf.if")
    return interpretIf(op);
  if (opName == "scf.for")
    return interpretFor(op);
  if (opName == "scf.while")
    return interpretWhile(op);

  // --- Triton core ops (builder/triton) ---
  if (opName == "tt.get_program_id")
    return handleTtGetProgramId(*this, op);
  if (opName == "tt.make_range")
    return handleTtMakeRange(*this, op);
  if (opName == "tt.splat")
    return handleTtSplat(*this, op);
  if (opName == "tt.expand_dims")
    return handleTtExpandDims(*this, op);
  if (opName == "tt.broadcast")
    return handleTtBroadcast(*this, op);
  if (opName == "tt.dot")
    return handleTtDot(*this, op);
  if (opName == "tt.addptr")
    return handleTtAddPtr(*this, op);
  if (opName == "tt.load")
    return handleTtLoad(*this, op);
  if (opName == "tt.store")
    return handleTtStore(*this, op);
  if (opName == "tt.reduce")
    return handleTtReduce(*this, op);

  // --- Arithmetic ops ---
  if (opName == "arith.constant")
    return handleArithConstant(*this, op);
  if (opName == "arith.addi")
    return handleArithAddi(*this, op);
  if (opName == "arith.subi")
    return handleArithSubi(*this, op);
  if (opName == "arith.muli")
    return handleArithMuli(*this, op);
  if (opName == "arith.divsi")
    return handleArithDivsi(*this, op);
  if (opName == "arith.remsi")
    return handleArithRemsi(*this, op);
  if (opName == "arith.andi")
    return handleArithAndi(*this, op);
  if (opName == "arith.extsi")
    return handleArithExtsi(*this, op);
  if (opName == "arith.cmpi")
    return handleArithCmpi(*this, op);
  if (opName == "arith.addf")
    return handleArithAddf(*this, op);
  if (opName == "arith.subf")
    return handleArithSubf(*this, op);
  if (opName == "arith.mulf")
    return handleArithMulf(*this, op);
  if (opName == "arith.divf")
    return handleArithDivf(*this, op);
  if (opName == "arith.maxnumf")
    return handleArithMaxnumf(*this, op);

  // --- math dialect ---
  if (opName == "math.exp")
    return handleMathExp(*this, op);

  // --- Function terminators — no effect on state ---
  if (opName == "tt.return" || opName == "func.return")
    return *this;

  llvm_unreachable(
      ("interpretOp: unimplemented op '" + opName + "'").str().c_str());
}

//===----------------------------------------------------------------------===//
// Control flow framework
//===----------------------------------------------------------------------===//

State State::interpretIf(mlir::Operation *op) const {
  // Plan:
  //   z3::expr cond = std::get<Scalar>(env.lookup(op->getOperand(0))).e;
  //   State thenState = interpretBlock(ifOp.getThenRegion().front());
  //   State elseState = interpretBlock(ifOp.getElseRegion().front());
  //
  //   // Merge env: for each result r_i,
  //   //   merged.env[r_i] = ite(cond, thenState.env[r_i], elseState.env[r_i])
  //   // Merge per-arg memories:
  //   //   merged.memState[a].array =
  //   //       ite(cond, thenState.memState[a].array,
  //   elseState.memState[a].array)
  (void)op;
  llvm_unreachable("interpretIf: not yet implemented");
}

State State::interpretFor(mlir::Operation *op) const {
  assert(op->getNumRegions() == 1 && op->getNumOperands() >= 3);
  auto constantBound = [&](mlir::Value value) -> int64_t {
    auto expr = std::get<Scalar>(env.lookup(value)).e.simplify();
    int64_t result;
    if (!Z3_get_numeral_int64(ctx, expr, &result))
      llvm_unreachable("scf.for: symbolic bounds need a different loop model");
    return result;
  };

  int64_t lb = constantBound(op->getOperand(0));
  int64_t ub = constantBound(op->getOperand(1));
  int64_t step = constantBound(op->getOperand(2));
  if (step <= 0 || (ub > lb && (ub - lb - 1) / step + 1 > 64))
    llvm_unreachable("scf.for: invalid or excessive static trip count");

  mlir::Block &body = op->getRegion(0).front();
  assert(body.getNumArguments() == op->getNumOperands() - 2);
  assert(body.back().getName().getStringRef() == "scf.yield");
  mlir::Operation &yieldOp = body.back();
  assert(yieldOp.getNumOperands() == op->getNumResults());

  std::vector<Value> carried;
  for (unsigned j = 0; j < op->getNumResults(); ++j)
    carried.push_back(env.lookup(op->getOperand(j + 3)));

  std::optional<State> current;
  current.emplace(*this);
  for (int64_t iv = lb; iv < ub; iv += step) {
    current->env.bind(body.getArgument(0),
                      context.constInt(iv, dtypeOf(body.getArgument(0).getType())));
    for (unsigned j = 0; j < carried.size(); ++j)
      current->env.bind(body.getArgument(j + 1), carried[j]);
    for (mlir::Operation &bodyOp : body) {
      if (&bodyOp == &yieldOp)
        break;
      current.emplace(current->interpretOp(&bodyOp));
    }
    carried.clear();
    for (mlir::Value value : yieldOp.getOperands())
      carried.push_back(current->env.lookup(value));
  }
  for (unsigned j = 0; j < carried.size(); ++j)
    current->env.bind(op->getResult(j), carried[j]);
  return std::move(*current);
}

State State::interpretWhile(mlir::Operation *op) const {
  (void)op;
  llvm_unreachable("interpretWhile: not yet implemented");
}
