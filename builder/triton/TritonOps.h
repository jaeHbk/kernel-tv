#ifndef TV_BUILDER_TRITON_TRITONOPS_H
#define TV_BUILDER_TRITON_TRITONOPS_H

#include "builder/mlir/State.h"
#include "mlir/IR/Operation.h"

namespace Semantics {

State handleTtGetProgramId(const State &s, mlir::Operation *op);
State handleTtMakeRange(const State &s, mlir::Operation *op);
State handleTtSplat(const State &s, mlir::Operation *op);
State handleTtExpandDims(const State &s, mlir::Operation *op);
State handleTtBroadcast(const State &s, mlir::Operation *op);
State handleTtDot(const State &s, mlir::Operation *op);
State handleTtAddPtr(const State &s, mlir::Operation *op);
State handleTtLoad(const State &s, mlir::Operation *op);
State handleTtStore(const State &s, mlir::Operation *op);
State handleTtReduce(const State &s, mlir::Operation *op);

} // namespace Semantics

#endif // TV_BUILDER_TRITON_TRITONOPS_H
