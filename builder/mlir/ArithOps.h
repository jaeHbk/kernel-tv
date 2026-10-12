#ifndef TV_BUILDER_MLIR_ARITHOPS_H
#define TV_BUILDER_MLIR_ARITHOPS_H

#include "builder/mlir/State.h"
#include "mlir/IR/Operation.h"

namespace Semantics {

State handleArithConstant(const State &s, mlir::Operation *op);
State handleArithAddi(const State &s, mlir::Operation *op);
State handleArithSubi(const State &s, mlir::Operation *op);
State handleArithMuli(const State &s, mlir::Operation *op);
State handleArithDivsi(const State &s, mlir::Operation *op);
State handleArithRemsi(const State &s, mlir::Operation *op);
State handleArithAndi(const State &s, mlir::Operation *op);
State handleArithExtsi(const State &s, mlir::Operation *op);
State handleArithCmpi(const State &s, mlir::Operation *op);
State handleArithAddf(const State &s, mlir::Operation *op);
State handleArithSubf(const State &s, mlir::Operation *op);
State handleArithMulf(const State &s, mlir::Operation *op);
State handleArithDivf(const State &s, mlir::Operation *op);
State handleArithMaxnumf(const State &s, mlir::Operation *op);

// math dialect (FP elementwise, modeled via the FP model uninterpreted fns).
State handleMathExp(const State &s, mlir::Operation *op);

} // namespace Semantics

#endif // TV_BUILDER_MLIR_ARITHOPS_H
