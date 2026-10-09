#ifndef KERNEL_TV_TYPES_H
#define KERNEL_TV_TYPES_H

// kernel-tv core — neutral scalar types, independent of any source IR.
// This header (and the whole kernel-tv core) MUST NOT include MLIR/Triton; it
// depends only on Z3 and the C++ standard library.

#include <cstdint>
#include <utility>
#include <vector>

#include <z3++.h>

namespace kernel_smt {

// Scalar data type. Neutral replacement for a source IR's scalar type in the
// core (a builder maps its own IR types onto DType).
enum class DType { I1, I8, I16, I32, I64, F16, BF16, F32, F64, Ptr };

// Tensor shape (row-major dimensions).
using Shape = std::vector<int64_t>;

// Opaque handle identifying which memory / address space a pointer belongs to.
// A (language-specific) builder maps its own provenance — e.g. the SSA value of
// a pointer kernel argument — onto a MemId; the core never sees source IR.
enum class MemId : uint32_t {};

// Floating-point encoding strategy (pluggable). Abstract is the validator
// default. FPA supports constants and basic arithmetic in the core; Real and
// IntegerRange remain placeholders. FPA does not preserve NaN payload bits.
enum class FPMode { Abstract, Real, IntegerRange, FPA };

// Byte width of a scalar type as laid out in the byte-addressable heap.
// i1 occupies 1 byte; pointers 8.
unsigned getByteWidth(DType);

// {exponent bits, significand bits (incl. the hidden bit)} for a float DType.
std::pair<unsigned, unsigned> fpExpSigBits(DType);

// Z3 sort used to encode a single element of `ty` under the given FP mode.
z3::sort getElemSort(z3::context &ctx, DType ty, FPMode fpMode);

} // namespace kernel_smt

#endif // KERNEL_TV_TYPES_H
