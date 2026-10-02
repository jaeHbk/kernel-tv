"""CPU witness for the accumulator-order change in ttir-broad/HIT-0007.

This checks the numerical mechanism only. It does not compile Triton IR or
reproduce the original GPU result. Multiplication by 1 is exact, so rounding
each addition to binary32 gives the same values as a binary32 FMA chain for
these operands.
"""

import ctypes
import math
import struct


def f32(value: float) -> float:
    return ctypes.c_float(value).value


def bits(value: float) -> str:
    return f"0x{struct.unpack('<I', struct.pack('<f', value))[0]:08x}"


def main() -> None:
    product = f32(3.0e38)
    accumulator = float("-inf")

    dot = f32(0.0)
    for _ in range(2):
        dot = f32(dot + product)
    reference = f32(accumulator + dot)

    candidate = f32(accumulator)
    for _ in range(2):
        candidate = f32(candidate + product)

    assert math.isfinite(product)
    assert math.isinf(dot) and dot > 0
    assert math.isnan(reference)
    assert math.isinf(candidate) and candidate < 0
    print(f"reference: NaN ({bits(reference)})")
    print(f"candidate: -Inf ({bits(candidate)})")


if __name__ == "__main__":
    main()
