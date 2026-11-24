"""Check the ps_mullit semantic fixture with runtime SGPR inputs and RGBA32F readback.

python -I runtime/test/shaders/build.py <work> vs_ids ps_mullit
python -I runtime/test/shaders/mullit_readback.py <work> <harness directory>
"""
import argparse
import itertools
import os
from pathlib import Path
import random
import struct
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check import EXE, run

SIGN = 0x80000000
INF = 0x7f800000
MIN_NORMAL = 0x00800000
ONE = 0x3f800000
LIGHT_OFF = 0xff7fffff
PAYLOAD_MASK = 0x40000000


def lighting_product(x, y, gate):
    """Return the expected f32 word using integer significands throughout."""
    ax, ay = x & ~SIGN, y & ~SIGN
    if not MIN_NORMAL <= gate <= INF or y in (LIGHT_OFF, SIGN | INF) or ay > INF:
        return LIGHT_OFF
    if min(ax, ay) < MIN_NORMAL:
        return 0
    if ax > INF:
        return x
    sign = (x ^ y) & SIGN
    if INF in (ax, ay):
        return sign | INF

    product = (MIN_NORMAL + ax % MIN_NORMAL) * (MIN_NORMAL + ay % MIN_NORMAL)
    shift = product.bit_length() - 24
    unit = 1 << shift
    mantissa, remainder = divmod(product, unit)
    if 2 * remainder > unit or (2 * remainder == unit and mantissa % 2):
        mantissa += 1
    exponent = (ax >> 23) + (ay >> 23) - 127 + shift - 23
    if mantissa == 2 * MIN_NORMAL:
        mantissa //= 2
        exponent += 1
    # Round before flushing. A product just below MIN_NORMAL can carry into it.
    if exponent <= 0:
        return sign
    if exponent >= 255:
        return sign | INF
    return sign | (exponent << 23) | (mantissa - MIN_NORMAL)


def operand_triples():
    gates = (ONE, INF, MIN_NORMAL, 1, MIN_NORMAL - 1, 0, SIGN, SIGN | ONE, 0x7fc23456, 0xffa54321)
    yield from ((0x40400000, 0xc0200000, gate) for gate in gates)
    yield from itertools.product(
        (0, SIGN, 1, SIGN | (MIN_NORMAL - 1), 0x7fa12345, 0xffc12345, INF),
        (0, SIGN | 1, LIGHT_OFF, INF, SIGN | INF, 0x7fc09876, ONE), (ONE,))
    yield from itertools.product(
        (ONE - 4, ONE - 3, ONE - 2, ONE - 1, ONE, SIGN | ONE, 0xbf7ffffe, 0xbf7fffff),
        (MIN_NORMAL, MIN_NORMAL + 1, MIN_NORMAL + 2), (ONE,))
    for x, y in ((INF - 1, 0x40000000), (LIGHT_OFF, 0x40000000), (INF - 1, MIN_NORMAL),
                 (MIN_NORMAL, 0x3f000000), (SIGN | MIN_NORMAL, 0x3f000000),
                 (INF, SIGN | ONE), (ONE, LIGHT_OFF - 1)):
        yield x, y, ONE
    # Halfway products round up to 0x3fc00002 and down to 0x3fc00004.
    yield 0x3fc00000, ONE + 1, ONE
    yield 0x3fc00000, ONE + 3, ONE
    rng = random.Random(578)
    for _ in range(28):
        x = rng.randrange(MIN_NORMAL, INF) | rng.choice((0, SIGN))
        y = rng.randrange(MIN_NORMAL, INF) | rng.choice((0, SIGN))
        yield x, y, ONE


def mismatch(data, x, y, gate):
    result = lighting_product(x, y, gate)
    modified = lighting_product(x ^ SIGN, y & ~SIGN, gate)
    expected = (result, modified, LIGHT_OFF, result ^ PAYLOAD_MASK)
    if len(data) != 4 * 4 * 16:
        return f"expected 256 bytes, got {len(data)}"
    for i, pixel in enumerate(struct.iter_unpack("<4I", data)):
        for channel, want, got in zip("RGBA", expected, pixel):
            # A float attachment can change a NaN payload. Alpha exposes the raw
            # R payload as a finite word in exactly those cases.
            if want & ~SIGN > INF and got & ~SIGN > INF:
                continue
            if got != want:
                return f"pixel {i} {channel}: got {got:#010x}, expected {want:#010x}"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("build", type=Path)
    args = parser.parse_args()
    spirv = args.work / "spirv"
    cases = dict.fromkeys(operand_triples())
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "pixels.raw"
        for inputs in cases:
            raw.unlink(missing_ok=True)
            words = ",".join(map(hex, inputs))
            code, output = run(args.build / ("vk_harness" + EXE), spirv / "vs_ids.spv",
                               spirv / "ps_mullit.spv", raw, "4", "4", env=dict(os.environ, VKH_SGPRS=words))
            if code:
                print(f"{words}: vk_harness exited {code}\n{output}", file=sys.stderr)
                return code
            problem = mismatch(raw.read_bytes(), *inputs) if raw.is_file() else "missing readback"
            if problem:
                sys.exit(f"{words}: {problem}")
    print(f"{len(cases)} operand triples, {16 * len(cases)} pixels matched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
