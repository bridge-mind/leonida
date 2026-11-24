"""The run-time FORMAT decode of buffer_load_format (Gen.decode_any_format) branches to one format's decode.

Every code goes through the translator and the mini interpreter, which follows the branches, and is compared with a
reference decoded here from the raw bits with exact quotients. A code outside FORMATS gives zeros.
"""
import struct
import unittest
from fractions import Fraction

import support
from spirv_mini_interp import half_to_float, run_fragment

import rdna2spirv

BASE = 0x1000
WORDS = (0x8123C4F7, 0x3E9A7F01, 0xC0490FDB, 0x5AA5FF80)


def f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def reference(code):
    if code not in rdna2spirv.Gen.FORMATS:
        return (0, 0, 0, 0)
    size, kind, widths = rdna2spirv.Gen.FORMATS[code]
    raw, bit, out = sum(w << (32 * i) for i, w in enumerate(WORDS)), 0, []
    for w in widths:
        v = raw >> bit & ((1 << w) - 1)
        bit += w
        signed = v - (1 << w) if v >> (w - 1) else v
        if kind in ("f32", "u32", "i32", "uint"):
            out.append(v)
        elif kind == "sint":
            out.append(signed & 0xFFFFFFFF)
        elif kind == "f16":
            # A NaN half widens as sign | 0x7F800000 | payload << 13, quiet bit as stored: measured on gfx1036 for
            # formats 13, 29 and 71 (#4623, tools/f16_widen_probe.py). Not struct's '<e', whose NaN payload
            # depends on the Python version (CPython gh-130317).
            if v & 0x7FFF > 0x7C00:
                out.append((v & 0x8000) << 16 | 0x7F800000 | (v & 0x3FF) << 13)
            else:
                out.append(f32_bits(half_to_float(v)))
        elif kind == "unorm":
            out.append(f32_bits(float(Fraction(v, (1 << w) - 1))))
        elif kind == "snorm":
            out.append(f32_bits(float(max(Fraction(-1), Fraction(signed, (1 << (w - 1)) - 1)))))
        else:
            return None                 # a kind this reference doesn't model: covered by its own tests
    return tuple(out[c % len(out)] for c in range(4))


def load(code):
    lines = [f"s_mov_b32 s0, {BASE:#x}", "s_mov_b32 s1, 0", "s_mov_b32 s2, 0x100",
             f"s_mov_b32 s3, {code << 12 | 0x30000FAC:#x}", "v_mov_b32 v1, 0",
             "buffer_load_format_xyzw v[2:5], v1, s[0:3], 0 offen", "exp mrt0 v2, v3, v4, v5 done vm", "s_endpgm"]
    spv = support.translate_lines("ps_fmt_branch", lines)
    return tuple(run_fragment(spv, memory={BASE + 4 * i: w for i, w in enumerate(WORDS)})["%out_mrt0"])


class FormatDecodeBranchTests(unittest.TestCase):
    def test_every_format_and_the_codes_between(self):
        for code in range(max(rdna2spirv.Gen.FORMATS) + 2):
            want = reference(code)
            if want is not None:
                self.assertEqual(load(code), want, f"format {code}")

    def test_only_the_chosen_decode_runs(self):
        gen = rdna2spirv.Gen("ps")
        gen.decode_any_format("%fmt", ["%w0", "%w1", "%w2", "%w3"])
        lines = [ln.strip() for ln in gen.body]
        merges = []
        for ln in lines:
            if ln.startswith("OpSelectionMerge"):
                self.assertIn("DontFlatten", ln)
                merges.append(ln.split()[1])
            elif merges and ln == f"{merges[-1]} = OpLabel":
                merges.pop()
            elif "OpBitFieldUExtract" in ln or "OpIMul" in ln:
                self.assertTrue(merges, f"decode work outside every branch: {ln}")


if __name__ == "__main__":
    unittest.main()
