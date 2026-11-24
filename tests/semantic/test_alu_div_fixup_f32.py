"""Semantic tests for RDNA2 v_div_fixup_f32 against native gfx1036 measurements.

Every row in MEASURED ran natively on an AMD Radeon(TM) Graphics RDNA2 GPU (gfx1036,
Windows AMD driver 32.0.21042.62) through tools/hw_oracle in both wave32 and wave64
(identical on every row) under the game's inferred MODE (IEEE=0, denorm32=0, DX10_CLAMP=1).
"""
import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
BASE = 0x1000

# (instruction, v1=S0, v2=S1, v3=S2, word gfx1036 left in v4)
MEASURED = [
    # 1. NaN(S2) wins over everything (including sNaN S1, 0/0, inf/inf, NaN S0); unchanged under IEEE=0
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x3F800000, 0x3F800000, 0x7FC00022, 0x7FC00022),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x3F800000, 0x3F800000, 0xFFC00033, 0xFFC00033),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x3F800000, 0x3F800000, 0x7F800001, 0x7F800001),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x3F800000, 0x7F800007, 0x7FC00022, 0x7FC00022),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x7FC00099, 0x7FC00011, 0x7F800005, 0x7F800005),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x3F800000, 0x00000000, 0xFF800009, 0xFF800009),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x3F800000, 0x7F800000, 0x7FC00044, 0x7FC00044),
    # 2. NaN(S1) wins over non-NaN S2; unchanged under IEEE=0
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x3F800000, 0x7FC00011, 0x3F800000, 0x7FC00011),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x3F800000, 0xFF800003, 0x00000000, 0xFF800003),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x7FC00099, 0x7F800001, 0x7F800000, 0x7F800001),
    # 3. 0/0 (including denormal S1/S2 under denorm32=0) and inf/inf -> canonical -qNaN (0xFFC00000)
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x3F800000, 0x00000000, 0x00000000, 0xFFC00000),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x3F800000, 0x80000000, 0x00000000, 0xFFC00000),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x3F800000, 0x00000000, 0x80000000, 0xFFC00000),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x3F800000, 0x80000000, 0x80000000, 0xFFC00000),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x3F800000, 0x00000001, 0x807FFFFF, 0xFFC00000),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x3F800000, 0x7F800000, 0x7F800000, 0xFFC00000),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x3F800000, 0xFF800000, 0x7F800000, 0xFFC00000),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x3F800000, 0x7F800000, 0xFF800000, 0xFFC00000),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x3F800000, 0xFF800000, 0xFF800000, 0xFFC00000),
    # 4. zero(S2) || inf(S1) -> +-0.0 (including 0/inf and denormal S2)
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x3F800000, 0x40000000, 0x00000000, 0x00000000),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x3F800000, 0xC0000000, 0x00000000, 0x80000000),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x3F800000, 0x40000000, 0x80000000, 0x80000000),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x3F800000, 0xC0000000, 0x80000000, 0x00000000),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x3F800000, 0x40000000, 0x80000001, 0x80000000),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x3F800000, 0x7F800000, 0x40000000, 0x00000000),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x3F800000, 0xFF800000, 0x40000000, 0x80000000),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x3F800000, 0xFF800000, 0xC0000000, 0x00000000),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x7FC00011, 0xFF800000, 0x00000000, 0x80000000),
    # 5. zero(S1) || inf(S2) -> +-inf (including inf/0 and denormal S1)
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x3F800000, 0x00000000, 0x40000000, 0x7F800000),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x3F800000, 0x80000000, 0x40000000, 0xFF800000),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x3F800000, 0x00000000, 0xC0000000, 0xFF800000),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x3F800000, 0x80000000, 0xC0000000, 0x7F800000),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x3F800000, 0x807FFFFF, 0x40000000, 0xFF800000),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x3F800000, 0x40000000, 0x7F800000, 0x7F800000),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x3F800000, 0xC0000000, 0x7F800000, 0xFF800000),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x3F800000, 0x80000000, 0x7F800000, 0xFF800000),
    # 6. Exponent-difference underflow boundary: e2 - e1 <= -151 -> +-0 (beats NaN(S0) & inf(S0)); no overflow check
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x3F800000, 0x4B800000, 0x00800000, 0x3F800000),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x3F800000, 0x4C000000, 0x00800000, 0x00000000),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x3F800000, 0xCC000000, 0x00800000, 0x80000000),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x3F800000, 0x7F7FFFFF, 0x34000000, 0x3F800000),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x3F800000, 0x7F000000, 0x33FFFFFF, 0x00000000),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x7FC00055, 0x64000000, 0x00800000, 0x00000000),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x7F800000, 0xE4000000, 0x00800000, 0x80000000),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x3F800000, 0x00800000, 0x7F000000, 0x3F800000),
    # 7. NaN(S0) -> +-inf
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x7FC00001, 0x40000000, 0x3F800000, 0x7F800000),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0xFFC00001, 0xC0000000, 0x3F800000, 0xFF800000),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x7F800001, 0xC0000000, 0xBF800000, 0x7F800000),
    # 8. Fallback copysign(S0, sign_out), preserving S0 subnormals even under denorm32=0
    ("v_div_fixup_f32 v4, v1, v2, v3", 0xBF400000, 0x40000000, 0x40400000, 0x3F400000),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x3F400000, 0xC0000000, 0x40400000, 0xBF400000),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x7F800000, 0xC0000000, 0x3F800000, 0xFF800000),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x00000000, 0xC0000000, 0x3F800000, 0x80000000),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x00000001, 0x3F800000, 0x3F800000, 0x00000001),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x00000001, 0xBF800000, 0x3F800000, 0x80000001),
    ("v_div_fixup_f32 v4, v1, v2, v3", 0x807FFFFF, 0xBF800000, 0xBF800000, 0x007FFFFF),
    # 9. VOP3 source modifiers abs / neg / -|x| (bitwise before classification, including NaNs)
    ("v_div_fixup_f32 v4, v1, -v2, v3", 0x3F800000, 0x40000000, 0x3F800000, 0xBF800000),
    ("v_div_fixup_f32 v4, v1, |v2|, v3", 0x3F800000, 0xC0000000, 0x3F800000, 0x3F800000),
    ("v_div_fixup_f32 v4, v1, -|v2|, v3", 0x3F800000, 0x40000000, 0x3F800000, 0xBF800000),
    ("v_div_fixup_f32 v4, v1, v2, -v3", 0x3F800000, 0x3F800000, 0x7FC00022, 0xFFC00022),
    ("v_div_fixup_f32 v4, v1, v2, |v3|", 0x3F800000, 0x3F800000, 0xFF800005, 0x7F800005),
    ("v_div_fixup_f32 v4, v1, -|v2|, v3", 0x3F800000, 0x7F800003, 0x3F800000, 0xFF800003),
    ("v_div_fixup_f32 v4, |v1|, v2, v3", 0xBF800000, 0xBF800000, 0x3F800000, 0xBF800000),
    # 10. Output modifiers: mul:2, mul:4, div:2, clamp (DX10_CLAMP=1 maps NaN -> +0.0), and combined omod+clamp
    ("v_div_fixup_f32 v4, v1, v2, v3 mul:2", 0x3F400000, 0x3F800000, 0x3F800000, 0x3FC00000),
    ("v_div_fixup_f32 v4, v1, v2, v3 mul:4", 0x3F000000, 0xBF800000, 0x3F800000, 0xC0000000),
    ("v_div_fixup_f32 v4, v1, v2, v3 div:2", 0x40000000, 0x3F800000, 0x3F800000, 0x3F800000),
    ("v_div_fixup_f32 v4, v1, v2, v3 clamp", 0x40400000, 0x3F800000, 0x3F800000, 0x3F800000),
    ("v_div_fixup_f32 v4, v1, v2, v3 clamp", 0x3F000000, 0xBF800000, 0x3F800000, 0x00000000),
    ("v_div_fixup_f32 v4, v1, v2, v3 clamp", 0x3F800000, 0x00000000, 0x00000000, 0x00000000),
    ("v_div_fixup_f32 v4, v1, v2, v3 clamp", 0x3F800000, 0x3F800000, 0x7F800001, 0x00000000),
    ("v_div_fixup_f32 v4, v1, v2, v3 clamp", 0x3F800000, 0x00000000, 0x3F800000, 0x3F800000),
    ("v_div_fixup_f32 v4, v1, v2, v3 clamp", 0x3F800000, 0x80000000, 0x3F800000, 0x00000000),
    ("v_div_fixup_f32 v4, v1, v2, v3 clamp", 0x3F800000, 0xBF800000, 0x00000000, 0x00000000),
    ("v_div_fixup_f32 v4, v1, v2, v3 clamp mul:2", 0x3F400000, 0x3F800000, 0x3F800000, 0x3F800000),
    ("v_div_fixup_f32 v4, v1, v2, v3 clamp div:2", 0x3FC00000, 0x3F800000, 0x3F800000, 0x3F400000),
]


def _translate(lines, name="ps_div_fixup_f32"):
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        src = os.path.join(tmp, "disasm", f"{name}.s")
        with open(os.path.join(tmp, f"{name}.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(src, "w", encoding="utf-8") as f:
            f.write(f"; {name}\n" + "".join(f"\t{x}\n" for x in lines))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            return rdna2spirv.translate(src)


def _run_body(body_lines, count):
    header = [
        f"s_mov_b32 s0, {BASE:#x}",
        "s_mov_b32 s1, 0",
        "s_mov_b32 s2, 0x1000",
        "s_mov_b32 s3, 0x31016fac",
    ]
    tail = ["exp mrt0 v4, v4, v4, v4 done vm", "s_endpgm"]
    spv = _translate(header + body_lines + tail)
    memory = {}
    si.run_fragment(spv, memory=memory)
    return spv, [memory.get(BASE + 4 * k) for k in range(count)]


class TestDivFixupF32(unittest.TestCase):
    def test_measured_gfx1036_rows(self):
        body = []
        for k, (insn, s0, s1, s2, _) in enumerate(MEASURED):
            body += [
                f"v_mov_b32 v1, {s0:#x}",
                f"v_mov_b32 v2, {s1:#x}",
                f"v_mov_b32 v3, {s2:#x}",
                "v_mov_b32 v4, 0xdeadbeef",
                insn,
                f"buffer_store_dword v4, off, s[0:3], 0 offset:{4 * k}",
            ]
        _, words = _run_body(body, len(MEASURED))
        for (insn, s0, s1, s2, want), got in zip(MEASURED, words):
            with self.subTest(insn=insn, s0=hex(s0), s1=hex(s1), s2=hex(s2)):
                self.assertEqual(got, want)

    def test_operand_forms_and_e64(self):
        body = [
            # SGPR S0, inline float S1/S2, _e64 spelling
            "s_mov_b32 s4, 0x3f400000",
            "v_div_fixup_f32_e64 v4, s4, -1.0, 2.0",
            "buffer_store_dword v4, off, s[0:3], 0 offset:0",
            # VOP3 FP modifiers on constants (-|1.0| = -1.0, |0xbf800000| = +1.0 -> neg sign)
            "v_mov_b32 v1, 0x3f800000",
            "v_div_fixup_f32 v4, v1, -|1.0|, |0xbf800000|",
            "buffer_store_dword v4, off, s[0:3], 0 offset:4",
            # Hex literal NaN in S2
            "v_mov_b32 v2, 0x3f800000",
            "v_div_fixup_f32 v4, v1, v2, 0x7fc00022",
            "buffer_store_dword v4, off, s[0:3], 0 offset:8",
            # null destination discards result cleanly without disturbing v4
            "v_mov_b32 v4, 0x12345678",
            "v_div_fixup_f32 null, v1, v2, v2",
            "buffer_store_dword v4, off, s[0:3], 0 offset:12",
        ]
        _, words = _run_body(body, 4)
        self.assertEqual(words, [0xBF400000, 0xBF800000, 0x7FC00022, 0x12345678])

    def test_inactive_lane_preserves_dst(self):
        body = [
            "v_mov_b32 v1, 0x3f800000",
            "v_mov_b32 v2, 0x40000000",
            "v_mov_b32 v3, 0x40400000",
            "v_mov_b32 v4, 0xdeadbeef",
            "s_mov_b32 exec_lo, 0",
            "v_div_fixup_f32 v4, v1, v2, v3",
            "s_mov_b32 exec_lo, -1",
            "buffer_store_dword v4, off, s[0:3], 0 offset:0",
        ]
        _, words = _run_body(body, 1)
        self.assertEqual(words, [0xDEADBEEF])

    def test_spirv_structure_and_unsupported(self):
        plain_spv = _translate([
            "v_mov_b32 v1, 0x3f800000",
            "v_mov_b32 v2, 0x40000000",
            "v_mov_b32 v3, 0x40400000",
            "v_div_fixup_f32 v4, v1, v2, v3",
            "exp mrt0 v4, v4, v4, v4 done vm",
            "s_endpgm",
        ], name="ps_div_fixup_plain")
        self.assertNotIn("OpFMul", plain_spv)
        self.assertNotIn("FClamp", plain_spv)
        self.assertNotIn("OpCapability Float64", plain_spv)

        clamp_spv = _translate([
            "v_mov_b32 v1, 0x3f800000",
            "v_mov_b32 v2, 0x00000000",
            "v_mov_b32 v3, 0x00000000",
            "v_div_fixup_f32 v4, v1, v2, v3 clamp",
            "exp mrt0 v4, v4, v4, v4 done vm",
            "s_endpgm",
        ], name="ps_div_fixup_clamp")
        self.assertIn("OpIsNan", clamp_spv)
        self.assertIn("FClamp", clamp_spv)

        with self.assertRaises(rdna2spirv.Unsupported):
            _translate([
                "v_div_fixup_f32 v4, v1, v2, v3 op_sel:[0,0,0,0]",
                "exp mrt0 v4, v4, v4, v4 done vm",
                "s_endpgm",
            ], name="ps_div_fixup_bad_mod")


if __name__ == "__main__":
    unittest.main()
