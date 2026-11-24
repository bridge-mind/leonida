"""Semantic tests for RDNA2 scalar bit test and zero-counting instructions.

Covers:
  - s_bitcmp0_b32: SCC = ((S0 >> (S1 & 31)) & 1) == 0
  - s_bitcmp1_b32: SCC = ((S0 >> (S1 & 31)) & 1) == 1
  - s_bitcmp0_b64: SCC = ((S0_64 >> (S1 & 63)) & 1) == 0
  - s_bitcmp1_b64: SCC = ((S0_64 >> (S1 & 63)) & 1) == 1
  - s_bcnt0_i32_b32: D = count_zero_bits(S0), SCC = (D != 0)
  - s_bcnt0_i32_b64: D = count_zero_bits(S0_64), SCC = (D != 0)
  - s_ff0_i32_b32: D = find_first_zero_bit(S0), -1 if none
  - s_ff0_i32_b64: D = find_first_zero_bit(S0_64), -1 if none
  - s_ff1_i32_b32/b64: D = find_first_set_bit(S0), -1 if none
  - Non-mask `_b32` sources for s_ff0/s_ff1 are scalar values, including float immediates and m0
  - Lane mask awareness: testing bits on lane-mask registers (exec / vcc)
  - SPIR-V translation validation via rdna2spirv.Gen
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import rdna2spirv

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


def ref_bitcmp0_b32(s0: int, s1: int) -> int:
    bit = ((s0 & MASK32) >> (s1 & 31)) & 1
    return 1 if bit == 0 else 0


def ref_bitcmp1_b32(s0: int, s1: int) -> int:
    bit = ((s0 & MASK32) >> (s1 & 31)) & 1
    return 1 if bit == 1 else 0


def ref_bitcmp0_b64(s0_64: int, s1: int) -> int:
    bit = ((s0_64 & MASK64) >> (s1 & 63)) & 1
    return 1 if bit == 0 else 0


def ref_bitcmp1_b64(s0_64: int, s1: int) -> int:
    bit = ((s0_64 & MASK64) >> (s1 & 63)) & 1
    return 1 if bit == 1 else 0


def ref_bcnt0_b32(s0: int) -> tuple[int, int]:
    val = (s0 & MASK32) ^ MASK32
    count = bin(val).count("1")
    return count, 1 if count != 0 else 0


def ref_bcnt0_b64(s0_64: int) -> tuple[int, int]:
    val = (s0_64 & MASK64) ^ MASK64
    count = bin(val).count("1")
    return count, 1 if count != 0 else 0


def ref_ff0_b32(s0: int) -> int:
    val = (s0 & MASK32) ^ MASK32
    if val == 0:
        return -1
    return (val & -val).bit_length() - 1


def ref_ff0_b64(s0_64: int) -> int:
    val = (s0_64 & MASK64) ^ MASK64
    if val == 0:
        return -1
    return (val & -val).bit_length() - 1


class TestBitcmpAndBcnt0Semantics(unittest.TestCase):
    """Unit tests for reference models of bit test and zero-counting instructions."""

    def test_bitcmp_b32_all_bits(self):
        for bit_idx in range(32):
            pattern = 1 << bit_idx
            self.assertEqual(ref_bitcmp1_b32(pattern, bit_idx), 1)
            self.assertEqual(ref_bitcmp0_b32(pattern, bit_idx), 0)
            # Test inverted pattern
            self.assertEqual(ref_bitcmp1_b32(~pattern, bit_idx), 0)
            self.assertEqual(ref_bitcmp0_b32(~pattern, bit_idx), 1)

    def test_bitcmp_b64_all_bits(self):
        for bit_idx in range(64):
            pattern = 1 << bit_idx
            self.assertEqual(ref_bitcmp1_b64(pattern, bit_idx), 1)
            self.assertEqual(ref_bitcmp0_b64(pattern, bit_idx), 0)
            # Test inverted pattern
            self.assertEqual(ref_bitcmp1_b64(~pattern, bit_idx), 0)
            self.assertEqual(ref_bitcmp0_b64(~pattern, bit_idx), 1)

    def test_bitcmp_modulo_shift(self):
        # 32-bit: bits [31:5] are ignored (s1 & 31)
        self.assertEqual(ref_bitcmp1_b32(0x80000000, 31 + 64), 1)
        self.assertEqual(ref_bitcmp0_b32(0x80000000, 31 + 64), 0)
        # 64-bit: bits [31:6] are ignored (s1 & 63)
        self.assertEqual(ref_bitcmp1_b64(0x8000000000000000, 63 + 128), 1)
        self.assertEqual(ref_bitcmp0_b64(0x8000000000000000, 63 + 128), 0)

    def test_bcnt0_b32_boundaries(self):
        self.assertEqual(ref_bcnt0_b32(0), (32, 1))
        self.assertEqual(ref_bcnt0_b32(MASK32), (0, 0))
        self.assertEqual(ref_bcnt0_b32(0x55555555), (16, 1))
        self.assertEqual(ref_bcnt0_b32(0x0000FFFF), (16, 1))

    def test_bcnt0_b64_boundaries(self):
        self.assertEqual(ref_bcnt0_b64(0), (64, 1))
        self.assertEqual(ref_bcnt0_b64(MASK64), (0, 0))
        self.assertEqual(ref_bcnt0_b64(0x5555555555555555), (32, 1))
        self.assertEqual(ref_bcnt0_b64(0x00000000FFFFFFFF), (32, 1))

    def test_ff0_b32(self):
        self.assertEqual(ref_ff0_b32(MASK32), -1)
        self.assertEqual(ref_ff0_b32(0), 0)
        self.assertEqual(ref_ff0_b32(1), 1)
        self.assertEqual(ref_ff0_b32(0b111), 3)
        self.assertEqual(ref_ff0_b32(0x7FFFFFFF), 31)

    def test_ff0_b64(self):
        self.assertEqual(ref_ff0_b64(MASK64), -1)
        self.assertEqual(ref_ff0_b64(0), 0)
        self.assertEqual(ref_ff0_b64(1), 1)
        self.assertEqual(ref_ff0_b64(MASK32), 32)
        self.assertEqual(ref_ff0_b64(0x7FFFFFFFFFFFFFFF), 63)


class TestRdna2SpirvBitcmpEmission(unittest.TestCase):
    """Test SPIR-V code generation in rdna2spirv for bitcmp, bcnt0, ff0."""

    def test_s_bitcmp1_b32_emission(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_bitcmp1_b32",
            "args": [
                {"k": "sgpr", "r": 0},
                {"k": "imm", "v": 7},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)
        self.assertIn("OpShiftRightLogical %uint", emitted)
        self.assertIn("OpBitwiseAnd %uint", emitted)
        self.assertIn("OpINotEqual %bool", emitted)
        self.assertIn("OpStore %scc", emitted)

    def test_s_bitcmp0_b32_emission(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_bitcmp0_b32",
            "args": [
                {"k": "sgpr", "r": 2},
                {"k": "sgpr", "r": 3},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)
        self.assertIn("OpShiftRightLogical %uint", emitted)
        self.assertIn("OpBitwiseAnd %uint", emitted)
        self.assertIn("OpIEqual %bool", emitted)
        self.assertIn("OpStore %scc", emitted)

    def test_s_bitcmp1_b64_emission(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_bitcmp1_b64",
            "args": [
                {"k": "sgpr", "r": 4, "n": 2},
                {"k": "imm", "v": 45},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)
        self.assertIn("OpUGreaterThanEqual %bool", emitted)
        self.assertIn("OpSelect %uint", emitted)
        self.assertIn("OpShiftRightLogical %uint", emitted)
        self.assertIn("OpStore %scc", emitted)

    def test_s_bitcmp1_b64_exec_lane_mask(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_bitcmp1_b64",
            "args": [
                {"k": "special", "t": "exec"},
                {"k": "imm", "v": 16},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)
        self.assertIn("OpGroupNonUniformBallot %v4uint_bal", emitted)
        self.assertIn("%c20", emitted)
        self.assertIn("OpStore %scc", emitted)

    def test_s_bcnt0_i32_b32_emission(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_bcnt0_i32_b32",
            "args": [
                {"k": "sgpr", "r": 0},
                {"k": "sgpr", "r": 1},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)
        self.assertIn("OpNot %uint", emitted)
        self.assertIn("OpBitCount %uint", emitted)
        self.assertIn("OpStore %scc", emitted)

    def test_s_bcnt0_i32_b64_emission(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_bcnt0_i32_b64",
            "args": [
                {"k": "sgpr", "r": 0},
                {"k": "sgpr", "r": 2, "n": 2},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)
        self.assertIn("OpNot %uint", emitted)
        self.assertIn("OpBitCount %uint", emitted)
        self.assertIn("OpIAdd %uint", emitted)
        self.assertIn("OpStore %scc", emitted)

    def test_s_ff0_i32_b32_emission(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_ff0_i32_b32",
            "args": [
                {"k": "sgpr", "r": 0},
                {"k": "sgpr", "r": 1},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)
        self.assertIn("OpNot %uint", emitted)
        self.assertIn("FindILsb", emitted)

    def test_s_ff0_i32_b32_immediate_is_scalar(self):
        for value in (5, 61):
            with self.subTest(value=value):
                g = rdna2spirv.Gen("ps")
                ins = {
                    "op": "s_ff0_i32_b32",
                    "args": [
                        {"k": "sgpr", "r": 0},
                        {"k": "imm", "v": value},
                    ],
                    "mods": [],
                }
                g.insn(ins)
                emitted = "\n".join(g.body)
                self.assertIn(f"OpNot %uint %c{value:x}", emitted)
                self.assertNotIn("OpNot %uint %cffffffff", emitted)

    def test_s_ff0_i32_b32_float_immediate_is_scalar(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_ff0_i32_b32",
            "args": [
                {"k": "sgpr", "r": 0},
                {"k": "fimm", "v": 1.0},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)
        self.assertIn("OpNot %uint %c3f80000", emitted)
        self.assertNotIn("OpNot %uint %cffffffff", emitted)

    def test_s_ff0_i32_b32_m0_is_scalar(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_ff0_i32_b32",
            "args": [
                {"k": "sgpr", "r": 0},
                {"k": "special", "t": "m0"},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)
        self.assertIn("OpLoad %uint %m0", emitted)
        self.assertIn("FindILsb", emitted)
        self.assertNotIn("OpNot %uint %cffffffff", emitted)

    def test_s_ff1_i32_b32_immediate_is_scalar(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_ff1_i32_b32",
            "args": [
                {"k": "sgpr", "r": 0},
                {"k": "imm", "v": 5},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)
        self.assertIn("FindILsb", emitted)
        self.assertIn("%c5", emitted)
        self.assertNotIn("%cffffffff", emitted)

    def test_s_ff0_i32_b64_immediate_is_scalar(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_ff0_i32_b64",
            "args": [
                {"k": "sgpr", "r": 0},
                {"k": "imm", "v": 5},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)
        self.assertIn("OpNot %uint %c5", emitted)
        self.assertIn("OpNot %uint %c0", emitted)
        self.assertNotIn("OpNot %uint %cffffffff", emitted)

    def test_s_ff1_i32_b64_immediate_is_scalar(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_ff1_i32_b64",
            "args": [
                {"k": "sgpr", "r": 0},
                {"k": "imm", "v": 5},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)
        self.assertIn("FindILsb", emitted)
        self.assertIn("OpBitcast %int %c5", emitted)
        self.assertNotIn("OpBitcast %int %cffffffff", emitted)

    def test_s_ff0_i32_b64_emission(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_ff0_i32_b64",
            "args": [
                {"k": "sgpr", "r": 0},
                {"k": "sgpr", "r": 2, "n": 2},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)
        self.assertIn("OpNot %uint", emitted)
        self.assertIn("FindILsb", emitted)
        self.assertIn("OpIAdd %uint", emitted)

    def test_s_ff0_s_ff1_b64_negative_inline_and_literal_immediates(self):
        """s_ff0_i32_b64 / s_ff1_i32_b64 sign-extend negative inline constants (-1..-16) and zero-extend literals."""
        import shutil
        import struct
        import subprocess
        import tempfile
        import warnings
        if HERE not in sys.path:
            sys.path.insert(0, HERE)
        import spirv_mini_interp as si

        stub_dxbc = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
        vsharp = [0, 0, 0xFFFFFFFF, 0x31016FAC]
        cases = [
            # (instruction, expected u32 result)
            ("s_ff0_i32_b64 s10, -1", 0xFFFFFFFF),           # 0xFFFFFFFF_FFFFFFFF -> no 0 bit
            ("s_ff0_i32_b64 s10, -2", 0),                    # 0xFFFFFFFF_FFFFFFFE -> bit 0 is 0
            ("s_ff0_i32_b64 s10, -16", 0),                   # 0xFFFFFFFF_FFFFFFF0 -> bit 0 is 0
            ("s_ff0_i32_b64 s10, 0xffffffff", 32),           # 0x00000000_FFFFFFFF -> bit 32 is first 0
            ("s_ff0_i32_b64 s10, 0", 0),                     # 0x00000000_00000000 -> bit 0 is 0
            ("s_ff0_i32_b64 s10, 0x5", 1),                   # 0b101 -> bit 1 is 0
            ("s_ff1_i32_b64 s10, -1", 0),                    # 0xFFFFFFFF_FFFFFFFF -> bit 0 is 1
            ("s_ff1_i32_b64 s10, -16", 4),                   # 0xFFFFFFFF_FFFFFFF0 -> bit 4 is 1
            ("s_ff1_i32_b64 s10, 0", 0xFFFFFFFF),            # no 1 bit
            ("s_ff1_i32_b64 s10, 0x80000000", 31),           # literal bit 31
            ("s_ff0_i32_b64 s10, null", 0),                  # null -> 0
            ("s_ff1_i32_b64 s10, null", 0xFFFFFFFF),         # null -> 0
        ]
        asm_lines = []
        for idx, (insn, _) in enumerate(cases):
            asm_lines += [
                insn,
                "v_mov_b32_e32 v1, s10",
                f"v_mov_b32_e32 v2, {idx * 4}",
                "buffer_store_dword v1, v2, s[0:3], 0 offen",
            ]
        asm_lines.append("s_endpgm")
        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(os.path.join(tmp, "disasm"))
            os.mkdir(os.path.join(tmp, "fssm"))
            src = os.path.join(tmp, "disasm", "cs_ff_b64_imm.s")
            with open(src, "w", encoding="utf-8", newline="\n") as f:
                f.write("; cs_ff_b64_imm\n" + "".join(f"\t{ln}\n" for ln in asm_lines))
            with open(os.path.join(tmp, "cs_ff_b64_imm.dxbc"), "wb") as f:
                f.write(stub_dxbc)
            with open(os.path.join(tmp, "fssm", "cs_ff_b64_imm.fssm"), "wb") as f:
                f.write(bytes(0x138) + struct.pack("<4I", 0x8090, 64, 1, 1))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spv = rdna2spirv.translate(src)
            spirv_as = shutil.which("spirv-as")
            spirv_val = shutil.which("spirv-val")
            if spirv_as and spirv_val:
                spvasm = os.path.join(tmp, "m.spvasm")
                spvbin = os.path.join(tmp, "m.spv")
                with open(spvasm, "w", encoding="utf-8", newline="\n") as f:
                    f.write(spv)
                r1 = subprocess.run([spirv_as, "--target-env", "vulkan1.2", spvasm, "-o", spvbin],
                                    capture_output=True, text=True)
                self.assertEqual(r1.returncode, 0, r1.stderr)
                r2 = subprocess.run([spirv_val, "--target-env", "vulkan1.2", spvbin],
                                    capture_output=True, text=True)
                self.assertEqual(r2.returncode, 0, r2.stderr)
        mem = si.run_workgroup(
            spv, 64, 64,
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
            push_constants=vsharp + [0] * 28, memory={},
        )
        for idx, (insn, want) in enumerate(cases):
            with self.subTest(insn=insn):
                self.assertEqual(mem[idx * 4], want)

    def test_s_ff1_i32_b64_high_word_immediate(self):
        """Synthetic 64-bit immediate with only high-word bits set uses hi via src64_data."""
        g = rdna2spirv.Gen("ps")
        g.insn({
            "op": "s_ff1_i32_b64",
            "args": [{"k": "sgpr", "r": 0}, {"k": "imm", "v": 0x0000000100000000}],
            "mods": [],
        })
        emitted = "\n".join(g.body)
        self.assertIn("OpBitcast %int %c0", emitted)
        self.assertIn("OpBitcast %int %c1", emitted)


if __name__ == "__main__":
    unittest.main()
