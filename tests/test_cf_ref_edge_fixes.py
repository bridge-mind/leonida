"""Regression tests for tools/cf_ref.py s_cmp_* validation and exec_hi/vcc_hi mask ops.

Standard library only; uses hand-written synthetic .s listings (no game data, GPU, or llvm-mc).

Run from the repository root:
    python -I tests/test_cf_ref_edge_fixes.py -v
"""
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import cf_ref  # noqa: E402
import cpu_ref  # noqa: E402
from shader_ir import parse_file  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
_VOP3_NO_E64 = {"v_mul_lo_u32", "v_mbcnt_lo_u32_b32", "v_mbcnt_hi_u32_b32"}


def _enc_bytes(size):
    return "; encoding: [" + ",".join(["0x00"] * size) + "]"


def _guess_size(text):
    op = text.split()[0]
    if op == "exp" or op.endswith("_e64") or op in _VOP3_NO_E64:
        return 8
    for tok in text.replace(",", " ").split()[1:]:
        if tok.startswith("0x") or (tok.lstrip("-").isdigit() and not (-16 <= int(tok) <= 64)):
            return 8
    return 4


def assemble(items, name="test"):
    pc = 0
    labels = {}
    sized = []
    for item in items:
        if isinstance(item, tuple) and item[0] == "lbl":
            labels[item[1]] = pc
            continue
        if isinstance(item, tuple) and item[0] == "br":
            sized.append((item, pc, 4))
            pc += 4
        else:
            text, size = item if isinstance(item, tuple) else (item, _guess_size(item))
            sized.append((text, pc, size))
            pc += size

    lines = [f"; {name}  code 0x0..0x{pc:x}"]
    for item, at, size in sized:
        if isinstance(item, tuple) and item[0] == "br":
            _, op, target = item
            delta = labels[target] - (at + 4)
            if delta % 4 != 0:
                raise ValueError(f"branch target {target} is not 4-byte aligned")
            text = f"{op} {delta // 4}"
        else:
            text = item
        lines.append(f"\t{text} {_enc_bytes(size)}")
    return "\n".join(lines) + "\n"


def lane_id_prologue(wave):
    insns = ["v_mbcnt_lo_u32_b32 v0, -1, 0"]
    if wave == 64:
        insns.append("v_mbcnt_hi_u32_b32 v0, -1, v0")
    return insns


class CfRefEdgeFixesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="cf-ref-edge-")
        self.addCleanup(self.tmp.cleanup)
        self.n = 0

    def write_items(self, items):
        self.n += 1
        path = os.path.join(self.tmp.name, f"t{self.n}.s")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(assemble(items, name=f"t{self.n}"))
        return path

    def run_items(self, items, wave=32):
        return cf_ref.run(self.write_items(items), wave=wave)

    def test_s_cmp_unknown_condition_raises_unsupported(self):
        bad_ops = [
            "s_cmp_bad_u32 s0, s1",
            "s_cmp_bad_i32 s0, s1",
            "s_cmp_tru_u32 s0, s1",
            "s_cmp_f_i32 s0, s1",
            "s_cmp_foo s0, s1",
            "s_cmp_eq_u64 s[0:1], s[2:3]",
            "s_cmp_lt_f32 s0, s1",
        ]
        for insn in bad_ops:
            mnemonic = insn.split()[0]
            with self.subTest(insn=insn):
                with self.assertRaises(cpu_ref.Unsupported) as cm:
                    self.run_items([insn, "s_endpgm"], wave=32)
                self.assertIn(mnemonic, str(cm.exception))

    def test_wave64_exec_hi_mov_preserves_lo_and_controls_upper_lanes(self):
        # Set exec_lo to lanes {0, 1} and exec_hi to lanes {32, 34}, preserving exec_lo.
        # Then verify active lanes 32..63 run, inactive ones do not, and exec_hi clears cleanly.
        res = self.run_items(lane_id_prologue(64) + [
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "s_cmp_eq_u32 1, 1",                 # SCC = 1; s_mov_b32 exec_hi must preserve SCC
            "s_mov_b32 exec_lo, 0x00000003",
            "s_mov_b32 exec_hi, 0x00000005",
            "s_mov_b32 s0, scc",
            "v_mov_b32 v1, 77",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_mov_b32 exec_hi, 0",
            "s_endpgm",
        ], wave=64)
        self.assertEqual(res["sgpr"][0], 1)
        self.assertEqual(sorted(res["exports"]), [0, 1, 32, 34])
        for lane in (0, 1, 32, 34):
            self.assertEqual(res["exports"][lane], (lane, 77, 0, 0))
        self.assertEqual(res["exec"], 0x0000000000000003)

        # exec_lo == 0 and exec_hi != 0 must keep EXECZ == 0 (s_cbranch_execnz taken).
        res_nz = self.run_items([
            "s_mov_b32 s0, 0",
            "s_mov_b32 exec_lo, 0",
            "s_mov_b32 exec_hi, 0x00010000",
            ("br", "s_cbranch_execz", "bad"),
            ("br", "s_cbranch_execnz", "ok"),
            ("lbl", "bad"),
            "s_mov_b32 s0, 999",
            ("lbl", "ok"),
            "s_add_u32 s0, s0, 1",
            "s_mov_b32 exec_hi, 0",
            ("br", "s_cbranch_execnz", "bad2"),
            "s_add_u32 s0, s0, 2",
            ("lbl", "bad2"),
            "s_endpgm",
        ], wave=64)
        self.assertEqual(res_nz["sgpr"][0], 3)
        self.assertEqual(res_nz["exec"], 0)

    def test_wave64_vcc_hi_mov_preserves_lo_and_drives_upper_lanes(self):
        # Write vcc_lo (lanes 1, 3) then vcc_hi (lanes 33, 63); low 32 bits must stay intact
        # and VOP2 v_cndmask_b32 must see vcc[63:32] on lanes 32..63.
        res = self.run_items(lane_id_prologue(64) + [
            "s_cmp_eq_u32 1, 1",                 # SCC = 1; s_mov_b32 vcc_hi must preserve SCC
            "s_mov_b32 vcc_lo, 0x0000000a",
            "s_mov_b32 vcc_hi, 0x80000002",
            "s_mov_b32 s0, scc",
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 20",
            "v_cndmask_b32 v3, v1, v2",
            "v_mov_b32 v4, 0",
            "exp mrt0 v0, v3, v4, v4 done vm",
            "s_endpgm",
        ], wave=64)
        self.assertEqual(res["sgpr"][0], 1)
        self.assertEqual(res["vcc"], (0x80000002 << 32) | 0x0000000A)
        for lane in range(64):
            want = 20 if lane in (1, 3, 33, 63) else 10
            self.assertEqual(res["exports"][lane][1], want)

        # vcc_lo == 0 and vcc_hi != 0 must keep VCCZ == 0 in wave64; clearing vcc_hi sets VCCZ == 1.
        res_vccz = self.run_items([
            "s_mov_b32 s0, 0",
            "s_mov_b32 vcc_lo, 0",
            "s_mov_b32 vcc_hi, 0x10",
            ("br", "s_cbranch_vccz", "bad"),
            "s_add_u32 s0, s0, 1",
            "s_mov_b32 vcc_hi, 0",
            ("br", "s_cbranch_vccnz", "bad"),
            "s_add_u32 s0, s0, 2",
            ("br", "s_cbranch_vccz", "done"),
            ("lbl", "bad"),
            "s_mov_b32 s0, 999",
            ("lbl", "done"),
            "s_endpgm",
        ], wave=64)
        self.assertEqual(res_vccz["sgpr"][0], 3)
        self.assertEqual(res_vccz["vcc"], 0)

    def test_read_exec_hi_and_vcc_hi_in_scalar_ops_wave64(self):
        res = self.run_items([
            "s_mov_b32 exec_lo, 0x11223344",
            "s_mov_b32 exec_hi, 0x55667788",
            "s_mov_b32 vcc_lo, 0xaabbccdd",
            "s_mov_b32 vcc_hi, 0x0f0f0f0f",
            "s_cmp_eq_u32 1, 2",                 # SCC = 0; s_mov_b32 must preserve SCC
            "s_mov_b32 s4, vcc_hi",
            "s_mov_b32 s5, exec_hi",
            "s_mov_b32 s6, scc",
            "s_and_b32 s7, exec_hi, 0x00ff00ff",
            "s_mov_b32 s8, scc",
            "s_or_b32 vcc_hi, vcc_hi, 0xf0000000",
            "s_mov_b32 s9, scc",
            "s_xnor_b32 s10, exec_hi, 0x55667788",  # 32-bit exact complement of XOR -> 0xffffffff -> SCC=1
            "s_mov_b32 s11, scc",
            "s_xor_b32 s12, exec_hi, 0x55667788",   # 0 -> SCC=0
            "s_mov_b32 s13, scc",
            "s_xnor_b32 s14, exec_hi, 0xaa998877",  # 0 -> SCC=0 (verifies 32-bit mask, not 64-bit)
            "s_mov_b32 s15, scc",
            "s_cmp_eq_u32 vcc_hi, 0xff0f0f0f",
            "s_mov_b32 s16, scc",
            "s_cmp_lg_u32 exec_hi, 0x55667788",
            "s_mov_b32 s17, scc",
            "s_mov_b32 exec_lo, -1",
            "s_mov_b32 exec_hi, 0",
            "s_xnor_b32 s18, exec_hi, -1",         # 32-bit -1 -> ~(0 ^ 0xffffffff) = 0, SCC=0
            "s_mov_b32 s19, scc",
            "s_mov_b32 exec_hi, -1",
            "s_xor_b32 s20, exec_hi, -1",          # 32-bit -1 -> 0xffffffff ^ 0xffffffff = 0, SCC=0
            "s_mov_b32 s21, scc",
            "s_endpgm",
        ], wave=64)
        self.assertEqual(res["sgpr"][4], 0x0F0F0F0F)
        self.assertEqual(res["sgpr"][5], 0x55667788)
        self.assertEqual(res["sgpr"][6], 0)
        self.assertEqual(res["sgpr"][7], 0x00660088)
        self.assertEqual(res["sgpr"][8], 1)
        self.assertEqual(res["sgpr"][9], 1)
        self.assertEqual(res["sgpr"][10], 0xFFFFFFFF)
        self.assertEqual(res["sgpr"][11], 1)
        self.assertEqual(res["sgpr"][12], 0)
        self.assertEqual(res["sgpr"][13], 0)
        self.assertEqual(res["sgpr"][14], 0)
        self.assertEqual(res["sgpr"][15], 0)
        self.assertEqual(res["sgpr"][16], 1)
        self.assertEqual(res["sgpr"][17], 0)
        self.assertEqual(res["sgpr"][18], 0)
        self.assertEqual(res["sgpr"][19], 0)
        self.assertEqual(res["sgpr"][20], 0)
        self.assertEqual(res["sgpr"][21], 0)
        self.assertEqual(res["vcc"], (0xFF0F0F0F << 32) | 0xAABBCCDD)
        self.assertEqual(res["exec"], MASK64)

        # Full-wave 64-bit register names in _b32 SOP2 ops remain rejected in wave64.
        with self.assertRaises(cpu_ref.Unsupported):
            self.run_items(["s_and_b32 vcc, vcc, exec", "s_endpgm"], wave=64)

    def test_wave32_exec_hi_keeps_upper_half_zero_and_vcc_hi_is_scalar(self):
        # Wave32 runs with EXEC[63:32] == 0 so lanes 32-63 never execute even if exec_hi is written.
        # vcc_hi holds its own 32-bit value independently of wave32 VCC[31:0] / VCCZ.
        items = lane_id_prologue(32) + [
            "s_mov_b32 exec_lo, 0x0000000f",
            "s_mov_b32 exec_hi, 0xffffffff",
            "s_or_b32 exec_hi, exec_hi, 0x1",
            "s_mov_b32 s0, exec_hi",
            "s_and_b32 s1, exec_hi, 1",
            "s_mov_b32 s2, scc",
            "s_mov_b32 vcc_lo, 0x00000005",
            "s_mov_b32 vcc_hi, 0xdeadbeef",
            "s_mov_b32 s3, vcc_hi",
            "s_and_b32 s4, vcc_hi, 0x0000ffff",
            "s_mov_b32 vcc_lo, 0",
            ("br", "s_cbranch_vccnz", "bad"),
            "s_and_saveexec_b32 vcc_hi, 0x00000003",
            "s_mov_b32 s5, vcc_hi",
            "s_mov_b32 exec_lo, vcc_hi",
            "v_mov_b32 v1, 1",
            "v_mov_b32 v2, 2",
            "v_mov_b32 v3, 3",
            "exp mrt0 v0, v1, v2, v3 done vm",
            ("br", "s_branch", "done"),
            ("lbl", "bad"),
            "s_mov_b32 s3, 0",
            ("lbl", "done"),
            "s_endpgm",
        ]
        path = self.write_items(items)
        machine = cf_ref.WaveMachine(parse_file(path), wave=32)
        res = machine.run()
        self.assertEqual(machine.exec >> 32, 0)
        self.assertEqual(res["exec"], 0x0000000F)
        self.assertEqual(res["vcc"], 0)
        self.assertEqual(res["sgpr"][0], 0)
        self.assertEqual(res["sgpr"][1], 0)
        self.assertEqual(res["sgpr"][2], 0)
        self.assertEqual(res["sgpr"][3], 0xDEADBEEF)
        self.assertEqual(res["sgpr"][4], 0x0000BEEF)
        self.assertEqual(res["sgpr"][5], 0x0000000F)
        self.assertEqual(sorted(res["exports"]), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
