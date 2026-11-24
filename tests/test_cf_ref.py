"""Unit tests for tools/cf_ref.py (standard library only; no game data, GPU, or llvm-mc).

Run from the repository root:
    python -I tests/test_cf_ref.py -v
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


def assemble(items, name="test", code_start=0):
    """Expand instructions, ('lbl', name) markers, and ('br', op, label) branches into a .s listing."""
    pc = code_start
    labels = {}
    sized = []
    for item in items:
        if isinstance(item, tuple) and item[0] == "lbl":
            labels[item[1]] = pc
            continue
        if isinstance(item, tuple) and item[0] == "br":
            size = 4
            sized.append((item, pc, size))
        else:
            text, size = item if isinstance(item, tuple) else (item, _guess_size(item))
            sized.append((text, pc, size))
        pc += size

    lines = [f"; {name}  code 0x{code_start:x}..0x{pc:x}"]
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
    """Instructions that populate v0 with each lane's hardware ID (0..wave-1)."""
    insns = ["v_mbcnt_lo_u32_b32 v0, -1, 0"]
    if wave == 64:
        insns.append("v_mbcnt_hi_u32_b32 v0, -1, v0")
    return insns


def load_u64(lo_reg, hi_reg, value):
    """Emit s_mov_b32 pairs to load a 64-bit constant into s[lo_reg:hi_reg]."""
    return [
        f"s_mov_b32 s{lo_reg}, 0x{value & MASK32:08x}",
        f"s_mov_b32 s{hi_reg}, 0x{(value >> 32) & MASK32:08x}",
    ]


class CfRefTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="cf-ref-")
        self.addCleanup(self.tmp.cleanup)
        self.n = 0

    def write_text(self, text):
        self.n += 1
        path = os.path.join(self.tmp.name, f"t{self.n}.s")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        return path

    def run_items(self, items, wave=32, max_steps=200000, code_start=0):
        path = self.write_text(assemble(items, name=f"t{self.n + 1}", code_start=code_start))
        return cf_ref.run(path, wave=wave, max_steps=max_steps)

    def run_raw(self, raw_lines, wave=32, max_steps=200000, header="; raw  code 0x0..0x100"):
        body = "\n".join(raw_lines) + "\n"
        text = f"{header}\n{body}" if header is not None else body
        return cf_ref.run(self.write_text(text), wave=wave, max_steps=max_steps)


class WaveModeAndMaskWidthTests(CfRefTestCase):
    def test_wave32_exec_upper_half_is_zero_and_only_low_32_lanes_run(self):
        items = lane_id_prologue(32) + [
            "v_mov_b32 v1, 0x1234",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        path = self.write_text(assemble(items))
        machine = cf_ref.WaveMachine(parse_file(path), wave=32)
        self.assertEqual(machine.exec, MASK32)
        self.assertEqual(machine.exec >> 32, 0)
        res = machine.run()
        self.assertEqual(res["wave"], 32)
        self.assertEqual(res["exec"], MASK32)
        self.assertEqual(res["vcc"], 0)
        self.assertEqual(sorted(res["exports"]), list(range(32)))
        for lane in range(32):
            self.assertEqual(res["exports"][lane], (lane, 0x1234, 0, 0))

    def test_wave64_all_lanes_active_and_64bit_masks(self):
        items = lane_id_prologue(64) + [
            "v_cmp_le_u32_e32 vcc, 40, v0",
            "v_mov_b32 v1, 1",
            "v_mov_b32 v2, 2",
            "v_mov_b32 v3, 3",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        path = self.write_text(assemble(items))
        machine = cf_ref.WaveMachine(parse_file(path), wave=64)
        self.assertEqual(machine.exec, MASK64)
        res = machine.run()
        self.assertEqual(res["wave"], 64)
        self.assertEqual(res["exec"], MASK64)
        self.assertEqual(res["vcc"], sum(1 << lane for lane in range(40, 64)))
        self.assertEqual(sorted(res["exports"]), list(range(64)))
        for lane in range(64):
            self.assertEqual(res["exports"][lane], (lane, 1, 2, 3))

    def test_mbcnt_lo_and_hi_partial_mask_and_accumulator(self):
        # Mask 0x55555555 has bits set at even positions 0, 2, 4, ...
        items = [
            "v_mbcnt_lo_u32_b32 v0, 0x55555555, 10",
            "v_mbcnt_hi_u32_b32 v1, 0x55555555, v0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        res = self.run_items(items, wave=64)
        for lane in range(64):
            lo_expected = 10 + ((min(lane, 32) + 1) // 2)
            hi_delta = (max(lane - 32, 0) + 1) // 2
            self.assertEqual(res["exports"][lane][:2], (lo_expected, lo_expected + hi_delta))

    def test_s_mov_mask_operations_and_scc_preservation(self):
        # Wave32: moving between vcc_lo/vcc, exec_lo/exec, SGPR, inline imm, and null must leave SCC alone.
        res32 = self.run_items([
            "s_cmp_eq_u32 1, 1",  # SCC = 1
            "s_mov_b32 vcc_lo, 0x0f0f0f0f",
            "s_mov_b32 s0, vcc",
            "s_mov_b32 s1, exec_lo",
            "s_mov_b32 null, vcc_lo",
            "s_mov_b32 s2, scc",
            "s_cmp_eq_u32 1, 2",  # SCC = 0
            "s_mov_b32 vcc, -1",
            "s_mov_b32 exec_lo, 0x00ff00ff",
            "s_mov_b32 s3, scc",
            "s_endpgm",
        ], wave=32)
        self.assertEqual(res32["sgpr"][0], 0x0F0F0F0F)
        self.assertEqual(res32["sgpr"][1], MASK32)
        self.assertEqual(res32["sgpr"][2], 1)
        self.assertEqual(res32["sgpr"][3], 0)
        self.assertEqual(res32["vcc"], MASK32)
        self.assertEqual(res32["exec"], 0x00FF00FF)
        self.assertEqual(res32["scc"], 0)

        # Wave64: s_mov_b64 handles 64-bit masks without altering SCC, and *_lo writes preserve upper 32 bits.
        res64 = self.run_items(
            load_u64(0, 1, 0x123456789ABCDEF0) + [
                "s_cmp_eq_u32 5, 5",  # SCC = 1
                "s_mov_b64 vcc, s[0:1]",
                "s_mov_b64 s[2:3], vcc",
                "s_mov_b64 null, exec",
                "s_mov_b32 vcc_lo, 0x11223344",
                "s_mov_b32 s4, scc",
                "s_mov_b64 s[6:7], -1",
                "s_mov_b64 exec, 0",
                "s_endpgm",
            ],
            wave=64,
        )
        self.assertEqual(res64["sgpr"][2], 0x9ABCDEF0)
        self.assertEqual(res64["sgpr"][3], 0x12345678)
        self.assertEqual(res64["sgpr"][4], 1)
        self.assertEqual(res64["sgpr"][6], MASK32)
        self.assertEqual(res64["sgpr"][7], MASK32)
        self.assertEqual(res64["vcc"], 0x1234567811223344)
        self.assertEqual(res64["exec"], 0)

        res64_exec_lo = self.run_items([
            "s_mov_b32 exec_lo, 0x5",
            "s_endpgm",
        ], wave=64)
        self.assertEqual(res64_exec_lo["exec"], 0xFFFFFFFF00000005)

    def test_v_cndmask_b32_implicit_and_explicit_conditions(self):
        # Wave32: VOP2 3-arg and 4-arg forms use low 32 bits of VCC; _e64 uses SGPR.
        res32 = self.run_items(lane_id_prologue(32) + [
            "s_mov_b32 vcc_lo, 0x0000000a",
            "s_mov_b32 s0, 0x00000005",
            "v_mov_b32 v1, 100",
            "v_mov_b32 v2, 200",
            "v_cndmask_b32 v3, v1, v2",
            "v_cndmask_b32 v4, v1, v2, vcc_lo",
            "v_cndmask_b32_e64 v5, v1, v2, s0",
            "v_mov_b32 v6, 0",
            "exp mrt0 v3, v4, v5, v6 done vm",
            "s_endpgm",
        ], wave=32)
        for lane in range(32):
            vcc_sel = 200 if lane in (1, 3) else 100
            sgpr_sel = 200 if lane in (0, 2) else 100
            self.assertEqual(res32["exports"][lane][:3], (vcc_sel, vcc_sel, sgpr_sel))

        # Wave64: VOP2 implicit VCC reads full 64-bit VCC including lanes >= 32.
        res64 = self.run_items(
            lane_id_prologue(64) + load_u64(0, 1, (1 << 2) | (1 << 45)) + [
                "s_mov_b64 vcc, s[0:1]",
                "v_mov_b32 v1, 10",
                "v_mov_b32 v2, 20",
                "v_cndmask_b32 v3, v1, v2",
                "v_cndmask_b32_e64 v4, v1, v2, s[0:1]",
                "v_mov_b32 v5, 0",
                "v_mov_b32 v6, 0",
                "exp mrt0 v3, v4, v5, v6 done vm",
                "s_endpgm",
            ],
            wave=64,
        )
        for lane in range(64):
            want = 20 if lane in (2, 45) else 10
            self.assertEqual(res64["exports"][lane][:2], (want, want))

    def test_mask_width_mismatches_and_invalid_wave_raise(self):
        cases = [
            (32, ["s_and_b64 vcc, vcc, exec", "s_endpgm"]),
            (32, ["s_and_saveexec_b64 s[0:1], vcc", "s_endpgm"]),
            (32, ["s_mov_b64 exec, -1", "s_endpgm"]),
            (64, ["s_and_b32 vcc, vcc, exec", "s_endpgm"]),
            (64, ["s_and_saveexec_b32 s0, vcc_lo", "s_endpgm"]),
            (64, ["v_cmp_eq_u32_e64 s0, v0, 0", "s_endpgm"]),
        ]
        for wave, insns in cases:
            with self.subTest(wave=wave, insn=insns[0]):
                with self.assertRaises(cpu_ref.Unsupported):
                    self.run_items(insns, wave=wave)

        path = self.write_text(assemble(["s_endpgm"]))
        with self.assertRaises(ValueError):
            cf_ref.run(path, wave=16)


class BranchTests(CfRefTestCase):
    def test_pc_relative_forward_zero_and_mixed_instruction_sizes(self):
        # code_start=0x100 exercises non-zero base PCs; mix 4-byte and 8-byte encodings.
        res = self.run_raw([
            f"\ts_mov_b32 s0, 1 {_enc_bytes(4)}",             # 0x100..0x104
            f"\ts_branch 0 {_enc_bytes(4)}",                  # 0x104..0x108 -> next_pc 0x108
            f"\ts_mov_b32 s1, 2 {_enc_bytes(4)}",             # 0x108..0x10c (executed)
            f"\ts_branch 2 {_enc_bytes(4)}",                  # 0x10c..0x110 -> 0x110 + 8 = 0x118
            f"\ts_mov_b32 s2, 99 {_enc_bytes(4)}",            # 0x110..0x114 (skipped)
            f"\ts_mov_b32 s3, 99 {_enc_bytes(4)}",            # 0x114..0x118 (skipped)
            f"\ts_branch 2 {_enc_bytes(4)}",                  # 0x118..0x11c -> 0x11c + 8 = 0x124
            f"\tv_mul_lo_u32 v0, v1, 3 {_enc_bytes(8)}",      # 0x11c..0x124 (one 8-byte insn skipped)
            f"\ts_mov_b32 s4, 7 {_enc_bytes(4)}",             # 0x124..0x128
            f"\ts_endpgm {_enc_bytes(4)}",                    # 0x128..0x12c
        ], header="; forward  code 0x100..0x12c")
        self.assertEqual(res["sgpr"], {0: 1, 1: 2, 4: 7})
        self.assertEqual(res["steps"], 7)

    def test_backward_branch_negative_simm16_representations(self):
        # All three encodings of -3 (decimal -3, hex 0xfffd, unsigned 65533) must sign-extend identically.
        for off_text in ("-3", "0xfffd", "65533"):
            with self.subTest(off=off_text):
                res = self.run_raw([
                    f"\ts_mov_b32 s0, 0 {_enc_bytes(4)}",            # 0x00
                    f"\ts_add_u32 s0, s0, 1 {_enc_bytes(4)}",        # 0x04 (loop head)
                    f"\ts_cmp_lt_u32 s0, 3 {_enc_bytes(4)}",         # 0x08
                    f"\ts_cbranch_scc1 {off_text} {_enc_bytes(4)}",  # 0x0c -> 0x10 + 4*(-3) = 0x04
                    f"\ts_endpgm {_enc_bytes(4)}",                   # 0x10
                ])
                self.assertEqual(res["sgpr"], {0: 3})
                self.assertEqual(res["steps"], 1 + 3 * 3 + 1)

        # 0xffff (-1) jumps to the branch instruction itself (next_pc - 4 == pc).
        with self.assertRaises(cf_ref.ProgramError) as cm:
            self.run_raw([
                f"\ts_cmp_eq_u32 1, 1 {_enc_bytes(4)}",      # 0x00
                f"\ts_cbranch_scc1 0xffff {_enc_bytes(4)}",  # 0x04 -> 0x08 - 4 = 0x04
                f"\ts_endpgm {_enc_bytes(4)}",
            ], max_steps=3)
        self.assertIn("at pc 0x4", str(cm.exception))

    def test_backward_branch_with_8byte_body_instruction(self):
        # 8-byte v_mul_lo_u32 + two 4-byte scalar insns + 4-byte branch = 20 bytes -> simm16 = -5.
        res = self.run_items([
            "s_mov_b32 s0, 0",
            "v_mov_b32 v0, 1",
            ("lbl", "loop"),
            "v_mul_lo_u32 v0, v0, 3",
            "s_add_u32 s0, s0, 1",
            "s_cmp_lt_u32 s0, 4",
            ("br", "s_cbranch_scc1", "loop"),
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ], wave=32)
        self.assertEqual(res["sgpr"][0], 4)
        self.assertEqual(res["exports"][0][0], 81)

    def test_conditional_branches_taken_and_fallthrough(self):
        # Exercise every conditional branch opcode in both taken and fallthrough states.
        res = self.run_items([
            "s_mov_b32 s0, 0",
            "s_cmp_eq_u32 1, 1",                 # SCC = 1
            ("br", "s_cbranch_scc0", "bad"),     # not taken
            ("br", "s_cbranch_scc1", "k1"),      # taken
            ("lbl", "k1"),
            "s_add_u32 s0, s0, 1",
            "s_cmp_eq_u32 1, 2",                 # SCC = 0
            ("br", "s_cbranch_scc1", "bad"),     # not taken
            ("br", "s_cbranch_scc0", "k2"),      # taken
            ("lbl", "k2"),
            "s_add_u32 s0, s0, 2",
            "s_mov_b32 vcc_lo, 0",               # VCCZ = 1
            ("br", "s_cbranch_vccnz", "bad"),    # not taken
            ("br", "s_cbranch_vccz", "k3"),      # taken
            ("lbl", "k3"),
            "s_add_u32 s0, s0, 4",
            "s_mov_b32 vcc_lo, 8",               # VCCZ = 0
            ("br", "s_cbranch_vccz", "bad"),     # not taken
            ("br", "s_cbranch_vccnz", "k4"),     # taken
            ("lbl", "k4"),
            "s_add_u32 s0, s0, 8",
            ("br", "s_cbranch_execz", "bad"),    # EXEC != 0 -> not taken
            ("br", "s_cbranch_execnz", "k5"),    # taken
            ("lbl", "k5"),
            "s_add_u32 s0, s0, 16",
            "s_mov_b32 exec_lo, 0",              # EXECZ = 1
            ("br", "s_cbranch_execnz", "bad"),   # not taken
            ("br", "s_cbranch_execz", "done"),   # taken
            ("lbl", "bad"),
            "s_mov_b32 s0, 999",
            ("lbl", "done"),
            "s_add_u32 s0, s0, 32",
            "s_endpgm",
        ], wave=32)
        self.assertEqual(res["sgpr"][0], 1 + 2 + 4 + 8 + 16 + 32)

    def test_vccz_and_execz_check_only_low_32_bits_in_wave32_and_all_64_in_wave64(self):
        # Wave64: bit 40 set in VCC or EXEC must make *z false and *nz true.
        res64 = self.run_items(load_u64(0, 1, 1 << 40) + [
            "s_mov_b32 s2, 0",
            "s_mov_b64 vcc, s[0:1]",
            ("br", "s_cbranch_vccz", "skip1"),
            "s_add_u32 s2, s2, 1",
            ("lbl", "skip1"),
            ("br", "s_cbranch_vccnz", "skip2"),
            "s_add_u32 s2, s2, 100",
            ("lbl", "skip2"),
            "s_mov_b64 exec, s[0:1]",
            ("br", "s_cbranch_execz", "skip3"),
            "s_add_u32 s2, s2, 2",
            ("lbl", "skip3"),
            ("br", "s_cbranch_execnz", "done"),
            "s_add_u32 s2, s2, 100",
            ("lbl", "done"),
            "s_endpgm",
        ], wave=64)
        self.assertEqual(res64["sgpr"][2], 3)

        # Wave32: even if upper 32 bits of machine.vcc/exec are non-zero, *z/*nz inspect only bits [31:0].
        vcc_path = self.write_text(assemble([
            "s_mov_b32 s0, 0",
            ("br", "s_cbranch_vccnz", "fail"),
            "s_add_u32 s0, s0, 1",
            ("br", "s_cbranch_vccz", "ok"),
            ("lbl", "fail"),
            "s_mov_b32 s0, 999",
            ("lbl", "ok"),
            "s_endpgm",
        ]))
        m_vcc = cf_ref.WaveMachine(parse_file(vcc_path), wave=32)
        m_vcc.vcc = 1 << 40
        r_vcc = m_vcc.run()
        self.assertEqual(r_vcc["sgpr"][0], 1)
        self.assertEqual(r_vcc["vcc"], 0)

        exec_path = self.write_text(assemble([
            "s_mov_b32 s0, 0",
            ("br", "s_cbranch_execnz", "fail"),
            "s_add_u32 s0, s0, 2",
            ("br", "s_cbranch_execz", "ok"),
            ("lbl", "fail"),
            "s_mov_b32 s0, 999",
            ("lbl", "ok"),
            "s_endpgm",
        ]))
        m_exec = cf_ref.WaveMachine(parse_file(exec_path), wave=32)
        m_exec.exec = 1 << 40
        r_exec = m_exec.run()
        self.assertEqual(r_exec["sgpr"][0], 2)
        self.assertEqual(r_exec["exec"], 0)

    def test_wave32_vcc_mask_writes_preserve_vcc_hi_and_scalar_ops_ignore_exec_hi(self):
        res_vcc = self.run_items([
            "s_mov_b32 vcc_hi, 0xdeadbeef",
            "s_mov_b32 vcc, 5",
            "s_mov_b32 s0, vcc_hi",
            "s_and_b32 vcc, vcc, 1",
            "s_mov_b32 s1, vcc_hi",
            "s_and_saveexec_b32 vcc, 3",
            "s_mov_b32 s2, vcc_hi",
            "s_mov_b32 s3, vcc_lo",
            "s_endpgm",
        ], wave=32)
        self.assertEqual(res_vcc["sgpr"][0], 0xDEADBEEF)
        self.assertEqual(res_vcc["sgpr"][1], 0xDEADBEEF)
        self.assertEqual(res_vcc["sgpr"][2], 0xDEADBEEF)
        self.assertEqual(res_vcc["sgpr"][3], 0xFFFFFFFF)

        res_exec = self.run_items([
            "s_add_u32 exec_hi, 1, 0",
            "s_lshl_b32 exec_hi, 1, 2",
            "s_cselect_b32 exec_hi, 7, 9",
            "s_mov_b32 s0, exec_hi",
            "s_add_u32 s1, exec_hi, 5",
            "v_mov_b32 v0, 1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ], wave=32)
        self.assertEqual(res_exec["sgpr"][0], 0)
        self.assertEqual(res_exec["sgpr"][1], 5)
        self.assertEqual(sorted(res_exec["exports"]), list(range(32)))


class CompareAndFlagTests(CfRefTestCase):
    def test_v_cmp_predicates_u32_and_i32_and_inactive_lanes_write_zero(self):
        # Pre-seed VCC with all 1s and restrict EXEC to lanes 0..3 so inactive lanes 4..31 must clear to 0.
        # Lanes 0..3 get [0, 1, 0xffffffff, 0x80000000] compared against 0/1/-1 to test u32 vs i32.
        items = lane_id_prologue(32) + [
            "s_mov_b32 vcc_lo, -1",
            "v_mov_b32 v1, 0",
            "v_cmp_eq_u32_e32 vcc_lo, 1, v0",
            "v_cndmask_b32 v1, v1, 1, vcc_lo",
            "v_cmp_eq_u32_e32 vcc_lo, 2, v0",
            "v_cndmask_b32 v1, v1, -1, vcc_lo",
            "v_cmp_eq_u32_e32 vcc_lo, 3, v0",
            "v_cndmask_b32 v1, v1, 0x80000000, vcc_lo",
            "s_mov_b32 exec_lo, 0x0f",
            "s_mov_b32 vcc_lo, -1",
            "v_cmp_eq_u32_e32 vcc, v1, 0",       # lane 0 only (wave32 maps vcc -> vcc_lo)
            "v_cmp_ne_u32_e64 s0, v1, 0",        # lanes 1, 2, 3
            "v_cmp_lg_u32_e64 s1, v1, 0",        # lanes 1, 2, 3
            "v_cmp_lt_u32_e64 s2, v1, 1",        # lane 0
            "v_cmp_le_u32_e64 s3, v1, 1",        # lanes 0, 1
            "v_cmp_gt_u32_e64 s4, v1, 1",        # lanes 2, 3
            "v_cmp_ge_u32_e64 s5, v1, 1",        # lanes 1, 2, 3
            "v_cmp_lt_i32_e64 s6, v1, 0",        # lanes 2, 3 (-1 and INT_MIN)
            "v_cmp_le_i32_e64 s7, v1, 0",        # lanes 0, 2, 3
            "v_cmp_gt_i32_e64 s8, v1, -1",       # lanes 0, 1
            "v_cmp_ge_i32_e64 s9, v1, -1",       # lanes 0, 1, 2
            "v_cmp_eq_i32_e64 null, v1, 0",      # discarded cleanly
            "s_endpgm",
        ]
        res = self.run_items(items, wave=32)
        self.assertEqual(res["vcc"], 0b0001)
        self.assertEqual(res["sgpr"], {
            0: 0b1110,
            1: 0b1110,
            2: 0b0001,
            3: 0b0011,
            4: 0b1100,
            5: 0b1110,
            6: 0b1100,
            7: 0b1101,
            8: 0b0011,
            9: 0b0111,
        })

    def test_v_cmp_wave64_sgpr_pair_and_inactive_lanes(self):
        items = lane_id_prologue(64) + load_u64(0, 1, (1 << 5) | (1 << 40)) + [
            "s_mov_b64 exec, s[0:1]",
            "v_cmp_ge_u32_e64 s[2:3], v0, 0",    # true for every lane, but only lanes 5 and 40 are active
            "s_endpgm",
        ]
        res = self.run_items(items, wave=64)
        self.assertEqual(res["sgpr"][2], 1 << 5)
        self.assertEqual(res["sgpr"][3], 1 << (40 - 32))

    def test_v_cmpx_updates_exec_without_touching_vcc(self):
        # Wave32: pre-active mask 0x0f0f, v_cmpx keeps lanes where 10 > v0 -> lanes 0..3 and 8..9.
        res32 = self.run_items(lane_id_prologue(32) + [
            "s_mov_b32 vcc_lo, 0xdeadbeef",
            "s_mov_b32 exec_lo, 0x0f0f",
            "v_cmpx_gt_u32_e32 10, v0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ], wave=32)
        self.assertEqual(res32["vcc"], 0xDEADBEEF)
        self.assertEqual(res32["exec"], 0x030F)
        self.assertEqual(sorted(res32["exports"]), [0, 1, 2, 3, 8, 9])

        # Wave64: signed compare with negative values; VCC remains untouched.
        res64 = self.run_items(
            lane_id_prologue(64) + load_u64(0, 1, 0x123456789ABCDEF0) + load_u64(2, 3, (1 << 1) | (1 << 40) | (1 << 41)) + [
                "s_mov_b64 vcc, s[0:1]",
                "s_mov_b64 exec, s[2:3]",
                "v_sub_nc_u32 v0, v0, 41",       # lane 1 -> -40, lane 40 -> -1, lane 41 -> 0
                "v_cmpx_ge_i32_e32 -1, v0",      # -1 >= signed(v0) -> lanes 1 and 40 survive, lane 41 drops
                "v_mov_b32 v1, 0",
                "v_mov_b32 v2, 0",
                "v_mov_b32 v3, 0",
                "exp mrt0 v0, v1, v2, v3 done vm",
                "s_endpgm",
            ],
            wave=64,
        )
        self.assertEqual(res64["vcc"], 0x123456789ABCDEF0)
        self.assertEqual(res64["exec"], (1 << 1) | (1 << 40))
        self.assertEqual(sorted(res64["exports"]), [1, 40])

    def test_s_cmp_predicates_u32_and_i32(self):
        cases = [
            ("s_cmp_eq_u32", "5", "5", 1),
            ("s_cmp_eq_u32", "5", "6", 0),
            ("s_cmp_ne_u32", "5", "6", 1),
            ("s_cmp_lg_u32", "5", "5", 0),
            ("s_cmp_lt_u32", "1", "0xffffffff", 1),
            ("s_cmp_lt_i32", "1", "0xffffffff", 0),
            ("s_cmp_le_u32", "0x80000000", "0x7fffffff", 0),
            ("s_cmp_le_i32", "0x80000000", "0x7fffffff", 1),
            ("s_cmp_gt_u32", "0xffffffff", "0", 1),
            ("s_cmp_gt_i32", "0xffffffff", "0", 0),
            ("s_cmp_ge_i32", "-5", "-5", 1),
            ("s_cmp_ge_i32", "-6", "-5", 0),
        ]
        insns = []
        for idx, (op, a, b, _) in enumerate(cases):
            insns.append(f"{op} {a}, {b}")
            insns.append(f"s_mov_b32 s{idx + 1}, scc")
        insns.append("s_endpgm")
        res = self.run_items(insns, wave=32)
        for idx, (_, _, _, want) in enumerate(cases):
            self.assertEqual(res["sgpr"][idx + 1], want, cases[idx])

    def test_scalar_bitwise_ops_and_scc_wave32_and_wave64(self):
        # All 8 SOP2 bitwise ops on SGPRs and masks in wave32.
        a32, b32 = 0xF0F00F0F, 0xFF00FF00
        ops32 = [
            ("and", a32 & b32),
            ("or", a32 | b32),
            ("xor", a32 ^ b32),
            ("andn2", a32 & ~b32 & MASK32),
            ("orn2", (a32 | ~b32) & MASK32),
            ("nand", ~(a32 & b32) & MASK32),
            ("nor", ~(a32 | b32) & MASK32),
            ("xnor", ~(a32 ^ b32) & MASK32),
        ]
        insns32 = [f"s_mov_b32 s0, 0x{a32:08x}", f"s_mov_b32 s1, 0x{b32:08x}"]
        for idx, (name, _) in enumerate(ops32):
            dst = 10 + idx * 2
            insns32.append(f"s_{name}_b32 s{dst}, s0, s1")
            insns32.append(f"s_mov_b32 s{dst + 1}, scc")
        insns32 += [
            "s_nand_b32 s2, -1, -1",             # result 0 -> SCC transitions 1 -> 0
            "s_mov_b32 s3, scc",
            "s_mov_b32 vcc_lo, 0x00ff00ff",
            "s_and_b32 null, vcc_lo, exec_lo",   # mask-touching op -> SCC = 1
            "s_mov_b32 s4, scc",
            "s_and_b32 vcc, 0, exec_lo",         # zero mask -> VCC = 0, SCC = 0
            "s_mov_b32 s5, scc",
            "s_endpgm",
        ]
        res32 = self.run_items(insns32, wave=32)
        for idx, (name, want) in enumerate(ops32):
            dst = 10 + idx * 2
            self.assertEqual(res32["sgpr"][dst], want, name)
            self.assertEqual(res32["sgpr"][dst + 1], 1 if want else 0, name)
        self.assertEqual(res32["sgpr"][2], 0)
        self.assertEqual(res32["sgpr"][3], 0)
        self.assertEqual(res32["sgpr"][4], 1)
        self.assertEqual(res32["sgpr"][5], 0)
        self.assertEqual(res32["vcc"], 0)

        # Wave64: all 8 _b64 mask ops + _b32 SGPR-only logic + high-half-only non-zero SCC check.
        a64, b64 = 0x0000000100000000, 0x0000000300000000
        ops64 = [
            ("and", a64 & b64),
            ("or", a64 | b64),
            ("xor", a64 ^ b64),
            ("andn2", a64 & ~b64 & MASK64),
            ("orn2", (a64 | ~b64) & MASK64),
            ("nand", ~(a64 & b64) & MASK64),
            ("nor", ~(a64 | b64) & MASK64),
            ("xnor", ~(a64 ^ b64) & MASK64),
        ]
        insns64 = load_u64(0, 1, a64) + load_u64(2, 3, b64) + [
            "s_and_b32 s4, 0xf0, 0x33",
            "s_mov_b32 s5, scc",
        ]
        for idx, (name, _) in enumerate(ops64):
            pair_lo = 10 + idx * 2
            scc_reg = 30 + idx
            insns64.append(f"s_{name}_b64 s[{pair_lo}:{pair_lo + 1}], s[0:1], s[2:3]")
            insns64.append(f"s_mov_b32 s{scc_reg}, scc")
        insns64 += ["s_xor_b64 vcc, s[0:1], s[0:1]", "s_endpgm"]
        res64 = self.run_items(insns64, wave=64)
        self.assertEqual(res64["sgpr"][4], 0x30)
        self.assertEqual(res64["sgpr"][5], 1)
        for idx, (name, want) in enumerate(ops64):
            pair_lo = 10 + idx * 2
            got = (res64["sgpr"][pair_lo + 1] << 32) | res64["sgpr"][pair_lo]
            self.assertEqual(got, want, name)
            self.assertEqual(res64["sgpr"][30 + idx], 1 if want else 0, name)
        self.assertEqual(res64["vcc"], 0)
        self.assertEqual(res64["scc"], 0)

    def test_saveexec_all_variants_wave32_and_wave64(self):
        # All 10 saveexec variants in wave32 and wave64, checking saved old EXEC, new EXEC, and SCC.
        s0_32, e0_32 = 0x0F0F0F0F, 0x00FF00FF
        s0_64, e0_64 = 0x123456789ABCDEF0, 0x00FF00FF0F0F0F0F
        variants = {
            "and": lambda s, e, m: s & e,
            "or": lambda s, e, m: s | e,
            "xor": lambda s, e, m: s ^ e,
            "andn2": lambda s, e, m: s & ~e & m,
            "orn2": lambda s, e, m: (s | ~e) & m,
            "nand": lambda s, e, m: ~(s & e) & m,
            "nor": lambda s, e, m: ~(s | e) & m,
            "xnor": lambda s, e, m: ~(s ^ e) & m,
            "andn1": lambda s, e, m: ~s & e & m,
            "orn1": lambda s, e, m: (~s | e) & m,
        }
        for name, fn in variants.items():
            with self.subTest(wave=32, variant=name):
                res = self.run_items([
                    f"s_mov_b32 exec_lo, 0x{e0_32:08x}",
                    f"s_mov_b32 s0, 0x{s0_32:08x}",
                    f"s_{name}_saveexec_b32 s0, s0",  # alias dst and src: must read old s0 and save old EXEC
                    "s_endpgm",
                ], wave=32)
                want = fn(s0_32, e0_32, MASK32) & MASK32
                self.assertEqual(res["sgpr"][0], e0_32)
                self.assertEqual(res["exec"], want)
                self.assertEqual(res["scc"], 1 if want else 0)

            with self.subTest(wave=64, variant=name):
                res = self.run_items(
                    load_u64(0, 1, e0_64) + load_u64(2, 3, s0_64) + [
                        "s_mov_b64 exec, s[0:1]",
                        f"s_{name}_saveexec_b64 s[4:5], s[2:3]",
                        "s_endpgm",
                    ],
                    wave=64,
                )
                want = fn(s0_64, e0_64, MASK64) & MASK64
                self.assertEqual((res["sgpr"][5] << 32) | res["sgpr"][4], e0_64)
                self.assertEqual(res["exec"], want)
                self.assertEqual(res["scc"], 1 if want else 0)

        # Explicitly verify SCC transitions to 0 when new EXEC becomes 0 (while old EXEC was non-zero).
        res_zero = self.run_items([
            "s_cmp_eq_u32 1, 1",
            "s_mov_b32 exec_lo, 0x0f0f",
            "s_and_saveexec_b32 s0, 0x00f0",
            "s_endpgm",
        ], wave=32)
        self.assertEqual(res_zero["sgpr"][0], 0x0F0F)
        self.assertEqual(res_zero["exec"], 0)
        self.assertEqual(res_zero["scc"], 0)

    def test_scalar_add_sub_u32_and_i32_scc_carry_borrow_overflow(self):
        cases = [
            ("s_add_u32", "0xffffffff", "1", 0, 1),
            ("s_add_u32", "0x7fffffff", "1", 0x80000000, 0),
            ("s_sub_u32", "0", "1", 0xFFFFFFFF, 1),
            ("s_sub_u32", "1", "2", 0xFFFFFFFF, 1),
            ("s_sub_u32", "2", "2", 0, 0),
            ("s_sub_u32", "5", "2", 3, 0),
            ("s_add_i32", "0x7fffffff", "1", 0x80000000, 1),     # +max + +1 -> signed overflow
            ("s_add_i32", "0x80000000", "-1", 0x7FFFFFFF, 1),    # -min + -1 -> signed overflow
            ("s_add_i32", "-1", "1", 0, 0),                      # opposite signs -> no signed overflow
            ("s_add_i32", "-2", "-3", 0xFFFFFFFB, 0),            # negative without overflow
            ("s_sub_i32", "0x80000000", "1", 0x7FFFFFFF, 1),     # INT_MIN - 1 -> signed overflow
            ("s_sub_i32", "0x7fffffff", "-1", 0x80000000, 1),    # INT_MAX - (-1) -> signed overflow
            ("s_sub_i32", "0", "1", 0xFFFFFFFF, 0),              # same sign -> no signed overflow
            ("s_sub_i32", "-1", "-2", 1, 0),                     # same sign -> no signed overflow
        ]
        for op, a, b, want_val, want_scc in cases:
            with self.subTest(op=op, a=a, b=b):
                res = self.run_items([
                    f"s_mov_b32 s0, {a}",
                    f"s_mov_b32 s1, {b}",
                    f"{op} s2, s0, s1",
                    "s_endpgm",
                ], wave=32)
                self.assertEqual(res["sgpr"][2], want_val)
                self.assertEqual(res["scc"], want_scc)

    def test_s_cselect_b32_selects_and_preserves_scc(self):
        res = self.run_items([
            "s_cmp_eq_u32 3, 3",             # SCC = 1
            "s_cselect_b32 s0, 11, 22",      # picks 11, SCC stays 1
            "s_cselect_b32 s1, 33, 44",      # still sees SCC = 1 -> picks 33
            "s_mov_b32 s2, scc",
            "s_cmp_eq_u32 3, 4",             # SCC = 0
            "s_cselect_b32 s3, 11, 22",      # picks 22, SCC stays 0
            "s_cselect_b32 s4, 33, 44",      # still sees SCC = 0 -> picks 44
            "s_mov_b32 s5, scc",
            "s_endpgm",
        ], wave=32)
        self.assertEqual(res["sgpr"], {0: 11, 1: 33, 2: 1, 3: 22, 4: 44, 5: 0})
        self.assertEqual(res["scc"], 0)

    def test_vector_alu_and_cpu_ref_fallback_respect_exec(self):
        # Active only on lane 1 (mask 0x2); lane 0 must retain its pre-mask sentinel values.
        res = self.run_items([
            "v_mov_b32 v0, 10",
            "v_mov_b32 v1, 3",
            "v_mov_b32 v2, 77",
            "v_mov_b32 v3, 77",
            "v_mov_b32 v4, 77",
            "v_mov_b32 v5, 77",
            "s_mov_b32 exec_lo, 0x2",
            "v_add_nc_u32 v2, v0, v1",
            "v_sub_nc_u32 v3, v0, v1",
            "v_subrev_nc_u32 v4, v0, v1",
            "v_mul_lo_u32 v5, v0, v1",
            "v_xor_b32 v2, 1, v2",               # falls through to cpu_ref.Machine.step
            "s_mov_b32 exec_lo, 0x3",
            "exp mrt0 v2, v3, v4, v5 done vm",
            "s_endpgm",
        ], wave=32)
        self.assertEqual(res["exports"][0], (77, 77, 77, 77))
        self.assertEqual(res["exports"][1], (12, 7, 0xFFFFFFF9, 30))


class ExportHaltStepAndErrorTests(CfRefTestCase):
    def test_exp_records_only_active_lanes_across_divergent_exports(self):
        # Lanes 0..7 take the then-branch and export (lane, 100, 0, 0); lanes 8..15 take else and export (lane, 200, 0, 0);
        # lanes 16..31 stay inactive during both exports and must not appear in res["exports"].
        res = self.run_items(lane_id_prologue(32) + [
            "s_mov_b32 exec_lo, 0x0000ffff",
            "v_cmp_lt_u32_e32 vcc_lo, v0, 8",
            "s_and_saveexec_b32 s0, vcc_lo",
            "v_mov_b32 v1, 100",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_xor_b32 exec_lo, exec_lo, s0",
            "v_mov_b32 v1, 200",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_or_b32 exec_lo, exec_lo, s0",
            "s_endpgm",
        ], wave=32)
        self.assertEqual(sorted(res["exports"]), list(range(16)))
        for lane in range(8):
            self.assertEqual(res["exports"][lane], (lane, 100, 0, 0))
        for lane in range(8, 16):
            self.assertEqual(res["exports"][lane], (lane, 200, 0, 0))

        # Exporting with EXEC == 0 records nothing.
        res_empty = self.run_items([
            "s_mov_b32 exec_lo, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ], wave=32)
        self.assertEqual(res_empty["exports"], {})

    def test_s_endpgm_halts_before_trailing_instructions(self):
        res = self.run_items([
            "s_mov_b32 s0, 42",
            "s_endpgm",
            "s_mov_b32 s0, 99",
            "v_pk_fma_f16 v0, v1, v2, v3",
            "exp mrt0 v0, v1, v2, v3 done vm",
        ], wave=32)
        self.assertEqual(res["sgpr"], {0: 42})
        self.assertEqual(res["exports"], {})
        self.assertEqual(res["steps"], 2)

    def test_max_steps_boundary_and_exceeded(self):
        # Exactly 5 dynamic instructions: s_mov, s_add, s_cmp, s_cbranch (not taken), s_endpgm.
        items = [
            "s_mov_b32 s0, 0",
            ("lbl", "loop"),
            "s_add_u32 s0, s0, 1",
            "s_cmp_lt_u32 s0, 1",
            ("br", "s_cbranch_scc1", "loop"),
            "s_endpgm",
        ]
        ok = self.run_items(items, max_steps=5)
        self.assertEqual(ok["steps"], 5)
        self.assertEqual(ok["sgpr"], {0: 1})

        with self.assertRaises(cf_ref.ProgramError) as cm:
            self.run_items(items, max_steps=4)
        self.assertIn("step limit 4", str(cm.exception))

        # Tight infinite loop also stops cleanly with ProgramError at max_steps.
        with self.assertRaises(cf_ref.ProgramError):
            self.run_raw([f"\ts_branch -1 {_enc_bytes(4)}", f"\ts_endpgm {_enc_bytes(4)}"], max_steps=10)

        with self.assertRaises(ValueError):
            self.run_items(["s_endpgm"], max_steps=0)

    def test_unsupported_instructions_and_operands_raise_cpu_ref_unsupported(self):
        self.assertIs(cf_ref.Unsupported, cpu_ref.Unsupported)
        self.assertIs(cf_ref.ProgramError, cpu_ref.ProgramError)
        unsupported_programs = [
            (["v_pk_fma_f16 v0, v1, v2, v3", "s_endpgm"], 32),
            (["ds_read_b32 v0, v1", "s_endpgm"], 32),
            (["s_waitcnt vmcnt(0)", "s_endpgm"], 32),
            (["s_andn1_wrexec_b32 s0, vcc_lo", "s_endpgm"], 32),
            (["s_cmp_eq_u64 s[0:1], s[2:3]", "s_endpgm"], 64),
            (["s_cmp_eq_u32 v0, 0", "s_endpgm"], 32),
            (["v_cmp_eq_f32_e32 vcc, v0, v1", "s_endpgm"], 32),
            (["v_cmpx_eq_u32_e64 s[0:1], v0, v1", "s_endpgm"], 64),
            (["exp pos0 v0, v1, v2, v3 done", "s_endpgm"], 32),
            (["exp mrt0 off, off, off, off done vm", "s_endpgm"], 32),
            (["s_branch L_missing", "s_endpgm"], 32),
            (["s_branch s0", "s_endpgm"], 32),
            (["s_mov_saveexec_b32 s0, vcc_lo", "s_endpgm"], 32),
            (["s_and_b32 vcc_lo, vcc_lo, scc", "s_endpgm"], 32),
            (["v_cmp_eq_u32_e64 v0, v0, 0", "s_endpgm"], 32),
            (["v_cmp_eq_u32_e32 vcc, v0", "s_endpgm"], 32),
            (["v_add_nc_u32_e64 v2, v1, 1 clamp", "s_endpgm"], 32),
            (["exp mrt0 v0, v0, v1, v1 done compr vm", "s_endpgm"], 32),
            (["v_cmp_eq_u32_e32 vcc_lo, v0, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf", "s_endpgm"], 32),
            (["s_mov_b32 vcc, 1", "s_endpgm"], 64),
            (["s_mov_b32 exec, 1", "s_endpgm"], 64),
        ]
        for insns, wave in unsupported_programs:
            with self.subTest(insn=insns[0], wave=wave):
                with self.assertRaises(cpu_ref.Unsupported):
                    self.run_items(insns, wave=wave)

    def test_program_errors_for_missing_encodings_header_endpgm_and_bad_branch_targets(self):
        # Missing ; encoding: comment or empty ; encoding: []
        with self.assertRaises(cf_ref.ProgramError):
            self.run_raw(["\ts_endpgm"])
        with self.assertRaises(cf_ref.ProgramError):
            self.run_raw(["\ts_endpgm ; encoding: []"])
        # Missing header line
        with self.assertRaises(cf_ref.ProgramError):
            self.run_raw([f"\ts_endpgm {_enc_bytes(4)}"], header=None)
        # Empty listing (no instructions)
        with self.assertRaises(cf_ref.ProgramError):
            self.run_raw([])
        # Missing s_endpgm
        with self.assertRaises(cf_ref.ProgramError):
            self.run_raw([f"\ts_mov_b32 s0, 1 {_enc_bytes(4)}"])
        # Branch out of bounds
        with self.assertRaises(cf_ref.ProgramError):
            self.run_raw([f"\ts_branch 10 {_enc_bytes(4)}", f"\ts_endpgm {_enc_bytes(4)}"])
        # Branch into the middle of an 8-byte instruction
        with self.assertRaises(cf_ref.ProgramError):
            self.run_raw([
                f"\ts_branch 1 {_enc_bytes(4)}",
                f"\tv_mul_lo_u32 v0, v0, 2 {_enc_bytes(8)}",
                f"\ts_endpgm {_enc_bytes(4)}",
            ])


if __name__ == "__main__":
    unittest.main()
