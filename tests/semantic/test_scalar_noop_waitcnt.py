"""Semantic and translation tests for RDNA2 scalar no-op, hint, waitcnt, and mode instructions.

Covers the 11 named scalar instructions and their syntax variants in tools/shader_ir.py and tools/rdna2spirv.py:
  * In rdna2spirv.NOOPS (emits no SPIR-V instructions, preserves VGPRs, SGPRs, VCC, SCC, and EXEC):
    s_clause, s_delay_alu (RDNA3 hint kept in NOOPS), s_inst_prefetch, s_setprio, s_sleep,
    s_waitcnt_depctr, s_waitcnt_expcnt, s_waitcnt_lgkmcnt, s_waitcnt_vmcnt,
    plus the sibling wait/no-op forms s_waitcnt, s_waitcnt_vscnt, s_nop, and v_nop.
  * s_round_mode and s_setreg_imm32_b32 can select nearest-even or toward-zero FP16/64 rounding while
    retaining nearest-even FP32. These writes leave VGPRs, SGPRs, VCC, SCC, and EXEC intact. Tests also cover
    rejection of unmodelled modes and the no-op behavior of non-MODE s_setreg* writes.

Standard library only (runs with python -I); hand-written RDNA2 listings, no GPU or game data.
"""
from __future__ import annotations

import gc
import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
for _p in (HERE, TOOLS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rdna2spirv  # noqa: E402
import shader_ir  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

M32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
# Raw buffer V# (OOB_SELECT 3, stride 0): any dword offset is in bounds in spirv_mini_interp.
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]

# Handled scalar no-op/hint/waitcnt instructions and explicit syntax variants accepted by shader_ir.py.
SCALAR_NOOP_VARIANTS = [
    "s_clause 0x1",
    "s_clause 0",
    "s_clause 15",
    "s_delay_alu 0",
    "s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)",
    "s_delay_alu instid0(VALU_DEP_2)",
    "s_inst_prefetch 0x0",
    "s_inst_prefetch 0x1",
    "s_inst_prefetch -1",
    "s_setprio 0x0",
    "s_setprio 0x2",
    "s_setprio 3",
    "s_sleep 0x0",
    "s_sleep 0x1",
    "s_sleep 7",
    "s_waitcnt_depctr 0xffe3",
    "s_waitcnt_depctr 0x0",
    "s_waitcnt_depctr depctr_vm_vsrc(0) depctr_va_sdst(0)",
    "s_waitcnt_depctr depctr_hold_cnt(0) depctr_sa_sdst(0) depctr_va_vdst(0)",
    "s_waitcnt_expcnt null, 0x0",
    "s_waitcnt_expcnt null, 0x7",
    "s_waitcnt_expcnt 0x0",
    "s_waitcnt_lgkmcnt null, 0x0",
    "s_waitcnt_lgkmcnt null, 0x3f",
    "s_waitcnt_lgkmcnt 0x0",
    "s_waitcnt_vmcnt null, 0x0",
    "s_waitcnt_vmcnt null, 0x3f",
    "s_waitcnt_vmcnt 0x0",
    # Sibling wait/no-op and non-MODE setreg forms handled alongside them in NOOPS / mode_write.
    "s_waitcnt vmcnt(0)",
    "s_waitcnt expcnt(0)",
    "s_waitcnt lgkmcnt(0)",
    "s_waitcnt vmcnt(1) lgkmcnt(2)",
    "s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)",
    "s_waitcnt 0",
    "s_waitcnt_vscnt null, 0x0",
    "s_waitcnt_vscnt vscnt(0)",
    "s_nop 0",
    "s_nop 0x7",
    "v_nop",
    "v_nop_e32",
    "v_nop_e64",
    "s_setreg_imm32_b32 hwreg(HW_REG_FLAT_SCR_LO, 0, 32), 0",
    "s_setreg_b32 hwreg(HW_REG_IB_STS), s2",
]

# Representative instruction per handled mnemonic from the 9-op scalar NOOPS group + s_waitcnt / s_nop.
REPRESENTATIVE_NOOPS = [
    "s_clause 0x1",
    "s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)",
    "s_inst_prefetch 0x0",
    "s_setprio 0x2",
    "s_sleep 0x1",
    "s_waitcnt_depctr 0xffe3",
    "s_waitcnt_expcnt null, 0x0",
    "s_waitcnt_lgkmcnt null, 0x0",
    "s_waitcnt_vmcnt null, 0x0",
    "s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)",
    "s_waitcnt_vscnt null, 0x0",
    "s_nop 0",
]

REFUSED_MODE_VARIANTS = [
    "s_denorm_mode 0x3",
    "s_denorm_mode 0",
    "s_round_mode 0x3",
    "s_round_mode 1",
    "s_round_mode 2",
    "s_round_mode 4",
    "s_round_mode 8",
]

MODELLED_MODE_VARIANTS = [
    ("s_round_mode 0", "%c0"),
    ("s_round_mode 12", "%c3"),
    ("s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0", "%c0"),
    ("s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 3", "%c3"),
]

_TMP_DIR: tempfile.TemporaryDirectory | None = None


def setUpModule() -> None:
    global _TMP_DIR
    _TMP_DIR = tempfile.TemporaryDirectory()
    os.mkdir(os.path.join(_TMP_DIR.name, "disasm"))
    os.mkdir(os.path.join(_TMP_DIR.name, "fssm"))


def tearDownModule() -> None:
    global _TMP_DIR
    if _TMP_DIR is not None:
        gc.collect()
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def _ensure_dir() -> str:
    if _TMP_DIR is None:
        setUpModule()
    assert _TMP_DIR is not None
    return _TMP_DIR.name


def make_fssm(threads: int, user_sgprs: int = 4) -> bytes:
    """Synthetic compute FSSM header: user_sgprs USER_SGPRs, TGID_X_EN=0, TIDIG_COMP_CNT=0, size=(threads,1,1)."""
    blob = bytearray(0x148)
    rsrc2 = (user_sgprs & 31) << 1
    struct.pack_into("<4I", blob, 0x138, rsrc2, threads, 1, 1)
    return bytes(blob)


def translate_ps(name: str, lines: list[str]) -> str:
    root = _ensure_dir()
    src = os.path.join(root, "disasm", name + ".s")
    with open(src, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"; {name}\n" + "".join(f"\t{ln}\n" for ln in lines))
    with open(os.path.join(root, name + ".dxbc"), "wb") as f:
        f.write(STUB_DXBC)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(src)


def translate_cs(name: str, lines: list[str], threads: int) -> str:
    root = _ensure_dir()
    src = os.path.join(root, "disasm", name + ".s")
    with open(src, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"; {name}\n" + "".join(f"\t{ln}\n" for ln in lines))
    with open(os.path.join(root, name + ".dxbc"), "wb") as f:
        f.write(STUB_DXBC)
    with open(os.path.join(root, "fssm", name + ".fssm"), "wb") as f:
        f.write(make_fssm(threads))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(src)


def ps_program(middle: list[str], *, scc_one: bool = True, vcc_one: bool = True,
               exec_lo: int = 0xAAAA5555, add_k: int = 0x10) -> list[str]:
    """Synthetic PS testing VGPRs, SGPRs, SCC, VCC, and EXEC across interleaved arithmetic and `middle`."""
    s_rhs = 5 if scc_one else 0
    s_lhs = 0xFFFFFFFF if scc_one else 0x12340000
    cmp_val = (0x11112222 + add_k) & M32 if vcc_one else 0x00000000
    return [
        "v_mov_b32 v0, 0x11112222",
        "v_mov_b32 v1, 0x33334444",
        "v_mov_b32 v2, 0x55556666",
        "v_mov_b32 v3, 0x77778888",
        f"s_mov_b32 s2, {s_lhs:#x}",
        f"s_mov_b32 s3, {s_rhs:#x}",
        "s_add_u32 s4, s2, s3",                      # SCC = 1 iff s_lhs + s_rhs overflows 32 bits
        f"v_add_nc_u32 v0, v0, {add_k:#x}",
        f"v_cmp_eq_u32 vcc_lo, v0, {cmp_val:#x}",    # VCC = 1 iff vcc_one
        f"s_mov_b32 exec_lo, {exec_lo:#x}",
        *middle,
        "s_cselect_b32 s5, 0x10000001, 0x20000002",  # reads SCC
        "s_add_u32 s6, s4, s5",                      # reads SGPR s4 and selected s5
        "v_cndmask_b32 v1, 0x99990000, v1, vcc_lo",  # reads VCC (and VGPR v1 when VCC=1)
        "v_add_nc_u32 v2, v2, s6",                   # reads EXEC, VGPR v2, SGPR s6
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]


def expected_ps_mrt0(*, scc_one: bool = True, vcc_one: bool = True,
                     exec_lo: int = 0xAAAA5555, add_k: int = 0x10) -> tuple[int, int, int, int]:
    """Plain-Python reference for ps_program()."""
    active = bool(exec_lo & 1)
    s_lhs = 0xFFFFFFFF if scc_one else 0x12340000
    s_rhs = 5 if scc_one else 0
    s4 = (s_lhs + s_rhs) & M32
    scc = (s_lhs + s_rhs) > M32
    v0 = ((0x11112222 + add_k) & M32)          # executed while EXEC=1 initially
    vcc = vcc_one                              # evaluated while EXEC=1 initially
    s5 = 0x10000001 if scc else 0x20000002
    s6 = (s4 + s5) & M32
    v1 = (0x33334444 if vcc else 0x99990000) if active else 0x33334444
    v2 = ((0x55556666 + s6) & M32) if active else 0x55556666
    v3 = 0x77778888
    if not active:
        return (0, 0, 0, 0)
    return (v0, v1, v2, v3)


def cs_program(middle: list[str], *, wave64: bool, scc_one: bool = True,
               exec_lo: int = 0xA5A55A5A, exec_hi: int = 0x3C3C6969) -> list[str]:
    """Synthetic wave32/wave64 CS checking VGPRs, SGPRs, SCC, VCC, and EXEC across all lanes."""
    s_lhs = 0xFFFFFFFF if scc_one else 0x10203040
    s_rhs = 3 if scc_one else 1
    if wave64:
        lane_setup = [
            "s_mov_b64 exec, -1",
            "v_mbcnt_lo_u32_b32 v0, -1, 0",
            "v_mbcnt_hi_u32_b32 v0, -1, v0",
        ]
        vcc_cmp = "v_cmp_eq_u32 vcc, v4, 0"
        set_exec = [f"s_mov_b32 exec_lo, {exec_lo:#x}", f"s_mov_b32 exec_hi, {exec_hi:#x}"]
        reset_exec = "s_mov_b64 exec, -1"
        vcc_cnd = "v_cndmask_b32 v2, 0x99990000, v2, vcc"
    else:
        lane_setup = []                        # v0 is seeded directly from LocalInvocationId.x
        vcc_cmp = "v_cmp_eq_u32 vcc_lo, v4, 0"
        set_exec = [f"s_mov_b32 exec_lo, {exec_lo:#x}"]
        reset_exec = "s_mov_b32 exec_lo, -1"
        vcc_cnd = "v_cndmask_b32 v2, 0x99990000, v2, vcc_lo"

    return [
        *lane_setup,
        "v_mov_b32 v1, v0",
        "v_add_nc_u32 v1, v1, 10",
        "v_mov_b32 v2, 0x77770000",
        "v_add_nc_u32 v2, v2, v0",
        "v_mov_b32 v3, 0x1000",
        "v_and_b32 v4, v0, 1",
        vcc_cmp,                               # even lanes: VCC = 1; odd lanes: VCC = 0
        "s_mov_b32 s8, 0x01020304",
        "s_mov_b32 s9, 0x55667788",
        f"s_mov_b32 s10, {s_lhs:#x}",
        f"s_add_u32 s11, s10, {s_rhs:#x}",     # SCC = 1 iff scc_one
        *set_exec,
        *middle,
        "v_add_nc_u32 v1, v1, 0x100",          # updates v1 only in lanes where EXEC=1
        "s_cselect_b32 s12, 0x10000, 0x20000",  # reads SCC
        reset_exec,
        vcc_cnd,                               # reads VCC across all lanes (even keeps v2, odd gets 0x99990000)
        "v_add_nc_u32 v3, v3, s8",             # reads SGPR s8
        "v_add_nc_u32 v3, v3, s12",            # reads s12 (SCC outcome)
        "v_add_nc_u32 v3, v3, s11",            # reads s11 (s_add_u32 sum)
        "v_mov_b32 v4, s9",                    # reads SGPR s9
        "v_lshlrev_b32 v5, 4, v0",             # byte offset = 16 * lane
        "buffer_store_dwordx4 v[1:4], v5, s[0:3], 0 offen",
        "s_endpgm",
    ]


def expected_cs_words(lanes: int, *, scc_one: bool = True,
                      exec_lo: int = 0xA5A55A5A, exec_hi: int = 0x3C3C6969) -> list[tuple[int, int, int, int]]:
    """Plain-Python reference for cs_program()."""
    s_lhs = 0xFFFFFFFF if scc_one else 0x10203040
    s_rhs = 3 if scc_one else 1
    s11 = (s_lhs + s_rhs) & M32
    s12 = 0x10000 if ((s_lhs + s_rhs) > M32) else 0x20000
    exec_mask = (exec_hi << 32) | exec_lo
    out = []
    for lane in range(lanes):
        active = bool((exec_mask >> lane) & 1)
        w0 = (lane + 10 + (0x100 if active else 0)) & M32
        w1 = (0x77770000 + lane) if (lane % 2 == 0) else 0x99990000
        w2 = (0x1000 + 0x01020304 + s12 + s11) & M32
        w3 = 0x55667788
        out.append((w0, w1, w2, w3))
    return out


def run_cs(spv: str, lanes: int) -> list[tuple[int, int, int, int]]:
    mem = si.run_workgroup(
        spv, lanes, lanes,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=VSHARP + [0] * 28,
        memory={},
    )
    return [tuple(mem[16 * lane + 4 * c] for c in range(4)) for lane in range(lanes)]


class ScalarNoopWaitcntParsingTests(unittest.TestCase):
    """Exact syntax and operand parsing in tools/shader_ir.py for all 11 scalar ops and variants."""

    def test_parse_all_scalar_noop_waitcnt_and_mode_forms(self):
        cases = [
            ("s_clause 0x1", "s_clause", [{"k": "imm", "v": 1}]),
            ("s_delay_alu 0", "s_delay_alu", [{"k": "imm", "v": 0}]),
            (
                "s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)",
                "s_delay_alu",
                [{"k": "special", "t": "instid0(VALU_DEP_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)"}],
            ),
            ("s_denorm_mode 0x3", "s_denorm_mode", [{"k": "imm", "v": 3}]),
            ("s_inst_prefetch 0x0", "s_inst_prefetch", [{"k": "imm", "v": 0}]),
            ("s_inst_prefetch -1", "s_inst_prefetch", [{"k": "imm", "v": -1}]),
            ("s_round_mode 0x3", "s_round_mode", [{"k": "imm", "v": 3}]),
            ("s_setprio 0x2", "s_setprio", [{"k": "imm", "v": 2}]),
            ("s_sleep 0x1", "s_sleep", [{"k": "imm", "v": 1}]),
            ("s_waitcnt_depctr 0xffe3", "s_waitcnt_depctr", [{"k": "imm", "v": 0xFFE3}]),
            (
                "s_waitcnt_depctr depctr_vm_vsrc(0) depctr_va_sdst(0)",
                "s_waitcnt_depctr",
                [{"k": "special", "t": "depctr_vm_vsrc(0)"}, {"k": "special", "t": "depctr_va_sdst(0)"}],
            ),
            (
                "s_waitcnt_depctr depctr_hold_cnt(0) depctr_sa_sdst(0) depctr_va_vdst(0)",
                "s_waitcnt_depctr",
                [
                    {"k": "special", "t": "depctr_hold_cnt(0)"},
                    {"k": "special", "t": "depctr_sa_sdst(0)"},
                    {"k": "special", "t": "depctr_va_vdst(0)"},
                ],
            ),
            (
                "s_waitcnt_expcnt null, 0x0",
                "s_waitcnt_expcnt",
                [{"k": "special", "t": "null", "neg": False, "abs": False}, {"k": "imm", "v": 0}],
            ),
            (
                "s_waitcnt_lgkmcnt null, 0x3f",
                "s_waitcnt_lgkmcnt",
                [{"k": "special", "t": "null", "neg": False, "abs": False}, {"k": "imm", "v": 0x3F}],
            ),
            (
                "s_waitcnt_vmcnt null, 0x0",
                "s_waitcnt_vmcnt",
                [{"k": "special", "t": "null", "neg": False, "abs": False}, {"k": "imm", "v": 0}],
            ),
            (
                "s_waitcnt_vscnt null, 0x0",
                "s_waitcnt_vscnt",
                [{"k": "special", "t": "null", "neg": False, "abs": False}, {"k": "imm", "v": 0}],
            ),
            ("s_waitcnt vmcnt(0)", "s_waitcnt", [{"k": "special", "t": "vmcnt(0)"}]),
            ("s_waitcnt expcnt(0)", "s_waitcnt", [{"k": "special", "t": "expcnt(0)"}]),
            ("s_waitcnt lgkmcnt(0)", "s_waitcnt", [{"k": "special", "t": "lgkmcnt(0)"}]),
            (
                "s_waitcnt vmcnt(1) lgkmcnt(2)",
                "s_waitcnt",
                [{"k": "special", "t": "vmcnt(1)"}, {"k": "special", "t": "lgkmcnt(2)"}],
            ),
            (
                "s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)",
                "s_waitcnt",
                [
                    {"k": "special", "t": "vmcnt(0)"},
                    {"k": "special", "t": "expcnt(0)"},
                    {"k": "special", "t": "lgkmcnt(0)"},
                ],
            ),
            ("s_waitcnt 0", "s_waitcnt", [{"k": "imm", "v": 0}]),
            (
                "s_setreg_b32 hwreg(HW_REG_MODE, 2, 2), s0",
                "s_setreg_b32",
                [
                    {"k": "special", "t": "hwreg(HW_REG_MODE, 2, 2)"},
                    {"k": "sgpr", "r": 0, "n": 1, "neg": False, "abs": False},
                ],
            ),
            (
                "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 3",
                "s_setreg_imm32_b32",
                [{"k": "special", "t": "hwreg(HW_REG_MODE, 2, 2)"}, {"k": "imm", "v": 3}],
            ),
            ("s_nop 0", "s_nop", [{"k": "imm", "v": 0}]),
            ("v_nop", "v_nop", []),
        ]
        for text, want_op, want_args in cases:
            with self.subTest(line=text):
                parsed = shader_ir.parse_line("\t" + text)
                self.assertIsNotNone(parsed)
                self.assertEqual(parsed["op"], want_op)
                self.assertEqual(parsed["args"], want_args)
                self.assertEqual(parsed["mods"], [])
                self.assertTrue(all(arg["k"] != "raw" for arg in parsed["args"]))


class PixelShaderNoopWaitcntTests(unittest.TestCase):
    """PS translation and spirv_mini_interp execution across all handled NOOPS and waitcnt variants."""

    def test_variants_emit_no_extra_spirv_and_preserve_state_active_scc1_vcc1(self):
        baseline = translate_ps("ps_noop_base_11", ps_program([], scc_one=True, vcc_one=True))
        want = expected_ps_mrt0(scc_one=True, vcc_one=True)
        self.assertEqual(want, (0x11112232, 0x33334444, 0x6555666B, 0x77778888))
        self.assertEqual(tuple(si.run_fragment(baseline, inputs={"%sg_lane": 0})["%out_mrt0"]), want)
        self.assertNotIn("OpControlBarrier", baseline)
        self.assertNotIn("OpMemoryBarrier", baseline)

        for variant in SCALAR_NOOP_VARIANTS:
            with self.subTest(insn=variant):
                spv = translate_ps("ps_noop_var", ps_program([variant], scc_one=True, vcc_one=True))
                self.assertEqual(spv, baseline, f"{variant} emitted unexpected SPIR-V operations")
                self.assertEqual(tuple(si.run_fragment(spv, inputs={"%sg_lane": 0})["%out_mrt0"]), want)

    def test_variants_preserve_scc0_and_vcc0(self):
        baseline = translate_ps("ps_noop_base_00", ps_program([], scc_one=False, vcc_one=False))
        want = expected_ps_mrt0(scc_one=False, vcc_one=False)
        self.assertEqual(want, (0x11112232, 0x99990000, 0x87896668, 0x77778888))
        self.assertEqual(tuple(si.run_fragment(baseline, inputs={"%sg_lane": 0})["%out_mrt0"]), want)

        for variant in SCALAR_NOOP_VARIANTS:
            with self.subTest(insn=variant):
                spv = translate_ps("ps_noop_var00", ps_program([variant], scc_one=False, vcc_one=False))
                self.assertEqual(spv, baseline)
                self.assertEqual(tuple(si.run_fragment(spv, inputs={"%sg_lane": 0})["%out_mrt0"]), want)

    def test_interleaved_across_every_arithmetic_step(self):
        base_lines = ps_program([], scc_one=True, vcc_one=True)
        baseline = translate_ps("ps_interleave_base", base_lines)
        want = expected_ps_mrt0(scc_one=True, vcc_one=True)
        interleaved = [
            "v_mov_b32 v0, 0x11112222",
            "v_mov_b32 v1, 0x33334444",
            "v_mov_b32 v2, 0x55556666",
            "v_mov_b32 v3, 0x77778888",
            "s_clause 0x1",
            "s_mov_b32 s2, 0xffffffff",
            "s_mov_b32 s3, 5",
            "s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)",
            "s_add_u32 s4, s2, s3",
            "s_inst_prefetch 0x0",
            "s_setprio 0x2",
            "v_add_nc_u32 v0, v0, 0x10",
            "s_sleep 0x1",
            "v_cmp_eq_u32 vcc_lo, v0, 0x11112232",
            "s_waitcnt_depctr depctr_vm_vsrc(0) depctr_va_sdst(0)",
            "s_mov_b32 exec_lo, 0xaaaa5555",
            "s_waitcnt_expcnt null, 0x0",
            "s_waitcnt_lgkmcnt null, 0x0",
            "s_waitcnt_vmcnt null, 0x0",
            "s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)",
            "s_waitcnt_vscnt null, 0x0",
            "s_nop 0",
            "v_nop",
            "s_cselect_b32 s5, 0x10000001, 0x20000002",
            "s_add_u32 s6, s4, s5",
            "v_cndmask_b32 v1, 0x99990000, v1, vcc_lo",
            "v_add_nc_u32 v2, v2, s6",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_ps("ps_interleave", interleaved)
        self.assertEqual(spv, baseline)
        self.assertEqual(tuple(si.run_fragment(spv, inputs={"%sg_lane": 0})["%out_mrt0"]), want)

    def test_inactive_exec_stays_zero_for_vgpr_writes_and_exports(self):
        # Part 1: EXEC=0 around all handled no-ops blocks VGPR writes, then restoring EXEC exports intact values.
        base_vgpr = [
            "v_mov_b32 v0, 0x10000001",
            "v_mov_b32 v1, 0x20000002",
            "v_mov_b32 v2, 0x30000003",
            "v_mov_b32 v3, 0x40000004",
            "s_mov_b32 exec_lo, 0",
            "v_add_nc_u32 v2, v2, 0x1234",
            "v_mov_b32 v3, 0xdeadbeef",
            "s_mov_b32 exec_lo, -1",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        with_noops = base_vgpr[:5] + REPRESENTATIVE_NOOPS + base_vgpr[5:]
        spv_base = translate_ps("ps_exec0_vgpr_base", base_vgpr)
        spv_noops = translate_ps("ps_exec0_vgpr_noops", with_noops)
        self.assertEqual(spv_noops, spv_base)
        self.assertEqual(tuple(si.run_fragment(spv_noops, inputs={"%sg_lane": 0})["%out_mrt0"]),
                         (0x10000001, 0x20000002, 0x30000003, 0x40000004))

        # Part 2: EXEC=0 across all handled no-ops keeps the lane inactive through end of shader (OpKill -> {}).
        base_exp = [
            "v_mov_b32 v0, 0x10000001",
            "v_mov_b32 v1, 0x20000002",
            "v_mov_b32 v2, 0x30000003",
            "v_mov_b32 v3, 0x40000004",
            "exp mrt0 v0, v1, v2, v3 vm",
            "s_mov_b32 exec_lo, 0xaaaa5554",
            "v_mov_b32 v0, 0xbad00000",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv_exp_base = translate_ps("ps_exec0_exp_base", base_exp)
        spv_exp_noops = translate_ps("ps_exec0_exp_noops", base_exp[:6] + REPRESENTATIVE_NOOPS + base_exp[6:])
        self.assertEqual(spv_exp_noops, spv_exp_base)
        self.assertEqual(si.run_fragment(spv_exp_noops, inputs={"%sg_lane": 0}), {})


class ComputeShaderNoopWaitcntTests(unittest.TestCase):
    """FSSM compute shader translation and multi-lane execution (wave32 & wave64)."""

    def test_wave32_each_and_all_preserve_registers_vcc_scc_and_exec(self):
        for scc_one in (True, False):
            baseline = translate_cs("cs_w32_base", cs_program([], wave64=False, scc_one=scc_one), threads=32)
            want = expected_cs_words(32, scc_one=scc_one)
            self.assertEqual(run_cs(baseline, 32), want)
            self.assertNotIn("OpControlBarrier", baseline)
            self.assertNotIn("OpMemoryBarrier", baseline)

            for variant in SCALAR_NOOP_VARIANTS:
                with self.subTest(scc_one=scc_one, insn=variant):
                    spv = translate_cs("cs_w32_var", cs_program([variant], wave64=False, scc_one=scc_one), threads=32)
                    self.assertEqual(spv, baseline)
                    self.assertEqual(run_cs(spv, 32), want)

            spv_all = translate_cs("cs_w32_all", cs_program(REPRESENTATIVE_NOOPS, wave64=False, scc_one=scc_one),
                                   threads=32)
            self.assertEqual(spv_all, baseline)
            self.assertEqual(run_cs(spv_all, 32), want)

    def test_wave64_preserves_state_across_all_64_lanes(self):
        for scc_one in (True, False):
            baseline = translate_cs("cs_w64_base", cs_program([], wave64=True, scc_one=scc_one), threads=64)
            want = expected_cs_words(64, scc_one=scc_one)
            self.assertEqual(run_cs(baseline, 64), want)
            self.assertNotIn("OpControlBarrier", baseline)
            self.assertNotIn("OpMemoryBarrier", baseline)

            with self.subTest(scc_one=scc_one):
                spv_all = translate_cs("cs_w64_all", cs_program(REPRESENTATIVE_NOOPS, wave64=True, scc_one=scc_one),
                                       threads=64)
                self.assertEqual(spv_all, baseline)
                self.assertEqual(run_cs(spv_all, 64), want)


class ModeInstructionsTests(unittest.TestCase):
    """Tests for s_denorm_mode, s_round_mode, and s_setreg* hardware MODE writes."""

    def test_round_mode_switches_half_precision_color_rounding(self) -> None:
        # Each magnitude is three quarters of the way from half 1.0 to its next neighbour.
        spv = translate_ps("ps_immediate_round_mode", [
            "v_mov_b32 v0, 0x3f801800", "v_mov_b32 v1, 0xbf801800",
            "s_round_mode 0xfffc",  # only SIMM16[3:0] contributes to FP_ROUND
            "v_cvt_f16_f32 v2, v0", "v_cvt_f16_f32 v3, v1",
            "s_round_mode 0",
            "v_cvt_f16_f32 v4, v0", "v_cvt_f16_f32 v5, v1",
            "exp mrt0 v2, v3, v4, v5 done vm", "s_endpgm",
        ])
        self.assertEqual(tuple(si.run_fragment(spv)["%out_mrt0"]), (0x3c00, 0xbc00, 0x3c01, 0xbc01))

    def test_unmodelled_immediate_modes_raise_unsupported(self):
        for insn in REFUSED_MODE_VARIANTS:
            op = insn.split()[0]
            with self.subTest(stage="ps", insn=insn):
                with self.assertRaises(rdna2spirv.Unsupported) as ctx:
                    translate_ps("ps_refused_mode", ps_program([insn]))
                self.assertIn(op, str(ctx.exception))
                self.assertIn("mode" if op == "s_denorm_mode" else "rounding", str(ctx.exception))

            with self.subTest(stage="cs", insn=insn):
                with self.assertRaises(rdna2spirv.Unsupported) as ctx:
                    translate_cs("cs_refused_mode", cs_program([insn], wave64=False), threads=32)
                self.assertIn(op, str(ctx.exception))
                self.assertIn("mode" if op == "s_denorm_mode" else "rounding", str(ctx.exception))

    def test_fp_round_emits_only_mode_r16_and_preserves_state(self):
        ps_base = translate_ps("ps_mode_base", ps_program([]))
        want_ps = expected_ps_mrt0()
        cs_base = translate_cs("cs_mode_base", cs_program([], wave64=False), threads=32)
        want_cs = expected_cs_words(32)

        for insn, mode_const in MODELLED_MODE_VARIANTS:
            with self.subTest(stage="ps", insn=insn):
                spv = translate_ps("ps_mode_setreg", ps_program([insn]))
                self.assertIn("%mode_r16 = OpVariable %_ptr_Function_uint Function %c0", spv)
                self.assertIn(f"OpStore %mode_r16 {mode_const}", spv)
                self.assertNotIn("OpControlBarrier", spv)
                # Removing the two mode lines recovers the exact baseline SPIR-V (modulo unused %c3 if added).
                self.assertEqual(spv.count("OpStore %mode_r16"), 1)
                self.assertEqual(tuple(si.run_fragment(spv, inputs={"%sg_lane": 0})["%out_mrt0"]), want_ps)
                self.assertEqual(tuple(si.run_fragment(ps_base, inputs={"%sg_lane": 0})["%out_mrt0"]), want_ps)

            with self.subTest(stage="cs", insn=insn):
                spv = translate_cs("cs_mode_setreg", cs_program([insn], wave64=False), threads=32)
                self.assertIn("%mode_r16 = OpVariable %_ptr_Function_uint Function %c0", spv)
                self.assertIn(f"OpStore %mode_r16 {mode_const}", spv)
                self.assertEqual(run_cs(spv, 32), want_cs)
                self.assertEqual(run_cs(cs_base, 32), want_cs)

    def test_unmodelled_s_setreg_mode_forms_raise_unsupported(self):
        bad_mode_writes = [
            "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 0, 2), 3",
            "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 4), 15",
            "s_setreg_imm32_b32 hwreg(HW_REG_MODE), 0",
            "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 1",
            "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 2",
            "s_setreg_b32 hwreg(HW_REG_MODE, 2, 2), s2",
        ]
        for insn in bad_mode_writes:
            with self.subTest(insn=insn):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate_ps("ps_bad_mode", ps_program([insn]))


class NegativeControlTests(unittest.TestCase):
    """Negative controls: sabotaging arithmetic or using unknown mnemonics must be detected."""

    def test_sabotaging_arithmetic_around_noop_waitcnt_changes_output(self):
        good_spv = translate_ps("ps_neg_good", ps_program(["s_waitcnt_vmcnt null, 0x0", "s_clause 0x1"]))
        good_out = si.run_fragment(good_spv, inputs={"%sg_lane": 0})
        self.assertEqual(tuple(good_out["%out_mrt0"]), expected_ps_mrt0(scc_one=True, vcc_one=True))

        # Assembly-level sabotage around the waitcnt/clause instructions:
        sabotaged_cases = [
            ("SCC flipped 1->0", ps_program(["s_waitcnt_vmcnt null, 0x0", "s_clause 0x1"], scc_one=False)),
            ("VCC flipped 1->0", ps_program(["s_waitcnt_vmcnt null, 0x0", "s_clause 0x1"], vcc_one=False)),
            ("v_add_nc_u32 addend changed", ps_program(["s_waitcnt_vmcnt null, 0x0", "s_clause 0x1"], add_k=0x20)),
            ("EXEC bit 0 cleared", ps_program(["s_waitcnt_vmcnt null, 0x0", "s_clause 0x1"], exec_lo=0xAAAA5554)),
        ]
        for label, bad_lines in sabotaged_cases:
            with self.subTest(sabotage=label):
                bad_spv = translate_ps("ps_neg_bad", bad_lines)
                bad_out = si.run_fragment(bad_spv, inputs={"%sg_lane": 0})
                self.assertNotEqual(bad_out, good_out)

        # SPIR-V-level sabotage on the translated output around the no-op/waitcnt block:
        for old, new, label in (
            ("OpIAdd %uint", "OpISub %uint", "v_add_nc_u32/s_add_u32 turned into OpISub"),
            ("OpIEqual %bool", "OpINotEqual %bool", "v_cmp_eq_u32 inverted"),
        ):
            with self.subTest(spirv_sabotage=label):
                self.assertIn(old, good_spv)
                mutated = good_spv.replace(old, new)
                self.assertNotEqual(si.run_fragment(mutated, inputs={"%sg_lane": 0}), good_out)

        # CS multi-lane negative control:
        cs_good = translate_cs("cs_neg_good", cs_program(["s_sleep 0x1", "s_waitcnt_depctr 0xffe3"], wave64=False),
                               threads=32)
        cs_good_words = run_cs(cs_good, 32)
        cs_bad = translate_cs("cs_neg_bad", cs_program(["s_sleep 0x1", "s_waitcnt_depctr 0xffe3"],
                                                       wave64=False, scc_one=False), threads=32)
        self.assertNotEqual(run_cs(cs_bad, 32), cs_good_words)

    def test_unknown_waitcnt_and_noop_mnemonics_fail_with_unsupported(self):
        for bogus in ("s_bogus_waitcnt 0x0", "s_bogus_waitcnt", "s_bogus_noop 0"):
            with self.subTest(stage="ps", insn=bogus):
                with self.assertRaises(rdna2spirv.Unsupported) as ctx:
                    translate_ps("ps_bogus", ps_program([bogus]))
                self.assertIn(bogus.split()[0], str(ctx.exception))

            with self.subTest(stage="cs", insn=bogus):
                with self.assertRaises(rdna2spirv.Unsupported) as ctx:
                    translate_cs("cs_bogus", cs_program([bogus], wave64=False), threads=32)
                self.assertIn(bogus.split()[0], str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
