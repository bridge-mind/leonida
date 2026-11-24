"""Semantic SPIR-V and interpreter tests for RDNA2 v_frexp/v_ldexp_f16 VOP3 output modifiers.

Translates synthetic RDNA2 pixel shaders via tools/rdna2spirv.py, runs them in
tests/semantic/spirv_mini_interp.py against the gfx1036-measured model in tools/frexp_ldexp_f16_mods_probe.py,
checks byte-identical no-op omod SPIR-V emission, op_sel half preservation, inactive EXEC preservation,
Unsupported rejection of unmodelled modifiers, and spirv-as/spirv-val validation. Standard library only.
"""
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
for p in (TOOLS, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import frexp_ldexp_f16_mods_probe as probe  # noqa: E402
import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
_TMP = None
_SRC_PATH = None


def setUpModule():
    global _TMP, _SRC_PATH
    _TMP = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_frexp_ldexp_f16_mods.s")
    with open(os.path.join(_TMP.name, "ps_frexp_ldexp_f16_mods.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP, _SRC_PATH
    if _TMP is not None:
        _TMP.cleanup()
        _TMP = None
        _SRC_PATH = None


def _translate(lines: list[str]) -> str:
    if _TMP is None:
        setUpModule()
    asm = "; ps_frexp_ldexp_f16_mods\n" + "".join(f"\t{ln}\n" for ln in lines)
    with open(_SRC_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_SRC_PATH)


def _run_ps(lines: list[str]) -> tuple[int, str]:
    spv = _translate(lines + ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])
    return si.run_fragment(spv)["%out_mrt0"][0], spv


def _op_line(op: str, mod: str, *, s0: str = "v1", s1: str = "v2") -> str:
    suffix = f" {mod}" if mod else ""
    if op == "v_ldexp_f16":
        return f"v_ldexp_f16_e64 v0, {s0}, {s1}{suffix}"
    return f"{op}_e64 v0, {s0}{suffix}"


class FrexpLdexpF16ModsSemanticTests(unittest.TestCase):
    def test_edge_inputs_across_all_output_modifiers(self):
        ldexp_edge = (0x0000, 0x8000, 0x0001, 0x0400, 0x3800, 0x3C00, 0xBC00, 0x7BFF, 0x7C00, 0xFC00, 0x7C01, 0x7E00)
        for op in probe.OPS:
            if op == "v_ldexp_f16":
                pairs = [(h, 1) for h in probe.EDGE_F16] + [
                    (h, n) for h in ldexp_edge for n in probe.LDEXP_EXPS if n != 1
                ]
            else:
                pairs = [(h, 0) for h in probe.EDGE_F16]
            for mod in probe.MODS:
                for h, n in pairs:
                    want = probe.expected_word(op, h, n, mod=mod, init_dst=0xDEADBEEF)
                    lines = [
                        f"v_mov_b32 v1, {0xABCD0000 | h:#x}",
                        f"v_mov_b32 v2, {0x12340000 | (n & 0xFFFF):#x}",
                        "v_mov_b32 v0, 0xdeadbeef",
                        _op_line(op, mod),
                    ]
                    with self.subTest(op=op, mod=mod or "plain", h=hex(h), n=n):
                        got, _ = _run_ps(lines)
                        self.assertEqual(got, want)

    def test_source_modifier_cases(self):
        for c in probe.SRC_MOD_CASES:
            s0 = f"-|v1|" if (c.src0_abs and c.src0_neg) else (f"|v1|" if c.src0_abs else (f"-v1" if c.src0_neg else "v1"))
            want = probe.expected_word(
                c.op, c.h, c.n, mod=c.mod, init_dst=c.init_dst, src0_abs=c.src0_abs, src0_neg=c.src0_neg
            )
            lines = [
                f"v_mov_b32 v1, {0xABCD0000 | c.h:#x}",
                f"v_mov_b32 v2, {0x12340000 | (c.n & 0xFFFF):#x}",
                f"v_mov_b32 v0, {c.init_dst:#x}",
                _op_line(c.op, c.mod, s0=s0),
            ]
            with self.subTest(case=c.name):
                got, _ = _run_ps(lines)
                self.assertEqual(got, want)

    def test_noop_omod_emits_byte_identical_spirv(self):
        preamble = ["v_mov_b32 v1, 0xabcdbc00", "v_mov_b32 v2, 0x12340001", "v_mov_b32 v0, 0xdeadbeef"]
        for op in ("v_frexp_mant_f16", "v_ldexp_f16"):
            _, spv_plain = _run_ps(preamble + [_op_line(op, "")])
            for omod in ("mul:2", "mul:4", "div:2"):
                with self.subTest(op=op, omod=omod):
                    _, spv_omod = _run_ps(preamble + [_op_line(op, omod)])
                    self.assertEqual(spv_omod, spv_plain)
            _, spv_clamp = _run_ps(preamble + [_op_line(op, "clamp")])
            for omod in ("mul:2", "mul:4", "div:2"):
                with self.subTest(op=op, mod=f"clamp {omod}"):
                    _, spv_cm = _run_ps(preamble + [_op_line(op, f"clamp {omod}")])
                    self.assertEqual(spv_cm, spv_clamp)

        _, spv_exp_plain = _run_ps(preamble + [_op_line("v_frexp_exp_i16_f16", "")])
        for mod in probe.MODS[1:]:
            with self.subTest(op="v_frexp_exp_i16_f16", mod=mod):
                _, spv_exp_mod = _run_ps(preamble + [_op_line("v_frexp_exp_i16_f16", mod)])
                self.assertEqual(spv_exp_mod, spv_exp_plain)

    def test_opsel_with_output_modifiers_preserves_other_half(self):
        # v1 = {hi: 8.0 (0x4800), lo: -1.0 (0xBC00)}; v2 = {hi: 2, lo: 1}
        v1 = (0x4800 << 16) | 0xBC00
        v2 = (2 << 16) | 1
        init = 0x11112222
        cases = [
            ("v_frexp_mant_f16", "op_sel:[1,0] mul:2", (init & 0xFFFF0000) | 0x3800),
            ("v_frexp_mant_f16", "op_sel:[1,1] clamp div:2", (0x3800 << 16) | (init & 0xFFFF)),
            ("v_frexp_mant_f16", "op_sel:[0,1] clamp mul:4", (0x0000 << 16) | (init & 0xFFFF)),
            ("v_frexp_exp_i16_f16", "op_sel:[1,0] clamp mul:2", (init & 0xFFFF0000) | 0x0004),
            ("v_frexp_exp_i16_f16", "op_sel:[1,1] clamp div:2", (0x0004 << 16) | (init & 0xFFFF)),
            ("v_ldexp_f16", "op_sel:[1,1,0] clamp mul:2", (init & 0xFFFF0000) | 0x3C00),
            ("v_ldexp_f16", "op_sel:[0,1,1] clamp div:2", (0x0000 << 16) | (init & 0xFFFF)),
            ("v_ldexp_f16", "op_sel:[1,0,1] mul:4", (0x4C00 << 16) | (init & 0xFFFF)),
        ]
        for op, mod, want in cases:
            with self.subTest(op=op, mod=mod):
                got, _ = _run_ps([f"v_mov_b32 v1, {v1:#x}", f"v_mov_b32 v2, {v2:#x}",
                                  f"v_mov_b32 v0, {init:#x}", _op_line(op, mod)])
                self.assertEqual(got, want)

    def test_inactive_exec_preserves_dst(self):
        sentinel = 0xA5A55A5A
        for op in probe.OPS:
            for mod in ("clamp", "mul:2", "clamp div:2"):
                with self.subTest(op=op, mod=mod):
                    got, _ = _run_ps([
                        f"v_mov_b32 v0, {sentinel:#x}",
                        "v_mov_b32 v1, 0x3c00",
                        "v_mov_b32 v2, 1",
                        "s_mov_b32 s10, exec_lo",
                        "s_mov_b32 exec_lo, 0",
                        _op_line(op, mod),
                        "s_mov_b32 exec_lo, s10",
                    ])
                    self.assertEqual(got, sentinel)

    def test_unmodelled_modifiers_raise_unsupported(self):
        for op in probe.OPS:
            for bad in ("mul:8", "div:4", "op_sel_hi:[1,0]", "neg_lo:[1]", "neg_hi:[1]"):
                with self.subTest(op=op, mod=bad), self.assertRaises(rdna2spirv.Unsupported):
                    _translate([_op_line(op, bad), "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])

    def test_spirv_validates_with_spirv_val(self):
        if not shutil.which("spirv-as") or not shutil.which("spirv-val"):
            self.skipTest("spirv-as or spirv-val not on PATH")
        with tempfile.TemporaryDirectory() as d:
            for op in probe.OPS:
                for mod in ("clamp", "mul:2", "clamp mul:4", "div:2"):
                    spv_text = _translate([
                        "v_mov_b32 v1, 0xabcdbc00",
                        "v_mov_b32 v2, 0x12340001",
                        "v_mov_b32 v0, 0xdeadbeef",
                        _op_line(op, mod),
                        "exp mrt0 v0, v0, v0, v0 done vm",
                        "s_endpgm",
                    ])
                    sa_path = os.path.join(d, "test.spvasm")
                    spv_path = os.path.join(d, "test.spv")
                    with open(sa_path, "w", encoding="utf-8", newline="\n") as f:
                        f.write(spv_text)
                    rdna2spirv.assemble_spirv(sa_path, spv_path)
                    res = subprocess.run(
                        ["spirv-val", "--target-env", "vulkan1.3", spv_path],
                        capture_output=True, text=True,
                    )
                    self.assertEqual(res.returncode, 0, f"{op} {mod}: {res.stderr}")


if __name__ == "__main__":
    unittest.main()
