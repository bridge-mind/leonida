"""Unit tests for tools/frexp_ldexp_f16_mods_probe.py (reference model, kernel assembly, FakeOracle verdicts, CLI).

Standard library only; no GPU, no LLVM required (the clang assembly test skips when AMD clang is unavailable),
and no game files.
"""
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.dont_write_bytecode = True
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import frexp_ldexp_f16_mods_probe as pr  # noqa: E402
from hw_oracle import asm, runner  # noqa: E402


class FakeOracle:
    """Simulated hw_oracle backend that parses raw GFX10 probe kernels and emits expected RDNA2 words."""

    arch = "gfx1036"
    device = "fake gfx1036"

    def __init__(
        self,
        *,
        bad_grid: bool = False,
        bad_flush: bool = False,
        bad_omod: bool = False,
        wave64_diff: bool = False,
        error_msg: str | None = None,
    ):
        self.bad_grid = bad_grid
        self.bad_flush = bad_flush
        self.bad_omod = bad_omod
        self.wave64_diff = wave64_diff
        self.error_msg = error_msg
        self.closed = False

    def run(self, kernels):
        out = []
        flush_map = {
            (c.op, c.h, c.n, c.mod): (c.init_dst & 0xFFFF0000) | c.want_lo
            for c in pr.FLUSH_CONTROL_CASES
        }
        for k in kernels:
            if self.error_msg is not None:
                out.append(self.error_msg)
                continue
            mode_val = pr.MODE_CONSOLE
            for ln in k.lines:
                m_mode = re.match(r"s_setreg_imm32_b32 hwreg\(HW_REG_MODE, 0, 10\), (0x[0-9a-fA-F]+)", ln)
                if m_mode:
                    mode_val = int(m_mode.group(1), 16)
                    break

            v1 = v2 = v8 = 0
            words = []
            for ln in k.lines:
                m_mov = re.match(r"v_mov_b32_e32 v([128]), (0x[0-9a-fA-F]+)", ln)
                if m_mov:
                    reg = int(m_mov.group(1))
                    val = int(m_mov.group(2), 16)
                    if reg == 1:
                        v1 = val
                    elif reg == 2:
                        v2 = val
                    else:
                        v8 = val
                    continue

                m_op = re.match(
                    r"(v_frexp_mant_f16_e64|v_frexp_exp_i16_f16_e64|v_ldexp_f16)\s+v8,\s+([^,\s]+)(?:,\s+v2)?(?:\s+(.*))?$",
                    ln,
                )
                if not m_op:
                    continue
                op = m_op.group(1).removesuffix("_e64")
                s0 = m_op.group(2)
                mod = (m_op.group(3) or "").strip()
                src0_neg = s0.startswith("-")
                src0_abs = "|" in s0
                h = v1 & 0xFFFF
                n = v2 & 0xFFFF
                if mode_val == pr.MODE_FLUSH16 and op != "v_frexp_exp_i16_f16":
                    w = flush_map[(op, h, n, mod)]
                    if self.bad_flush and mod == "mul:2":
                        w ^= 0x0400
                else:
                    w = pr.expected_word(
                        op, h, n, mod=mod, init_dst=v8, src0_abs=src0_abs, src0_neg=src0_neg
                    )
                    if self.bad_grid and mode_val == pr.MODE_CONSOLE and not words and op == "v_frexp_mant_f16":
                        w ^= 1
                    if self.bad_omod and mode_val == pr.MODE_CONSOLE and mod == "mul:2" and not words:
                        w ^= 0x0100
                    if self.wave64_diff and k.wave == 64 and not words and mode_val == pr.MODE_CONSOLE:
                        w ^= 1
                words.append(w)
            out.append(words)
        return out

    def close(self):
        self.closed = True


def _run_main(oracle, *args):
    buf = io.StringIO()
    err = io.StringIO()
    with mock.patch.object(runner, "open_oracle", return_value=oracle), redirect_stdout(buf), redirect_stderr(err):
        rc = pr.main(list(args))
    return rc, buf.getvalue(), err.getvalue()


class ReferenceModelTests(unittest.TestCase):
    def test_self_test(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(pr.self_test(), 0)

    def test_clamp_f16_rules(self):
        self.assertEqual(pr.ref_clamp_f16(0x0000), 0x0000)
        self.assertEqual(pr.ref_clamp_f16(0x8000), 0x0000)
        self.assertEqual(pr.ref_clamp_f16(0x0001), 0x0001)
        self.assertEqual(pr.ref_clamp_f16(0x8001), 0x0000)
        self.assertEqual(pr.ref_clamp_f16(0x3800), 0x3800)
        self.assertEqual(pr.ref_clamp_f16(0x3C00), 0x3C00)
        self.assertEqual(pr.ref_clamp_f16(0x3C01), 0x3C00)
        self.assertEqual(pr.ref_clamp_f16(0x7C00), 0x3C00)
        self.assertEqual(pr.ref_clamp_f16(0xFC00), 0x0000)
        self.assertEqual(pr.ref_clamp_f16(0x7C01), 0x0000)
        self.assertEqual(pr.ref_clamp_f16(0x7E00), 0x0000)
        self.assertEqual(pr.ref_clamp_f16(0xFE55), 0x0000)

    def test_expected_word_console_mode_and_high_half_preserved(self):
        # omod is a no-op under console MODE; clamp applies to mant and ldexp only.
        for mod in ("", "mul:2", "mul:4", "div:2"):
            self.assertEqual(pr.expected_word("v_frexp_mant_f16", 0xBC00, mod=mod, init_dst=0x12345678), 0x1234B800)
            self.assertEqual(pr.expected_word("v_ldexp_f16", 0x3C00, 1, mod=mod, init_dst=0x12345678), 0x12344000)
        for mod in ("clamp", "clamp mul:2", "clamp mul:4", "clamp div:2"):
            self.assertEqual(pr.expected_word("v_frexp_mant_f16", 0xBC00, mod=mod, init_dst=0x12345678), 0x12340000)
            self.assertEqual(pr.expected_word("v_frexp_mant_f16", 0x7C00, mod=mod, init_dst=0x12345678), 0x12343C00)
            self.assertEqual(pr.expected_word("v_ldexp_f16", 0x3C00, 1, mod=mod, init_dst=0x12345678), 0x12343C00)
            self.assertEqual(pr.expected_word("v_frexp_exp_i16_f16", 0x4800, mod=mod, init_dst=0x12345678), 0x12340004)
        # abs applied before neg
        self.assertEqual(
            pr.expected_word("v_frexp_mant_f16", 0xBC00, mod="clamp", src0_abs=True, src0_neg=True),
            0xDEAD0000,
        )
        self.assertEqual(
            pr.expected_word("v_frexp_mant_f16", 0xBC00, mod="clamp", src0_abs=True, src0_neg=False),
            0xDEAD3800,
        )


class KernelTests(unittest.TestCase):
    def test_kernel_structure(self):
        for op in pr.OPS:
            for mod in pr.MODS:
                lines, pairs = pr.mode_grid_kernel_lines(op, mod, pr.MODE_CONSOLE)
                self.assertEqual(lines[0], "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 0, 10), 0x1c0")
                self.assertEqual(lines[-2:], ["s_waitcnt vmcnt(0)", "s_endpgm"])
                expected_n = len(pr.EDGE_F16) * len(pr.LDEXP_EXPS) if op == "v_ldexp_f16" else len(pr.EDGE_F16)
                self.assertEqual(len(pairs), expected_n)

    def test_kernels_assemble_when_amd_clang_available(self):
        try:
            clang = runner.find_clang()
        except runner.Unavailable as e:
            self.skipTest(str(e))
        for wave in (32, 64):
            kernels, _ = pr.build_jobs(["v_frexp_mant_f16"], [wave])
            text, _, _ = asm.module_text(kernels[:4], "gfx1036", wave=wave)
            with tempfile.TemporaryDirectory() as d:
                s_path = Path(d) / "k.s"
                o_path = Path(d) / "k.o"
                s_path.write_text(text, encoding="utf-8", newline="\n")
                flags = ["-mwavefrontsize64"] if wave == 64 else []
                res = subprocess.run(
                    [clang, "--target=amdgcn-amd-amdhsa", "-mcpu=gfx1036", *flags, "-c", str(s_path), "-o", str(o_path)],
                    capture_output=True,
                    text=True,
                )
            self.assertEqual(res.returncode, 0, res.stderr[:800])


class ProbeCliTests(unittest.TestCase):
    def test_fake_oracle_pass_and_json_lf(self):
        oracle = FakeOracle()
        with tempfile.TemporaryDirectory() as d:
            report = os.path.join(d, "nested", "report.json")
            rc, out, err = _run_main(oracle, "--json", report)
            self.assertEqual(rc, 0, out + err)
            self.assertTrue(oracle.closed)
            self.assertIn("mismatches=0, wave_diffs=0, omod_diffs=0", out)
            with open(report, "rb") as f:
                raw = f.read()
            self.assertNotIn(b"\r\n", raw)
            doc = json.loads(raw.decode("utf-8"))
            self.assertEqual(doc["arch"], "gfx1036")
            self.assertEqual(doc["waves"], [32, 64])
            self.assertEqual(doc["ops"], list(pr.OPS))
            self.assertEqual(doc["mismatches"], 0)
            self.assertEqual(doc["wave_diffs"], 0)
            self.assertEqual(doc["omod_diffs"], 0)
            self.assertEqual(doc["total_words"], 3466)

    def test_fake_oracle_grid_mismatch_exits_1(self):
        rc, out, _ = _run_main(FakeOracle(bad_grid=True), "--ops", "v_frexp_mant_f16", "--waves", "32")
        self.assertEqual(rc, 1, out)
        self.assertIn("MISMATCH grid", out)

    def test_fake_oracle_flush_control_mismatch_exits_1(self):
        rc, out, _ = _run_main(FakeOracle(bad_flush=True), "--ops", "v_frexp_mant_f16", "--waves", "32")
        self.assertEqual(rc, 1, out)
        self.assertIn("MISMATCH flush_ctrl", out)

    def test_fake_oracle_omod_diff_exits_1(self):
        rc, out, _ = _run_main(FakeOracle(bad_omod=True), "--ops", "v_frexp_mant_f16", "--waves", "32")
        self.assertEqual(rc, 1, out)
        self.assertIn("OMOD DIFF", out)

    def test_fake_oracle_wave_diff_exits_1(self):
        rc, out, _ = _run_main(FakeOracle(wave64_diff=True), "--ops", "v_frexp_mant_f16", "--waves", "32,64")
        self.assertEqual(rc, 1, out)
        self.assertIn("WAVE DIFF", out)

    def test_fake_oracle_harness_error_exits_2(self):
        rc, out, _ = _run_main(FakeOracle(error_msg="assembler rejected: test"), "--ops", "v_frexp_mant_f16", "--waves", "32")
        self.assertEqual(rc, 2, out)
        self.assertIn("harness error", out)

    def test_bad_cli_and_json_paths(self):
        with tempfile.TemporaryDirectory() as d:
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as ctx:
                    pr.main(["--ops", "v_bogus", "--waves", "32"])
                self.assertEqual(ctx.exception.code, 2)
                with self.assertRaises(SystemExit) as ctx2:
                    pr.main(["--waves", "16"])
                self.assertEqual(ctx2.exception.code, 2)
                with self.assertRaises(SystemExit) as ctx3:
                    _run_main(FakeOracle(), "--json", d)
                self.assertEqual(ctx3.exception.code, 2)

            blocker = os.path.join(d, "file_not_dir")
            with open(blocker, "wb") as f:
                f.write(b"keep")
            rc, _, err = _run_main(
                FakeOracle(), "--ops", "v_frexp_mant_f16", "--waves", "32", "--json", os.path.join(blocker, "r.json")
            )
            self.assertEqual(rc, 2)
            self.assertIn("ERROR: cannot write", err)
            self.assertNotIn("Traceback", err)

    def test_non_gfx10_arch_exits_77(self):
        oracle = FakeOracle()
        oracle.arch = "gfx1100"
        rc, out, _ = _run_main(oracle)
        self.assertEqual(rc, 77, out)
        self.assertTrue(oracle.closed)

    def test_unavailable_oracle_exits_77(self):
        buf = io.StringIO()
        with mock.patch.object(runner, "open_oracle", side_effect=runner.Unavailable("missing")), redirect_stdout(buf):
            rc = pr.main(["--ops", "v_frexp_mant_f16", "--waves", "32"])
        self.assertEqual(rc, 77)
        self.assertIn("unavailable", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
