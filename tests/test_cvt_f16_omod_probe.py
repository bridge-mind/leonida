"""Unit tests for tools/cvt_f16_omod_probe.py (reference model, kernel assembly, FakeOracle verdicts, CLI).

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

import cvt_f16_omod_probe as pr  # noqa: E402
from hw_oracle import asm, runner  # noqa: E402


class FakeOracle:
    """Simulated hw_oracle backend that parses kernel_lines() and answers from expected_cvt_f16_omod."""

    arch = "gfx1036"
    device = "fake gfx1036"

    def __init__(
        self,
        *,
        bad_lo: bool = False,
        bad_hi: bool = False,
        wave64_diff: bool = False,
        error_msg: str | None = None,
    ):
        self.bad_lo = bad_lo
        self.bad_hi = bad_hi
        self.wave64_diff = wave64_diff
        self.error_msg = error_msg
        self.closed = False

    def run(self, kernels):
        out = []
        for k in kernels:
            if self.error_msg is not None:
                out.append(self.error_msg)
                continue
            mdict = k.mode or {}
            setreg_rtz = any("hwreg(HW_REG_MODE, 2, 2), 3" in ln for ln in k.lines)
            rtz = (mdict.get("round16_64", 0) == 3) or setreg_rtz
            denorm_keep = mdict.get("denorm16_64", 0) == 3
            ieee = mdict.get("ieee", 0) == 1
            dx10_clamp = mdict.get("dx10_clamp", 1) == 1
            words = []
            for ln in k.lines:
                m_src = re.match(r"v_mov_b32_e32 v1, (0x[0-9a-fA-F]+)", ln)
                if m_src:
                    cur_src = int(m_src.group(1), 16)
                    continue
                m_cvt = re.match(r"v_cvt_f16_f32_e64 v2, v1(?:\s+(.*))?$", ln)
                if m_cvt:
                    mod = (m_cvt.group(1) or "").strip()
                    w = pr.expected_cvt_f16_omod(
                        cur_src,
                        mod,
                        rtz=rtz,
                        denorm_keep=denorm_keep,
                        ieee=ieee,
                        dx10_clamp=dx10_clamp,
                    )
                    if self.bad_lo and len(words) == 0:
                        w ^= 1
                    if self.bad_hi and len(words) == 0:
                        w = (w & 0xFFFF) | (0x1234 << 16)
                    if self.wave64_diff and k.wave == 64 and len(words) == 0:
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

    def test_critical_edge_cases(self):
        f32 = pr._f32_bits
        # RNE vs RTZ overflow
        self.assertEqual(
            pr.expected_cvt_f16_omod(f32(32768.0), "mul:2", rtz=False, denorm_keep=False, ieee=False),
            0xAAAA7C00,
        )
        self.assertEqual(
            pr.expected_cvt_f16_omod(f32(32768.0), "mul:2", rtz=True, denorm_keep=False, ieee=False),
            0xAAAA7BFF,
        )
        self.assertEqual(
            pr.expected_cvt_f16_omod(f32(-32768.0), "mul:2", rtz=True, denorm_keep=False, ieee=False),
            0xAAAAFBFF,
        )
        # +-65520 div:2 rounds to f16 first (+-inf RNE, +-65504 RTZ), then divides
        self.assertEqual(
            pr.expected_cvt_f16_omod(f32(65520.0), "div:2", rtz=False, denorm_keep=False, ieee=False),
            0xAAAA7C00,
        )
        self.assertEqual(
            pr.expected_cvt_f16_omod(f32(65520.0), "div:2", rtz=True, denorm_keep=False, ieee=False),
            0xAAAA77FF,
        )
        self.assertEqual(
            pr.expected_cvt_f16_omod(f32(-65520.0), "div:2", rtz=True, denorm_keep=False, ieee=False),
            0xAAAAF7FF,
        )
        # True +-inf stays +-inf even under RTZ + mul:2/mul:4
        self.assertEqual(
            pr.expected_cvt_f16_omod(0x7F800000, "mul:2", rtz=True, denorm_keep=False, ieee=False),
            0xAAAA7C00,
        )
        self.assertEqual(
            pr.expected_cvt_f16_omod(0xFF800000, "mul:4", rtz=True, denorm_keep=False, ieee=False),
            0xAAAAFC00,
        )
        # div:2 underflow from exponent 1 preserves sign (-0.0 = 0x8000); exponent 0 gives +0.0 (0x0000)
        for neg_exp1 in (f32(-(2.0 ** -14)), f32(-1.5 * (2.0 ** -14)), 0xB8FFE000):
            self.assertEqual(
                pr.expected_cvt_f16_omod(neg_exp1, "div:2", rtz=False, denorm_keep=False, ieee=False),
                0xAAAA8000,
            )
        self.assertEqual(
            pr.expected_cvt_f16_omod(0x80000000, "div:2", rtz=False, denorm_keep=False, ieee=False),
            0xAAAA0000,
        )
        self.assertEqual(
            pr.expected_cvt_f16_omod(f32(-(2.0 ** -15)), "div:2", rtz=False, denorm_keep=False, ieee=False),
            0xAAAA0000,
        )
        # clamp still produces +0.0 from -0.0 and acts in every mode
        self.assertEqual(
            pr.expected_cvt_f16_omod(f32(-(2.0 ** -14)), "clamp div:2", rtz=False, denorm_keep=False, ieee=False),
            0xAAAA0000,
        )
        self.assertEqual(
            pr.expected_cvt_f16_omod(f32(0.75), "clamp mul:2", rtz=True, denorm_keep=False, ieee=False),
            0xAAAA3C00,
        )
        self.assertEqual(
            pr.expected_cvt_f16_omod(f32(0.75), "clamp mul:2", rtz=True, denorm_keep=True, ieee=False),
            0xAAAA3A00,
        )
        self.assertEqual(
            pr.expected_cvt_f16_omod(f32(0.75), "clamp mul:2", rtz=True, denorm_keep=False, ieee=True),
            0xAAAA3A00,
        )


class KernelTests(unittest.TestCase):
    def test_kernel_structure(self):
        for mode_name in pr.MODES:
            for mod in pr.MODS:
                lines = pr.kernel_lines(mode_name, mod)
                self.assertEqual(lines[-2:], ["s_waitcnt vmcnt(0)", "s_endpgm"])
                if mode_name == "dyn_rtz_d0_i0":
                    self.assertEqual(lines[0], "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 3")
                else:
                    self.assertTrue(lines[0].startswith("v_mov_b32_e32 v1, "))
                cvt_lines = [ln for ln in lines if ln.startswith("v_cvt_f16_f32_e64")]
                self.assertEqual(len(cvt_lines), len(pr.INPUTS))

    def test_kernels_assemble_when_amd_clang_available(self):
        try:
            clang = runner.find_clang()
        except runner.Unavailable as e:
            self.skipTest(str(e))
        for wave in (32, 64):
            kernels, _ = pr.build_jobs(list(pr.MODES), [wave])
            text, _, _ = asm.module_text(kernels[:8], "gfx1036", wave=wave)
            with tempfile.TemporaryDirectory() as d:
                s_path = Path(d) / "k.s"
                o_path = Path(d) / "k.o"
                s_path.write_text(text, encoding="utf-8")
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
            self.assertIn("lo_mismatches=0, hi_mismatches=0, wave_diffs=0, dyn_rtz_diffs=0", out)
            with open(report, "rb") as f:
                raw = f.read()
            self.assertNotIn(b"\r\n", raw)
            doc = json.loads(raw.decode("utf-8"))
            self.assertEqual(doc["arch"], "gfx1036")
            self.assertEqual(doc["waves"], [32, 64])
            self.assertEqual(doc["inputs"], len(pr.INPUTS))
            self.assertEqual(doc["total_cases"], 2 * len(pr.MODES) * len(pr.MODS) * len(pr.INPUTS))
            self.assertEqual(doc["lo_mismatches"], 0)
            self.assertEqual(doc["hi_mismatches"], 0)
            self.assertEqual(doc["wave_diffs"], 0)

    def test_fake_oracle_lo_mismatch_exits_1(self):
        rc, out, _ = _run_main(FakeOracle(bad_lo=True), "--modes", "rne_d0_i0", "--waves", "32")
        self.assertEqual(rc, 1, out)
        self.assertIn("MISMATCH", out)

    def test_fake_oracle_hi_clobber_exits_1(self):
        rc, out, _ = _run_main(FakeOracle(bad_hi=True), "--modes", "rne_d0_i0", "--waves", "32")
        self.assertEqual(rc, 1, out)
        self.assertIn("hi_mismatches=8", out)

    def test_fake_oracle_wave_diff_exits_1(self):
        rc, out, _ = _run_main(FakeOracle(wave64_diff=True), "--modes", "rne_d0_i0", "--waves", "32,64")
        self.assertEqual(rc, 1, out)
        self.assertIn("WAVE DIFF", out)

    def test_fake_oracle_harness_error_exits_2(self):
        rc, out, _ = _run_main(FakeOracle(error_msg="assembler rejected: test"), "--modes", "rne_d0_i0", "--waves", "32")
        self.assertEqual(rc, 2, out)
        self.assertIn("harness error", out)

    def test_json_directory_refused_and_unwritable_path_exits_2(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                _run_main(FakeOracle(), "--json", d)
            blocker = os.path.join(d, "file_not_dir")
            with open(blocker, "wb") as f:
                f.write(b"keep")
            rc, _, err = _run_main(FakeOracle(), "--modes", "rne_d0_i0", "--waves", "32", "--json", os.path.join(blocker, "r.json"))
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
            rc = pr.main(["--modes", "rne_d0_i0", "--waves", "32"])
        self.assertEqual(rc, 77)
        self.assertIn("unavailable", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
