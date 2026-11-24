"""Unit tests for tools/cvt_pknorm_f16_probe.py (reference model, kernels, evaluator, CLI, live gfx1036).

Standard library only; no game files. Live hardware check runs on gfx1036 when available and skips otherwise.
"""
import io
import json
import os
import re
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

import cvt_pknorm_f16_probe as pr  # noqa: E402
from hw_oracle import asm, runner  # noqa: E402

_INLINE_VALS = {
    "0.5": 0x00003800,
    "-0.5": 0x0000B800,
    "1.0": 0x00003C00,
    "1": 0x00000001,
    "-1": 0xFFFFFFFF,
    "0x3800": 0x00003800,
}


def _parse_src(tok: str, regs: dict[str, int]) -> tuple[int, bool, bool]:
    is_neg = tok.startswith("-")
    if is_neg:
        tok = tok[1:]
    is_abs = tok.startswith("|") and tok.endswith("|")
    if is_abs:
        tok = tok[1:-1]
    if tok in regs:
        val = regs[tok]
    elif tok in _INLINE_VALS:
        val = _INLINE_VALS[tok]
    elif tok.startswith("0x"):
        val = int(tok, 16)
    else:
        val = int(tok, 0)
    return val, is_abs, is_neg


class FakeOracle:
    """Simulated hw_oracle backend executing sweep and case kernels against pr.expected_pknorm_f16."""

    def __init__(
        self,
        *,
        arch: str = "gfx1036",
        device: str = "fake gfx1036",
        mismatch_sweep: bool = False,
        mismatch_case: bool = False,
        wave_diff: bool = False,
        clamp_diff: bool = False,
        raw_omod_diff: bool = False,
        error_msg: str | None = None,
    ):
        self.arch = arch
        self.device = device
        self.mismatch_sweep = mismatch_sweep
        self.mismatch_case = mismatch_case
        self.wave_diff = wave_diff
        self.clamp_diff = clamp_diff
        self.raw_omod_diff = raw_omod_diff
        self.error_msg = error_msg
        self.closed = False

    def _run_sweep(self, k) -> list[int]:
        ins = next(ln for ln in k.lines if ln.startswith("v_cvt_pknorm_"))
        op = ins.split()[0]
        dhi = "op_sel:[0,0,1]" in ins
        clamp = "clamp" in ins
        half = 1 if any("0x8000" in ln for ln in k.lines) else 0
        base = half * 32768
        tab = pr._REF_TABLES[op == "v_cvt_pknorm_i16_f16"]
        words = []
        for idx in range(k.words):
            lo = tab[base + 2 * idx]
            hi = tab[base + 2 * idx + 1]
            w = ((lo << 16) if dhi else (lo | (hi << 16))) & 0xFFFFFFFF
            words.append(w)
        if self.mismatch_sweep and half == 0:
            words[0] ^= 1
        if self.wave_diff and k.wave == 64 and half == 0:
            words[1] ^= 1
        if self.clamp_diff and clamp and half == 0:
            words[2] ^= 1
        return words

    def _run_cases(self, k) -> list[int]:
        regs: dict[str, int] = {}
        words: list[int] = []
        for ln in k.lines:
            m_mov = re.match(r"^(?:v_mov_b32_e32|s_mov_b32)\s+([vs]\d+),\s+(0x[0-9a-fA-F]+)$", ln)
            if m_mov:
                regs[m_mov.group(1)] = int(m_mov.group(2), 16)
                continue
            m_long = re.match(r"^\.long\s+(0x[0-9a-fA-F]+),\s+(0x[0-9a-fA-F]+)$", ln)
            if m_long:
                dw0 = int(m_long.group(1), 16)
                dw1 = int(m_long.group(2), 16)
                opc = (dw0 >> 16) & 0xFF
                clamp = bool((dw0 >> 15) & 1)
                omod = (dw1 >> 27) & 3
                op = "v_cvt_pknorm_i16_f16" if opc == 0x12 else "v_cvt_pknorm_u16_f16"
                w = pr.expected_pknorm_f16(op, regs["v1"], regs["v2"], clamp=clamp)
                if self.raw_omod_diff and omod == 2:
                    w ^= 0x100
                if self.clamp_diff and clamp:
                    w ^= 1
                words.append(w)
                continue
            m_op = re.match(r"^(v_cvt_pknorm_[iu]16_f16)\s+v8,\s+([^,]+),\s+([^\s]+)(?:\s+(.*))?$", ln)
            if m_op:
                op, s0_tok, s1_tok, tail = m_op.groups()
                tail = tail or ""
                s0_val, s0_abs, s0_neg = _parse_src(s0_tok, regs)
                s1_val, s1_abs, s1_neg = _parse_src(s1_tok, regs)
                sel = (0, 0)
                dhi = False
                m_sel = re.search(r"op_sel:\[(\d),(\d),(\d)\]", tail)
                if m_sel:
                    sel = (int(m_sel.group(1)), int(m_sel.group(2)))
                    dhi = bool(int(m_sel.group(3)))
                clamp = "clamp" in tail.split()
                w = pr.expected_pknorm_f16(
                    op, s0_val, s1_val, sel=sel, dhi=dhi, clamp=clamp,
                    src0_abs=s0_abs, src0_neg=s0_neg, src1_abs=s1_abs, src1_neg=s1_neg,
                )
                if self.mismatch_case and not words:
                    w ^= 1
                if self.wave_diff and k.wave == 64 and not words:
                    w ^= 2
                if self.clamp_diff and clamp and not words:
                    w ^= 4
                words.append(w)
        return words

    def run(self, kernels):
        out = []
        for k in kernels:
            if self.error_msg is not None:
                out.append(self.error_msg)
            elif k.words == pr.SWEEP_WORDS_PER_HALF:
                out.append(self._run_sweep(k))
            else:
                out.append(self._run_cases(k))
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

    def test_double_rounding_nans_infs_and_modifiers(self):
        # Single-rounded traps near +-0.5 ulp
        self.assertEqual(pr.ref_norm16_f16(0x3801, True), 16399)
        self.assertEqual(pr.ref_norm16_f16(0xB801, True), (-16399) & 0xFFFF)
        self.assertEqual(pr.ref_norm16_f16(0x3801, False), 32799)
        self.assertEqual(pr.ref_norm16_f16(0x37FF, True), 16376)
        self.assertEqual(pr.ref_norm16_f16(0x37FF, False), 32752)
        # NaNs, +-inf, +-0, denormals
        for nan in (0x7E00, 0xFE00, 0x7C01, 0xFC01):
            self.assertEqual(pr.ref_norm16_f16(nan, True), 0)
            self.assertEqual(pr.ref_norm16_f16(nan, False), 0)
        self.assertEqual(pr.ref_norm16_f16(0x7C00, True), 32767)
        self.assertEqual(pr.ref_norm16_f16(0xFC00, True), (-32767) & 0xFFFF)
        self.assertEqual(pr.ref_norm16_f16(0x7C00, False), 65535)
        self.assertEqual(pr.ref_norm16_f16(0xFC00, False), 0)
        self.assertEqual(pr.ref_norm16_f16(0x8000, True), 0)
        self.assertEqual(pr.ref_norm16_f16(0x0001, False), 0)
        # All 8 op_sel permutations + clamp no-op + abs/neg
        s0, s1 = 0x3C003400, 0xBC003801
        want_opsel = {
            (0, 0, 0): 0x400F2000,
            (0, 0, 1): 0x20000000,
            (0, 1, 0): 0x80012000,
            (0, 1, 1): 0x20000000,
            (1, 0, 0): 0x400F7FFF,
            (1, 0, 1): 0x7FFF0000,
            (1, 1, 0): 0x80017FFF,
            (1, 1, 1): 0x7FFF0000,
        }
        for (a, b, d), want in want_opsel.items():
            for clamp in (False, True):
                self.assertEqual(
                    pr.expected_pknorm_f16("v_cvt_pknorm_i16_f16", s0, s1, sel=(a, b), dhi=d, clamp=clamp),
                    want,
                )
        self.assertEqual(
            pr.expected_pknorm_f16(
                "v_cvt_pknorm_i16_f16", s0, s1, sel=(1, 1), dhi=True, src0_abs=True, src1_neg=True
            ),
            0x7FFF0000,
        )


class BuildJobsTests(unittest.TestCase):
    def test_job_counts_labels_and_word_bounds(self):
        kernels, meta = pr.build_jobs(list(pr.OPS), [32, 64], exhaustive=True)
        # 2 waves * 2 ops * (4 sweep modes * 2 halves + 2 case kernels) = 40 kernels
        self.assertEqual(len(kernels), 40)
        self.assertEqual(len(meta), 40)
        labels = []
        for k, entry in zip(kernels, meta):
            self.assertTrue(k.raw)
            self.assertLessEqual(k.words, 16384)
            self.assertEqual(k.wave, entry[1])
            for ln in k.lines:
                if ln.startswith(".L_pknorm_sweep_") and ln.endswith(":"):
                    labels.append(ln)
        self.assertEqual(len(labels), 32)
        self.assertEqual(len(set(labels)), 32)

    def test_no_exhaustive_jobs(self):
        kernels, meta = pr.build_jobs(["v_cvt_pknorm_i16_f16"], [32], exhaustive=False)
        self.assertEqual(len(kernels), 2)
        self.assertTrue(all(m[0] == "cases" for m in meta))


class MockedCLITests(unittest.TestCase):
    def test_pass_and_json_lf_endings(self):
        oracle = FakeOracle()
        with tempfile.TemporaryDirectory() as tmp:
            rep = Path(tmp) / "sub" / "report.json"
            rc, out, err = _run_main(oracle, "--json", str(rep))
            self.assertEqual(rc, 0, out + err)
            self.assertTrue(oracle.closed)
            self.assertIn("mismatches=0, wave_diffs=0, clamp_diffs=0, raw_omod_diffs=0", out)
            raw = rep.read_bytes()
            self.assertNotIn(b"\r\n", raw)
            doc = json.loads(raw.decode("utf-8"))
            self.assertEqual(doc["sweep_halves"], 1048576)
            self.assertEqual(doc["sweep_words"], 524288)
            self.assertEqual(doc["case_words"], 8 * len(pr.CASES))
            self.assertEqual(doc["mismatches"], 0)
            self.assertEqual(doc["wave_diffs"], 0)
            self.assertEqual(doc["clamp_diffs"], 0)
            self.assertEqual(doc["raw_omod_diffs"], 0)

    def test_failure_modes_and_exit_codes(self):
        self.assertEqual(_run_main(FakeOracle(mismatch_sweep=True))[0], 1)
        self.assertEqual(_run_main(FakeOracle(mismatch_case=True), "--no-exhaustive")[0], 1)
        self.assertEqual(_run_main(FakeOracle(wave_diff=True))[0], 1)
        self.assertEqual(_run_main(FakeOracle(clamp_diff=True), "--no-exhaustive")[0], 1)
        self.assertEqual(_run_main(FakeOracle(raw_omod_diff=True), "--no-exhaustive")[0], 1)

        oracle_err = FakeOracle(error_msg="assemble failed")
        rc, out, _ = _run_main(oracle_err, "--no-exhaustive")
        self.assertEqual(rc, 2)
        self.assertIn("harness error:", out)
        self.assertTrue(oracle_err.closed)

        non_gfx10 = FakeOracle(arch="gfx1100", device="fake gfx1100")
        rc, out, _ = _run_main(non_gfx10)
        self.assertEqual(rc, 77)
        self.assertIn("not GFX10", out)

        buf = io.StringIO()
        with mock.patch.object(runner, "open_oracle", side_effect=runner.Unavailable("missing clang")), redirect_stdout(buf):
            rc = pr.main([])
        self.assertEqual(rc, 77)

        with tempfile.TemporaryDirectory() as tmp, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                pr.main(["--json", tmp])
            self.assertEqual(cm.exception.code, 2)


class LiveHardwareTests(unittest.TestCase):
    def test_live_gfx1036_oracle(self):
        buf = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(err):
            rc = pr.main(["--hw-arch", "gfx1036"])
        if rc == 77:
            self.skipTest(buf.getvalue().strip() or "gfx1036 hardware oracle unavailable")
        self.assertEqual(rc, 0, buf.getvalue() + err.getvalue())
        self.assertIn("mismatches=0, wave_diffs=0, clamp_diffs=0, raw_omod_diffs=0", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
