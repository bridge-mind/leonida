"""Unit tests for tools/cvt_norm_mods_probe.py (reference model, kernel assembly, FakeOracle verdicts, CLI).

Standard library only; no GPU, no LLVM required (live assembly tests skip when unavailable), and no game files.
"""
import io
import json
import os
import re
import shutil
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

import cvt_norm_mods_probe as pr  # noqa: E402
from hw_oracle import asm, runner  # noqa: E402


def _eval_f32_src_mod(bits: int, expr: str) -> int:
    if "|" in expr:
        bits &= 0x7FFFFFFF
    if expr.startswith("-"):
        bits ^= 0x80000000
    return bits & pr.M32


def _eval_f16_src_mod(bits16: int, expr: str) -> int:
    bits16 &= pr.M16
    if "|" in expr:
        bits16 &= 0x7FFF
    if expr.startswith("-"):
        bits16 ^= 0x8000
    return bits16


class FakeOracle:
    """Simulated hw_oracle backend parsing probe kernels and computing reference outputs."""

    arch = "gfx1036"
    device = "fake gfx1036"

    def __init__(self, *, bad_word: bool = False, wave64_diff: bool = False, error_msg: str | None = None):
        self.bad_word = bad_word
        self.wave64_diff = wave64_diff
        self.error_msg = error_msg
        self.closed = False

    def run(self, kernels, words=4):
        del words
        out = []
        for k in kernels:
            if self.error_msg is not None:
                out.append(self.error_msg)
                continue
            lines = k.lines
            joined = "\n".join(lines)
            m_sweep = re.search(r"v_add_nc_u32_e32 v11, (\d+), v10", joined)
            if m_sweep:
                base_word = int(m_sweep.group(1))
                m_op = re.search(r"(v_cvt_norm_[iu]16_f16)_e64", joined)
                assert m_op is not None, joined
                res = pr.expected_sweep_words(m_op.group(1))[base_word : base_word + 16384]
                if self.bad_word:
                    res[0] ^= 1
                if self.wave64_diff and k.wave == 64:
                    res[0] ^= 1
                out.append(res)
                continue

            regs: dict[str, int] = {}
            res = []
            for ln in lines:
                m_mov = re.match(r"v_mov_b32_e32 (v\d+), (0x[0-9a-fA-F]+|\d+)$", ln)
                if m_mov:
                    regs[m_mov.group(1)] = int(m_mov.group(2), 0) & pr.M32
                    continue
                m_sdwa = re.match(r"(v_cvt_norm_[iu]16_f16)_sdwa v0, ([^ ]+)(.*)", ln)
                if m_sdwa:
                    op, src_expr, tail = m_sdwa.groups()
                    snorm = op == "v_cvt_norm_i16_f16"
                    src = regs["v1"]
                    h = (src >> 16) & pr.M16 if "src0_sel:WORD_1" in tail else (src & pr.M16)
                    h = _eval_f16_src_mod(h, src_expr)
                    val16 = pr.expected_norm_f16(h, snorm)
                    seed = regs["v0"]
                    if "dst_sel:WORD_1" in tail:
                        w = (val16 << 16) | (seed & pr.M16)
                    elif "UNUSED_PRESERVE" in tail:
                        w = (seed & 0xFFFF0000) | val16
                    elif "UNUSED_SEXT" in tail:
                        w = (0xFFFF0000 | val16) if (val16 & 0x8000) else val16
                    else:
                        w = val16
                    res.append(w & pr.M32)
                    continue
                m_e64 = re.match(r"(v_cvt_norm_[iu]16_f16)_e64 v0, ([^ ]+)", ln)
                if m_e64:
                    op, src_expr = m_e64.groups()
                    snorm = op == "v_cvt_norm_i16_f16"
                    h = _eval_f16_src_mod(regs["v1"] & pr.M16, src_expr)
                    val16 = pr.expected_norm_f16(h, snorm)
                    res.append((regs["v0"] & 0xFFFF0000) | val16)
                    continue
                m_pk = re.match(r"(v_cvt_pknorm_[iu]16_f32) v0, ([^,]+), ([^ ]+)(?: clamp)?$", ln)
                if m_pk:
                    op, s0_expr, s1_expr = m_pk.groups()
                    s0 = _eval_f32_src_mod(regs["v1"], s0_expr)
                    s1 = _eval_f32_src_mod(regs["v2"], s1_expr)
                    res.append(pr.expected_pknorm_f32(op, s0, s1))
                    continue

            if res:
                if self.bad_word:
                    res[0] ^= 1
                if self.wave64_diff and k.wave == 64:
                    res[0] ^= 1
            out.append(res)
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

    def test_norm_f16_single_rounding_ties_and_edges(self):
        # Single-rounding near +-0.5 vs f32 double-rounding
        self.assertEqual(pr.expected_norm_f16(0x3800, True), 16384)
        self.assertEqual(pr.expected_norm_f16(0x3801, True), 16399)
        self.assertEqual(pr.expected_norm_f16(0xB801, True), (-16399) & 0xFFFF)
        self.assertEqual(pr.expected_norm_f16(0x37FF, True), 16376)
        self.assertEqual(pr.expected_norm_f16(0xB7FF, True), (-16376) & 0xFFFF)
        self.assertEqual(pr.expected_norm_f16(0x3800, False), 32768)
        self.assertEqual(pr.expected_norm_f16(0x3801, False), 32799)
        self.assertEqual(pr.expected_norm_f16(0x37FF, False), 32752)
        # Out-of-range, infinities, NaNs, subnormals
        self.assertEqual(pr.expected_norm_f16(0x4000, True), 0x7FFF)
        self.assertEqual(pr.expected_norm_f16(0xC000, True), 0x8001)
        self.assertEqual(pr.expected_norm_f16(0xC000, False), 0)
        self.assertEqual(pr.expected_norm_f16(0x7C00, False), 0xFFFF)
        self.assertEqual(pr.expected_norm_f16(0xFC00, True), 0x8001)
        self.assertEqual(pr.expected_norm_f16(0x7C01, True), 0)
        self.assertEqual(pr.expected_norm_f16(0xFE00, False), 0)
        self.assertEqual(pr.expected_norm_f16(0x0001, True), 0)
        self.assertEqual(pr.expected_norm_f16(0x03FF, False), 4)

    def test_pknorm_f32_edges_and_near_ties(self):
        self.assertEqual(pr.expected_pknorm_f32("v_cvt_pknorm_i16_f32", 0x38400180, 0x00000000), 0x00000001)
        self.assertEqual(pr.expected_pknorm_f32("v_cvt_pknorm_i16_f32", 0x3C004101, 0xBF800000), 0x80010101)
        self.assertEqual(pr.expected_pknorm_f32("v_cvt_pknorm_u16_f32", 0x37C000C0, 0x40000000), 0xFFFF0001)
        self.assertEqual(pr.expected_pknorm_f32("v_cvt_pknorm_u16_f32", 0x3B008081, 0xBF800000), 0x00000081)
        self.assertEqual(pr.expected_pknorm_f32("v_cvt_pknorm_i16_f32", 0x7F800001, 0xFF800000), 0x80010000)

    def test_kernel_builders_have_unique_labels_and_endpgm(self):
        kernels, meta = pr.build_jobs(list(pr.OPS), [32, 64])
        self.assertEqual(len(kernels), len(meta))
        labels = []
        for k in kernels:
            self.assertEqual(k.lines[-2:], ["s_waitcnt vmcnt(0)", "s_endpgm"])
            for ln in k.lines:
                if ln.endswith(":"):
                    labels.append(ln[:-1])
        self.assertEqual(len(labels), len(set(labels)), "duplicate loop labels across kernels")

    def test_kernels_assemble_when_amd_clang_available(self):
        try:
            clang = runner.find_clang()
        except runner.Unavailable as e:
            self.skipTest(str(e))
        for wave in (32, 64):
            kernels, _ = pr.build_jobs(list(pr.OPS), [wave])
            text, _, _ = asm.module_text(kernels, "gfx1036", wave=wave)
            with tempfile.TemporaryDirectory() as d:
                s_path = Path(d) / "k.s"
                o_path = Path(d) / "k.o"
                with open(s_path, "w", encoding="utf-8", newline="\n") as f:
                    f.write(text)
                flags = ["-mwavefrontsize64"] if wave == 64 else []
                res = subprocess.run(
                    [clang, "--target=amdgcn-amd-amdhsa", "-mcpu=gfx1036", *flags, "-c", str(s_path), "-o", str(o_path)],
                    capture_output=True,
                    text=True,
                )
            self.assertEqual(res.returncode, 0, res.stderr[:800])

    def test_kernels_assemble_when_llvm_mc_available(self):
        llvm_mc = os.environ.get("LLVM_MC") or shutil.which("llvm-mc")
        if not llvm_mc or not os.path.isfile(llvm_mc):
            self.skipTest("llvm-mc not found")
        for wave in (32, 64):
            kernels, _ = pr.build_jobs(list(pr.OPS), [wave])
            text, _, _ = asm.module_text(kernels, "gfx1036", wave=wave)
            flags = ["-mattr=+wavefrontsize64"] if wave == 64 else []
            res = subprocess.run(
                [llvm_mc, "-triple=amdgcn-amd-amdhsa", "-mcpu=gfx1036", *flags, "-filetype=obj", "-o", os.devnull],
                input=text,
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
            self.assertIn("mismatches=0, wave_diffs=0", out)
            with open(report, "rb") as f:
                raw = f.read()
            self.assertNotIn(b"\r\n", raw)
            doc = json.loads(raw.decode("utf-8"))
            self.assertEqual(doc["arch"], "gfx1036")
            self.assertEqual(doc["waves"], [32, 64])
            self.assertEqual(doc["ops"], list(pr.OPS))
            self.assertEqual(doc["mismatches"], 0)
            self.assertEqual(doc["wave_diffs"], 0)
            self.assertEqual(doc["total_words"], 1050708)

    def test_fake_oracle_word_mismatch_exits_1(self):
        rc, out, _ = _run_main(FakeOracle(bad_word=True), "--ops", "v_cvt_pknorm_i16_f32", "--waves", "32")
        self.assertEqual(rc, 1, out)
        self.assertIn("MISMATCH", out)

    def test_fake_oracle_wave_diff_exits_1(self):
        rc, out, _ = _run_main(FakeOracle(wave64_diff=True), "--ops", "v_cvt_pknorm_u16_f32", "--waves", "32,64")
        self.assertEqual(rc, 1, out)
        self.assertIn("WAVE DIFF", out)

    def test_fake_oracle_harness_error_exits_2(self):
        rc, out, _ = _run_main(
            FakeOracle(error_msg="assembler rejected: test"), "--ops", "v_cvt_pknorm_i16_f32", "--waves", "32"
        )
        self.assertEqual(rc, 2, out)
        self.assertIn("harness error", out)

    def test_bad_cli_args_exit_2(self):
        for bad_waves in ("", "16", "32,", "32,,64", "abc"):
            with self.subTest(waves=bad_waves):
                rc, _, err = _run_main(FakeOracle(), "--waves", bad_waves)
                self.assertEqual(rc, 2)
                self.assertIn("--waves", err)
        for bad_ops in ("", "v_not_an_op", "v_cvt_norm_i16_f16,"):
            with self.subTest(ops=bad_ops):
                rc, _, err = _run_main(FakeOracle(), "--ops", bad_ops)
                self.assertEqual(rc, 2)
                self.assertIn("--ops", err)

    def test_json_directory_refused_and_unwritable_path_exits_2(self):
        with tempfile.TemporaryDirectory() as d:
            rc, _, err = _run_main(FakeOracle(), "--ops", "v_cvt_pknorm_i16_f32", "--waves", "32", "--json", d)
            self.assertEqual(rc, 2)
            self.assertIn("ERROR:", err)
            self.assertNotIn("Traceback", err)
            blocker = os.path.join(d, "file_not_dir")
            with open(blocker, "wb") as f:
                f.write(b"keep")
            rc, _, err = _run_main(
                FakeOracle(), "--ops", "v_cvt_pknorm_i16_f32", "--waves", "32", "--json", os.path.join(blocker, "r.json")
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
            rc = pr.main(["--ops", "v_cvt_pknorm_i16_f32", "--waves", "32"])
        self.assertEqual(rc, 77)
        self.assertIn("unavailable", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
