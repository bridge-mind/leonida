"""Unit tests for tools/f64_omod_probe.py (gate model, simulated oracle verdicts, CLI).

Standard library only; no GPU and no game files. The simulated oracle follows the gate the probe measures, and the
broken variants exist so the probe has to prove it reports a device that disagrees.
"""
import io
import json
import math
import os
import struct
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

import f64_omod_probe as pr  # noqa: E402
from hw_oracle import runner  # noqa: E402

OP_KINDS = {op: kind for op, _form, kind, _cases in pr.OPS}


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def plain_of(op, regs, mode):
    """Plain (no-modifier) result of one case, as a Python model of the operation."""
    kind = OP_KINDS[op]
    if op == "v_cvt_f64_i32":
        raw = regs[1]
        return kind, float(raw - (1 << 32) if raw & (1 << 31) else raw)
    if op == "v_cvt_f64_u32":
        return kind, float(regs[1])
    double = struct.unpack("<d", struct.pack("<Q", (regs[2] << 32) | regs[1]))[0]
    if op == "v_cvt_f32_f64":
        return kind, f32(double)
    if op == "v_frexp_mant_f64":
        return kind, math.frexp(double)[0]
    if op == "v_div_fixup_f64":
        sign = math.copysign(1.0, regs[4]) * math.copysign(1.0, regs[6])
        return kind, math.copysign(double, sign)
    sign16 = math.copysign(1.0, regs[2]) * math.copysign(1.0, regs[3])
    quotient = struct.unpack("<e", struct.pack("<H", regs[1] & 0xFFFF))[0]
    return kind, math.copysign(f32(quotient), sign16)


class FakeOracle:
    """Simulated hw_oracle backend: parses the raw probe kernels and returns words for each instruction group."""

    arch = "gfx1036"
    device = "fake gfx1036"

    def __init__(self, *, apply_scaling_everywhere=False, wave64_differs=False, arch=None):
        self.apply_scaling_everywhere = apply_scaling_everywhere
        self.wave64_differs = wave64_differs
        self.arch = arch or self.arch
        self.closed = False

    def run(self, kernels):
        out = []
        for kernel in kernels:
            mode = kernel.mode
            regs = {}
            words = []
            for line in kernel.lines:
                if line.startswith("v_mov_b32_e32 "):
                    _, dst, value = line.replace(",", " ").split()
                    regs[int(dst[1:])] = int(value, 16)
                    continue
                op = line.split()[0].split("_e64")[0]
                if op not in OP_KINDS:
                    continue
                mod = ""
                parts = line.split()
                if parts[-1] in pr.MODS and len(parts) > 1:
                    mod = parts[-1]
                kind, value = plain_of(op, regs, mode)
                scaling_on = True if self.apply_scaling_everywhere else pr.scaling_applies(kind, mode)
                if mod in pr.SCALE and scaling_on and math.isfinite(value):
                    value = value * pr.SCALE[mod]
                elif mod == "clamp" and math.isfinite(value):
                    value = min(max(value, 0.0), 1.0)
                words.extend(pr.scalar_bits(value, kind)[:pr.word_count(kind)])
            if self.wave64_differs and kernel.wave == 64 and words:
                words[0] ^= 1
            out.append(words)
        return out

    def close(self):
        self.closed = True


def run_main(argv, oracle):
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.object(runner, "open_oracle", return_value=oracle):
        with redirect_stdout(out), redirect_stderr(err):
            code = pr.main(argv)
    return code, out.getvalue(), err.getvalue()


class GateModelTests(unittest.TestCase):
    def test_gate_is_per_result_precision(self):
        game, flush = pr.MODES["game_i0_d3"], pr.MODES["i0_d0"]
        keep32, ieee = pr.MODES["i0_d3_f32keep"], pr.MODES["i1_d0"]
        for kind in ("f64", "f16"):
            self.assertFalse(pr.scaling_applies(kind, game))
            self.assertTrue(pr.scaling_applies(kind, flush))
            self.assertFalse(pr.scaling_applies(kind, keep32))
            self.assertFalse(pr.scaling_applies(kind, ieee))
        # An f32 destination is flushed by denorm32, so it scales in the game mode too.
        self.assertTrue(pr.scaling_applies("f32", game))
        self.assertTrue(pr.scaling_applies("f32", flush))
        self.assertFalse(pr.scaling_applies("f32", keep32))
        self.assertFalse(pr.scaling_applies("f32", ieee))

    def test_clamp_and_non_finite_results(self):
        game = pr.MODES["game_i0_d3"]
        self.assertEqual(pr.expected_words(pr.scalar_bits(-0.875, "f64"), "clamp", "f64", game),
                         pr.scalar_bits(0.0, "f64")[:2])
        self.assertEqual(pr.expected_words(pr.scalar_bits(2.0, "f16"), "clamp", "f16", game)[:1],
                         pr.scalar_bits(1.0, "f16")[:1])
        inf = pr.scalar_bits(float("inf"), "f64")[:2]
        self.assertEqual(pr.expected_words(inf, "mul:2", "f64", pr.MODES["i0_d0"]), inf)
        self.assertEqual(pr.expected_words(inf, "clamp", "f64", game), inf)

    def test_self_test_reports_the_case_counts(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(pr.main(["--self-test"]), 0)
        self.assertIn("self-test ok", out.getvalue())


class ProbeCliTests(unittest.TestCase):
    def test_device_following_the_gate_is_accepted(self):
        with tempfile.TemporaryDirectory() as work:
            report = os.path.join(work, "probe.json")
            code, out, err = run_main(["--waves", "32,64", "--json", report], FakeOracle())
            self.assertEqual((code, err), (0, ""))
            self.assertIn("mismatches=0, wave_diffs=0", out)
            with open(report, encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertEqual(data["arch"], "gfx1036")
            self.assertEqual(data["modes"], list(pr.MODES))
            self.assertEqual(data["mismatches"], 0)
            self.assertEqual(data["wave_diffs"], 0)
            # The 16/64-bit results keep the scaling only in the flushing mode; the f32 result keeps it in two.
            scaled_modes = {mode for mode in pr.MODES if pr.scaling_applies("f64", pr.MODES[mode])}
            self.assertEqual(scaled_modes, {"i0_d0"})
            self.assertEqual({mode for mode in pr.MODES if pr.scaling_applies("f32", pr.MODES[mode])},
                             {"i0_d0", "game_i0_d3"})
            self.assertGreater(data["scaling_gated_on"], 0)
            self.assertGreater(data["scaling_gated_off"], data["scaling_gated_on"])

    def test_device_that_ignores_the_mode_gate_is_reported(self):
        code, out, err = run_main(["--waves", "32"], FakeOracle(apply_scaling_everywhere=True))
        self.assertEqual(code, 1)
        self.assertIn("MISMATCH game_i0_d3", out)
        self.assertIn("v_cvt_f64_i32", out)

    def test_wave32_wave64_difference_is_reported(self):
        code, out, _err = run_main(["--waves", "32,64"], FakeOracle(wave64_differs=True))
        self.assertEqual(code, 1)
        self.assertIn("WAVE DIFF", out)

    def test_unavailable_or_non_gfx10_oracle_exits_77(self):
        with mock.patch.object(runner, "open_oracle", side_effect=runner.Unavailable("no device")):
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(pr.main([]), 77)
            self.assertIn("hardware oracle unavailable", out.getvalue())
        code, out, _err = run_main([], FakeOracle(arch="gfx1100"))
        self.assertEqual((code, out.strip()), (77, "hardware oracle unavailable: gfx1100 is not GFX10"))

    def test_unknown_mode_is_a_cli_error(self):
        with self.assertRaises(SystemExit) as caught:
            with redirect_stderr(io.StringIO()):
                pr.main(["--modes", "game_i0_d3,not_a_mode"])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
