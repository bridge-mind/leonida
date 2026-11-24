"""tools/d16_image_probe.py: the textures and samplers, the kernels (they assemble for gfx1031), the comparison and its
exit codes through a fake oracle that plays the GPU, and the exit-77 contract. The last class runs on the hardware
when the oracle tools are present and the GPU is GFX10, and is skipped otherwise. Standard library only."""
import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import d16_format_probe as fp  # noqa: E402
import d16_image_probe as ip  # noqa: E402
from hw_oracle import asm, runner  # noqa: E402


class Textures(unittest.TestCase):
    def test_self_test(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(ip.self_test(), 0)

    def test_every_value_of_a_16_bit_format_is_one_texel(self):
        seen = [v for b in range(ip.blocks("unorm16")) for v in ip.block_values("unorm16", b)]
        self.assertEqual(seen, list(range(65536)))

    def test_the_float32_set_starts_with_the_two_unorm_halfway_quotients(self):
        v, ties = ip.float32_texels()
        self.assertEqual([fp.f32_bits(fp.to_f32(fp.Fraction(65519, 65535))),
                          fp.f32_bits(fp.to_f32(fp.Fraction(65455, 65535)))], [v[0], v[2]])
        self.assertTrue(all((b & 0x1FFF) == 0x1000 for b in v[:ties]), "the ties come first")
        self.assertEqual(len(v), ip.TEXELS)

    def test_t_sharp_carries_the_format(self):
        for fmt, (code, _, _, width) in ip.FORMATS.items():
            w = ip.t_words("gfx1031", fmt)
            self.assertEqual((w[1] >> 20) & 0x1FF, code, fmt)
            self.assertEqual(w[2] >> 31, 1)                        # RESOURCE_LEVEL on GFX10
            self.assertEqual(((w[2] & 0xFFF) << 2 | w[1] >> 30) + 1, width)


class Kernels(unittest.TestCase):
    def test_kernels_assemble_for_rdna2(self):
        try:
            clang = runner.find_clang()
        except runner.Unavailable as e:
            self.skipTest(str(e))
        for wave in (32, 64):
            kernels = [asm.Kernel(ip.kernel_lines("gfx1031", f"{f}_0", f, ip.block_values(f, 0), 0.5), wave,
                                  ip.TEXELS, ip.TEXELS * ip.WORDS_PER_LANE, raw=True, vgprs=24, sgprs=32)
                       for f in ip.FORMATS]
            text, _, _ = asm.module_text(kernels, "gfx1031", wave=wave)
            with tempfile.TemporaryDirectory() as d:
                s = Path(d) / "k.s"
                s.write_text(text, encoding="utf-8")
                r = subprocess.run([clang, "--target=amdgcn-amd-amdhsa", "-mcpu=gfx1031",
                                    *(["-mwavefrontsize64"] if wave == 64 else []), "-c", str(s), "-o",
                                    str(Path(d) / "k.o")], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[:800])
            self.assertIn("image_sample_lz v13, v[16:17], s[8:15], s[20:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm d16",
                          text)


def _val(kind, n, v, rule):
    return fp.models16(kind, n, v)[rule]


class FakeOracle:
    """Plays the GPU from each kernel's label (.Lpc<fmt>_<block>) and its texture: d16 results by `rule`, f32 loads
    correctly rounded, bilinear f32 the rounded exact mix and its d16 rounded from that f32 to nearest even."""
    arch = "gfx1031"
    device = "fake gfx1031"

    def __init__(self, rule="rte", f32_off=False, wave64_flip=False, tie="even"):
        self.rule, self.f32_off, self.wave64_flip, self.tie = rule, f32_off, wave64_flip, tie

    def run(self, kernels, words=4):
        out = []
        for k in kernels:
            fmt, block = re.match(r"\.Lpc(\w+?)_(\d+):", k.lines[1]).groups()
            _, kind, n, width = ip.FORMATS[fmt]
            vals = ip.block_values(fmt, int(block))
            frac = struct.unpack("<f", struct.pack("<I", int(re.search(
                r"v_add_f32 v16, 0x([0-9a-f]+)", "\n".join(k.lines)).group(1), 16)))[0]
            res = []
            for lane, v in enumerate(vals):
                right = vals[lane + 1] if lane % width + 1 < width else v
                if kind in ("unorm", "snorm"):
                    load = fp.models32(kind, n, v)["f32_rte"] + (1 if self.f32_off else 0)
                    d16 = _val(kind, n, v, self.rule)
                    mix = fp.quotient(kind, n, v) * (1 - fp.Fraction(frac)) + fp.quotient(kind, n, right) * \
                        fp.Fraction(frac)
                elif kind == "f16":
                    load = fp.f32_bits(struct.unpack("<e", struct.pack("<H", v))[0]) if (v & 0x7C00) != 0x7C00 \
                        else 0x7FC00000
                    d16, mix = v, fp.Fraction(0)
                else:
                    x = fp.Fraction(struct.unpack("<f", struct.pack("<I", v))[0])
                    load = v
                    d16 = fp.f16_bits(fp.to_f16(x, self.tie == "down" and (v & 0x1FFF) == 0x1000))
                    mix = fp.Fraction(0)
                if self.wave64_flip and k.wave == 64 and lane == 5:
                    d16 ^= 1
                b32 = fp.f32_bits(fp.to_f32(mix))
                b16 = fp.f16_bits(fp.to_f16(fp.to_f32(mix)))
                hi = ip.SEED & 0xFFFF0000
                res += [load, hi | d16, load, hi | d16, b32, hi | b16, lane, 0]
            out.append(res)
        return out

    def close(self):
        pass


def run_main(oracle, *args):
    buf = io.StringIO()
    with mock.patch.object(runner, "open_oracle", return_value=oracle), redirect_stdout(buf):
        rc = ip.main(list(args))
    return rc, buf.getvalue()


class Compare(unittest.TestCase):
    def test_single_rounding_hardware_is_four_values_off_the_translator(self):
        with tempfile.TemporaryDirectory() as d:
            j = os.path.join(d, "r.json")
            rc, out = run_main(FakeOracle(), "--formats", "unorm16,snorm16", "--waves", "32", "--json", j)
            with open(j, encoding="utf-8") as f:
                rep = json.load(f)
        self.assertEqual(rc, 0, out)
        for fmt in ("unorm16", "snorm16"):
            e = rep["formats"][fmt]
            self.assertEqual(e["fields"]["load_d16"]["off_f32_rte"], 2)
            self.assertEqual(e["fields"]["load_d16"]["match"]["rte"], 65536)
            self.assertEqual(e["fields"]["bilinear"]["d16_is_rte_of_own_f32"], 65536)
            self.assertEqual(e["f32_load_is_rounded_quotient"], 65536)
        self.assertEqual(rep["formats"]["unorm16"]["fields"]["load_d16"]["first_off"][0]["value"], "0xffaf")

    def test_float32_halfway_points_are_counted_apart(self):
        for tie, key in (("even", "to_even"), ("down", "toward_zero")):
            with tempfile.TemporaryDirectory() as d:
                j = os.path.join(d, "r.json")
                rc, out = run_main(FakeOracle(tie=tie), "--formats", "float32", "--waves", "32", "--json", j)
                with open(j, encoding="utf-8") as f:
                    h = json.load(f)["formats"]["float32"]["fields"]["load_d16"]["halfway"]
            self.assertEqual(rc, 0, out)
            self.assertEqual(h[key] + h["either"], h["inputs"], tie)
            self.assertEqual(h["other"], 0)
            self.assertGreaterEqual(h[key], 20)                 # ties whose even neighbour is away from zero
            self.assertEqual(h["inputs"], ip.float32_texels()[1])

    def test_an_f32_load_one_ulp_off_is_counted_not_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            j = os.path.join(d, "r.json")
            rc, out = run_main(FakeOracle(f32_off=True), "--formats", "unorm8", "--waves", "32", "--json", j)
            with open(j, encoding="utf-8") as f:
                e = json.load(f)["formats"]["unorm8"]
        self.assertEqual(rc, 0, out)
        self.assertEqual(e["f32_load_is_rounded_quotient"], 0)

    def test_a_short_runner_result_is_a_harness_error(self):
        """zip() would drop the tail and the counts would quietly cover fewer texels (BUILD-733 in d16_format_probe)."""
        short = FakeOracle()
        full = short.run

        def truncated(kernels, words=4):
            return full(kernels, words)[:-1]

        short.run = truncated
        rc, out = run_main(short, "--formats", "unorm16", "--waves", "32")
        self.assertEqual(rc, 2, out)
        self.assertIn("returned 63 results for 64 kernels", out)

    def test_a_wave_difference_exits_1(self):
        rc, out = run_main(FakeOracle(wave64_flip=True), "--formats", "unorm8,float16")
        self.assertEqual(rc, 1, out)

    def test_not_gfx10_exits_77(self):
        o = FakeOracle()
        o.arch = "gfx1150"
        rc, out = run_main(o, "--formats", "unorm8")
        self.assertEqual(rc, 77, out)


class Cli(unittest.TestCase):
    def test_exits_77_without_the_tools(self):
        env = dict(os.environ, LEONIDA_AMD_CLANG=str(TOOLS / "no-such-clang.exe"), LEONIDA_HW_DRM_NODE="")
        r = subprocess.run([sys.executable, "-I", str(TOOLS / "d16_image_probe.py"), "--formats", "unorm8"],
                           capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 77, r.stdout + r.stderr)

    def test_bad_arguments_are_a_usage_error(self):
        for args in (["--formats", "unorm12"], ["--frac", "1.5"]):
            r = subprocess.run([sys.executable, "-I", str(TOOLS / "d16_image_probe.py"), *args],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 2, args)


class OnHardware(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.oracle = runner.open_oracle()
        except runner.Unavailable as e:
            raise unittest.SkipTest(f"hardware oracle unavailable: {e}") from None
        if asm.family(cls.oracle.arch) != "gfx10":
            cls.oracle.close()
            raise unittest.SkipTest(f"{cls.oracle.arch} is not GFX10")

    @classmethod
    def tearDownClass(cls):
        cls.oracle.close()

    def test_unorm8_controls_hold(self):
        hw = ip.sweep(self.oracle, "unorm8", 32, 0.5)
        res = ip.classify("unorm8", hw, 0.5)
        self.assertEqual(res["control"], [])
        self.assertEqual(hw["load"][255], 0x3F800000)
        self.assertEqual(hw["load_d16"][255] & 0xFFFF, 0x3C00)


if __name__ == "__main__":
    unittest.main()
