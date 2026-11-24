"""The GPU fixtures for docs/design.md rules that used to be checked on the CPU only (TEST-175).

Each fixture in runtime/test/shaders writes four mrt0 words that its .asm header derives by hand ("Expected mrt0
bits"). These tests keep the three places that state those words in step with the translator: the interpreter run of
the translated fixture (tests/semantic/fixtures/disasm, made by regen_fixtures.py), check.py's PIXEL_BITS and
scripts/run_gpu_tests.py's table. A GPU run of check.py or run_gpu_tests.py then compares the same words with what
the device writes. The two interp fixtures are drawn with vs_interp_consts, whose params are constant over the
triangle; the interpreter gets those constants, read back from that vertex shader's source.
"""
import os
import re
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SHADERS = os.path.join(ROOT, "runtime", "test", "shaders")
sys.path.insert(0, HERE)
sys.path.insert(0, SHADERS)

import support  # noqa: E402
import check    # noqa: E402  (runtime/test/shaders/check.py; importing it runs nothing)

FIXTURES = ("ps_cmp_nlg", "ps_cvt_off", "ps_fmac_f16_mods", "ps_vop3_const_mods", "ps_interp_f32_mods",
            "ps_interp_mov_mods")
INTERP_VERTEX = "vs_interp_consts"
INLINE_F32 = {"0.5": 0x3F000000, "-0.5": 0xBF000000, "1.0": 0x3F800000, "-1.0": 0xBF800000, "2.0": 0x40000000,
              "-2.0": 0xC0000000, "4.0": 0x40800000, "-4.0": 0xC0800000, "0.25": 0x3E800000, "-0.25": 0xBE800000}


def read(name):
    with open(os.path.join(SHADERS, name + ".asm"), encoding="utf-8") as f:
        return f.read()


def header_words(name):
    m = re.search(r"Expected mrt0 bits: (.*)", read(name))
    return tuple(int(x, 16) for x in m.group(1).split(","))


def runner_words(name):
    """The tuple scripts/run_gpu_tests.py expects for `name` (its table is local to run_suite, so read the source)."""
    with open(os.path.join(ROOT, "scripts", "run_gpu_tests.py"), encoding="utf-8") as f:
        m = re.search(r'"%s": \(([^)]*)\)' % re.escape(name), f.read())
    return tuple(int(x, 16) for x in m.group(1).split(",")) if m else None


def vertex_params(name):
    """{param n: four f32 values} exported by a vertex shader made of v_mov_b32 constants and `exp paramN` lines."""
    regs, params = {}, {}
    for line in read(name).splitlines():
        line = line.split(";")[0].strip()
        m = re.match(r"v_mov_b32(?:_e32)? v(\d+), (\S+)$", line)
        if m:
            v = m.group(2)
            regs[int(m.group(1))] = INLINE_F32[v] if v in INLINE_F32 else int(v, 0) & 0xFFFFFFFF
            continue
        m = re.match(r"exp param(\d+) v(\d+), v(\d+), v(\d+), v(\d+)", line)
        if m:
            params[int(m.group(1))] = tuple(struct.unpack("<f", struct.pack("<I", regs[int(r)]))[0]
                                            for r in m.groups()[1:])
    return params


def interpreter_words(name):
    spv = support.translate_fixture(name)
    kw = {}
    if name in check.PIXEL_VERTEX and check.PIXEL_VERTEX[name] == INTERP_VERTEX:
        params = vertex_params(INTERP_VERTEX)
        kw["inputs"] = {f"%in_attr{n}": v for n, v in params.items() if f"%in_attr{n}" in spv}
    return tuple(support.mrt0_words(spv, **kw))


class RuleFixtureTests(unittest.TestCase):
    def test_interpreter_gives_the_header_words(self):
        for name in FIXTURES:
            with self.subTest(name=name):
                self.assertEqual([f"{w:#010x}" for w in interpreter_words(name)],
                                 [f"{w:#010x}" for w in header_words(name)])

    def test_both_gpu_runners_expect_the_header_words(self):
        for name in FIXTURES:
            with self.subTest(name=name):
                self.assertEqual(check.PIXEL_BITS.get(name), header_words(name))
                self.assertEqual(runner_words(name), header_words(name))

    def test_no_word_is_a_float_pattern_a_render_target_could_change(self):
        # vk_harness renders to R32G32B32A32_SFLOAT: keep every word a normal float or +-0, never NaN, inf or denormal.
        for name in FIXTURES:
            for w in header_words(name):
                exp = (w >> 23) & 0xFF
                with self.subTest(name=name, word=f"{w:#010x}"):
                    self.assertTrue(w & 0x7FFFFFFF == 0 or 0 < exp < 0xFF)

    def test_interp_fixtures_are_drawn_with_the_constant_vertex_shader(self):
        params = vertex_params(INTERP_VERTEX)
        self.assertEqual(params[0], (1.5, -0.25, 0.375, 0.625))
        self.assertEqual(params[1], (1.5, -0.5, 0.25, 4.0))
        self.assertEqual(params[2], (0.25, 3.0, 0.0, 0.0))
        for name in ("ps_interp_f32_mods", "ps_interp_mov_mods"):
            self.assertEqual(check.PIXEL_VERTEX[name], INTERP_VERTEX)
            self.assertIn(f"vk_harness {INTERP_VERTEX}.spv {name}.spv", read(name))

    def test_attr1_is_flat_and_the_others_interpolate(self):
        # attr1 is read only with v_interp_mov, so the translator declares it Flat; attr2 goes through p1/p2.
        spv = support.translate_fixture("ps_interp_mov_mods")
        self.assertIn("OpDecorate %in_attr1 Flat", spv)
        self.assertNotIn("OpDecorate %in_attr2 Flat", spv)
        self.assertNotIn("PerVertexKHR", spv)


if __name__ == "__main__":
    unittest.main()
