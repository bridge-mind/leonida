"""Every GFX10 buffer data format the translator knows, decoded and encoded, against tools/buffer_format_fuzz.py's MODEL.

MODEL (decode_element / encode_element there) is what the Radeon 880M (gfx1150, RDNA3.5, not RDNA2) did for every
pattern of tools/buffer_format_fuzz.py --oracle hw: 1.33 million validated elements over 63 formats on the 880M and the same on a real RDNA2, loads through a V# and
through tbuffer's own format, stores through both (docs/buffer_formats.md, docs/design.md "Typed buffer formats"). This
file holds the translated SPIR-V to it: the SPIR-V mini interpreter runs `buffer_load_format_xyzw` (the V# format and
tbuffer's own) and `buffer_store_format_xyzw` of every format added with the SCALED / SINT / 10_11_11 / 11_11_10 /
2_10_10_10 / 10_10_10_2 families, with edge values, NaN, infinity and denormals, and compares every word.

Older formats are checked for their table entry and, for stores, against the model too (StoresOfTheOlderFormats); their
loads are covered by test_buffer_load_format.py, test_norm_decode.py and the GPU fixtures.
"""
import os
import random
import sys
import unittest

import support
import spirv_mini_interp as si
import test_value_semantics as tv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, support.TOOLS_DIR)
import buffer_format_fuzz as model  # noqa: E402
import rdna2spirv  # noqa: E402

IN_BASE, OUT_BASE = 0x1000, 0x2000
OUT_V = 0x31016FAC                      # s[0:3] of the compute fixtures: raw, base 0
SENTINEL = 0xDEAD0000
DST_XYZW = 4 | 5 << 3 | 6 << 6 | 7 << 9

# Added with this change: every code the SCALED, SINT, small-float and 10:10:10:2 families add to Gen.FORMATS.
ADDED = (3, 4, 5, 6, 9, 10, 11, 12, 16, 17, 18, 19, 25, 26, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
         45, 46, 47, 48, 49, 51, 52, 53, 54, 55,
         58, 59, 61, 67, 68, 70)
# Of the formats above, the ones whose store the table covers (all; a SCALED store converts as its NORM format).
GFX10_ONLY = (30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 46, 47)     # no GFX11 spelling: a real RDNA2 validated them


def v_sharp_lines(reg, base, code):
    w3 = 0x30000000 | code << 12 | DST_XYZW
    return [f"s_mov_b32 s{reg}, {base:#x}", f"s_mov_b32 s{reg + 1}, 0", f"s_mov_b32 s{reg + 2}, 0x100",
            f"s_mov_b32 s{reg + 3}, {w3:#x}"]


def translate_load(code, path):
    if path == "mubuf":
        fetch = "buffer_load_format_xyzw v[2:5], v1, s[4:7], 0 offen"
        sharp = code
    else:                                       # tbuffer takes the format from the instruction, not the V# (32_FLOAT)
        fetch = f"tbuffer_load_format_xyzw v[2:5], v1, s[4:7], 0 format:[BUF_FMT_{model.GFX10_NAMES[code]}] offen"
        sharp = 22
    lines = ["v_cmpx_eq_u32_e32 0, v0", *v_sharp_lines(4, IN_BASE, sharp), "v_mov_b32 v1, 0", fetch,
             "s_waitcnt vmcnt(0)", "buffer_store_dwordx4 v[2:5], off, s[0:3], 0", "s_endpgm"]
    return support.translate_lines("cs_fmt_all_load", lines, fssm=tv.CS_BOUNDS_FSSM)


def translate_store(code, instr="xyzw"):
    lines = ["v_cmpx_eq_u32_e32 0, v0", *v_sharp_lines(4, OUT_BASE, code), *v_sharp_lines(8, IN_BASE, 22),
             "v_mov_b32 v1, 0", "buffer_load_dwordx4 v[2:5], v1, s[8:11], 0 offen", "s_waitcnt vmcnt(0)",
             f"buffer_store_format_{instr} v[2:{5 if instr == 'xyzw' else 3}], v1, s[4:7], 0 offen", "s_endpgm"]
    return support.translate_lines("cs_fmt_all_store", lines, fssm=tv.CS_BOUNDS_FSSM)


def run(spv, memory):
    support.run_fragment(spv, inputs={"%wg_id": (0, 0, 0), "%local_id": (0, 0, 0)},
                         push_constants=[0, 0, 0x100, OUT_V] + [0] * 28, memory=memory)
    return memory


def words_of(data):
    return [int.from_bytes(data[4 * i:4 * i + 4], "little") for i in range(4)]


SPECIAL = {          # field values worth a lane each, whatever the format: the model's own edges plus these
    "FLOAT": (0, 1, 0x3F, 0x40, 0x7BF, 0x7C0, 0x7C1, 0x7DF, 0x7FF, 0x3DF, 0x3E0, 0x3E1, 0x3FF, 0x20, 0x1F, 0x400, 0x3C00),
}


def load_elements(name, rng):
    elems = model.load_patterns(name, rng, 3)
    widths, nfmt = model.parse_name(name)
    keep = elems[:2] + rng.sample(elems[2:], 24)
    if nfmt == "FLOAT":
        for c, w in enumerate(widths):
            for v in SPECIAL["FLOAT"]:
                if v < 1 << w:
                    comps = [rng.getrandbits(x) for x in widths]
                    comps[c] = v
                    keep.append(model.pack(widths, comps, rng, 0))
    for c, w in enumerate(widths):              # every field at its own extremes, the other fields random
        for v in (0, 1, (1 << w) - 1, 1 << (w - 1), (1 << (w - 1)) - 1, (1 << (w - 1)) + 1):
            comps = [rng.getrandbits(x) for x in widths]
            comps[c] = v
            keep.append(model.pack(widths, comps, rng, 0))
    return keep


class FormatTable(unittest.TestCase):
    def test_every_entry_agrees_with_its_gfx10_name(self):
        """FORMATS[code] = (bytes, kind, widths from bit 0) must be what the name says, for the old entries too."""
        kinds = {"UNORM": "unorm", "SNORM": "snorm", "USCALED": "uscaled", "SSCALED": "sscaled", "SINT": "sint"}
        for code, (size, kind, widths) in rdna2spirv.Gen.FORMATS.items():
            with self.subTest(code=code):
                want_widths, nfmt = model.parse_name(model.GFX10_NAMES[code])
                self.assertEqual(widths, want_widths)
                self.assertEqual(size, sum(widths) // 8)
                if nfmt == "UINT":
                    self.assertIn(kind, ("uint", "u32"))
                elif nfmt == "SINT":
                    self.assertIn(kind, ("sint", "i32") if widths != (32,) * len(widths) else ("i32",))
                elif nfmt == "FLOAT":
                    self.assertIn(kind, ("f32", "f16", "ufloat"))
                else:
                    self.assertEqual(kind, kinds[nfmt])

    def test_every_buffer_code_is_in_the_table(self):
        for code in ADDED:
            self.assertIn(code, rdna2spirv.Gen.FORMATS)
        self.assertEqual([c for c in range(1, 78) if c not in rdna2spirv.Gen.FORMATS], [])
        self.assertEqual(len(rdna2spirv.Gen.FORMATS), 77)


class LoadsMatchTheModel(unittest.TestCase):
    def check(self, code, path):
        name = model.GFX10_NAMES[code]
        spv = translate_load(code, path)
        rng = random.Random(code * 7 + (path == "mubuf"))
        count = 0
        for e in load_elements(name, rng):
            memory = {IN_BASE + 4 * i: w for i, w in enumerate(words_of(e))}
            run(spv, memory)
            got = [memory.get(4 * k, 0) for k in range(4)]
            self.assertEqual(got, model.decode_element(name, e, path), (name, path, e[:model.element_size(name)].hex()))
            count += 1
        return count

    def test_loads_through_the_v_sharp(self):
        for code in ADDED:
            with self.subTest(format=model.GFX10_NAMES[code]):
                self.check(code, "mubuf")

    def test_loads_through_tbuffer_formats(self):
        for code in ADDED:
            with self.subTest(format=model.GFX10_NAMES[code]):
                self.check(code, "mtbuf")


class StoresMatchTheModel(unittest.TestCase):
    def test_stores_through_the_v_sharp(self):
        for code in ADDED:
            name = model.GFX10_NAMES[code]
            with self.subTest(format=name):
                spv = translate_store(code)
                rng = random.Random(code * 13)
                rows = model.store_inputs(name, rng)
                for row in rng.sample(rows, 48):
                    data = b"".join(w.to_bytes(4, "little") for w in row)
                    memory = {IN_BASE + 4 * i: w for i, w in enumerate(row)}
                    memory.update({OUT_BASE + 4 * i: SENTINEL for i in range(4)})
                    run(spv, memory)
                    got = b"".join(memory.get(OUT_BASE + 4 * i, 0).to_bytes(4, "little") for i in range(4))
                    want = bytearray(b"".join(SENTINEL.to_bytes(4, "little") for _ in range(4)))
                    size = model.element_size(name)
                    want[:size] = model.encode_element(name, row)
                    self.assertEqual(got, bytes(want), (name, [f"{w:08x}" for w in row]))


class HandWorkedValues(unittest.TestCase):
    """Values worked out by hand from the format definitions, so the model is not only checked against itself."""

    def load(self, code, data, path="mubuf"):
        memory = {IN_BASE + 4 * i: w for i, w in enumerate(words_of(bytes(data) + bytes(16 - len(data))))}
        run(translate_load(code, path), memory)
        return [memory.get(4 * k, 0) for k in range(4)]

    def test_uint_sint_8_zero_and_sign_extend(self):
        self.assertEqual(self.load(5, [0xFF])[0], 0xFF)                  # 8_UINT
        self.assertEqual(self.load(6, [0xFF])[0], 0xFFFFFFFF)            # 8_SINT: -1
        self.assertEqual(self.load(6, [0x80])[0], 0xFFFFFF80)
        self.assertEqual(self.load(12, [0x00, 0x80])[0], 0xFFFF8000)     # 16_SINT

    def test_scaled_is_the_integer_as_float(self):
        self.assertEqual(self.load(3, [0xFF])[0], 0x437F0000)            # 8_USCALED: 255.0
        self.assertEqual(self.load(4, [0x80])[0], 0xC3000000)            # 8_SSCALED: -128.0
        self.assertEqual(self.load(10, [0x00, 0x80])[0], 0xC7000000)     # 16_SSCALED: -32768.0

    def test_small_floats(self):
        # 11_11_10_FLOAT (43): x is the 10 bit field in bits 9:0 (5 bit mantissa), y and z are 11 bit
        # 10_11_11_FLOAT (36): x is the 11 bit field in bits 10:0 (6 bit mantissa), z the 10 bit one in 31:22
        one11 = 15 << 6                                                  # 1.0 in 11 bits: exponent 15
        self.assertEqual(self.load(36, one11.to_bytes(4, "little"))[0], 0x3F800000)
        self.assertEqual(self.load(43, (15 << 5).to_bytes(4, "little"))[0], 0x3F800000)
        self.assertEqual(self.load(36, (0x7C0).to_bytes(4, "little"))[0], 0x7F800000)       # infinity
        self.assertEqual(self.load(36, (0x7C1).to_bytes(4, "little"))[0], 0x7F820000)       # NaN, payload kept
        self.assertEqual(self.load(36, (1).to_bytes(4, "little"))[0], 0x35800000)           # 2^-20, a denormal
        self.assertEqual(self.load(43, (1 << 22).to_bytes(4, "little"))[2], 0x36000000)    # z field 2: 2 * 2^-20
        self.assertEqual(self.load(36, ((15 << 5) << 22).to_bytes(4, "little"))[2], 0x3F800000)   # z, 10 bits: 15 << 5

    def test_2_10_10_10_sint_and_snorm(self):
        word = (1 << 30) | (0x201 << 20) | (0x1FF << 10) | 0x200        # w = 1, z = 0x201, y = 0x1FF, x = 0x200
        data = word.to_bytes(4, "little")
        self.assertEqual(self.load(55, data), [0xFFFFFE00, 511, 0xFFFFFE01, 1])
        self.assertEqual(self.load(51, data)[0], 0xBF800000)             # SNORM: the most negative code, -512, is -1
        self.assertEqual(self.load(51, data)[1], 0x3F800000)             # +511 / 511
        self.assertEqual(self.load(51, data)[2], 0xBF800000)             # 0x201 is -511: -511 / 511

    def test_stores_hand_worked(self):
        def store(code, regs):
            memory = {IN_BASE + 4 * i: w for i, w in enumerate(regs)}
            memory.update({OUT_BASE + 4 * i: SENTINEL for i in range(4)})
            run(translate_store(code), memory)
            return b"".join(memory[OUT_BASE + 4 * i].to_bytes(4, "little") for i in range(4))

        self.assertEqual(store(6, [0x100, 0, 0, 0])[:2], bytes([0x7F, 0x00]))     # 8_SINT saturates to 127
        self.assertEqual(store(6, [0xFFFFFF00, 0, 0, 0])[0], 0x80)                # ... and to -128
        self.assertEqual(store(5, [0x1FF, 0, 0, 0])[0], 0xFF)                     # 8_UINT saturates
        self.assertEqual(store(3, [0x3F800000, 0, 0, 0])[0], 0xFF)                # 8_USCALED stores like 8_UNORM
        self.assertEqual(store(36, [0x3F800000, 0, 0, 0])[:4], (15 << 6).to_bytes(4, "little"))
        self.assertEqual(store(36, [0x7F800000, 0, 0, 0])[:2], (0x7C0).to_bytes(2, "little"))   # +inf
        self.assertEqual(store(36, [0xBF800000, 0, 0, 0])[:2], bytes(2))          # negative gives 0
        self.assertEqual(store(36, [0x47FFE000, 0, 0, 0])[:2], (0x7BF).to_bytes(2, "little"))   # 65504: the maximum
        self.assertEqual(store(36, [0x4B000000, 0, 0, 0])[:2], (0x7BF).to_bytes(2, "little"))   # too large: the maximum
        self.assertEqual(store(45, [0x3F000000, 0, 0, 0])[0] & 3, 1)              # 10_10_10_2_SNORM, 2 bit x: 0.5 -> 1


class StoresOfTheOlderFormats(unittest.TestCase):
    """Stores of formats that were in the table before the SCALED / SINT / small-float families, which used to differ from the
    hardware in three places and now follow it (docs/buffer_formats.md): 16-bit float stores round toward zero and saturate,
    UNORM / SNORM stores round the exact product once, and a two-component format takes channels (x, w) through a V# with identity selectors.
    Measured on the Radeon 880M (gfx1150) and on a real RDNA2, gfx1036 (@AwkGrep256, #2022)."""

    def store(self, code, regs, instr="xyzw"):
        memory = {IN_BASE + 4 * i: w for i, w in enumerate(regs)}
        memory.update({OUT_BASE + 4 * i: SENTINEL for i in range(4)})
        run(translate_store(code, instr), memory)
        return memory[OUT_BASE]

    def test_two_component_v_sharp_store_takes_x_and_w(self):
        # buffer_store_format_xyzw of (1, 2, 3, 4) into 8_8_SINT writes 01 04, buffer_store_format_xy writes 01 00
        self.assertEqual(self.store(19, [1, 2, 3, 4]) & 0xFFFF, 0x0401)
        self.assertEqual(self.store(19, [1, 2, 3, 4], "xy") & 0xFFFF, 0x0001)
        self.assertEqual(self.store(27, [0x1111, 0x2222, 0x3333, 0x4444]), 0x44441111)    # 16_16_UINT

    def test_two_component_store_with_the_runtime_selectors_takes_x_and_y(self):
        # The runtime gives a two-component format DST_SEL X, Y, 0, 1 (gfx10_descriptors.h): channel y then, as measured natively
        # (gfx1036, test_buffer_format_narrow_int.py; gfx1150: XY01, XY0W and XY00 all give (x, y), XYZW and XYZ1 give (x, w))
        for sel, want in ((4 | 5 << 3 | 0 << 6 | 1 << 9, 0x0201), (4 | 5 << 3 | 0 << 6 | 7 << 9, 0x0201),
                          (4 | 5 << 3 | 6 << 6 | 1 << 9, 0x0401), (4 | 0 << 3 | 0 << 6 | 1 << 9, 0x0401)):
            lines = ["v_cmpx_eq_u32_e32 0, v0", *v_sharp_lines(4, OUT_BASE, 19), "v_mov_b32 v1, 0",
                     f"s_mov_b32 s7, {0x30000000 | 19 << 12 | sel:#x}", "v_mov_b32 v2, 1", "v_mov_b32 v3, 2",
                     "v_mov_b32 v4, 3", "v_mov_b32 v5, 4", "buffer_store_format_xyzw v[2:5], v1, s[4:7], 0 offen", "s_endpgm"]
            spv = support.translate_lines("cs_fmt_all_sel", lines, fssm=tv.CS_BOUNDS_FSSM)
            memory = {OUT_BASE: SENTINEL}
            run(spv, memory)
            self.assertEqual(memory[OUT_BASE] & 0xFFFF, want, hex(sel))

    def test_two_component_tbuffer_store_takes_x_and_y(self):
        lines = ["v_cmpx_eq_u32_e32 0, v0", *v_sharp_lines(4, OUT_BASE, 22), "v_mov_b32 v1, 0",
                 "v_mov_b32 v2, 1", "v_mov_b32 v3, 2", "v_mov_b32 v4, 3", "v_mov_b32 v5, 4",
                 "tbuffer_store_format_xyzw v[2:5], v1, s[4:7], 0 format:[BUF_FMT_8_8_SINT] offen", "s_endpgm"]
        spv = support.translate_lines("cs_fmt_all_tbuf", lines, fssm=tv.CS_BOUNDS_FSSM)
        memory = {OUT_BASE: SENTINEL}
        run(spv, memory)
        self.assertEqual(memory[OUT_BASE] & 0xFFFF, 0x0201)

    def test_16_float_store_rounds_toward_zero_and_saturates(self):
        # 65520.0 is the tie between 65504 and infinity: nearest even says infinity, the hardware 0x7BFF
        self.assertEqual(self.store(13, [0x47FFE000, 0, 0, 0]) & 0xFFFF, 0x7BFF)
        self.assertEqual(self.store(13, [0x3EFFFFFF, 0, 0, 0]) & 0xFFFF, 0x37FF)     # 0.49999997: truncated, not 0x3800
        self.assertEqual(self.store(13, [0x7149F2CA, 0, 0, 0]) & 0xFFFF, 0x7BFF)     # 1e30
        self.assertEqual(self.store(13, [0x7F800001, 0, 0, 0]) & 0xFFFF, 0x7C01)     # a NaN stays one
        self.assertEqual(self.store(13, [0xFF800000, 0, 0, 0]) & 0xFFFF, 0xFC00)     # -infinity

    def test_unorm_store_rounds_the_exact_product(self):
        # 0x37C000C0 * 65535 is a hair under 1.5: the hardware stores 1, an f32 product rounds up to 1.5 and stores 2
        self.assertEqual(self.store(7, [0x37C000C0, 0, 0, 0]) & 0xFFFF, 1)
        self.assertEqual(self.store(7, [0x3E99999A, 0, 0, 0]) & 0xFFFF, 19661)      # 0.3 * 65535

    def test_every_older_format_matches_the_model(self):
        older = [c for c, (_, kind, _) in rdna2spirv.Gen.FORMATS.items()
                 if c not in ADDED and kind not in ("f32", "u32", "i32")]
        for code in older:
            name = model.GFX10_NAMES[code]
            with self.subTest(format=name):
                spv = translate_store(code)
                rng = random.Random(code * 29)
                for row in rng.sample(model.store_inputs(name, rng), 32):
                    memory = {IN_BASE + 4 * i: w for i, w in enumerate(row)}
                    memory.update({OUT_BASE + 4 * i: SENTINEL for i in range(4)})
                    run(spv, memory)
                    got = b"".join(memory[OUT_BASE + 4 * i].to_bytes(4, "little") for i in range(4))
                    want = bytearray(b"".join(SENTINEL.to_bytes(4, "little") for _ in range(4)))
                    want[:model.element_size(name)] = model.encode_element(name, row)
                    self.assertEqual(got, bytes(want), (name, [f"{w:08x}" for w in row]))


if __name__ == "__main__":
    unittest.main()
