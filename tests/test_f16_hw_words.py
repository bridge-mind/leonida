"""The ps_f16_* GPU test shaders: expected words measured on RDNA2 hardware, checked here without a GPU.

HARDWARE holds, per shader, the four mrt0 words an AMD RX 6800 (gfx1030) returned when the shader's instructions
ran as a raw compute kernel (registers renamed, nothing else changed), in wave32 and in wave64, with IEEE mode off,
DX10 clamp on, denormals preserved and round to nearest even. They are not derived from the translator.

Three things are checked for every shader:
  * the words in the .asm header, in check.py and in scripts/run_gpu_tests.py are those hardware words;
  * a plain-Python reference written from the rule in docs/design.md ("16-bit results") gives the same words;
  * the translator's SPIR-V, run by tests/spirv_interp.py, gives the same words.
"""
import os
import re
import struct
import sys

import pytest

import rdna2spirv
import spirv_interp as si
from listing import EMPTY_DXBC, translate_listing

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHADER_DIR = os.path.join(ROOT, "runtime", "test", "shaders")
sys.path.insert(0, SHADER_DIR)
import check  # noqa: E402  (path set above)

HARDWARE = {
    "ps_f16_med3_opsel": (0xAAAA4100, 0xAAAA4200, 0xAAAA3E00, 0x40005555),
    "ps_f16_med3_mods": (0x41005555, 0x3C005555, 0x41003800, 0xAAAABA00),
    "ps_f16_vop2": (0xAAAA4000, 0xAAAAC000, 0xAAAA3A00, 0x42003F00),
    "ps_f16_vop2b": (0xAAAA3E00, 0xAAAAB800, 0xAAAA3C00, 0x42003400),
    "ps_f16_vop1": (0xAAAA4000, 0xAAAABC00, 0xAAAAC000, 0xAAAA8000),
    "ps_f16_vop1b": (0xAAAA3400, 0xAAAA4000, 0xAAAAB400, 0xAAAA3800),
    "ps_f16_const": (0xAAAA3D00, 0x38005555, 0xAAAA3800, 0x3F000000),
    "ps_f16_const2": (0x42003D00, 0x3D004200, 0xAAAABC00, 3),
    "ps_f16_pack": (0x34003C00, 0x38003400, 0x34000000, 0x44004200),
    # neg/abs on the selected half: a sign flip / clear before packing (SHADER-260; the rule was measured on
    # gfx1036 and these words follow it, they are not a direct readback of this shader)
    "ps_f16_pack_mods": (0x4400B400, 0xC4003400, 0x4400C200, 0xB400B400),
}
OLD = 0xAAAA5555


def h(x):
    """f16 bits of x; every value used here is exact in f16."""
    b = struct.unpack("<H", struct.pack("<e", x))[0]
    assert struct.unpack("<e", struct.pack("<H", b))[0] == x
    return b


def lo(w):
    return struct.unpack("<e", struct.pack("<H", w & 0xFFFF))[0]


def hi(w):
    return lo(w >> 16)


def to_lo(old, x):
    """Partial write: x into the low half, the high half of the destination kept."""
    return old & 0xFFFF0000 | h(x)


def to_hi(old, x):
    """op_sel destination bit: x into the high half, the low half kept."""
    return h(x) << 16 | old & 0xFFFF


def med(a, b, c):
    return sorted((a, b, c))[1]


def clamp(x):
    return min(max(x, 0.0), 1.0)


def ref_med3_opsel():
    v0, v1, v2 = 0x41003C00, 0x48004000, 0x3E004200
    return (to_lo(OLD, med(hi(v0), lo(v1), lo(v2))), to_lo(OLD, med(lo(v0), hi(v1), lo(v2))),
            to_lo(OLD, med(lo(v0), lo(v1), hi(v2))), to_hi(OLD, med(lo(v0), lo(v1), lo(v2))))


def ref_med3_mods():
    v0, v1, v2 = 0xC1003800, 0xC8004000, 0x3A003C00
    return (to_hi(OLD, med(-hi(v0), abs(hi(v1)), lo(v2))),
            to_hi(OLD, clamp(med(abs(hi(v0)), abs(hi(v1)), hi(v2)))),
            to_hi(v0, med(-hi(v0), lo(v1), abs(hi(v1)))),             # the destination is a copy of v0
            to_lo(OLD, med(lo(v0), hi(v1), -abs(hi(v2)))))


def ref_vop2():
    v0, v1, acc = 0x44003E00, 0x40003800, 0x42003C00
    return (to_lo(OLD, lo(v0) + lo(v1)), to_lo(OLD, -lo(v0) - abs(lo(v1))), to_lo(OLD, lo(v0) * lo(v1)),
            to_lo(acc, lo(v0) * lo(v1) + lo(acc)))


def ref_vop2b():
    v0, v1, acc = 0x44003E00, 0x40003800, 0x42003C00
    return (to_lo(OLD, max(lo(v0), lo(v1))), to_lo(OLD, min(lo(v0), -lo(v1))), to_lo(OLD, clamp(lo(v0) + lo(v1))),
            to_lo(acc, -lo(v0) * abs(lo(v1)) + lo(acc)))


def ref_vop1():
    v0, v1 = 0xBE004100, 0x4300BA00
    assert (lo(v0), lo(v1)) == (2.5, -0.75)
    #        2.5 -> 2 (even)   floor(-0.75)     ceil(-2.5)        trunc(-0.75) = -0.0
    return to_lo(OLD, 2.0), to_lo(OLD, -1.0), to_lo(OLD, -2.0), OLD & 0xFFFF0000 | 0x8000


def ref_vop1b():
    v0, v1 = 0x34004400, 0x3E00C180
    assert (lo(v0), lo(v1)) == (4.0, -2.75)
    return to_lo(OLD, lo(v1) - (-3.0)), to_lo(OLD, lo(v0) ** 0.5), to_lo(OLD, 1 / -lo(v0)), to_lo(OLD, lo(v0) ** -0.5)


def const(x):
    """An inline float constant in an f16 op: the f16 value in the low half, zero in the high half."""
    return h(x)


def ref_const():
    v0 = 0x42003400
    return (to_lo(OLD, lo(const(1.0)) + lo(v0)), to_hi(OLD, med(hi(v0), lo(const(-4.0)), lo(const(0.5)))),
            to_lo(OLD, lo(v0) * lo(const(2.0)) + hi(const(0.5))),
            struct.unpack("<I", struct.pack("<f", lo(const(0.5))))[0])


def ref_const2():
    v0, one = 0x42003400, const(1.0)
    return (h(hi(one) + hi(v0)) << 16 | h(lo(one) + lo(v0)),        # v_pk_add_f16: each half on its own
            h(lo(one) + lo(v0)) << 16 | h(hi(one) + hi(v0)),        # op_sel [1,1], op_sel_hi [0,0]: halves swapped
            to_lo(OLD, -1.0),                                       # cos(0.5 turns)
            7 if lo(const(0.5)) < lo(v0) else 3)


def ref_pack():
    v0, v1, w = 0x42003400, 0x34004400, 0xFFFF
    return ((v0 & w) << 16 | const(1.0) & w, (const(0.5) & w) << 16 | v0 & w,
            (v0 & w) << 16 | const(2.0) >> 16, (v1 & w) << 16 | v0 >> 16)


def ref_pack_mods():
    # -v / |v| on the selected half: NEG flips bit 15, ABS clears it, ABS first.
    v0, v1, w = 0x42003400, 0x34004400, 0xFFFF
    lo0, hi0, lo1, hi1 = v0 & w, v0 >> 16, v1 & w, v1 >> 16
    return (lo0 ^ 0x8000 | (lo1 & 0x7FFF) << 16, lo0 | (lo1 ^ 0x8000) << 16,
            hi0 ^ 0x8000 | lo1 << 16, lo0 ^ 0x8000 | (hi1 ^ 0x8000) << 16)


REFERENCE = {"ps_f16_med3_opsel": ref_med3_opsel, "ps_f16_med3_mods": ref_med3_mods, "ps_f16_vop2": ref_vop2,
             "ps_f16_vop2b": ref_vop2b, "ps_f16_vop1": ref_vop1, "ps_f16_vop1b": ref_vop1b,
             "ps_f16_const": ref_const, "ps_f16_const2": ref_const2, "ps_f16_pack": ref_pack,
             "ps_f16_pack_mods": ref_pack_mods}
NAMES = sorted(HARDWARE)


def source(name):
    with open(os.path.join(SHADER_DIR, name + ".asm"), encoding="utf-8") as f:
        return f.read()


def translated_words(tmp_path, name):
    """The four words the translated shader writes to mrt0, per tests/spirv_interp.py."""
    insns = [ln.strip() for ln in source(name).splitlines() if ln.strip() and not ln.startswith(";")]
    (tmp_path / "disasm").mkdir(exist_ok=True)
    (tmp_path / f"{name}.dxbc").write_bytes(EMPTY_DXBC)
    path = tmp_path / "disasm" / f"{name}.s"
    path.write_text(f"; {name} code 0x0..0x100\n" + "".join("\t" + ln + "\n" for ln in insns))
    out = si.run(rdna2spirv.translate(str(path))).outputs["out_mrt0"]
    return tuple(si.f2u(x) for x in out)


def test_every_f16_shader_is_listed():
    # ps_f16_fnan_* are the NaN-operand med3 shaders, with their own words and test (tests/semantic/test_float_nan_fract.py)
    on_disk = sorted(f[:-4] for f in os.listdir(SHADER_DIR)
                     if f.startswith("ps_f16_") and f.endswith(".asm") and not f.startswith("ps_f16_fnan_"))
    assert on_disk == NAMES == sorted(REFERENCE)


@pytest.mark.parametrize("name", NAMES)
def test_header_and_checkers_hold_the_hardware_words(name):
    header = re.search(r"^; Expected mrt0 bits: (.*)$", source(name), re.M).group(1)
    assert tuple(int(x, 0) for x in header.split(",")) == HARDWARE[name]
    assert check.PIXEL_BITS[name] == HARDWARE[name]
    with open(os.path.join(ROOT, "scripts", "run_gpu_tests.py"), encoding="utf-8") as f:
        m = re.search(rf'"{name}": \(([^)]*)\)', f.read())
    assert m and tuple(int(x, 0) for x in m.group(1).split(",")) == HARDWARE[name]


@pytest.mark.parametrize("name", NAMES)
def test_rule_reference_gives_the_hardware_words(name):
    assert tuple(f"{w:#010x}" for w in REFERENCE[name]()) == tuple(f"{w:#010x}" for w in HARDWARE[name])


@pytest.mark.parametrize("name", NAMES)
def test_translator_gives_the_hardware_words(name, tmp_path):
    got = translated_words(tmp_path, name)
    assert tuple(f"{w:#010x}" for w in got) == tuple(f"{w:#010x}" for w in HARDWARE[name])


# 0x3118 for 0.15915494 (1 / 2pi) is also what the hardware gave for v_max_f16 d, 0.15915494, v with v.lo = -4.0.
@pytest.mark.parametrize("text,bits", [("0.5", 0x3800), ("1.0", 0x3C00), ("2.0", 0x4000), ("4.0", 0x4400),
                                       ("-0.5", 0xB800), ("-1.0", 0xBC00), ("-2.0", 0xC000), ("-4.0", 0xC400),
                                       ("0.15915494", 0x3118)])
def test_inline_float_constant_is_an_f16_value(text, bits, tmp_path):
    """Every float constant the assembler prints for an f16 operand, read through a v_max_f16 that keeps it."""
    asm = f"v_mov_b32 v1, 0xfc00fc00\nv_mov_b32 v3, {OLD:#x}\nv_max_f16 v3, {text}, v1"      # max(c, -inf) = c
    assert si.run(translate_listing(tmp_path, asm)).vgpr[3] == OLD & 0xFFFF0000 | bits
