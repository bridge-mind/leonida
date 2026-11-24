"""A half NaN in a typed buffer load widens by shifting its payload (measured on RDNA2).

The widening had never been measured: tools/mem_diff.py decoded 16_FLOAT with struct's '<e' (payload kept from
Python 3.14) while the translator decoded it with UnpackHalf2x16, which the mini interpreter canonicalises to
0x7FC00000. Measured natively on the AMD Radeon integrated GPU of a Ryzen 9 9950X3D (gfx1036, RDNA2) with
tools/f16_widen_probe.py (AMD clang 23, wave32 and wave64 alike; formats 13 16_FLOAT, 29 16_16_FLOAT and
71 16_16_16_16_FLOAT; buffer_load_format_* and tbuffer_load_format_*): every half NaN becomes f32 with
sign | 0x7F800000 | (payload << 13). The sign is kept, the 10-bit payload moves into the f32 mantissa unchanged,
and the quiet bit is NOT forced: the signalling half 0x7DDB gives 0x7FBB6000, bit 22 clear.

v_cvt_f32_f16, which a driver lowers UnpackHalf2x16 to, differs on an sNaN: measured 0x7FBB6000 for the typed
load but 0x7FFB6000 (quieted, payload kept) for the ALU instruction. The translator therefore builds the NaN case
from the bits instead of trusting UnpackHalf2x16, whose result for a NaN is undefined.

MEASURED rows: (half word, f32 word). The same f32 word came back in every component of every format and form.
"""

import os
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(32)


BASE = 0x1000
DST_SEL_XYZW = 4 | 5 << 3 | 6 << 6 | 7 << 9
FMT_F16, FMT_F16X2, FMT_F16X4 = 13, 29, 71

MEASURED = (
    (0x7DDB, 0x7FBB6000),  # sNaN 7ddb: payload kept, quiet bit still clear
    (0x7FE6, 0x7FFCC000),  # qNaN 7fe6
    (0x7FBB, 0x7FF76000),  # qNaN with another payload
    (0xFFB2, 0xFFF64000),  # negative qNaN, sign kept
    (0x7FCB, 0x7FF96000),
    (0xFF96, 0xFFF2C000),
    (0x7C01, 0x7F802000),  # sNaN with the smallest payload
    (0xFC01, 0xFF802000),  # negative sNaN with the smallest payload
    (0x3C00, 0x3F800000),  # 1.0, control
    (0xC000, 0xC0000000),  # -2.0, control
)

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_buf_f16_widen.s")
    with open(os.path.join(_TMP_DIR.name, "ps_buf_f16_widen.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def load_mubuf(fmt, memory, ncomp=4):
    """Four MRT words of buffer_load_format_* with DST_SEL XYZW on format `fmt`."""
    if _TMP_DIR is None:
        setUpModule()
    vdata = "v2" if ncomp == 1 else f"v[2:{1 + ncomp}]"
    regs = [f"v{2 + min(i, ncomp - 1)}" for i in range(4)]
    asm = (
        "; test_buffer_f16_widen_hw\n"
        f"\ts_mov_b32 s0, {BASE:#x}\n"
        "\ts_mov_b32 s1, 0\n"
        "\ts_mov_b32 s2, 0x100\n"
        f"\ts_mov_b32 s3, {fmt << 12 | DST_SEL_XYZW | 0x30000000:#x}\n"
        "\tv_mov_b32 v1, 0\n"
        f"\tbuffer_load_format_{'xyzw'[:ncomp]} {vdata}, v1, s[0:3], 0 offen\n"
        "\ts_waitcnt vmcnt(0)\n"
        f"\texp mrt0 {', '.join(regs)} done vm\n"
        "\ts_endpgm\n"
    )
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return si.run_fragment(spv, memory=dict(memory))["%out_mrt0"]


def load_mtbuf(buf_fmt, memory, ncomp=4):
    """Four MRT words of tbuffer_load_format_*; the MTBUF selectors are the identity, so absent channels are 0."""
    if _TMP_DIR is None:
        setUpModule()
    vdata = "v2" if ncomp == 1 else f"v[2:{1 + ncomp}]"
    regs = [f"v{2 + min(i, ncomp - 1)}" for i in range(4)]
    asm = (
        "; test_buffer_f16_widen_hw\n"
        f"\ts_mov_b32 s0, {BASE:#x}\n"
        "\ts_mov_b32 s1, 0\n"
        "\ts_mov_b32 s2, 0x100\n"
        f"\ts_mov_b32 s3, {FMT_F16 << 12 | DST_SEL_XYZW | 0x30000000:#x}\n"
        "\tv_mov_b32 v1, 0\n"
        f"\ttbuffer_load_format_{'xyzw'[:ncomp]} {vdata}, v1, s[0:3], 0"
        f" format:[BUF_FMT_{buf_fmt}] offen\n"
        "\ts_waitcnt vmcnt(0)\n"
        f"\texp mrt0 {', '.join(regs)} done vm\n"
        "\ts_endpgm\n"
    )
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return si.run_fragment(spv, memory=dict(memory))["%out_mrt0"]


class MeasuredWiden(unittest.TestCase):
    def test_mubuf_16_float_x(self):
        for half, want in MEASURED:
            with self.subTest(half=hex(half)):
                self.assertEqual(load_mubuf(FMT_F16, {BASE: half}, 1)[0], want)

    def test_mtbuf_16_float_x(self):
        for half, want in MEASURED:
            with self.subTest(half=hex(half)):
                self.assertEqual(load_mtbuf("16_FLOAT", {BASE: half}, 1)[0], want)

    def test_mubuf_16_16_float_xy(self):
        for half, want in MEASURED:
            for other, other_want in ((0x7FE6, 0x7FFCC000), (0xC000, 0xC0000000)):
                with self.subTest(half=hex(half), other=hex(other)):
                    got = load_mubuf(FMT_F16X2, {BASE: other << 16 | half}, 2)
                    self.assertEqual(got[:2], (want, other_want))

    def test_mubuf_16_float_repeats_x_in_every_lane(self):
        # One-component format, DST_SEL XYZW: hardware repeats X (measured), and so does the translator.
        for half, want in MEASURED:
            with self.subTest(half=hex(half)):
                self.assertEqual(load_mubuf(FMT_F16, {BASE: half}, 4), (want,) * 4)

    def test_mubuf_16_16_16_16_float_xyzw(self):
        words = (0x7FE6 << 16) | 0x7DDB, (0xFF96 << 16) | 0xFFB2
        memory = {BASE: words[0], BASE + 4: words[1]}
        got = load_mubuf(FMT_F16X4, memory)
        self.assertEqual(got, (0x7FBB6000, 0x7FFCC000, 0xFFF64000, 0xFFF2C000))
        self.assertEqual(
            load_mtbuf("16_16_16_16_FLOAT", memory),
            (0x7FBB6000, 0x7FFCC000, 0xFFF64000, 0xFFF2C000),
        )

    def test_mtbuf_absent_channels_stay_zero(self):
        # The identity MTBUF selectors zero-fill absent channels, even for a NaN in X.
        self.assertEqual(
            load_mtbuf("16_16_FLOAT", {BASE: 0x7FE6 << 16 | 0x7DDB}, 4),
            (0x7FBB6000, 0x7FFCC000, 0, 0),
        )

    def test_snan_quiet_bit_is_not_forced(self):
        # The one bit that separates the typed load from v_cvt_f32_f16 (measured 0x7FFB6000 there).
        got = load_mubuf(FMT_F16, {BASE: 0x7DDB}, 1)[0]
        self.assertEqual(got, 0x7FBB6000)
        self.assertEqual(got & 0x00400000, 0)


if __name__ == "__main__":
    unittest.main()
