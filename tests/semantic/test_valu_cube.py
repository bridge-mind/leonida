"""Semantic value tests for RDNA2 cube-map VALU ops: v_cubeid_f32, v_cubesc_f32, v_cubetc_f32, v_cubema_f32.

Each test translates a synthetic pixel shader with tools/rdna2spirv.py, runs it in tests/semantic/spirv_mini_interp.py,
and compares the four exported mrt0 binary32 words (face_id, sc, tc, ma) against a plain-Python reference oracle.

RDNA2 ISA cube-map formulas for inputs (S0.f, S1.f, S2.f) = (x, y, z):
  Major axis & face selection (Z wins ties over Y and X; Y wins ties over X):
    if |z| >= |x| and |z| >= |y|:  major = Z   (z < 0 -> face 5.0 [-Z], else 4.0 [+Z])
    elif |y| >= |x|:               major = Y   (y < 0 -> face 3.0 [-Y], else 2.0 [+Y])
    else:                          major = X   (x < 0 -> face 1.0 [-X], else 0.0 [+X])

  Projected face coordinates (sc, tc) and major axis (ma):
    +X (0.0): sc = -z, tc = -y, ma = 2.0 * x
    -X (1.0): sc = +z, tc = -y, ma = 2.0 * x
    +Y (2.0): sc = +x, tc = +z, ma = 2.0 * y
    -Y (3.0): sc = +x, tc = -z, ma = 2.0 * y
    +Z (4.0): sc = +x, tc = -y, ma = 2.0 * z
    -Z (5.0): sc = -x, tc = -y, ma = 2.0 * z

Note on v_cubema_f32: RDNA2 ISA returns 2.0 * signed_major_axis (negative on -X/-Y/-Z faces), so |v_cubema_f32| equals
2.0 * max(|x|, |y|, |z|). That is why hardware cube-sampling sequences such as runtime/test/shaders/ps_cube_sample.asm
follow v_cubema_f32 with `v_rcp_f32_e64 dst, |src|`.
"""
import itertools
import math
import os
import struct
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

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def f32(x: float) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bits_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def ref_cube(x: float, y: float, z: float) -> tuple[float, float, float, float]:
    """Independent RDNA2 ISA reference returning binary32 (face_id, sc, tc, ma)."""
    x, y, z = f32(x), f32(y), f32(z)
    ax, ay, az = abs(x), abs(y), abs(z)
    # Measured natively on RDNA2 (gfx1036): v_cubema_f32 returns +0.0 (never -0.0) when the major axis is ±0.0.
    ma = lambda v: 0.0 if v == 0.0 else f32(2.0 * v)  # noqa: E731
    if az >= ax and az >= ay:
        if z < 0.0:
            return (5.0, f32(-x), f32(-y), ma(z))
        return (4.0, x, f32(-y), ma(z))
    if ay >= ax:
        if y < 0.0:
            return (3.0, x, f32(-z), ma(y))
        return (2.0, x, z, ma(y))
    if x < 0.0:
        return (1.0, z, f32(-y), ma(x))
    return (0.0, f32(-z), f32(-y), ma(x))


def ref_cube_words(x: float, y: float, z: float) -> tuple[int, int, int, int]:
    return tuple(f32_bits(v) for v in ref_cube(x, y, z))


# Hand-worked (x, y, z) -> expected (face_id, sc, tc, ma) constants across all 6 faces and ties.
HAND_WORKED = (
    # All 6 faces with asymmetric coordinates
    ((4.0, -1.5, 0.75), (0.0, -0.75, 1.5, 8.0)),      # +X
    ((-4.0, 1.5, -0.75), (1.0, -0.75, -1.5, -8.0)),   # -X
    ((1.25, 3.0, -0.5), (2.0, 1.25, -0.5, 6.0)),      # +Y
    ((-1.25, -3.0, 0.5), (3.0, -1.25, -0.5, -6.0)),   # -Y
    ((-0.5, 1.25, 2.5), (4.0, -0.5, -1.25, 5.0)),     # +Z
    ((0.5, -1.25, -2.5), (5.0, -0.5, 1.25, -5.0)),    # -Z
    # Tie cases: |x| == |y| (> |z|) -> Y wins; |y| == |z| (> |x|) -> Z wins; |x| == |y| == |z| -> Z wins
    ((2.0, 2.0, 0.5), (2.0, 2.0, 0.5, 4.0)),          # +Y beats +X
    ((2.0, -2.0, 0.5), (3.0, 2.0, -0.5, -4.0)),       # -Y beats +X
    ((-2.0, 2.0, -0.5), (2.0, -2.0, -0.5, 4.0)),      # +Y beats -X
    ((-2.0, -2.0, -0.5), (3.0, -2.0, 0.5, -4.0)),     # -Y beats -X
    ((0.5, 2.0, 2.0), (4.0, 0.5, -2.0, 4.0)),         # +Z beats +Y
    ((0.5, -2.0, -2.0), (5.0, -0.5, 2.0, -4.0)),      # -Z beats -Y
    ((1.5, -1.5, 1.5), (4.0, 1.5, 1.5, 3.0)),         # +Z wins 3-way tie
    ((-1.5, 1.5, -1.5), (5.0, 1.5, -1.5, -3.0)),      # -Z wins 3-way tie
)


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_cube.s")
    with open(os.path.join(_TMP_DIR.name, "ps_cube.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_lines(lines: list[str]) -> tuple[int, int, int, int]:
    """Translate `lines` with exp mrt0 v0, v1, v5, v6 and return the four 32-bit words."""
    if _TMP_DIR is None:
        setUpModule()
    asm = ("; test_valu_cube\n"
           + "".join(f"\t{ln}\n" for ln in lines)
           + "\texp mrt0 v0, v1, v5, v6 done vm\n\ts_endpgm\n")
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return tuple(si.run_fragment(spv)["%out_mrt0"])


def run_cube(x_bits: int, y_bits: int, z_bits: int, suffix: str = "", srcs: str = "v2, v3, v4",
             mods: str = "") -> tuple[int, int, int, int]:
    mod_s = f" {mods}" if mods else ""
    return run_lines([
        f"v_mov_b32 v2, {x_bits:#x}",
        f"v_mov_b32 v3, {y_bits:#x}",
        f"v_mov_b32 v4, {z_bits:#x}",
        f"v_cubeid_f32{suffix} v0, {srcs}{mod_s}",
        f"v_cubesc_f32{suffix} v1, {srcs}{mod_s}",
        f"v_cubetc_f32{suffix} v5, {srcs}{mod_s}",
        f"v_cubema_f32{suffix} v6, {srcs}{mod_s}",
    ])


class OracleTests(unittest.TestCase):
    """Hand-worked constants agree with the plain-Python reference oracle."""

    def test_hand_worked_cases(self):
        for xyz, want in HAND_WORKED:
            with self.subTest(xyz=xyz):
                self.assertEqual(ref_cube(*xyz), want)
                self.assertEqual(ref_cube_words(*xyz), tuple(f32_bits(v) for v in want))

    def test_signed_zero_negations_and_origin(self):
        # (1, 0, 0) on +X: sc = -z = -0.0, tc = -y = -0.0
        self.assertEqual(ref_cube_words(1.0, 0.0, 0.0), (0x00000000, 0x80000000, 0x80000000, 0x40000000))
        # (0, 0, 0) picks +Z (|z| >= |x| and |z| >= |y|, z not < 0): tc = -y = -0.0
        self.assertEqual(ref_cube_words(0.0, 0.0, 0.0), (0x40800000, 0x00000000, 0x80000000, 0x00000000))


class CubeValuTests(unittest.TestCase):
    def assert_cube(self, x: float, y: float, z: float, suffix: str = ""):
        xb, yb, zb = f32_bits(x), f32_bits(y), f32_bits(z)
        got = run_cube(xb, yb, zb, suffix=suffix)
        want = ref_cube_words(x, y, z)
        self.assertEqual(got, want, f"xyz=({x}, {y}, {z}) got={[hex(w) for w in got]} want={[hex(w) for w in want]}")
        # Also confirm |v_cubema_f32| == 2.0 * max(|x|, |y|, |z|).
        got_ma = bits_f32(got[3])
        self.assertEqual(abs(got_ma), f32(2.0 * max(abs(f32(x)), abs(f32(y)), abs(f32(z)))))

    def test_all_six_faces(self):
        faces = [
            ("+X", (4.0, -1.5, 0.75)),
            ("+X_unit", (1.0, 0.5, -0.25)),
            ("-X", (-4.0, 1.5, -0.75)),
            ("-X_unit", (-2.0, -0.5, 1.25)),
            ("+Y", (1.25, 3.0, -0.5)),
            ("+Y_asym", (-0.75, 2.5, 1.5)),
            ("-Y", (-1.25, -3.0, 0.5)),
            ("-Y_asym", (0.75, -2.5, -1.5)),
            ("+Z", (-0.5, 1.25, 2.5)),
            ("+Z_asym", (1.5, -0.75, 4.0)),
            ("-Z", (0.5, -1.25, -2.5)),
            ("-Z_asym", (-1.5, 0.75, -4.0)),
        ]
        for label, (x, y, z) in faces:
            with self.subTest(face=label, x=x, y=y, z=z):
                self.assert_cube(x, y, z)

    def test_major_axis_doubling_boundary(self):
        for major_bits, expected_ma in ((0x7EFFFFFF, 0x7F7FFFFF), (0x7F000000, 0x7F800000),
                                        (0xFEFFFFFF, 0xFF7FFFFF), (0xFF000000, 0xFF800000)):
            with self.subTest(major_bits=hex(major_bits)):
                major = bits_f32(major_bits)
                self.assertEqual(ref_cube_words(major, 1.0, 1.0)[3], expected_ma)
                self.assert_cube(major, 1.0, 1.0)

    def test_axis_tie_cases(self):
        # |x| == |y| with |z| < |x| -> Y wins over X for all sign combinations.
        for sx, sy, sz in itertools.product((-1.0, 1.0), repeat=3):
            x, y, z = 2.0 * sx, 2.0 * sy, 0.5 * sz
            with self.subTest(tie="xy", x=x, y=y, z=z):
                got = run_cube(f32_bits(x), f32_bits(y), f32_bits(z))
                self.assertIn(bits_f32(got[0]), (2.0, 3.0))
                self.assertEqual(got, ref_cube_words(x, y, z))

        # |y| == |z| with |x| < |z| -> Z wins over Y for all sign combinations.
        for sx, sy, sz in itertools.product((-1.0, 1.0), repeat=3):
            x, y, z = 0.5 * sx, 2.0 * sy, 2.0 * sz
            with self.subTest(tie="yz", x=x, y=y, z=z):
                got = run_cube(f32_bits(x), f32_bits(y), f32_bits(z))
                self.assertIn(bits_f32(got[0]), (4.0, 5.0))
                self.assertEqual(got, ref_cube_words(x, y, z))

        # |x| == |z| with |y| < |z| -> Z wins over X for all sign combinations.
        for sx, sy, sz in itertools.product((-1.0, 1.0), repeat=3):
            x, y, z = 2.0 * sx, 0.5 * sy, 2.0 * sz
            with self.subTest(tie="xz", x=x, y=y, z=z):
                got = run_cube(f32_bits(x), f32_bits(y), f32_bits(z))
                self.assertIn(bits_f32(got[0]), (4.0, 5.0))
                self.assertEqual(got, ref_cube_words(x, y, z))

        # 3-way tie |x| == |y| == |z| -> Z wins for all 8 sign combinations.
        for sx, sy, sz in itertools.product((-1.0, 1.0), repeat=3):
            x, y, z = 1.5 * sx, 1.5 * sy, 1.5 * sz
            with self.subTest(tie="xyz", x=x, y=y, z=z):
                got = run_cube(f32_bits(x), f32_bits(y), f32_bits(z))
                self.assertIn(bits_f32(got[0]), (4.0, 5.0))
                self.assertEqual(got, ref_cube_words(x, y, z))

    def test_zero_and_negative_coordinates(self):
        cases = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
            (-0.125, -0.375, -0.25),
            (-0.6, -0.3, -0.9),
            (-64.5, -12.25, -3.75),
            (0.3, -0.6, -0.15),
        ]
        for x, y, z in cases:
            with self.subTest(x=x, y=y, z=z):
                self.assert_cube(x, y, z)

    def test_signed_zero_triples_and_nan_cubema(self):
        # Measured natively on RDNA2 (gfx1036, wave32 & wave64, all 4 IEEE/DX10_CLAMP modes):
        # v_cubema_f32 returns +0.0 (0x00000000) for every (±0.0, ±0.0, ±0.0) triple, including
        # when z is -0.0 (or negated from +0.0 via -v4) and with output modifiers like mul:2.
        for xb, yb, zb in itertools.product((0x00000000, 0x80000000), repeat=3):
            with self.subTest(bits=(hex(xb), hex(yb), hex(zb))):
                got = run_cube(xb, yb, zb)
                want = ref_cube_words(bits_f32(xb), bits_f32(yb), bits_f32(zb))
                self.assertEqual(got, want)
                self.assertEqual(got[3], 0x00000000)
        self.assertEqual(run_cube(0x00000000, 0x00000000, 0x00000000, srcs="v2, v3, -v4")[3], 0x00000000)
        self.assertEqual(run_cube(0x00000000, 0x00000000, 0x80000000, mods="mul:2")[3], 0x00000000)
        # NaN on the selected major axis is preserved; -0.0 selected with NaN on other axes still gives +0.0.
        self.assertEqual(run_cube(0xFFC00000, 0x3F800000, 0x3F800000)[3], 0xFFC00000)
        self.assertEqual(run_cube(0x80000000, 0x7FC00000, 0x7FC00000)[3], 0x00000000)

    def test_vop3_e64_suffix_and_source_modifiers(self):
        # _e64 form matches unsuffixed VOP3 form.
        self.assert_cube(-0.75, 2.5, -3.25, suffix="_e64")

        # Source modifiers apply before face selection: (-v2, |v3|, -|v4|).
        vectors = [
            (1.0, -3.0, 2.0),
            (-4.0, -1.5, -0.5),
            (0.25, 0.5, -2.5),
        ]
        for x, y, z in vectors:
            with self.subTest(x=x, y=y, z=z):
                got = run_cube(f32_bits(x), f32_bits(y), f32_bits(z), srcs="-v2, |v3|, -|v4|")
                want = ref_cube_words(-f32(x), abs(f32(y)), -abs(f32(z)))
                self.assertEqual(got, want)

    def test_inline_constants_sgpr_and_dst_aliasing(self):
        # Inline float constants + SGPR operand.
        got = run_lines([
            f"s_mov_b32 s4, {f32_bits(-2.0):#x}",
            "v_cubeid_f32 v0, 1.0, s4, 0.5",
            "v_cubesc_f32 v1, 1.0, s4, 0.5",
            "v_cubetc_f32 v5, 1.0, s4, 0.5",
            "v_cubema_f32 v6, 1.0, s4, 0.5",
        ])
        self.assertEqual(got, ref_cube_words(1.0, -2.0, 0.5))

        # Destination register aliasing one of the source registers (v2).
        got_alias = run_lines([
            f"v_mov_b32 v2, {f32_bits(1.25):#x}",
            f"v_mov_b32 v3, {f32_bits(-3.0):#x}",
            f"v_mov_b32 v4, {f32_bits(0.5):#x}",
            "v_cubesc_f32 v1, v2, v3, v4",
            "v_cubetc_f32 v5, v2, v3, v4",
            "v_cubema_f32 v6, v2, v3, v4",
            "v_cubeid_f32 v2, v2, v3, v4",
            "v_mov_b32 v0, v2",
        ])
        self.assertEqual(got_alias, ref_cube_words(1.25, -3.0, 0.5))

    def test_output_modifiers_mul_div_clamp(self):
        x, y, z = 0.75, -0.5, -2.0      # -Z: id=5.0, sc=-0.75, tc=0.5, ma=-4.0
        # clamp saturates to [0.0, 1.0]: 5.0 -> 1.0, -0.75 -> 0.0, 0.5 -> 0.5, -4.0 -> 0.0
        got_clamp = run_cube(f32_bits(x), f32_bits(y), f32_bits(z), mods="clamp")
        self.assertEqual(got_clamp, tuple(f32_bits(v) for v in (1.0, 0.0, 0.5, 0.0)))

        # div:2 halves each result (giving signed major axis directly on v_cubema_f32).
        got_div2 = run_cube(f32_bits(x), f32_bits(y), f32_bits(z), mods="div:2")
        self.assertEqual(got_div2, tuple(f32_bits(v * 0.5) for v in ref_cube(x, y, z)))

        # mul:2 doubles each result.
        got_mul2 = run_cube(f32_bits(x), f32_bits(y), f32_bits(z), mods="mul:2")
        self.assertEqual(got_mul2, tuple(f32_bits(v * 2.0) for v in ref_cube(x, y, z)))

    def test_exec_mask_preserves_inactive_lane(self):
        got = run_lines([
            "v_mov_b32 v0, 0x11111111",
            "v_mov_b32 v1, 0x22222222",
            "v_mov_b32 v5, 0x33333333",
            "v_mov_b32 v6, 0x44444444",
            f"v_mov_b32 v2, {f32_bits(1.0):#x}",
            f"v_mov_b32 v3, {f32_bits(2.0):#x}",
            f"v_mov_b32 v4, {f32_bits(-3.0):#x}",
            "s_mov_b32 exec_lo, 0",
            "v_cubeid_f32 v0, v2, v3, v4",
            "v_cubesc_f32 v1, v2, v3, v4",
            "v_cubetc_f32 v5, v2, v3, v4",
            "v_cubema_f32 v6, v2, v3, v4",
            "s_mov_b32 exec_lo, 1",
        ])
        self.assertEqual(got, (0x11111111, 0x22222222, 0x33333333, 0x44444444))

    def test_negative_control_swapped_axes_changes_output(self):
        base = run_cube(f32_bits(1.25), f32_bits(-3.0), f32_bits(0.5))
        swapped = run_cube(f32_bits(1.25), f32_bits(-3.0), f32_bits(0.5), srcs="v3, v2, v4")
        self.assertNotEqual(base, swapped)


if __name__ == "__main__":
    unittest.main()
