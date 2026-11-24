from __future__ import annotations

"""Semantic interpreter tests for RDNA2 v_add_nc_i32 / v_sub_nc_i32 and the integer clamp bit.

v_add_nc_i32 / v_sub_nc_i32: D = S0 + S1 / S0 - S1, wrapping at 32 bits. With the VOP3 clamp bit an integer
add/sub saturates instead (GFX9+; LLVM selects uaddsat/usubsat to v_add/sub_nc_u32 clamp and saddsat/ssubsat to
v_add/sub_nc_i32 clamp): the unsigned ops to [0, 0xFFFFFFFF], the signed ones to [0x80000000, 0x7FFFFFFF].
v_subrev_nc_u32 is S1 - S0 and saturates the same way. Each case translates a synthetic pixel shader with
tools/rdna2spirv.py, runs it in tests/semantic/spirv_mini_interp.py and compares the result with a plain-Python
reference. D leaves through exp as two 16-bit halves ORed into 0x40000000, so no export channel is a NaN or denormal
pattern that a float round trip could change.
"""
import os
import random
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
import spirv_wave_interp as wi  # noqa: E402

M32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def s32(x: int) -> int:
    x &= M32
    return x - (1 << 32) if x >> 31 else x


def ref_add_nc_u32(x: int, y: int, clamp: bool = False) -> int:
    """Pure reference oracle for RDNA2 unsigned 32-bit addition with optional VOP3 clamp.

    Computes (x & M32) + (y & M32). Without clamp, the sum wraps modulo 2^32.
    With clamp, if the 33-bit sum exceeds 0xFFFFFFFF (generating a carry),
    the result saturates to 0xFFFFFFFF (GFX9+ uaddsat behavior).
    """
    a = x & M32
    b = y & M32
    full_sum = a + b
    if clamp:
        return min(full_sum, M32)
    return full_sum & M32


def oracle_add_nc_u32_analysis(x: int, y: int) -> dict[str, int | bool]:
    """Arithmetic analysis for v_add_nc_u32: full sum, carry, wrapped, and clamped results."""
    a = x & M32
    b = y & M32
    total = a + b
    has_carry = total > M32
    wrapped = total & M32
    clamped = M32 if has_carry else total
    return {
        "a": a,
        "b": b,
        "total": total,
        "carry": has_carry,
        "wrapped": wrapped,
        "clamped": clamped,
    }


def ref(op: str, x: int, y: int, clamp: bool) -> int:
    if op == "v_subrev_nc_u32":
        x, y, op = y, x, "v_sub_nc_u32"
    signed = op.endswith("_i32")
    a, b = (s32(x), s32(y)) if signed else (x & M32, y & M32)
    r = a - b if "_sub" in op else a + b
    if clamp:
        lo, hi = (-(1 << 31), (1 << 31) - 1) if signed else (0, M32)
        r = min(max(r, lo), hi)
    return r & M32


OPS = ["v_add_nc_i32", "v_sub_nc_i32", "v_add_nc_u32", "v_sub_nc_u32", "v_subrev_nc_u32"]
VALUES = [0, 1, 2, 0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFF, 0xFFFFFFFE, 0x40000000, 0xC0000000, 0x12345678]

_TMP_DIR = None
_SRC_PATH = None
_VS_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH, _VS_SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_add_nc_clamp.s")
    _VS_SRC_PATH = os.path.join(disasm_dir, "vs_add_nc_clamp.s")
    with open(os.path.join(_TMP_DIR.name, "ps_add_nc_clamp.dxbc"), "wb") as f:
        f.write(STUB_DXBC)
    with open(os.path.join(_TMP_DIR.name, "vs_add_nc_clamp.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR, _SRC_PATH, _VS_SRC_PATH
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None
        _SRC_PATH = None
        _VS_SRC_PATH = None


def run_wave_vs(lines: list[str], wave: int = 32, base_lane: int = 0) -> list[int | None]:
    """Run a synthetic vertex shader on a wave of lanes, returning param0 word 0 for each lane."""
    if _TMP_DIR is None:
        setUpModule()
    body = ["s_mov_b64 s[10:11], exec"] + list(lines) + [
        "v_mov_b32_e32 v5, 0",
        "exp pos0 v5, v5, v5, v5 done",
        "exp param0 v6, v5, v5, v5",
        "s_endpgm",
    ]
    asm = "; vs_add_nc_clamp\n" + "".join(f"\t{ln}\n" for ln in body)
    with open(_VS_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_VS_SRC_PATH)
    res = wi.run_wave(
        spv,
        wave=wave,
        builtins={
            "VertexIndex": lambda lane, b=base_lane: b + lane,
            "InstanceIndex": lambda _lane: 0,
        },
    )
    outputs: list[int | None] = []
    for out in res["outputs"]:
        if out is None:
            outputs.append(None)
        else:
            outputs.append(out["%out_param0"][0])
    return outputs


def run_lines(lines) -> int:
    """Run the lines, return v0 (exported as two 16-bit halves, see the module docstring)."""
    if _TMP_DIR is None:
        setUpModule()
    lines = list(lines) + ["v_and_b32 v5, 0xffff, v0", "v_lshrrev_b32 v6, 16, v0",
                           "v_or_b32 v5, 0x40000000, v5", "v_or_b32 v6, 0x40000000, v6"]
    asm = "; test_add_nc_clamp\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v5, v6, v5, v5 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    lo, hi = si.run_fragment(spv)["%out_mrt0"][:2]
    assert lo >> 16 == 0x4000 and hi >> 16 == 0x4000, (hex(lo), hex(hi))
    return (lo & 0xFFFF) | (hi & 0xFFFF) << 16


def run_op(op: str, x: int, y: int, clamp: bool) -> int:
    return run_lines([f"v_mov_b32 v1, {x:#x}", f"v_mov_b32 v2, {y:#x}",
                      f"{op}_e64 v0, v1, v2" + (" clamp" if clamp else "")])


class AddNcClampTests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(ref("v_add_nc_i32", 0x7FFFFFFF, 1, False), 0x80000000)
        self.assertEqual(ref("v_add_nc_i32", 0x7FFFFFFF, 1, True), 0x7FFFFFFF)
        self.assertEqual(ref("v_sub_nc_i32", 0x80000000, 1, True), 0x80000000)
        self.assertEqual(ref("v_add_nc_u32", M32, 2, True), M32)
        self.assertEqual(ref("v_sub_nc_u32", 1, 2, True), 0)
        self.assertEqual(ref("v_subrev_nc_u32", 2, 1, True), 0)

    def test_values(self):
        for op in OPS:
            for clamp in (False, True):
                for x in VALUES:
                    for y in VALUES[::2]:
                        with self.subTest(op=op, clamp=clamp, x=hex(x), y=hex(y)):
                            self.assertEqual(run_op(op, x, y, clamp), ref(op, x, y, clamp))

    def test_seeded(self):
        rng = random.Random(0xC1A)
        for _ in range(40):
            op, clamp = rng.choice(OPS), bool(rng.getrandbits(1))
            x, y = rng.getrandbits(32), rng.getrandbits(32)
            with self.subTest(op=op, clamp=clamp, x=hex(x), y=hex(y)):
                self.assertEqual(run_op(op, x, y, clamp), ref(op, x, y, clamp))

    def test_vop2_forms_wrap(self):
        # The VOP2 encodings have no clamp bit: they wrap.
        self.assertEqual(run_lines(["v_mov_b32 v1, -1", "v_add_nc_u32_e32 v0, 2, v1"]), 1)
        self.assertEqual(run_lines(["v_mov_b32 v1, 1", "v_sub_nc_u32_e32 v0, 0, v1"]), M32)


class TestAddNcU32ReferenceOracle(unittest.TestCase):
    """Reference oracle tests for vector addition with clamp saturation v_add_nc_u32."""

    def test_oracle_identity_zero(self) -> None:
        """Adding zero must preserve operand under both wrapping and clamping."""
        test_values = [0, 1, 2, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, 0xFFFFFFFF]
        for val in test_values:
            with self.subTest(val=hex(val)):
                self.assertEqual(ref_add_nc_u32(val, 0, clamp=False), val)
                self.assertEqual(ref_add_nc_u32(val, 0, clamp=True), val)
                self.assertEqual(ref_add_nc_u32(0, val, clamp=False), val)
                self.assertEqual(ref_add_nc_u32(0, val, clamp=True), val)

    def test_oracle_commutativity(self) -> None:
        """v_add_nc_u32 must be strictly commutative with and without clamp."""
        test_pairs = [
            (0, 0),
            (1, 0xFFFFFFFF),
            (0x12345678, 0x87654321),
            (0x7FFFFFFF, 0x80000000),
            (0xFFFFFFFE, 2),
            (0xAAAAAAAA, 0x55555555),
            (0xC0000000, 0x60000000),
            (0xFFFFFFFF, 0xFFFFFFFF),
        ]
        for a, b in test_pairs:
            for clamp in (False, True):
                with self.subTest(a=hex(a), b=hex(b), clamp=clamp):
                    self.assertEqual(ref_add_nc_u32(a, b, clamp), ref_add_nc_u32(b, a, clamp))

    def test_oracle_sub_saturation_equivalence(self) -> None:
        """When the sum does not exceed 0xFFFFFFFF, clamp and no-clamp results must be identical."""
        safe_pairs = [
            (0, 0),
            (10, 20),
            (0x1000, 0x2000),
            (0x7FFFFFFF, 0),
            (0x7FFFFFFF, 0x80000000),  # sum == 0xFFFFFFFF exactly
            (0x55555555, 0x22222222),
            (0xFFFFFFFE, 1),           # sum == 0xFFFFFFFF exactly
        ]
        for a, b in safe_pairs:
            with self.subTest(a=hex(a), b=hex(b)):
                wrapped = ref_add_nc_u32(a, b, clamp=False)
                clamped = ref_add_nc_u32(a, b, clamp=True)
                self.assertEqual(wrapped, clamped)
                self.assertEqual(clamped, a + b)

    def test_oracle_saturation_boundary_transitions(self) -> None:
        """At and across the 2^32 boundary, clamping must saturate to 0xFFFFFFFF while wrapping wraps."""
        cases = [
            # (a, b, expected_wrapped, expected_clamped, expected_carry)
            (0xFFFFFFFF, 0, 0xFFFFFFFF, 0xFFFFFFFF, False),
            (0xFFFFFFFE, 1, 0xFFFFFFFF, 0xFFFFFFFF, False),
            (0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0xFFFFFFFF, False),
            # Transition across boundary:
            (0xFFFFFFFF, 1, 0, 0xFFFFFFFF, True),
            (0xFFFFFFFE, 2, 0, 0xFFFFFFFF, True),
            (0xFFFFFFFF, 2, 1, 0xFFFFFFFF, True),
            (0x80000000, 0x80000000, 0, 0xFFFFFFFF, True),
            (0x80000001, 0x80000000, 1, 0xFFFFFFFF, True),
            (0xC0000000, 0x80000000, 0x40000000, 0xFFFFFFFF, True),
            (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFFF, True),
        ]
        for a, b, exp_wrap, exp_clamp, exp_carry in cases:
            with self.subTest(a=hex(a), b=hex(b)):
                analysis = oracle_add_nc_u32_analysis(a, b)
                self.assertEqual(analysis["carry"], exp_carry)
                self.assertEqual(analysis["wrapped"], exp_wrap)
                self.assertEqual(analysis["clamped"], exp_clamp)
                self.assertEqual(ref_add_nc_u32(a, b, clamp=False), exp_wrap)
                self.assertEqual(ref_add_nc_u32(a, b, clamp=True), exp_clamp)

    def test_oracle_monotonicity_under_saturation(self) -> None:
        """For a fixed base, clamped addition must be monotonically non-decreasing up to saturation."""
        base = 0xFFFFFFF0
        prev_clamped = 0
        for delta in range(32):
            clamped = ref_add_nc_u32(base, delta, clamp=True)
            self.assertGreaterEqual(clamped, prev_clamped)
            if delta < 15:
                self.assertEqual(clamped, base + delta)
            else:
                self.assertEqual(clamped, M32)
            prev_clamped = clamped

    def test_translation_execution_matches_ref_oracle(self) -> None:
        """SPIR-V translation and mini-interpreter execution must match ref_add_nc_u32."""
        sample_pairs = [
            (0, 0),
            (100, 200),
            (0x7FFFFFFF, 1),
            (0xFFFFFFFF, 1),
            (0xFFFFFFFE, 2),
            (0x80000000, 0x80000000),
            (0xFFFFFFFF, 0xFFFFFFFF),
        ]
        for a, b in sample_pairs:
            for clamp in (False, True):
                with self.subTest(a=hex(a), b=hex(b), clamp=clamp):
                    got = run_op("v_add_nc_u32", a, b, clamp)
                    want = ref_add_nc_u32(a, b, clamp)
                    self.assertEqual(got, want)


class TestAddNcU32SimdDivergence(unittest.TestCase):
    """Multi-lane SIMD divergence and clamp boundary verification for v_add_nc_u32."""

    def test_wave32_clamp_boundary_divergence(self) -> None:
        """In wave32, lanes cross the clamp boundary: non-saturating lanes match wrapped, saturated lanes clamp."""
        # Operands: v0 = lane index (0..31), v1 = 0xFFFFFFF0
        # Lanes 0..15: sums 0xFFFFFFF0..0xFFFFFFFF (below or at 2^32 boundary). Clamped == Wrapped == sum.
        # Lanes 16..31: sums 0x100000000..0x10000000F. Clamped == 0xFFFFFFFF, Wrapped == lane - 16.
        lines_clamped = [
            "v_mov_b32_e32 v1, 0xfffffff0",
            "v_add_nc_u32_e64 v6, v0, v1 clamp",
        ]
        lines_wrapped = [
            "v_mov_b32_e32 v1, 0xfffffff0",
            "v_add_nc_u32_e64 v6, v0, v1",
        ]
        clamped_outs = run_wave_vs(lines_clamped, wave=32)
        wrapped_outs = run_wave_vs(lines_wrapped, wave=32)

        self.assertEqual(len(clamped_outs), 32)
        self.assertEqual(len(wrapped_outs), 32)

        for lane in range(32):
            expected_clamped = ref_add_nc_u32(lane, 0xFFFFFFF0, clamp=True)
            expected_wrapped = ref_add_nc_u32(lane, 0xFFFFFFF0, clamp=False)
            self.assertEqual(clamped_outs[lane], expected_clamped, f"Mismatch at clamped lane {lane}")
            self.assertEqual(wrapped_outs[lane], expected_wrapped, f"Mismatch at wrapped lane {lane}")

            if lane <= 15:
                # Sub-saturation: clamp and wrap must be identical
                self.assertEqual(clamped_outs[lane], wrapped_outs[lane])
                self.assertEqual(clamped_outs[lane], 0xFFFFFFF0 + lane)
            else:
                # Saturation: clamp is 0xFFFFFFFF while wrap rolls over
                self.assertEqual(clamped_outs[lane], M32)
                self.assertEqual(wrapped_outs[lane], lane - 16)
                self.assertNotEqual(clamped_outs[lane], wrapped_outs[lane])

    def test_wave64_clamp_boundary_divergence(self) -> None:
        """In wave64, verify multi-lane divergence across the 64-lane wavefront."""
        # Operands: v0 = lane index (0..63), v1 = 0xFFFFFFE0
        # Lanes 0..31: sums 0xFFFFFFE0..0xFFFFFFFF (clamped == wrapped == 0xFFFFFFE0 + lane)
        # Lanes 32..63: sums 0x100000000..0x10000001F (clamped == 0xFFFFFFFF, wrapped == lane - 32)
        lines_clamped = [
            "v_mov_b32_e32 v1, 0xffffffe0",
            "v_add_nc_u32_e64 v6, v0, v1 clamp",
        ]
        lines_wrapped = [
            "v_mov_b32_e32 v1, 0xffffffe0",
            "v_add_nc_u32_e64 v6, v0, v1",
        ]
        clamped_outs = run_wave_vs(lines_clamped, wave=64)
        wrapped_outs = run_wave_vs(lines_wrapped, wave=64)

        self.assertEqual(len(clamped_outs), 64)
        self.assertEqual(len(wrapped_outs), 64)

        for lane in range(64):
            expected_clamped = ref_add_nc_u32(lane, 0xFFFFFFE0, clamp=True)
            expected_wrapped = ref_add_nc_u32(lane, 0xFFFFFFE0, clamp=False)
            self.assertEqual(clamped_outs[lane], expected_clamped, f"Wave64 mismatch at clamped lane {lane}")
            self.assertEqual(wrapped_outs[lane], expected_wrapped, f"Wave64 mismatch at wrapped lane {lane}")

            if lane <= 31:
                self.assertEqual(clamped_outs[lane], wrapped_outs[lane])
            else:
                self.assertEqual(clamped_outs[lane], M32)
                self.assertEqual(wrapped_outs[lane], lane - 32)
                self.assertNotEqual(clamped_outs[lane], wrapped_outs[lane])

    def test_wave32_inactive_exec_mask(self) -> None:
        """Disabled lanes under EXEC mask must not update their destination register."""
        sentinel = 0xDEADBEEF
        lines = [
            f"v_mov_b32_e32 v6, {sentinel:#x}",
            "s_mov_b32 exec_lo, 0x0000ffff",  # only lanes 0..15 active
            "v_mov_b32_e32 v1, 0xffffffe0",
            "v_add_nc_u32_e64 v6, v0, v1 clamp",
            "s_mov_b32 exec_lo, 0xffffffff",  # restore exec for export
        ]
        outs = run_wave_vs(lines, wave=32)
        for lane in range(16):
            # Active lanes: v0 + 0xffffffe0 with clamp
            self.assertEqual(outs[lane], ref_add_nc_u32(lane, 0xFFFFFFE0, clamp=True))
        for lane in range(16, 32):
            # Inactive lanes: destination register retains sentinel value
            self.assertEqual(outs[lane], sentinel, f"Lane {lane} was overwritten despite inactive EXEC")

    def test_simd_dual_divergent_operands(self) -> None:
        """Simultaneous per-lane variation on both source operands across clamp threshold."""
        # Lane v0 in 0..31
        # Let v1 = 0x80000000 + lane (ranges 0x80000000..0x8000001F)
        # Let v2 = 0x7FFFFFF0 + lane (ranges 0x7FFFFFF0..0x8000000F)
        # Total sum = 0xFFFFFFF0 + 2 * lane
        # Lanes 0..7: sums 0xFFFFFFF0..0xFFFFFFFE (below 2^32, clamped == wrapped)
        # Lanes 8..31: sums 0x100000000..0x10000002E (exceeds 2^32, clamped == 0xFFFFFFFF, wrapped == 2 * (lane - 8))
        lines_clamped = [
            "v_add_nc_u32_e32 v1, 0x80000000, v0",
            "v_add_nc_u32_e32 v2, 0x7ffffff0, v0",
            "v_add_nc_u32_e64 v6, v1, v2 clamp",
        ]
        lines_wrapped = [
            "v_add_nc_u32_e32 v1, 0x80000000, v0",
            "v_add_nc_u32_e32 v2, 0x7ffffff0, v0",
            "v_add_nc_u32_e64 v6, v1, v2",
        ]
        clamped_outs = run_wave_vs(lines_clamped, wave=32)
        wrapped_outs = run_wave_vs(lines_wrapped, wave=32)

        for lane in range(32):
            v1_val = 0x80000000 + lane
            v2_val = 0x7FFFFFF0 + lane
            exp_clamped = ref_add_nc_u32(v1_val, v2_val, clamp=True)
            exp_wrapped = ref_add_nc_u32(v1_val, v2_val, clamp=False)
            self.assertEqual(clamped_outs[lane], exp_clamped, f"Clamped mismatch at lane {lane}")
            self.assertEqual(wrapped_outs[lane], exp_wrapped, f"Wrapped mismatch at lane {lane}")
            if lane < 8:
                self.assertEqual(clamped_outs[lane], wrapped_outs[lane])
                self.assertEqual(clamped_outs[lane], 0xFFFFFFF0 + 2 * lane)
            else:
                self.assertEqual(clamped_outs[lane], M32)
                self.assertEqual(wrapped_outs[lane], (lane - 8) * 2)
                self.assertNotEqual(clamped_outs[lane], wrapped_outs[lane])


class TestAddNcU32EdgeCases(unittest.TestCase):
    """Overflow wrapping vs clamping edge cases and SPIR-V structure verification."""

    def test_edge_case_matrix(self) -> None:
        """Exhaustive boundary testing across extreme 32-bit integer patterns."""
        edge_values = [
            0,
            1,
            2,
            0x7FFFFFFF,
            0x80000000,
            0x80000001,
            0xFFFFFFFE,
            0xFFFFFFFF,
            0x55555555,
            0xAAAAAAAA,
        ]
        for a in edge_values:
            for b in (0, 1, 0x80000000, 0xFFFFFFFF):
                for clamp in (False, True):
                    with self.subTest(a=hex(a), b=hex(b), clamp=clamp):
                        got = run_op("v_add_nc_u32", a, b, clamp)
                        want = ref_add_nc_u32(a, b, clamp)
                        self.assertEqual(got, want)

    def test_vop2_e32_vs_vop3_e64_equivalence_without_clamp(self) -> None:
        """Without clamp, VOP3 e64 must produce bit-for-bit identical results to VOP2 e32."""
        pairs = [
            (0xFFFFFFFF, 1),
            (0xFFFFFFFF, 0xFFFFFFFF),
            (0x80000000, 0x80000000),
            (0x12345678, 0x9ABCDEF0),
        ]
        for a, b in pairs:
            vop2 = run_lines([f"v_mov_b32 v1, {b:#x}", f"v_add_nc_u32_e32 v0, {a:#x}, v1"])
            vop3 = run_lines([f"v_mov_b32 v1, {a:#x}", f"v_mov_b32 v2, {b:#x}", "v_add_nc_u32_e64 v0, v1, v2"])
            self.assertEqual(vop2, vop3)
            self.assertEqual(vop2, ref_add_nc_u32(a, b, clamp=False))

    def test_vcc_isolation(self) -> None:
        """v_add_nc_u32 must not write or clobber VCC carry condition."""
        # Set vcc_lo to -1 (all lanes true). Execute v_add_nc_u32 with carry.
        # Verify vcc_lo is still -1 via v_cndmask_b32.
        lines = [
            "s_mov_b32 vcc_lo, -1",
            "v_mov_b32 v1, 0xffffffff",
            "v_mov_b32 v2, 1",
            "v_add_nc_u32_e64 v3, v1, v2 clamp",
            "v_cndmask_b32_e64 v0, 0, 1, vcc_lo",
        ]
        res = run_lines(lines)
        self.assertEqual(res, 1, "VCC condition was clobbered by v_add_nc_u32")

    def test_spirv_lowering_instruction_pattern(self) -> None:
        """v_add_nc_u32 with clamp lowers to OpIAddCarry + OpSelect, without clamp to OpIAdd."""
        if _TMP_DIR is None:
            setUpModule()
        # Clamped form:
        lines_clamped = [
            "v_mov_b32 v1, 0xffffffff",
            "v_mov_b32 v2, 1",
            "v_add_nc_u32_e64 v0, v1, v2 clamp",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        asm = "; ps_spirv_check\n" + "".join(f"\t{ln}\n" for ln in lines_clamped)
        with open(_SRC_PATH, "w", encoding="utf-8") as f:
            f.write(asm)
        spv_clamp = rdna2spirv.translate(_SRC_PATH)
        self.assertIn("OpIAddCarry %umul", spv_clamp)
        self.assertIn("4294967295", spv_clamp)
        self.assertIn("OpCompositeExtract %uint", spv_clamp)

        # Unclamped form:
        lines_wrapped = [
            "v_mov_b32 v1, 0xffffffff",
            "v_mov_b32 v2, 1",
            "v_add_nc_u32_e64 v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        asm = "; ps_spirv_check\n" + "".join(f"\t{ln}\n" for ln in lines_wrapped)
        with open(_SRC_PATH, "w", encoding="utf-8") as f:
            f.write(asm)
        spv_wrap = rdna2spirv.translate(_SRC_PATH)
        self.assertIn("OpIAdd %uint", spv_wrap)
        self.assertNotIn("OpIAddCarry", spv_wrap)


class TestAddNcU32NegativeControls(unittest.TestCase):
    """Negative controls ensuring mutated arithmetic or incorrect saturation limits fail assertions."""

    def test_overflow_wrapping_vs_clamping_divergence_detected(self) -> None:
        """Whenever sum > 0xFFFFFFFF, wrapping and clamping must strictly diverge."""
        overflow_pairs = [
            (0xFFFFFFFF, 1),
            (0xFFFFFFFF, 0xFFFFFFFF),
            (0x80000000, 0x80000000),
            (0xFFFFFFFE, 2),
            (0xC0000000, 0x40000001),
        ]
        for a, b in overflow_pairs:
            with self.subTest(a=hex(a), b=hex(b)):
                wrapped = ref_add_nc_u32(a, b, clamp=False)
                clamped = ref_add_nc_u32(a, b, clamp=True)
                self.assertNotEqual(wrapped, clamped)
                self.assertEqual(clamped, M32)
                self.assertEqual(wrapped, (a + b) & M32)

    def test_signed_saturation_sabotage_detected(self) -> None:
        """Unsigned clamp must not be confused with signed clamp (saddsat limit 0x7FFFFFFF)."""
        a, b = 0xFFFFFFFF, 1
        correct_uaddsat = ref_add_nc_u32(a, b, clamp=True)
        sabotaged_saddsat = 0x7FFFFFFF
        self.assertEqual(correct_uaddsat, M32)
        self.assertNotEqual(correct_uaddsat, sabotaged_saddsat)

    def test_subtraction_saturation_sabotage_detected(self) -> None:
        """Addition clamp (saturates to 0xFFFFFFFF) must not be confused with subtraction clamp (saturates to 0)."""
        a, b = 0xFFFFFFFF, 1
        correct_add_clamp = ref_add_nc_u32(a, b, clamp=True)
        sabotaged_sub_clamp = 0
        self.assertNotEqual(correct_add_clamp, sabotaged_sub_clamp)

    def test_unmasked_64bit_sum_sabotage_detected(self) -> None:
        """Unmasked 64-bit integer overflow must not pass 32-bit wrapping tests."""
        a, b = 0xFFFFFFFF, 1
        correct_wrapped = ref_add_nc_u32(a, b, clamp=False)
        sabotaged_unmasked = a + b  # 0x100000000
        self.assertEqual(correct_wrapped, 0)
        self.assertNotEqual(correct_wrapped, sabotaged_unmasked)

    def test_off_by_one_clamp_limit_detected(self) -> None:
        """Saturation must be exactly 0xFFFFFFFF, not 0xFFFFFFFE."""
        a, b = 0xFFFFFFFF, 2
        correct_clamped = ref_add_nc_u32(a, b, clamp=True)
        sabotaged_limit = 0xFFFFFFFE
        self.assertEqual(correct_clamped, M32)
        self.assertNotEqual(correct_clamped, sabotaged_limit)

    def test_no_carry_addition_does_not_mask_stage_carries(self) -> None:
        """Oracle analysis accurately reports carry flag and distinct outputs."""
        analysis = oracle_add_nc_u32_analysis(0x80000000, 0x80000000)
        self.assertTrue(analysis["carry"])
        self.assertEqual(analysis["wrapped"], 0)
        self.assertEqual(analysis["clamped"], M32)
        self.assertNotEqual(analysis["wrapped"], analysis["clamped"])


if __name__ == "__main__":
    unittest.main()
