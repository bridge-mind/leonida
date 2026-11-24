"""Semantic interpreter tests for RDNA2 s_bfm_b64 and s_add_i32.

s_bfm_b64 (RDNA2 ISA): D = ((1 << (S0 & 63)) - 1) << (S1 & 63) as a 64-bit lane mask; SCC is not written.
Each case translates a synthetic wave64 pixel shader with tools/rdna2spirv.py, executes all 64 lanes in
tests/semantic/spirv_wave_interp.py, reads D via v_cndmask_b32_e64 and SCC via s_cselect_b32, and compares
all 64 lane bits against a plain-Python reference.

s_add_i32 (RDNA2 ISA): D = (S0 + S1) & 0xFFFFFFFF; SCC = signed 32-bit overflow.
Each case translates a synthetic pixel shader with tools/rdna2spirv.py, runs it in
tests/semantic/spirv_mini_interp.py, and compares (D, SCC) against a plain-Python signed 32-bit reference,
including direct contrast with s_add_u32 on unsigned wrap without signed overflow.
"""
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import spirv_mini_interp as si  # noqa: E402
import spirv_wave_interp as wi  # noqa: E402
import support  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF

# (size, offset) cases covering zero, low-half, high-half, straddling 31/32, truncation > 63, and mod-64 wrap.
BFM_CASES = [
    (0, 0),
    (0, 31),
    (0, 45),
    (1, 0),
    (1, 63),
    (8, 4),          # low half: offset < 32, offset + size <= 32
    (32, 0),         # full low half
    (12, 32),        # high half: offset >= 32
    (16, 40),        # high half: offset >= 32
    (32, 32),        # full high half
    (8, 28),         # straddles bit 31/32
    (2, 31),         # straddles bit 31/32 minimally
    (64, 0),         # size & 63 == 0 -> mask 0
    (63, 0),         # lanes 0..62
    (63, 1),         # lanes 1..63
    (32, 48),        # truncated past lane 63 (offset + size > 64)
    (63, 40),        # truncated past lane 63
    (65, 66),        # size=65 -> 1, offset=66 -> 2
    (128, 3),        # size=128 -> 0
    (0x105, 0x120),  # size=5, offset=32
]

ADD_BOUNDARY_CASES = [
    # (x, y, description)
    (0, 0, "zero + zero"),
    (0, 0x7FFFFFFF, "zero + INT32_MAX"),
    (0, 0x80000000, "zero + INT32_MIN"),
    (0x7FFFFFFF, 1, "positive overflow: INT32_MAX + 1"),
    (0x7FFFFFFF, 0x7FFFFFFF, "positive overflow: INT32_MAX + INT32_MAX"),
    (0x40000000, 0x40000000, "positive overflow: 2^30 + 2^30"),
    (0x80000000, 0x80000000, "negative overflow: INT32_MIN + INT32_MIN"),
    (0x80000000, 0xFFFFFFFF, "negative overflow: INT32_MIN + (-1)"),
    (0xC0000000, 0xBFFFFFFF, "negative overflow: -2^30 + (-2^30 - 1)"),
    (0xFFFFFFFF, 1, "unsigned wrap without signed overflow: (-1) + 1"),
    (0xFFFFFFFF, 0xFFFFFFFF, "unsigned wrap without signed overflow: (-1) + (-1)"),
    (0x80000000, 0x7FFFFFFF, "opposite signs: INT32_MIN + INT32_MAX"),
    (0x7FFFFFFF, 0x80000000, "opposite signs: INT32_MAX + INT32_MIN"),
    (0x12345678, 0xEDCBA988, "opposite signs cancelling to zero"),
    (0x40000000, 0x3FFFFFFF, "exactly INT32_MAX, no overflow"),
    (0xC0000000, 0xC0000000, "exactly INT32_MIN from two negatives, no overflow"),
    (0x80000001, 0xFFFFFFFF, "(INT32_MIN + 1) + (-1) = INT32_MIN, no overflow"),
    (0x80000000, 1, "INT32_MIN + 1, no overflow"),
]


def s32(x: int) -> int:
    x &= MASK32
    return x - (1 << 32) if x >> 31 else x


def ref_bfm_b64(size: int, offset: int) -> int:
    sz = size & 63
    off = offset & 63
    return (((1 << sz) - 1) << off) & MASK64


def ref_add_i32(x: int, y: int) -> tuple:
    total = s32(x) + s32(y)
    ovf = int(total < -(1 << 31) or total >= (1 << 31))
    return total & MASK32, ovf


def ref_add_u32(x: int, y: int) -> tuple:
    total = (x & MASK32) + (y & MASK32)
    return total & MASK32, int(total > MASK32)


def run_bfm_b64(setup_lines, size_op: str, off_op: str, scc_in: int = 0) -> tuple:
    lines = [
        "s_mov_b64 exec, -1",
        *setup_lines,
        f"s_mov_b32 s4, {scc_in}",
        "s_cmp_eq_u32 s4, 1",
        f"s_bfm_b64 s[0:1], {size_op}, {off_op}",
        "s_cselect_b32 s5, 1, 0",
        "v_cndmask_b32_e64 v0, 0, 1, s[0:1]",
        "v_mov_b32 v1, s5",
        "v_mov_b32 v2, 0",
        "v_mov_b32 v3, 0",
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_salu_bfm64", lines)
    res = wi.run_wave(spv, wave=64)
    outs = res["outputs"]
    assert len(outs) == 64
    mask = 0
    for lane, out in enumerate(outs):
        assert out is not None, f"lane {lane} killed"
        bit, scc_out = out["%out_mrt0"][:2]
        assert bit in (0, 1), f"lane {lane} unexpected mask bit {bit}"
        assert scc_out == scc_in, f"lane {lane} SCC changed from {scc_in} to {scc_out}"
        mask |= bit << lane
    return mask, outs[0]["%out_mrt0"][1]


def run_add_op(op: str, setup_lines, x_op: str, y_op: str, scc_in: int = 0) -> tuple:
    lines = [
        *setup_lines,
        f"s_mov_b32 s4, {scc_in}",
        "s_cmp_eq_u32 s4, 1",
        f"{op} s0, {x_op}, {y_op}",
        "s_cselect_b32 s1, 1, 0",
        "v_mov_b32 v0, s0",
        "v_mov_b32 v1, s1",
        "exp mrt0 v0, v1, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_salu_add", lines)
    out = si.run_fragment(spv)["%out_mrt0"]
    return out[0], out[1]


class BfmB64Tests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(ref_bfm_b64(0, 17), 0)
        self.assertEqual(ref_bfm_b64(8, 28), 0x0000000FF0000000)
        self.assertEqual(ref_bfm_b64(32, 48), 0xFFFF000000000000)
        self.assertEqual(ref_bfm_b64(65, 66), 1 << 2)

    def test_sgpr_operands_wave64(self):
        for size, offset in BFM_CASES:
            for scc_in in (0, 1):
                with self.subTest(size=size, offset=offset, scc_in=scc_in):
                    setup = [f"s_mov_b32 s2, {size:#x}", f"s_mov_b32 s3, {offset:#x}"]
                    got_mask, got_scc = run_bfm_b64(setup, "s2", "s3", scc_in=scc_in)
                    self.assertEqual(got_mask, ref_bfm_b64(size, offset))
                    self.assertEqual(got_scc, scc_in)

    def test_immediate_operands_wave64(self):
        imm_cases = [
            ("0", "0", 0, 0),
            ("8", "28", 8, 28),
            ("32", "48", 32, 48),
            ("65", "66", 65, 66),
            ("0x10", "0x24", 16, 36),
        ]
        for s_lit, o_lit, size, offset in imm_cases:
            with self.subTest(size=s_lit, offset=o_lit):
                got_mask, _ = run_bfm_b64([], s_lit, o_lit, scc_in=1)
                self.assertEqual(got_mask, ref_bfm_b64(size, offset))
                # Mixed SGPR + immediate
                got_mixed, _ = run_bfm_b64([f"s_mov_b32 s2, {size:#x}"], "s2", o_lit, scc_in=0)
                self.assertEqual(got_mixed, ref_bfm_b64(size, offset))

    def test_vcc_and_null_destinations_wave64(self):
        # D = vcc (read back as a lane mask by v_cndmask_b32_e32); D = null discards the mask and keeps SCC.
        for size, offset in ((8, 28), (20, 48), (32, 0)):
            for scc_in in (0, 1):
                with self.subTest(size=size, offset=offset, scc_in=scc_in):
                    spv = support.translate_lines("ps_salu_bfm64_vcc", [
                        "s_mov_b64 exec, -1",
                        "s_mov_b64 vcc, 0",
                        f"s_bfm_b64 vcc, {size}, {offset}",
                        f"s_mov_b32 s4, {scc_in}",
                        "s_cmp_eq_u32 s4, 1",
                        "s_bfm_b64 null, 63, 1",
                        "s_cselect_b32 s5, 1, 0",
                        "v_cndmask_b32_e32 v0, 0, 1, vcc",
                        "v_mov_b32 v1, s5",
                        "v_mov_b32 v2, 0",
                        "v_mov_b32 v3, 0",
                        "exp mrt0 v0, v1, v2, v3 done vm",
                        "s_endpgm",
                    ])
                    outs = wi.run_wave(spv, wave=64)["outputs"]
                    mask = sum(out["%out_mrt0"][0] << lane for lane, out in enumerate(outs))
                    self.assertEqual(mask, ref_bfm_b64(size, offset))
                    self.assertEqual({out["%out_mrt0"][1] for out in outs}, {scc_in})

    def test_seeded_wave64(self):
        rng = random.Random(0xBF64)
        for _ in range(20):
            size = rng.getrandbits(10)
            offset = rng.getrandbits(10)
            with self.subTest(size=size, offset=offset):
                setup = [f"s_mov_b32 s2, {size:#x}", f"s_mov_b32 s3, {offset:#x}"]
                got_mask, _ = run_bfm_b64(setup, "s2", "s3")
                self.assertEqual(got_mask, ref_bfm_b64(size, offset))


class AddI32Tests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(ref_add_i32(0x7FFFFFFF, 1), (0x80000000, 1))
        self.assertEqual(ref_add_i32(0x80000000, 0x80000000), (0, 1))
        self.assertEqual(ref_add_i32(0x80000000, 0xFFFFFFFF), (0x7FFFFFFF, 1))
        self.assertEqual(ref_add_i32(0xFFFFFFFF, 1), (0, 0))

    def test_boundaries_and_scc(self):
        for x, y, desc in ADD_BOUNDARY_CASES:
            want = ref_add_i32(x, y)
            # Seed SCC opposite to expected to verify s_add_i32 overwrites SCC in both directions.
            scc_in = 1 - want[1]
            with self.subTest(case=desc, x=hex(x), y=hex(y)):
                setup = [f"s_mov_b32 s2, {x:#x}", f"s_mov_b32 s3, {y:#x}"]
                self.assertEqual(run_add_op("s_add_i32", setup, "s2", "s3", scc_in=scc_in), want)

    def test_contrast_with_s_add_u32(self):
        contrast_cases = [
            (0xFFFFFFFF, 1, (0, 0), (0, 1)),          # unsigned wrap, no signed overflow
            (0x7FFFFFFF, 1, (0x80000000, 1), (0x80000000, 0)),  # signed overflow, no unsigned wrap
            (0x80000000, 0x80000000, (0, 1), (0, 1)), # both overflow
            (0x80000000, 0xFFFFFFFF, (0x7FFFFFFF, 1), (0x7FFFFFFF, 1)),  # both overflow
        ]
        for x, y, want_i32, want_u32 in contrast_cases:
            with self.subTest(x=hex(x), y=hex(y)):
                setup = [f"s_mov_b32 s2, {x:#x}", f"s_mov_b32 s3, {y:#x}"]
                self.assertEqual(run_add_op("s_add_i32", setup, "s2", "s3", scc_in=1 - want_i32[1]), want_i32)
                self.assertEqual(run_add_op("s_add_u32", setup, "s2", "s3", scc_in=1 - want_u32[1]), want_u32)
                self.assertEqual(want_i32, ref_add_i32(x, y))
                self.assertEqual(want_u32, ref_add_u32(x, y))

    def test_immediate_operands(self):
        self.assertEqual(
            run_add_op("s_add_i32", ["s_mov_b32 s2, 0x7fffffff"], "s2", "1", scc_in=0),
            (0x80000000, 1),
        )
        self.assertEqual(
            run_add_op("s_add_i32", ["s_mov_b32 s2, 0x80000000"], "s2", "-1", scc_in=0),
            (0x7FFFFFFF, 1),
        )
        self.assertEqual(
            run_add_op("s_add_i32", ["s_mov_b32 s2, 0xffffffff"], "s2", "1", scc_in=1),
            (0, 0),
        )

    def test_aliasing_m0_and_null_destination(self):
        cases = [
            # (lines, want (D, SCC)); D is s0 afterwards, SCC the last s_add_i32's.
            (["s_mov_b32 s0, 0x40000000", "s_add_i32 s0, s0, s0"], (0x80000000, 1)),      # D == S0 == S1
            (["s_mov_b32 s0, 0x80000000", "s_mov_b32 s1, -1", "s_add_i32 s0, s1, s0"], (0x7FFFFFFF, 1)),  # D == S1
            (["s_mov_b32 s0, 0x80000000", "s_mov_b32 m0, -1", "s_add_i32 s0, s0, m0"], (0x7FFFFFFF, 1)),  # M0 source
            # null D: the sum is dropped but SCC is still written (0 after an overflowing add left it 1).
            (["s_mov_b32 s0, 5", "s_add_i32 s2, 0x7fffffff, 1", "s_add_i32 null, -1, 1"], (5, 0)),
            (["s_mov_b32 s0, 5", "s_add_i32 s2, -1, 1", "s_add_i32 null, 0x80000000, -1"], (5, 1)),
        ]
        for body, want in cases:
            with self.subTest(body=body[-1]):
                spv = support.translate_lines("ps_salu_add_alias", body + [
                    "s_cselect_b32 s1, 1, 0",
                    "v_mov_b32 v0, s0",
                    "v_mov_b32 v1, s1",
                    "exp mrt0 v0, v1, v0, v0 done vm",
                    "s_endpgm",
                ])
                self.assertEqual(tuple(si.run_fragment(spv)["%out_mrt0"][:2]), want)

    def test_seeded_random(self):
        rng = random.Random(0xADD32)
        for _ in range(30):
            x, y = rng.getrandbits(32), rng.getrandbits(32)
            want = ref_add_i32(x, y)
            with self.subTest(x=hex(x), y=hex(y)):
                setup = [f"s_mov_b32 s2, {x:#x}", f"s_mov_b32 s3, {y:#x}"]
                self.assertEqual(run_add_op("s_add_i32", setup, "s2", "s3", scc_in=1 - want[1]), want)


if __name__ == "__main__":
    unittest.main()
