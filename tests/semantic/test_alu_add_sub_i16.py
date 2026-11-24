"""Semantic interpreter tests for RDNA2 16-bit signed add/sub (v_add_i16, v_sub_i16).

Covers both the VOP2 form (`v_add_i16`, `v_sub_i16`, which zeroes bits [31:16] of the
destination VGPR) and the VOP3 `_e64` form (`v_add_i16_e64`, `v_sub_i16_e64`, which
preserves the untouched half of the destination VGPR and supports `op_sel` and signed
`clamp` to [-32768, 32767]).

Standard library only; zero game data, no GPU hardware required.
"""
from __future__ import annotations

import os
import random
import re
import sys
import unittest

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

OLD_V0 = 0x5A5A1234


def s16(word: int, hi: int = 0) -> int:
    """Extract a 16-bit lane half as a signed Python integer in [-32768, 32767]."""
    h = (word >> 16 if hi else word) & 0xFFFF
    return h - 0x10000 if (h & 0x8000) else h


def ref_i16(
    op: str,
    s0: int,
    s1: int,
    *,
    old: int = OLD_V0,
    clamp: bool = False,
    sel: tuple[int, int, int] = (0, 0, 0),
) -> int:
    """Plain-Python reference matching tools/rdna2spirv.py for v_add_i16 / v_sub_i16."""
    e64 = op.endswith("_e64")
    x = s16(s0, sel[0])
    y = s16(s1, sel[1])
    res = x + y if "add" in op else x - y
    if clamp:
        res = max(-0x8000, min(0x7FFF, res))
    lo = res & 0xFFFF
    if e64 and sel[2]:
        return (old & 0xFFFF) | (lo << 16)
    if e64:
        return (old & 0xFFFF0000) | lo
    return lo


def translate_i16_ps(name: str, lines: list[str]) -> str:
    """Translate a minimal pixel shader ending with `exp mrt0` of v0."""
    prog = list(lines) + [
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    return support.translate_lines(name, prog)


def run_i16_op(
    op: str,
    s0: int,
    s1: int,
    *,
    old: int = OLD_V0,
    mods: str = "",
) -> int:
    """Translate and run a single `op v0, v1, v2` instruction and return v0."""
    suffix = f" {mods}" if mods else ""
    spv = translate_i16_ps(
        f"ps_{op}",
        [
            f"v_mov_b32 v1, {s0:#x}",
            f"v_mov_b32 v2, {s1:#x}",
            f"v_mov_b32 v0, {old:#x}",
            f"{op} v0, v1, v2{suffix}",
        ],
    )
    return si.run_fragment(spv)["%out_mrt0"][0]


class AluAddSubI16Tests(unittest.TestCase):
    def test_add_i16_basic_wrap_and_high_bits(self) -> None:
        cases = [
            (0, 0, 0x0000),
            (3, 4, 0x0007),
            (0x7FFF, 2, 0x8001),       # signed 16-bit overflow wrap
            (0xFFFF, 1, 0x0000),       # unsigned 16-bit wrap (-1 + 1 = 0)
            (0xFFFE, 5, 0x0003),
            (0x8000, 0x8000, 0x0000),
            (0xFFFF, 0xFFFF, 0xFFFE),  # -1 + -1 = -2
        ]
        for s0, s1, expected_lo in cases:
            with self.subTest(op="v_add_i16", s0=hex(s0), s1=hex(s1)):
                self.assertEqual(run_i16_op("v_add_i16", s0, s1), expected_lo)
            with self.subTest(op="v_add_i16_e64", s0=hex(s0), s1=hex(s1)):
                self.assertEqual(
                    run_i16_op("v_add_i16_e64", s0, s1),
                    (OLD_V0 & 0xFFFF0000) | expected_lo,
                )

    def test_add_i16_inline_negative_operands(self) -> None:
        for op in ("v_add_i16", "v_add_i16_e64"):
            with self.subTest(op=op, case="-1 + 0"):
                spv = translate_i16_ps(
                    f"ps_{op}_neg0",
                    [
                        f"v_mov_b32 v0, {OLD_V0:#x}",
                        "v_mov_b32 v1, 0",
                        f"{op} v0, -1, v1",
                    ],
                )
                got = si.run_fragment(spv)["%out_mrt0"][0]
                expected = 0x5A5AFFFF if op.endswith("_e64") else 0x0000FFFF
                self.assertEqual(got, expected)

            with self.subTest(op=op, case="10 + -15"):
                spv = translate_i16_ps(
                    f"ps_{op}_neg1",
                    [
                        f"v_mov_b32 v0, {OLD_V0:#x}",
                        "v_mov_b32 v1, 10",
                        f"{op} v0, v1, -15",
                    ],
                )
                got = si.run_fragment(spv)["%out_mrt0"][0]
                expected = 0x5A5AFFFB if op.endswith("_e64") else 0x0000FFFB
                self.assertEqual(got, expected)

    def test_sub_i16_basic_borrow_asymmetry_and_high_bits(self) -> None:
        cases = [
            (0, 0, 0x0000),
            (10, 3, 0x0007),
            (1, 2, 0xFFFF),            # borrow / underflow: 1 - 2 = -1
            (0, 1, 0xFFFF),
            (0x8000, 1, 0x7FFF),       # signed 16-bit underflow wrap
            (0x8000, 0x7FFF, 0x0001),
            (0x0005, 0xFFFF, 0x0006),  # 5 - (-1) = 6
        ]
        for s0, s1, expected_lo in cases:
            with self.subTest(op="v_sub_i16", s0=hex(s0), s1=hex(s1)):
                self.assertEqual(run_i16_op("v_sub_i16", s0, s1), expected_lo)
            with self.subTest(op="v_sub_i16_e64", s0=hex(s0), s1=hex(s1)):
                self.assertEqual(
                    run_i16_op("v_sub_i16_e64", s0, s1),
                    (OLD_V0 & 0xFFFF0000) | expected_lo,
                )

        # Asymmetry: S0 - S1 != S1 - S0
        for op in ("v_sub_i16", "v_sub_i16_e64"):
            ab = run_i16_op(op, 5, 12)
            ba = run_i16_op(op, 12, 5)
            self.assertNotEqual(ab, ba)
            self.assertEqual(ab & 0xFFFF, 0xFFF9)
            self.assertEqual(ba & 0xFFFF, 0x0007)

    def test_high_input_bits_do_not_pollute_16bit_result(self) -> None:
        pairs = [
            (0xDEAD0005, 0xBEEF0003),
            (0x12347FFF, 0x56780002),
            (0xABCDFFFF, 0xEF010001),
            (0x80008000, 0x7FFF0001),
        ]
        for op in ("v_add_i16", "v_add_i16_e64", "v_sub_i16", "v_sub_i16_e64"):
            for s0, s1 in pairs:
                with self.subTest(op=op, s0=hex(s0), s1=hex(s1)):
                    got = run_i16_op(op, s0, s1)
                    self.assertEqual(got, ref_i16(op, s0, s1))

    def test_sgpr_opsel_and_clamp(self) -> None:
        # SGPR scalar source broadcast
        for op in ("v_add_i16", "v_sub_i16", "v_add_i16_e64", "v_sub_i16_e64"):
            spv = translate_i16_ps(
                f"ps_{op}_sgpr",
                [
                    "s_mov_b32 s0, 0x1234000a",
                    "v_mov_b32 v1, 0xabcd0003",
                    f"v_mov_b32 v0, {OLD_V0:#x}",
                    f"{op} v0, s0, v1",
                ],
            )
            got = si.run_fragment(spv)["%out_mrt0"][0]
            self.assertEqual(got, ref_i16(op, 0x1234000A, 0xABCD0003))

        # Signed clamp on _e64: saturate to [-32768, 32767]
        self.assertEqual(
            run_i16_op("v_add_i16_e64", 0x7FFF, 2, mods="clamp"),
            (OLD_V0 & 0xFFFF0000) | 0x7FFF,
        )
        self.assertEqual(
            run_i16_op("v_sub_i16_e64", 0x8000, 1, mods="clamp"),
            (OLD_V0 & 0xFFFF0000) | 0x8000,
        )

        # op_sel on _e64: select high half of src0 and write to high half of dst
        got_opsel = run_i16_op(
            "v_add_i16_e64",
            0x000A0001,
            0x00020005,
            mods="op_sel:[1,0,1]",
        )
        self.assertEqual(
            got_opsel,
            ref_i16("v_add_i16_e64", 0x000A0001, 0x00020005, sel=(1, 0, 1)),
        )

    def test_seeded_random(self) -> None:
        rng = random.Random(0x116)
        ops = ("v_add_i16", "v_add_i16_e64", "v_sub_i16", "v_sub_i16_e64")
        for _ in range(40):
            op = rng.choice(ops)
            s0 = rng.getrandbits(32)
            s1 = rng.getrandbits(32)
            old = rng.getrandbits(32)
            clamp = op.endswith("_e64") and bool(rng.getrandbits(1))
            mods = "clamp" if clamp else ""
            with self.subTest(op=op, s0=hex(s0), s1=hex(s1), clamp=clamp):
                got = run_i16_op(op, s0, s1, old=old, mods=mods)
                self.assertEqual(got, ref_i16(op, s0, s1, old=old, clamp=clamp))

    def test_negative_controls_spirv_mutation(self) -> None:
        # Negative control 1: swapping OpIAdd <-> OpISub in memory changes the result.
        spv_add = translate_i16_ps(
            "ps_neg_ctrl_add",
            [
                "v_mov_b32 v1, 10",
                "v_mov_b32 v2, 3",
                "v_add_i16 v0, v1, v2",
            ],
        )
        self.assertEqual(si.run_fragment(spv_add)["%out_mrt0"][0], 13)
        self.assertIn("= OpIAdd %int ", spv_add)
        spv_add_mut = spv_add.replace("= OpIAdd %int ", "= OpISub %int ", 1)
        self.assertEqual(si.run_fragment(spv_add_mut)["%out_mrt0"][0], 7)

        spv_sub = translate_i16_ps(
            "ps_neg_ctrl_sub",
            [
                "v_mov_b32 v1, 10",
                "v_mov_b32 v2, 3",
                "v_sub_i16 v0, v1, v2",
            ],
        )
        self.assertEqual(si.run_fragment(spv_sub)["%out_mrt0"][0], 7)
        self.assertIn("= OpISub %int ", spv_sub)
        spv_sub_mut = spv_sub.replace("= OpISub %int ", "= OpIAdd %int ", 1)
        self.assertEqual(si.run_fragment(spv_sub_mut)["%out_mrt0"][0], 13)

        # Negative control 2: removing the final 0xFFFF (65535) output mask leaks
        # sign-extended bits [31:16] from the 32-bit signed sum (-1 + -1 = -2).
        spv_mask = translate_i16_ps(
            "ps_neg_ctrl_mask",
            [
                "v_mov_b32 v1, 0xffff",
                "v_mov_b32 v2, 0xffff",
                "v_add_i16 v0, v1, v2",
            ],
        )
        self.assertEqual(si.run_fragment(spv_mask)["%out_mrt0"][0], 0x0000FFFE)
        const_match = re.search(
            r"^(%\S+)\s*=\s*OpConstant\s+%uint\s+65535\b", spv_mask, re.MULTILINE
        )
        self.assertIsNotNone(const_match)
        c65535 = const_match.group(1)
        mask_matches = list(
            re.finditer(
                rf"= OpBitwiseAnd %uint (%\S+) {re.escape(c65535)}\b", spv_mask
            )
        )
        # one 16-bit mask at least: SHADER-315's constant folding removed the ones on the literal operands
        self.assertGreaterEqual(len(mask_matches), 1)
        last = mask_matches[-1]
        spv_mask_mut = (
            spv_mask[: last.start()]
            + f"= OpCopyObject %uint {last.group(1)}"
            + spv_mask[last.end() :]
        )
        self.assertEqual(si.run_fragment(spv_mask_mut)["%out_mrt0"][0], 0xFFFFFFFE)


if __name__ == "__main__":
    unittest.main()
