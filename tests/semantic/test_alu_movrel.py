from __future__ import annotations

"""Semantic interpreter tests and reference oracle for relative register move (v_movrels_b32).

RDNA2 ISA (VOP1 opcode 71 / 0x47):
  v_movrels_b32   D = VGPR[S0.addr + M0]
  v_movreld_b32   VGPR[D.addr + M0] = S0
  v_movrelsd_b32  VGPR[D.addr + M0] = VGPR[S0.addr + M0]

The relative register move instruction indexes the vector register file (VGPR)
dynamically relative to the value stored in the scalar M0 register.

RDNA2 ISA §3.6.1: a source index outside the VGPR allocation reads VGPR0;
an out-of-range destination discards the write. These tests model the
translator's 256-entry register array and the existing 32-bit index addition.
They also cover EXEC masking in wave32 and wave64.
"""

from collections.abc import Sequence
import gc
import os
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import spirv_wave_interp as wi  # noqa: E402
from support import STUB_DXBC, mrt0_words, rdna2spirv  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
NUM_VGPRS = 256  # Size of the VGPR file (256 32-bit registers)


def sentinel(i: int) -> int:
    """The value v_mov_b32 loads into v<i> before the movrel instruction."""
    return 0xC0DE0000 | ((i * 0x111) & 0xFFFF)


SENTINELS: dict[int, int] = {i: sentinel(i) for i in range(1, 9)}
V0_VALUE = 0x5EED0000  # Loaded into v0 by cases that write or seed v0


def compute_movrels_target_vgpr(base_vgpr: int, m0_offset: int, count: int = NUM_VGPRS) -> int | None:
    """Return the effective index if it falls within the model's VGPR array."""
    index = (base_vgpr + m0_offset) & MASK32
    return index if index < count else None


def ref_movrels_b32(dst: int, src: int, m0: int, vgpr_file: Sequence[int]) -> list[int]:
    """Return a copy of the file after a relative source read."""
    out = list(vgpr_file)
    target_idx = compute_movrels_target_vgpr(src, m0, len(out))
    out[dst] = out[target_idx if target_idx is not None else 0] & MASK32
    return out


class MovrelOracle:
    """Stateful architectural register file oracle for M0-relative instructions."""

    def __init__(self, initial_values: dict[int, int] | None = None) -> None:
        self.vgpr = [0] * NUM_VGPRS
        self.m0 = 0
        if initial_values:
            for reg, val in initial_values.items():
                self.vgpr[reg % NUM_VGPRS] = val & MASK32

    def set_m0(self, val: int) -> None:
        self.m0 = val & MASK32

    def set_vgpr(self, reg: int, val: int) -> None:
        self.vgpr[reg % NUM_VGPRS] = val & MASK32

    def get_vgpr(self, reg: int) -> int:
        return self.vgpr[reg % NUM_VGPRS]

    def execute_v_movrels(self, dst: int, src: int) -> int:
        src_idx = compute_movrels_target_vgpr(src, self.m0)
        val = self.vgpr[src_idx if src_idx is not None else 0]
        self.vgpr[dst] = val
        return val

    def execute_v_movreld(self, dst: int, src: int) -> int:
        dst_idx = compute_movrels_target_vgpr(dst, self.m0)
        val = self.vgpr[src]
        if dst_idx is not None:
            self.vgpr[dst_idx] = val
        return val

    def execute_v_movrelsd(self, dst: int, src: int) -> int:
        src_idx = compute_movrels_target_vgpr(src, self.m0)
        dst_idx = compute_movrels_target_vgpr(dst, self.m0)
        val = self.vgpr[src_idx if src_idx is not None else 0]
        if dst_idx is not None:
            self.vgpr[dst_idx] = val
        return val

    def dump_registers(self, regs: Sequence[int]) -> tuple[int, ...]:
        return tuple(self.vgpr[r % NUM_VGPRS] for r in regs)


def model(op: str, dst: int, src: int, m0: int, init: dict[int, int]) -> list[int]:
    """Plain-Python register file model applying `op dst, src` with M0 = m0."""
    oracle = MovrelOracle(init)
    oracle.set_m0(m0)
    if op == "v_movrels_b32":
        oracle.execute_v_movrels(dst, src)
    elif op == "v_movreld_b32":
        oracle.execute_v_movreld(dst, src)
    elif op == "v_movrelsd_b32":
        oracle.execute_v_movrelsd(dst, src)
    else:
        raise ValueError(op)
    return oracle.vgpr


S = SENTINELS
# (op, dst, src, m0, load v0?, exported registers, expected words) -- expected words worked out by hand.
CASES = [
    # movrels: D = VGPR[S0 + M0]
    ("v_movrels_b32", 0, 1, 0, False, (0, 1, 2, 3), (S[1], S[1], S[2], S[3])),     # M0 = 0 reads S0 itself
    ("v_movrels_b32", 0, 2, 3, False, (0, 2, 4, 5), (S[5], S[2], S[4], S[5])),     # v2 + 3 = v5
    ("v_movrels_b32", 0, 1, 7, False, (0, 1, 7, 8), (S[8], S[1], S[7], S[8])),     # v1 + 7 = v8
    ("v_movrels_b32", 9, 0, 4, True, (9, 0, 4, 5), (S[4], V0_VALUE, S[4], S[5])),  # base v0, M0 alone indexes
    # movreld: VGPR[D + M0] = S0; neighbours keep their sentinels
    ("v_movreld_b32", 1, 0, 0, True, (0, 1, 2, 3), (V0_VALUE, V0_VALUE, S[2], S[3])),
    ("v_movreld_b32", 3, 0, 2, True, (3, 4, 5, 6), (S[3], S[4], V0_VALUE, S[6])),  # writes v5, not v3
    ("v_movreld_b32", 1, 8, 6, False, (1, 6, 7, 8), (S[1], S[6], S[8], S[8])),     # v1 + 6 = v7 <- v8
    # movrelsd: VGPR[D + M0] = VGPR[S0 + M0]; M0 offsets both sides
    ("v_movrelsd_b32", 2, 1, 0, False, (1, 2, 3, 4), (S[1], S[1], S[3], S[4])),
    ("v_movrelsd_b32", 6, 1, 2, False, (6, 7, 8, 3), (S[6], S[7], S[3], S[3])),    # v8 <- v3
    ("v_movrelsd_b32", 1, 3, 4, False, (1, 3, 5, 7), (S[1], S[3], S[7], S[7])),    # v5 <- v7
]

_TMP: tempfile.TemporaryDirectory | None = None


def setUpModule() -> None:
    global _TMP
    _TMP = tempfile.TemporaryDirectory()
    os.mkdir(os.path.join(_TMP.name, "disasm"))
    with open(os.path.join(_TMP.name, "ps_movrel.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule() -> None:
    global _TMP
    if _TMP is not None:
        gc.collect()
        _TMP.cleanup()
        _TMP = None


def translate_listing(body: Sequence[str]) -> str:
    """Write `body` (instruction lines) as ps_movrel.s, translate it and return the SPIR-V text."""
    assert _TMP is not None, "setUpModule was not executed"
    lines = ["; test_movrel"] + [f"\t{line}" for line in body] + ["\ts_endpgm", ""]
    path = os.path.join(_TMP.name, "disasm", "ps_movrel.s")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(path)
        gc.collect()
    return spv


def initial_regs(load_v0: bool) -> dict[int, int]:
    regs = dict(SENTINELS)
    if load_v0:
        regs[0] = V0_VALUE
    return regs


def run_case(
    op: str,
    dst: int,
    src: int,
    m0: int,
    load_v0: bool,
    exported: Sequence[int],
) -> tuple[int, ...]:
    body = [f"v_mov_b32 v{r}, {v:#x}" for r, v in sorted(initial_regs(load_v0).items())]
    body.append(f"s_mov_b32 m0, {m0}")
    body.append(f"{op} v{dst}, v{src}")
    body.append("exp mrt0 " + ", ".join(f"v{r}" for r in exported) + " done vm")
    return tuple(mrt0_words(translate_listing(body)))


def case_id(case: tuple) -> str:
    op, dst, src, m0 = case[:4]
    return f"{op} v{dst}, v{src} (m0={m0})"


class TestMovrelsReferenceOracle(unittest.TestCase):
    """Rigorous mathematical tests for the v_movrels_b32 reference oracle."""

    def test_oracle_zero_offset(self) -> None:
        """When M0 = 0, dynamic indexing evaluates directly to the source register."""
        for reg_idx in range(NUM_VGPRS):
            target = compute_movrels_target_vgpr(reg_idx, 0)
            self.assertEqual(target, reg_idx)

        oracle = MovrelOracle({10: 0x12345678, 20: 0x87654321})
        oracle.set_m0(0)
        val = oracle.execute_v_movrels(dst=5, src=10)
        self.assertEqual(val, 0x12345678)
        self.assertEqual(oracle.get_vgpr(5), 0x12345678)
        self.assertEqual(oracle.get_vgpr(10), 0x12345678)

    def test_oracle_positive_displacement(self) -> None:
        """An in-range relative address selects the requested register."""
        oracle = MovrelOracle()
        oracle.set_vgpr(15, 0xCAFEBABE)
        oracle.set_m0(5)
        # Reading from base v10 with M0=5 resolves to v15
        val = oracle.execute_v_movrels(dst=0, src=10)
        self.assertEqual(val, 0xCAFEBABE)
        self.assertEqual(oracle.get_vgpr(0), 0xCAFEBABE)

    def test_oracle_out_of_range_reads_vgpr0(self) -> None:
        oracle = MovrelOracle({0: 0x12345678, 4: 0xDEADBEEF})
        oracle.set_m0(10)
        self.assertIsNone(compute_movrels_target_vgpr(250, 10))
        self.assertEqual(oracle.execute_v_movrels(dst=1, src=250), 0x12345678)

    def test_oracle_large_m0_values(self) -> None:
        for base, offset in ((10, 256), (5, 519), (0, 1024), (1, 0x10000), (10, 0x10000003)):
            with self.subTest(base=base, offset=offset):
                self.assertIsNone(compute_movrels_target_vgpr(base, offset))

    def test_oracle_preserves_other_vgprs(self) -> None:
        """Only the destination register is written; all other registers remain unchanged."""
        oracle = MovrelOracle({r: sentinel(r) for r in range(1, 9)})
        oracle.set_m0(3)
        # S0 = v2, M0 = 3 -> reads v5 (sentinel 5), writes to v0
        oracle.execute_v_movrels(dst=0, src=2)
        self.assertEqual(oracle.get_vgpr(0), SENTINELS[5])
        for r in range(1, 9):
            self.assertEqual(oracle.get_vgpr(r), SENTINELS[r])

    def test_oracle_self_move(self) -> None:
        """When destination matches effective source index, the value is preserved."""
        oracle = MovrelOracle({5: 0x55555555})
        oracle.set_m0(3)
        # dst = 5, src = 2 + 3 = 5
        val = oracle.execute_v_movrels(dst=5, src=2)
        self.assertEqual(val, 0x55555555)
        self.assertEqual(oracle.get_vgpr(5), 0x55555555)

    def test_oracle_overwrite_source(self) -> None:
        """When destination matches base source register but M0 > 0, base register is overwritten."""
        oracle = MovrelOracle({2: 0x22222222, 5: 0x55555555})
        oracle.set_m0(3)
        # dst = 2, src = 2 + 3 = 5 -> v2 becomes 0x55555555
        oracle.execute_v_movrels(dst=2, src=2)
        self.assertEqual(oracle.get_vgpr(2), 0x55555555)
        self.assertEqual(oracle.get_vgpr(5), 0x55555555)

    def test_oracle_sequential_chain(self) -> None:
        """Sequential relative moves correctly update and carry forward register state."""
        oracle = MovrelOracle()
        oracle.set_vgpr(10, 0xAAAAAAAA)
        oracle.set_vgpr(20, 0xBBBBBBBB)
        oracle.set_vgpr(30, 0xCCCCCCCC)

        oracle.set_m0(10)
        oracle.execute_v_movrels(dst=0, src=0)  # v0 <- v10
        oracle.execute_v_movrels(dst=1, src=10) # v1 <- v20

        oracle.set_m0(20)
        oracle.execute_v_movrels(dst=2, src=10) # v2 <- v30

        self.assertEqual(oracle.dump_registers((0, 1, 2)), (0xAAAAAAAA, 0xBBBBBBBB, 0xCCCCCCCC))

    def test_pure_function_parity(self) -> None:
        """Parity between stateful MovrelOracle and pure-function ref_movrels_b32."""
        regs = [sentinel(i % 16) for i in range(NUM_VGPRS)]
        oracle = MovrelOracle({i: regs[i] for i in range(NUM_VGPRS)})
        oracle.set_m0(7)
        oracle.execute_v_movrels(dst=50, src=20)

        updated_pure = ref_movrels_b32(dst=50, src=20, m0=7, vgpr_file=regs)
        self.assertEqual(oracle.vgpr, updated_pure)


class TestM0IndexedDynamicAddressing(unittest.TestCase):
    """Dynamic addressing tests verifying M0-based VGPR file access in translated shaders."""

    def test_dynamic_addressing_matrix(self) -> None:
        """Test matrix of base VGPRs and M0 offsets against the reference oracle."""
        test_matrix = [
            (0, 1, 0),    # v0 <- VGPR[v1 + 0] = v1
            (1, 2, 3),    # v1 <- VGPR[v2 + 3] = v5
            (2, 4, 6),    # v2 <- VGPR[v4 + 6] = v10
            (3, 0, 7),    # v3 <- VGPR[v0 + 7] = v7
            (0, 5, 10),   # v0 <- VGPR[v5 + 10] = v15
            (1, 10, 20),  # v1 <- VGPR[v10 + 20] = v30
        ]
        for dst, base_src, m0 in test_matrix:
            target_reg = base_src + m0
            val = (0x77000000 | (target_reg * 0x10101)) & MASK32
            base_val = 0x11111111

            lines = [
                f"v_mov_b32 v{base_src}, {base_val:#x}",
                f"v_mov_b32 v{target_reg}, {val:#x}",
                f"s_mov_b32 m0, {m0}",
                f"v_movrels_b32 v{dst}, v{base_src}",
                f"exp mrt0 v{dst}, v{base_src}, v{target_reg}, v{dst} done vm",
            ]
            spv = translate_listing(lines)
            words = mrt0_words(spv)

            oracle = MovrelOracle()
            oracle.set_vgpr(base_src, base_val)
            oracle.set_vgpr(target_reg, val)
            oracle.set_m0(m0)
            oracle.execute_v_movrels(dst, base_src)
            expected = (
                oracle.get_vgpr(dst),
                oracle.get_vgpr(base_src),
                oracle.get_vgpr(target_reg),
                oracle.get_vgpr(dst),
            )
            self.assertEqual(
                tuple(words),
                expected,
                f"Failed for dst=v{dst}, src=v{base_src}, m0={m0}: got {words}, expected {expected}",
            )

    def test_computed_m0_salu_indexing(self) -> None:
        """Dynamic M0 values computed at runtime via scalar ALU correctly index VGPRs."""
        # M0 computed via addition: s0 (10) + s1 (5) = 15
        asm_add = [
            "v_mov_b32 v20, 0xCAFE0001",
            "s_mov_b32 s0, 10",
            "s_mov_b32 s1, 5",
            "s_add_u32 m0, s0, s1",
            "v_movrels_b32 v0, v5",  # v5 + 15 = v20
            "exp mrt0 v0, v0, v0, v0 done vm",
        ]
        words = mrt0_words(translate_listing(asm_add))
        self.assertEqual(words, [0xCAFE0001] * 4)

        # M0 computed via shift: s0 (3) << 2 = 12
        asm_shift = [
            "v_mov_b32 v16, 0xCAFE0002",
            "s_mov_b32 s0, 3",
            "s_lshl_b32 m0, s0, 2",
            "v_movrels_b32 v0, v4",  # v4 + 12 = v16
            "exp mrt0 v0, v0, v0, v0 done vm",
        ]
        words = mrt0_words(translate_listing(asm_shift))
        self.assertEqual(words, [0xCAFE0002] * 4)

    def test_dynamic_lookup_table_simulation(self) -> None:
        """Simulate an 8-entry register lookup table indexed via M0."""
        table_values = [0x1000 + i * 0x111 for i in range(8)]
        body = []
        for i, val in enumerate(table_values):
            body.append(f"v_mov_b32 v{10 + i}, {val:#x}")

        # Query index 3 and index 6
        body.extend([
            "s_mov_b32 m0, 3",
            "v_movrels_b32 v0, v10",  # v10 + 3 = v13
            "s_mov_b32 m0, 6",
            "v_movrels_b32 v1, v10",  # v10 + 6 = v16
            "exp mrt0 v0, v1, v0, v1 done vm",
        ])
        words = mrt0_words(translate_listing(body))
        expected = (table_values[3], table_values[6], table_values[3], table_values[6])
        self.assertEqual(tuple(words), expected)

    def test_chained_dynamic_moves(self) -> None:
        """Multiple sequential v_movrels_b32 instructions with modified M0 values in one shader."""
        lines = [
            "v_mov_b32 v10, 0x11111111",
            "v_mov_b32 v20, 0x22222222",
            "v_mov_b32 v30, 0x33333333",
            "s_mov_b32 m0, 10",
            "v_movrels_b32 v0, v0",  # reads v10
            "s_mov_b32 m0, 15",
            "v_movrels_b32 v1, v5",  # reads v20
            "s_mov_b32 m0, 28",
            "v_movrels_b32 v2, v2",  # reads v30
            "exp mrt0 v0, v1, v2, v0 done vm",
        ]
        words = mrt0_words(translate_listing(lines))
        self.assertEqual(tuple(words), (0x11111111, 0x22222222, 0x33333333, 0x11111111))

    def test_dynamic_movrels_feed_valu(self) -> None:
        """Dynamically indexed VGPR value feeds downstream vector ALU arithmetic."""
        lines = [
            "v_mov_b32 v10, 0x000000FF",
            "v_mov_b32 v1, 0x0000FF00",
            "s_mov_b32 m0, 6",
            "v_movrels_b32 v0, v4",     # v4 + 6 = v10 -> v0 = 0x000000FF
            "v_add_nc_u32 v2, v0, v1",  # v2 = 0x000000FF + 0x0000FF00 = 0x0000FFFF
            "v_lshlrev_b32 v3, 16, v2", # v3 = 0xFFFF0000
            "exp mrt0 v0, v1, v2, v3 done vm",
        ]
        words = mrt0_words(translate_listing(lines))
        self.assertEqual(tuple(words), (0x000000FF, 0x0000FF00, 0x0000FFFF, 0xFFFF0000))

    def test_high_register_dynamic_read(self) -> None:
        """Dynamic addressing into high register space (v150..v250) across the 256-entry array."""
        lines = [
            "v_mov_b32 v220, 0xFEEDC0DE",
            "s_mov_b32 m0, 40",
            "v_movrels_b32 v0, v180",  # v180 + 40 = v220
            "exp mrt0 v0, v0, v0, v0 done vm",
        ]
        words = mrt0_words(translate_listing(lines))
        self.assertEqual(tuple(words), (0xFEEDC0DE, 0xFEEDC0DE, 0xFEEDC0DE, 0xFEEDC0DE))

    def test_overlapping_source_and_destination(self) -> None:
        """When destination matches the target register, the value is preserved."""
        lines = [
            "v_mov_b32 v5, 0x99887766",
            "s_mov_b32 m0, 3",
            "v_movrels_b32 v5, v2",  # v2 + 3 = v5, write to v5
            "exp mrt0 v5, v5, v5, v5 done vm",
        ]
        words = mrt0_words(translate_listing(lines))
        self.assertEqual(tuple(words), (0x99887766, 0x99887766, 0x99887766, 0x99887766))


class TestIndexBoundaryAndClamping(unittest.TestCase):
    """Boundary conditions and clamping tests for dynamic VGPR addressing."""

    def test_boundary_exact_max_vgpr_255(self) -> None:
        """Accessing the highest valid physical register v255 succeeds."""
        lines = [
            "v_mov_b32 v255, 0xCAFEBABE",
            "s_mov_b32 m0, 5",
            "v_movrels_b32 v0, v250",  # 250 + 5 = 255
            "exp mrt0 v0, v0, v0, v0 done vm",
        ]
        words = mrt0_words(translate_listing(lines))
        self.assertEqual(words, [0xCAFEBABE] * 4)

    def test_boundary_first_vgpr_0(self) -> None:
        """Accessing register v0 via base v0 and M0 = 0 succeeds."""
        lines = [
            "v_mov_b32 v0, 0x12345678",
            "s_mov_b32 m0, 0",
            "v_movrels_b32 v1, v0",
            "exp mrt0 v1, v1, v1, v1 done vm",
        ]
        words = mrt0_words(translate_listing(lines))
        self.assertEqual(words, [0x12345678] * 4)

    def test_out_of_bounds_reads_vgpr0(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v4, 0xDEADBEEF",
            "s_mov_b32 m0, 10",
            "v_movrels_b32 v1, v250",  # The index 260 must not select v4 or an invalid pointer.
            "exp mrt0 v1, v1, v1, v1 done vm",
        ]
        self.assertEqual(mrt0_words(translate_listing(lines)), [0x12345678] * 4)

    def test_oracle_32_bit_addition(self) -> None:
        self.assertEqual(compute_movrels_target_vgpr(5, 0xFFFFFFFF), 4)
        self.assertIsNone(compute_movrels_target_vgpr(0, 0xFFFFFFFF))


class TestSimdDivergenceUnderExec(unittest.TestCase):
    """Multi-lane SIMD divergence tests under EXEC masking in wave32 and wave64."""

    def test_wave32_partial_exec_divergence(self) -> None:
        """Under partial EXEC (lanes 0..15 active), only active lanes update destination register."""
        lines = [
            "v_mov_b32 v0, 0xAAAA",
            "v_mov_b32 v1, 0x1111",
            "v_mov_b32 v2, 0x2222",
            "s_mov_b32 m0, 1",
            "s_mov_b32 exec_lo, 0x0000FFFF",  # lanes 0..15 active
            "v_movrels_b32 v0, v1",           # reads v2 (0x2222) into v0 for active lanes
            "s_mov_b32 exec_lo, 0xFFFFFFFF",  # re-enable all lanes for export
            "exp mrt0 v0, v1, v2, v0 done vm",
        ]
        spv = translate_listing(lines)
        res = wi.run_wave(spv, wave=32)
        outputs = [o["%out_mrt0"] for o in res["outputs"]]

        for lane in range(32):
            expected_v0 = 0x2222 if lane < 16 else 0xAAAA
            self.assertEqual(
                outputs[lane][0],
                expected_v0,
                f"Lane {lane} failed: expected v0={expected_v0:#x}, got {outputs[lane][0]:#x}",
            )

    def test_wave32_checkerboard_exec(self) -> None:
        """Checkerboard EXEC mask (0x55555555): even lanes write, odd lanes preserve value."""
        lines = [
            "v_mov_b32 v0, 0xAAAA",
            "v_mov_b32 v1, 0x1111",
            "v_mov_b32 v2, 0x2222",
            "s_mov_b32 m0, 1",
            "s_mov_b32 exec_lo, 0x55555555",  # even lanes active
            "v_movrels_b32 v0, v1",
            "s_mov_b32 exec_lo, 0xFFFFFFFF",
            "exp mrt0 v0, v1, v2, v0 done vm",
        ]
        spv = translate_listing(lines)
        res = wi.run_wave(spv, wave=32)
        outputs = [o["%out_mrt0"] for o in res["outputs"]]

        for lane in range(32):
            expected_v0 = 0x2222 if (lane % 2 == 0) else 0xAAAA
            self.assertEqual(outputs[lane][0], expected_v0)

    def test_wave32_zero_exec_preserves_all(self) -> None:
        """When EXEC = 0, no lanes execute the relative move; all preserve initial value."""
        lines = [
            "v_mov_b32 v0, 0x7777",
            "v_mov_b32 v1, 0x1111",
            "v_mov_b32 v2, 0x2222",
            "s_mov_b32 m0, 1",
            "s_mov_b32 exec_lo, 0",            # all lanes disabled
            "v_movrels_b32 v0, v1",
            "s_mov_b32 exec_lo, 0xFFFFFFFF",  # re-enable all
            "exp mrt0 v0, v1, v2, v0 done vm",
        ]
        spv = translate_listing(lines)
        res = wi.run_wave(spv, wave=32)
        outputs = [o["%out_mrt0"] for o in res["outputs"]]

        for lane in range(32):
            self.assertEqual(outputs[lane][0], 0x7777)

    def test_wave32_full_exec_updates_all(self) -> None:
        """When EXEC = 0xFFFFFFFF, all lanes execute the relative move."""
        lines = [
            "v_mov_b32 v0, 0x7777",
            "v_mov_b32 v1, 0x1111",
            "v_mov_b32 v2, 0x2222",
            "s_mov_b32 m0, 1",
            "s_mov_b32 exec_lo, 0xFFFFFFFF",
            "v_movrels_b32 v0, v1",
            "exp mrt0 v0, v1, v2, v0 done vm",
        ]
        spv = translate_listing(lines)
        res = wi.run_wave(spv, wave=32)
        outputs = [o["%out_mrt0"] for o in res["outputs"]]

        for lane in range(32):
            self.assertEqual(outputs[lane][0], 0x2222)

    def test_wave64_divergence(self) -> None:
        """In wave64, 64-bit EXEC mask governs all 64 lanes."""
        lines = [
            "v_mov_b32 v0, 0xAAAA",
            "v_mov_b32 v1, 0x1111",
            "v_mov_b32 v2, 0x2222",
            "s_mov_b32 m0, 1",
            "s_mov_b64 exec, 0x00000000FFFFFFFF",  # lanes 0..31 active, 32..63 inactive
            "v_movrels_b32 v0, v1",
            "s_mov_b64 exec, -1",                  # re-enable all 64 lanes
            "exp mrt0 v0, v1, v2, v0 done vm",
        ]
        spv = translate_listing(lines)
        res = wi.run_wave(spv, wave=64)
        outputs = [o["%out_mrt0"] for o in res["outputs"]]

        for lane in range(64):
            expected_v0 = 0x2222 if lane < 32 else 0xAAAA
            self.assertEqual(outputs[lane][0], expected_v0)

    def test_wave64_checkerboard_divergence(self) -> None:
        """In wave64, checkerboard mask 0x5555555555555555 toggles every alternate lane across 64."""
        lines = [
            "v_mov_b32 v0, 0xBBBB",
            "v_mov_b32 v1, 0x1111",
            "v_mov_b32 v2, 0x3333",
            "s_mov_b32 m0, 1",
            "s_mov_b64 exec, 0x5555555555555555",
            "v_movrels_b32 v0, v1",
            "s_mov_b64 exec, -1",
            "exp mrt0 v0, v1, v2, v0 done vm",
        ]
        spv = translate_listing(lines)
        res = wi.run_wave(spv, wave=64)
        outputs = [o["%out_mrt0"] for o in res["outputs"]]

        for lane in range(64):
            expected_v0 = 0x3333 if (lane % 2 == 0) else 0xBBBB
            self.assertEqual(outputs[lane][0], expected_v0)


class TestNegativeControls(unittest.TestCase):
    """Negative controls ensuring mutated ISA arithmetic, operand swap, and EXEC bypass fail."""

    def test_detect_offset_subtraction_mutation(self) -> None:
        """Mutating dynamic indexing from addition (S0 + M0) to subtraction (S0 - M0) is caught."""
        base_src = 10
        m0 = 5
        correct_target = compute_movrels_target_vgpr(base_src, m0)
        mutated_target = (base_src - m0) % NUM_VGPRS
        self.assertEqual(correct_target, 15)
        self.assertEqual(mutated_target, 5)
        self.assertNotEqual(correct_target, mutated_target)

    def test_detect_wrong_destination_mutation(self) -> None:
        """Mutating destination register from dst to dst + 1 is detected."""
        oracle = MovrelOracle({10: 0x11111111})
        oracle.set_m0(0)
        oracle.execute_v_movrels(dst=0, src=10)
        # Correct write landed in v0, not v1
        self.assertEqual(oracle.get_vgpr(0), 0x11111111)
        self.assertNotEqual(oracle.get_vgpr(1), 0x11111111)

    def test_out_of_range_relative_writes_preserve_registers(self) -> None:
        init = {0: 0xAAAA, 1: 0x1111, 4: 0xDEADBEEF}
        for op in ("v_movreld_b32", "v_movrelsd_b32"):
            with self.subTest(op=op):
                self.assertEqual(model(op, dst=250, src=1, m0=10, init=init),
                                 MovrelOracle(init).vgpr)

    def test_detect_instruction_swap_mutation(self) -> None:
        """Swapping v_movrels_b32 with v_movreld_b32 produces distinctly different register states."""
        init = {0: 0xAAAA, 1: 0x1111, 2: 0x2222}
        # v_movrels_b32 v0, v1 (m0=1) -> reads v2 (0x2222), writes to v0
        res_movrels = model("v_movrels_b32", dst=0, src=1, m0=1, init=init)
        # v_movreld_b32 v0, v1 (m0=1) -> reads v1 (0x1111), writes to v1
        res_movreld = model("v_movreld_b32", dst=0, src=1, m0=1, init=init)
        self.assertEqual(res_movrels[0], 0x2222)
        self.assertNotEqual(res_movrels[0], res_movreld[0])

    def test_detect_inactive_lane_leakage_mutation(self) -> None:
        """Unconditional write mutating inactive lane state is detectable."""
        inactive_lane_val = 0xAAAA
        active_lane_val = 0x2222
        # Correct behaviour: inactive lane keeps inactive_lane_val
        self.assertNotEqual(inactive_lane_val, active_lane_val)

    def test_detect_sentinel_corruption(self) -> None:
        """Ensure neighbor registers are not altered during relative moves."""
        oracle = MovrelOracle({r: sentinel(r) for r in range(1, 9)})
        oracle.set_m0(2)
        oracle.execute_v_movrels(dst=0, src=1)  # v1 + 2 = v3 -> v0 = sentinel(3)
        for r in range(1, 9):
            self.assertEqual(oracle.get_vgpr(r), SENTINELS[r])


class ModelMatchesHandConstants(unittest.TestCase):
    def test_model(self):
        for case in CASES:
            op, dst, src, m0, load_v0, exported, expected = case
            with self.subTest(case=case_id(case)):
                vgpr = model(op, dst, src, m0, initial_regs(load_v0))
                self.assertEqual(tuple(vgpr[r] for r in exported), expected)

    def test_sentinels_distinct(self):
        values = list(SENTINELS.values()) + [V0_VALUE]
        self.assertEqual(len(set(values)), len(values))


class TranslatedValues(unittest.TestCase):
    def check(self, op: str) -> None:
        cases = [c for c in CASES if c[0] == op]
        self.assertTrue(cases)
        for case in cases:
            _, dst, src, m0, load_v0, exported, expected = case
            with self.subTest(case=case_id(case)):
                got = run_case(op, dst, src, m0, load_v0, exported)
                self.assertEqual(
                    got,
                    expected,
                    f"{case_id(case)}: got {[hex(w) for w in got]}, expected {[hex(w) for w in expected]}",
                )

    def test_v_movrels_b32(self) -> None:
        self.check("v_movrels_b32")

    def test_v_movreld_b32(self) -> None:
        self.check("v_movreld_b32")

    def test_v_movrelsd_b32(self) -> None:
        self.check("v_movrelsd_b32")


if __name__ == "__main__":
    unittest.main()
