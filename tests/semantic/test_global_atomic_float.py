"""Semantic tests for RDNA2 global_atomic_fmin and global_atomic_fmax (32-bit float).

Verifies the CAS loop lowering (OpAtomicCompareExchange + GLSL.std.450 NMin/NMax):
  - IEEE 754 float semantics: positives, negatives, signed zeros (-0.0 < +0.0), infinities,
    subnormals, and NaN handling (NaN operand returns the other operand).
  - Pre-op value return in vdst when 'glc' is present, and preservation of vdst when 'glc' is absent.
  - Addressing modes: saddr SGPR pair, saddr='off' with 64-bit vaddr, and signed 12-bit offsets (-2048..2044).
  - EXEC masking: inactive lanes bypass the CAS loop entirely, leave memory untouched, and retain vdst.
  - Multi-lane CAS contention: multiple active lanes racing on a single shared memory location.
  - Rejection of invalid modifiers (slc, dlc, lds), operand counts, and multi-dword registers.
  - Dependency analysis (read_before_write_vgprs).

Note on hardware fidelity (measured natively on gfx1036 for #2361 by @Pierre-Kookis):
  - The expectations below verify what rdna2spirv's CAS lowering with GLSL.std.450 NMin/NMax produces today
    with denormals kept (Vulkan default).
  - On native hardware under the game's MODE (f32 denormals flushed), a denormal operand counts as a signed zero.
  - On native hardware, an sNaN in either operand wins and is stored quieted (0x7fc00001), whereas GLSL NMin/NMax
    returns the non-NaN operand.

Standard library only; runs through spirv_mini_interp on the CPU without GPU, LLVM, or game files.
"""
from __future__ import annotations

import math
import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
from shader_ir import parse_line as parse  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x88, 32, 1, 1)  # 4 user SGPRs, workgroup id in s4, 32x1x1
FSSM = bytes(FSSM)

TARGET = 0x2000
OUT = 0x3000

# IEEE 754 32-bit float constants in raw hex
POS_ZERO = 0x00000000
NEG_ZERO = 0x80000000
ONE = 0x3F800000       # 1.0
TWO = 0x40000000       # 2.0
THREE = 0x40400000     # 3.0
HALF = 0x3F000000      # 0.5
NEG_ONE = 0xBF800000   # -1.0
NEG_TWO = 0xC0000000   # -2.0
POS_INF = 0x7F800000   # +inf
NEG_INF = 0xFF800000   # -inf
QNAN = 0x7FC00001      # Quiet NaN
SNAN = 0x7F800001      # Signalling NaN
NEG_QNAN = 0xFFC00000  # Negative Quiet NaN
SUBNORM_MIN = 0x00000001  # Smallest positive subnormal
SUBNORM_MAX = 0x007FFFFF  # Largest subnormal


def translate(body: list[str]) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        for d in ("disasm", "fssm"):
            os.mkdir(os.path.join(tmp, d))
        with open(os.path.join(tmp, "cs_gatomic_f.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", "cs_gatomic_f.fssm"), "wb") as f:
            f.write(FSSM)
        src = os.path.join(tmp, "disasm", "cs_gatomic_f.s")
        with open(src, "w", encoding="utf-8", newline="\n") as f:
            f.write("; cs_gatomic_f\n" + "".join(f"\t{ln}\n" for ln in body) + "\ts_endpgm\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            return rdna2spirv.translate(src)


def run_cs(body: list[str], memory: dict[int, int], nlanes: int = 1) -> dict[int, int]:
    spv = translate(body)
    mem = dict(memory)
    si.run_workgroup(spv, nlanes, 32,
                     inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                     push_constants=[0] * 36, memory=mem)
    return mem


def run_atomic(kind: str, mem_init: int, data_val: int, glc: bool = True,
               offset: int = 0, vaddr: int = TARGET) -> tuple[int | None, int | None]:
    body = [
        f"v_mov_b32_e32 v1, {vaddr - offset:#x}",
        f"v_mov_b32_e32 v2, {data_val:#x}",
        "v_mov_b32_e32 v5, 0xdeadbeef"
    ]
    off_str = f" offset:{offset}" if offset else ""
    if glc:
        body.append(f"global_atomic_{kind} v5, v1, v2, s[0:1]{off_str} glc")
    else:
        body.append(f"global_atomic_{kind} v1, v2, s[0:1]{off_str}")
    body += [
        "s_waitcnt vmcnt(0)",
        f"v_mov_b32_e32 v6, {OUT:#x}",
        "global_store_dword v6, v5, s[0:1]"
    ]
    res_mem = run_cs(body, {TARGET: mem_init})
    return res_mem.get(OUT), res_mem.get(TARGET)


class GlobalAtomicFloatSemantics(unittest.TestCase):
    def test_fmin_fmax_standard_values(self):
        cases = [
            # (kind, init_mem, data, expected_final)
            # Basic ordered values
            ("fmin", ONE, TWO, ONE),
            ("fmin", TWO, ONE, ONE),
            ("fmin", ONE, NEG_TWO, NEG_TWO),
            ("fmin", NEG_TWO, ONE, NEG_TWO),
            ("fmin", NEG_ONE, NEG_TWO, NEG_TWO),
            ("fmax", ONE, TWO, TWO),
            ("fmax", TWO, ONE, TWO),
            ("fmax", ONE, NEG_TWO, ONE),
            ("fmax", NEG_TWO, ONE, ONE),
            ("fmax", NEG_ONE, NEG_TWO, NEG_ONE),
            # Signed zeros (-0.0 < +0.0)
            ("fmin", POS_ZERO, NEG_ZERO, NEG_ZERO),
            ("fmin", NEG_ZERO, POS_ZERO, NEG_ZERO),
            ("fmax", POS_ZERO, NEG_ZERO, POS_ZERO),
            ("fmax", NEG_ZERO, POS_ZERO, POS_ZERO),
            # Infinities
            ("fmin", ONE, POS_INF, ONE),
            ("fmin", POS_INF, ONE, ONE),
            ("fmin", ONE, NEG_INF, NEG_INF),
            ("fmin", NEG_INF, ONE, NEG_INF),
            ("fmin", POS_INF, NEG_INF, NEG_INF),
            ("fmax", ONE, POS_INF, POS_INF),
            ("fmax", POS_INF, ONE, POS_INF),
            ("fmax", ONE, NEG_INF, ONE),
            ("fmax", NEG_INF, ONE, ONE),
            ("fmax", POS_INF, NEG_INF, POS_INF),
            # Subnormal numbers: holds with denormals kept (Vulkan default).
            # On gfx1036 in denormals-flushed MODE, denormals flush to signed zero (#2361).
            ("fmin", POS_ZERO, SUBNORM_MIN, POS_ZERO),
            ("fmin", SUBNORM_MAX, SUBNORM_MIN, SUBNORM_MIN),
            ("fmax", POS_ZERO, SUBNORM_MIN, SUBNORM_MIN),
            ("fmax", SUBNORM_MAX, SUBNORM_MIN, SUBNORM_MAX),
            # NaN handling: GLSL.std.450 NMin/NMax returns the non-NaN operand.
            # On native RDNA2 hardware (#2361), sNaN in either operand wins and is stored quieted.
            ("fmin", QNAN, ONE, ONE),
            ("fmin", ONE, QNAN, ONE),
            ("fmin", SNAN, NEG_TWO, NEG_TWO),
            ("fmin", NEG_TWO, SNAN, NEG_TWO),
            ("fmin", NEG_QNAN, HALF, HALF),
            ("fmax", QNAN, ONE, ONE),
            ("fmax", ONE, QNAN, ONE),
            ("fmax", SNAN, NEG_TWO, NEG_TWO),
            ("fmax", NEG_TWO, SNAN, NEG_TWO),
            ("fmax", NEG_QNAN, HALF, HALF),
        ]
        for kind, m_init, d_val, exp_final in cases:
            with self.subTest(op=kind, m=hex(m_init), d=hex(d_val)):
                out_vdst, final_mem = run_atomic(kind, m_init, d_val, glc=True)
                self.assertEqual(out_vdst, m_init, "glc must return pre-op memory value in vdst")
                self.assertEqual(final_mem, exp_final, "memory must reflect atomic fmin/fmax result")

    def test_both_nan_leaves_nan_in_memory(self):
        # When both memory and data are NaN, the result is NaN
        for kind in ("fmin", "fmax"):
            with self.subTest(op=kind):
                out_vdst, final_mem = run_atomic(kind, QNAN, SNAN, glc=True)
                self.assertEqual(out_vdst, QNAN)
                # Unpack as float32 to check NaN property
                f_val = struct.unpack("<f", struct.pack("<I", final_mem))[0]
                self.assertTrue(math.isnan(f_val), f"Expected NaN, got {hex(final_mem)}")

    def test_without_glc_vdst_preserved(self):
        # Without glc, the temporary register v5 must keep its sentinel 0xDEADBEEF
        for kind, m_init, d_val, exp_final in (("fmin", TWO, ONE, ONE), ("fmax", ONE, TWO, TWO)):
            with self.subTest(op=kind):
                out_vdst, final_mem = run_atomic(kind, m_init, d_val, glc=False)
                self.assertEqual(out_vdst, 0xDEADBEEF, "without glc, vdst register is preserved")
                self.assertEqual(final_mem, exp_final)

    def test_offsets_and_bounds(self):
        for offset in (-2048, -1024, -4, 4, 1024, 2044):
            with self.subTest(offset=offset):
                out_vdst, final_mem = run_atomic("fmin", TWO, ONE, offset=offset)
                self.assertEqual(out_vdst, TWO)
                self.assertEqual(final_mem, ONE)
                out_vdst, final_mem = run_atomic("fmax", ONE, TWO, offset=offset)
                self.assertEqual(out_vdst, ONE)
                self.assertEqual(final_mem, TWO)

    def test_off_addressing(self):
        # 64-bit vaddr with saddr = 'off'
        body = [
            f"v_mov_b32_e32 v8, {TARGET:#x}",
            "v_mov_b32_e32 v9, 0",
            f"v_mov_b32_e32 v2, {ONE:#x}",
            "v_mov_b32_e32 v5, 0xdeadbeef",
            "global_atomic_fmin v5, v[8:9], v2, off glc",
            "s_waitcnt vmcnt(0)",
            f"v_mov_b32_e32 v6, {OUT:#x}",
            "global_store_dword v6, v5, s[0:1]",
        ]
        mem = run_cs(body, {TARGET: TWO})
        self.assertEqual(mem.get(OUT), TWO)
        self.assertEqual(mem.get(TARGET), ONE)

    def test_exec_mask_inactive_lanes_preserve_memory_and_vdst(self):
        # 4-lane workgroup where only lane 1 is active
        # Lanes 0, 2, 3 must not write to TARGET and must preserve vdst=0xCAFE0000 | lane
        body = [
            f"v_mov_b32_e32 v1, {TARGET:#x}",
            f"v_mov_b32_e32 v2, {NEG_TWO:#x}",
            "v_mov_b32_e32 v5, 0xcafe0000",
            "v_or_b32_e32 v5, v5, v0",  # sentinel per lane
            "v_cmp_eq_u32_e32 vcc_lo, 1, v0",  # lane 1 active only
            "s_and_saveexec_b32 s4, vcc_lo",
            "global_atomic_fmin v5, v1, v2, s[0:1] glc",
            "s_mov_b32 exec_lo, s4",  # restore exec
            "s_waitcnt vmcnt(0)",
            "v_lshlrev_b32_e32 v6, 2, v0",
            f"v_add_nc_u32_e32 v6, {OUT:#x}, v6",
            "global_store_dword v6, v5, s[0:1]",
        ]
        mem = run_cs(body, {TARGET: ONE}, nlanes=4)
        self.assertEqual(mem.get(TARGET), NEG_TWO, "active lane 1 should have updated TARGET")
        self.assertEqual(mem.get(OUT), 0xCAFE0000, "inactive lane 0 preserves vdst")
        self.assertEqual(mem.get(OUT + 4), ONE, "active lane 1 gets pre-op memory value")
        self.assertEqual(mem.get(OUT + 8), 0xCAFE0002, "inactive lane 2 preserves vdst")
        self.assertEqual(mem.get(OUT + 12), 0xCAFE0003, "inactive lane 3 preserves vdst")

    def test_all_inactive_lanes_leave_memory_untouched(self):
        # Complete wave masked out (exec=0)
        body = [
            f"v_mov_b32_e32 v1, {TARGET:#x}",
            f"v_mov_b32_e32 v2, {NEG_TWO:#x}",
            "v_mov_b32_e32 v5, 0x12345678",
            "s_mov_b32 exec_lo, 0",  # disable all lanes
            "global_atomic_fmin v5, v1, v2, s[0:1] glc",
            "s_mov_b32 exec_lo, 1",  # re-enable lane 0 for output
            "s_waitcnt vmcnt(0)",
            f"v_mov_b32_e32 v6, {OUT:#x}",
            "global_store_dword v6, v5, s[0:1]",
        ]
        mem = run_cs(body, {TARGET: ONE}, nlanes=1)
        self.assertEqual(mem.get(TARGET), ONE, "memory must remain untouched when all lanes inactive")
        self.assertEqual(mem.get(OUT), 0x12345678, "vdst preserved when inactive")

    def test_multi_lane_contention(self):
        # 3 lanes concurrently updating TARGET: lane 0 proposes 1.0, lane 1 proposes 2.0, lane 2 proposes -1.0
        # For fmin, final must be -1.0; for fmax, final must be 3.0 (from init=3.0) or 2.0 (from init=0.0)
        vals = [ONE, TWO, NEG_ONE]
        body = [
            f"v_mov_b32_e32 v1, {TARGET:#x}",
            "v_mov_b32_e32 v5, 0",
            f"v_mov_b32_e32 v2, {vals[0]:#x}",
            "v_cmp_eq_u32_e32 vcc_lo, 1, v0",
            f"v_cndmask_b32_e32 v2, v2, {vals[1]:#x}, vcc_lo",
            "v_cmp_eq_u32_e32 vcc_lo, 2, v0",
            f"v_cndmask_b32_e32 v2, v2, {vals[2]:#x}, vcc_lo",
            "global_atomic_fmin v5, v1, v2, s[0:1] glc",
            "s_waitcnt vmcnt(0)",
            "v_lshlrev_b32_e32 v6, 2, v0",
            f"v_add_nc_u32_e32 v6, {OUT:#x}, v6",
            "global_store_dword v6, v5, s[0:1]",
        ]
        mem = run_cs(body, {TARGET: THREE}, nlanes=3)
        self.assertEqual(mem.get(TARGET), NEG_ONE, "CAS loop must resolve contention to minimum")
        # Every lane must have seen a valid pre-CAS state (one saw 3.0, others intermediate values)
        seen_vdsts = {mem.get(OUT), mem.get(OUT + 4), mem.get(OUT + 8)}
        self.assertIn(THREE, seen_vdsts, "At least one lane must see the initial memory value")

    def test_rejected_forms(self):
        invalid_instructions = [
            # Unsupported modifiers
            "global_atomic_fmin v5, v1, v2, s[0:1] slc glc",
            "global_atomic_fmax v5, v1, v2, s[0:1] dlc glc",
            "global_atomic_fmin v5, v1, v2, s[0:1] lds glc",
            # Multi-dword registers
            "global_atomic_fmin_x2 v[4:5], v1, v[2:3], s[0:1] glc",
            "global_atomic_fmax v5, v1, v[2:3], s[0:1] glc",
            "global_atomic_fmin v[4:5], v1, v2, s[0:1] glc",
            # Arity / glc mismatch
            "global_atomic_fmin v5, v1, v2, s[0:1]",        # vdst present without glc
            "global_atomic_fmax v1, v2, s[0:1] glc",        # glc present without vdst
        ]
        for ins in invalid_instructions:
            with self.subTest(instruction=ins):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate(["v_mov_b32_e32 v1, 0", "v_mov_b32_e32 v2, 0", ins])

    def test_read_before_write_analysis(self):
        # glc variant reads vaddr (1) and vdata (2), writes vdst (5)
        parsed_glc = [parse("\tglobal_atomic_fmin v5, v1, v2, s[0:1] glc")]
        self.assertEqual(rdna2spirv.read_before_write_vgprs(parsed_glc), {1, 2})

        # non-glc variant reads vaddr (1) and vdata (2), writes no VGPR
        parsed_no_glc = [parse("\tglobal_atomic_fmax v1, v2, s[0:1]")]
        self.assertEqual(rdna2spirv.read_before_write_vgprs(parsed_no_glc), {1, 2})


if __name__ == "__main__":
    unittest.main()
