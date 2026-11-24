"""Semantic interpreter tests for vector bitwise lane-mask tracking (`v_and_or_b32`, `v_xnor_b32`, `v_bfi_b32`)."""
import gc
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
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]  # OOB_SELECT 3 (raw): stores up to num_records
SENTINEL = 0xCAFEBABE


def _translate_cs(name, body_lines, wave_size=32, emulate=False):
    """Translate a synthetic compute shader and return `(spv, gen)`."""
    prev_emu = rdna2spirv.WAVE64_EMULATE
    try:
        rdna2spirv.WAVE64_EMULATE = emulate
        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(os.path.join(tmp, "disasm"))
            os.mkdir(os.path.join(tmp, "fssm"))
            s_path = os.path.join(tmp, "disasm", name + ".s")
            with open(s_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(f"; {name}\n")
                for ln in body_lines:
                    f.write(f"\t{ln}\n")
                f.write("\ts_endpgm\n")
            with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
                f.write(STUB_DXBC)
            # COMPUTE_PGM_RSRC2: 0x8088 -> 4 user SGPRs (s[0:3] = output V#), TGID_X.
            with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
                f.write(bytes(0x138) + struct.pack("<4I", 0x8088, wave_size, 1, 1))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spv = rdna2spirv.translate(s_path)
                gc.collect()
            gen = rdna2spirv.translate.last_gen
            assert gen.wave64 == (wave_size == 64), f"{name}: expected wave64={wave_size == 64}, got {gen.wave64}"
            return spv, gen
    finally:
        rdna2spirv.WAVE64_EMULATE = prev_emu


def _run_cs(name, body_lines, wave_size=32, emulate=False):
    """Translate a synthetic compute shader and execute every lane in spirv_mini_interp."""
    spv, _ = _translate_cs(name, body_lines, wave_size=wave_size, emulate=emulate)
    sg_size = 32 if emulate else wave_size
    pc = VSHARP + [0] * 28
    return si.run_workgroup(
        spv, wave_size, sg_size,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i % sg_size},
        push_constants=pc, memory={},
    )


def _bitwise_mask_lines(wave_size):
    """Build shader computing masks via v_and_or_b32, v_xnor_b32, and v_bfi_b32 under partial EXEC."""
    is64 = wave_size == 64
    r = {
        "exec": "exec" if is64 else "exec_lo",
        "mov": "s_mov_b64" if is64 else "s_mov_b32",
        "saveexec": "s_and_saveexec_b64" if is64 else "s_and_saveexec_b32",
        "bcnt": "s_bcnt1_i32_b64" if is64 else "s_bcnt1_i32_b32",
        "s4": "s[4:5]" if is64 else "s4",
        "s6": "s[6:7]" if is64 else "s6",
        "s8": "s[8:9]" if is64 else "s8",
        "s10": "s[10:11]" if is64 else "s10",
        "s12": "s[12:13]" if is64 else "s12",
        "s14": "s[14:15]" if is64 else "s14",
        "s16": "s[16:17]" if is64 else "s16",
        "s18": "s[18:19]" if is64 else "s18",
    }
    lines = [
        f"{r['mov']} {r['exec']}, -1",
        "v_and_b32 v1, 1, v0",
        f"v_cmp_eq_u32_e64 {r['s6']}, v1, 0",        # even lanes
        f"v_cmp_lt_u32_e64 {r['s8']}, v0, 10",       # lanes < 10
        "v_and_b32 v1, 2, v0",
        f"v_cmp_eq_u32_e64 {r['s10']}, v1, 0",       # (lane & 2) == 0
        f"v_cmp_lt_u32_e64 {r['s12']}, v0, 8",       # partial EXEC: lanes 0..7 active
        "v_mov_b32 v2, 0",
        "v_mov_b32 v3, 0",
        "v_mov_b32 v4, 0",
    ]
    if is64:
        lines += ["v_mov_b32 v5, 0", "v_mov_b32 v6, 0", "v_mov_b32 v7, 0"]
    lines += [
        f"{r['saveexec']} {r['s4']}, {r['s12']}",
        # Mask ops under partial EXEC must update inactive lanes too.
        "v_and_or_b32 v2, s6, s8, s10",
        "v_xnor_b32 v3, s6, s8",
        "v_bfi_b32 v4, s6, s8, s10",
    ]
    if is64:
        lines += [
            "v_and_or_b32 v5, s7, s9, s11",
            "v_xnor_b32 v6, s7, s9",
            "v_bfi_b32 v7, s7, s9, s11",
        ]
    lines += [
        f"{r['mov']} {r['exec']}, -1",
        "v_readfirstlane_b32 s14, v2",
        "v_readfirstlane_b32 s16, v3",
        "v_readfirstlane_b32 s18, v4",
    ]
    if is64:
        lines += [
            "v_readfirstlane_b32 s15, v5",
            "v_readfirstlane_b32 s17, v6",
            "v_readfirstlane_b32 s19, v7",
        ]
    lines += [
        f"{r['bcnt']} s20, {r['s14']}",
        f"{r['bcnt']} s21, {r['s16']}",
        f"{r['bcnt']} s22, {r['s18']}",
        f"v_cndmask_b32_e64 v8, 10, 20, {r['s14']}",
        f"v_cndmask_b32_e64 v9, 30, 40, {r['s16']}",
        f"v_cndmask_b32_e64 v10, 50, 60, {r['s18']}",
        "v_mov_b32 v11, s20",
        "v_mov_b32 v12, s21",
        "v_mov_b32 v13, s22",
        "v_mul_lo_u32 v14, v0, 36",
    ]
    for off, vreg in enumerate((2, 3, 4, 8, 9, 10, 11, 12, 13)):
        if off == 0:
            lines.append(f"buffer_store_dword v{vreg}, v14, s[0:3], 0 offen")
        else:
            lines.append(f"v_add_nc_u32 v15, {off * 4}, v14")
            lines.append(f"buffer_store_dword v{vreg}, v15, s[0:3], 0 offen")
    return lines


def _vgpr_stores(spv):
    """Return mapping `vgpr_idx -> [rhs_expression, ...]` for constant-index `%vgpr` OpStores."""
    const_uint = {}
    ptr_vgpr = {}
    defs = {}
    stores = {}
    for ln in spv.splitlines():
        s = ln.strip()
        if " = " in s:
            lhs, rhs = s.split(" = ", 1)
            defs[lhs] = rhs
            parts = rhs.split()
            if parts[:2] == ["OpConstant", "%uint"] and len(parts) == 3:
                const_uint[lhs] = int(parts[2])
            elif (
                len(parts) == 4
                and parts[0] == "OpAccessChain"
                and parts[1] == "%_ptr_Function_uint"
                and parts[2] == "%vgpr"
                and parts[3] in const_uint
            ):
                ptr_vgpr[lhs] = const_uint[parts[3]]
        elif s.startswith("OpStore "):
            parts = s.split()
            if len(parts) >= 3 and parts[1] in ptr_vgpr:
                idx = ptr_vgpr[parts[1]]
                stores.setdefault(idx, []).append(defs.get(parts[2], parts[2]))
    return stores


class _VectorBitwiseMixin:
    wave_size = 32

    def test_vector_bitwise_masks_under_partial_exec(self):
        mem = _run_cs(
            f"cs_vec_bitwise_w{self.wave_size}",
            _bitwise_mask_lines(self.wave_size),
            wave_size=self.wave_size,
            emulate=False,
        )
        want_andor = [((i & 1 == 0) and i < 10) or (i & 2 == 0) for i in range(self.wave_size)]
        want_xnor = [(i & 1 == 0) == (i < 10) for i in range(self.wave_size)]
        want_bfi = [(i < 10) if (i & 1 == 0) else (i & 2 == 0) for i in range(self.wave_size)]
        cnt_andor = sum(want_andor)
        cnt_xnor = sum(want_xnor)
        cnt_bfi = sum(want_bfi)
        for lane in range(self.wave_size):
            base = lane * 36
            self.assertEqual(mem[base], 0xFFFFFFFF if want_andor[lane] else 0, f"lane {lane} raw and_or")
            self.assertEqual(mem[base + 4], 0xFFFFFFFF if want_xnor[lane] else 0, f"lane {lane} raw xnor")
            self.assertEqual(mem[base + 8], 0xFFFFFFFF if want_bfi[lane] else 0, f"lane {lane} raw bfi")
            self.assertEqual(mem[base + 12], 20 if want_andor[lane] else 10, f"lane {lane} cndmask and_or")
            self.assertEqual(mem[base + 16], 40 if want_xnor[lane] else 30, f"lane {lane} cndmask xnor")
            self.assertEqual(mem[base + 20], 60 if want_bfi[lane] else 50, f"lane {lane} cndmask bfi")
            self.assertEqual(mem[base + 24], cnt_andor, f"lane {lane} bcnt and_or")
            self.assertEqual(mem[base + 28], cnt_xnor, f"lane {lane} bcnt xnor")
            self.assertEqual(mem[base + 32], cnt_bfi, f"lane {lane} bcnt bfi")


class TestVectorBitwiseMasksWave32(_VectorBitwiseMixin, unittest.TestCase):
    wave_size = 32


class TestVectorBitwiseMasksWave64Native(_VectorBitwiseMixin, unittest.TestCase):
    wave_size = 64


class TestVectorBitwiseMasksWave64Emulated(unittest.TestCase):
    """Structural SPIR-V checks for `--wave64-emulate` (spirv_mini_interp rejects OpControlBarrier)."""

    def test_vector_bitwise_masks_emulate_spirv_structure(self):
        spv, gen = _translate_cs(
            "cs_vec_bitwise_w64_emu",
            _bitwise_mask_lines(64),
            wave_size=64,
            emulate=True,
        )
        self.assertTrue(gen.wave64 and gen.emulate)
        self.assertEqual(len(gen.mask_writes), 6)
        self.assertNotIn("FindILsb", spv)
        self.assertIn("OpControlBarrier", spv)
        self.assertIn("OpStore %x_it", spv)
        stores = _vgpr_stores(spv)
        for vreg in (2, 5):  # v_and_or_b32 -> final OpBitwiseOr
            self.assertTrue(stores[vreg][-1].startswith("OpBitwiseOr %uint "), stores[vreg][-1])
        for vreg in (3, 6):  # v_xnor_b32 -> final OpNot
            self.assertTrue(stores[vreg][-1].startswith("OpNot %uint "), stores[vreg][-1])
        for vreg in (4, 7):  # v_bfi_b32 -> final OpBitwiseOr
            self.assertTrue(stores[vreg][-1].startswith("OpBitwiseOr %uint "), stores[vreg][-1])


class TestVectorBitwiseDataControl(unittest.TestCase):
    def test_non_mask_data_stays_exec_gated_wave32(self):
        lines = [
            "s_mov_b32 exec_lo, -1",
            "v_cmp_lt_u32_e64 s8, v0, 16",
            f"v_mov_b32 v2, {SENTINEL:#x}",
            f"v_mov_b32 v3, {SENTINEL:#x}",
            f"v_mov_b32 v4, {SENTINEL:#x}",
            "s_and_saveexec_b32 s6, s8",
            "v_and_or_b32 v2, v0, 1, 2",
            "v_xnor_b32 v3, v0, 3",
            "v_bfi_b32 v4, v0, 0x0f0f0f0f, 0x10101010",
            "s_mov_b32 exec_lo, s6",
            "v_mul_lo_u32 v10, v0, 12",
            "buffer_store_dword v2, v10, s[0:3], 0 offen",
            "v_add_nc_u32 v11, 4, v10",
            "buffer_store_dword v3, v11, s[0:3], 0 offen",
            "v_add_nc_u32 v11, 8, v10",
            "buffer_store_dword v4, v11, s[0:3], 0 offen",
        ]
        mem = _run_cs("cs_vec_bitwise_data_w32", lines, wave_size=32, emulate=False)
        for lane in range(32):
            base = lane * 12
            if lane < 16:
                self.assertEqual(mem[base], (lane & 1) | 2, f"lane {lane} data and_or")
                self.assertEqual(mem[base + 4], (~(lane ^ 3)) & 0xFFFFFFFF, f"lane {lane} data xnor")
                want_bfi = ((lane & 0x0F0F0F0F) | ((~lane) & 0x10101010)) & 0xFFFFFFFF
                self.assertEqual(mem[base + 8], want_bfi, f"lane {lane} data bfi")
            else:
                self.assertEqual(mem[base], SENTINEL, f"inactive lane {lane} and_or")
                self.assertEqual(mem[base + 4], SENTINEL, f"inactive lane {lane} xnor")
                self.assertEqual(mem[base + 8], SENTINEL, f"inactive lane {lane} bfi")


if __name__ == "__main__":
    unittest.main()
