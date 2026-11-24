"""Semantic value tests for buffer_atomic_{fmin,fmax} and atomic natural alignment / 64-bit bounds.

Covers two native RDNA2 (gfx1036, wave32 & wave64) hardware measurements:
1. buffer_atomic_{fmin,fmax}: 32-bit float min/max at the MUBUF address via the shared
   OpAtomicCompareExchange + GLSL.std.450 NMin/NMax CAS loop (no float-atomic extension),
   with V# bounds, 4-byte natural alignment, and glc return semantics.
2. Natural alignment across buffer_atomic*, buffer_atomic*_x2, global_atomic*, and
   global_atomic*_x2: 32-bit atomics require (addr & 3) == 0 and 64-bit _x2 atomics require
   (addr & 7) == 0; unaligned or OOB accesses leave memory untouched, active glc lanes receive
   0 / (0, 0), and inactive lanes (EXEC=0) preserve vdst/vdata. Also pins all-or-nothing
   8-byte bounds on buffer_atomic*_x2 across OOB_SELECT modes 0..3.

Standard library only; no GPU, no LLVM, no game files.
"""
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

M32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x88, 32, 1, 1)  # 4 user SGPRs, workgroup id in s4, 32x1x1
FSSM = bytes(FSSM)

RAW_W3 = 0x31016FAC     # OOB_SELECT = 3 (RAW)
MODE0_W3 = 0x01016FAC   # OOB_SELECT = 0 (STRUCTURED_WITH_OFFSET)
MODE1_W3 = 0x11016FAC   # OOB_SELECT = 1 (STRUCTURED)
MODE2_W3 = 0x21016FAC   # OOB_SELECT = 2 (DISABLED)

TARGET, OUT = 0x2000, 0x3000
SENTINEL, SENTINEL2 = 0xDEAD1234, 0xBEEF5678


def translate(body):
    with tempfile.TemporaryDirectory() as tmp:
        for d in ("disasm", "fssm"):
            os.mkdir(os.path.join(tmp, d))
        with open(os.path.join(tmp, "cs_atom.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", "cs_atom.fssm"), "wb") as f:
            f.write(FSSM)
        src = os.path.join(tmp, "disasm", "cs_atom.s")
        with open(src, "w", encoding="utf-8") as f:
            f.write("; cs_atom\n" + "".join(f"\t{ln}\n" for ln in body) + "\ts_endpgm\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            return rdna2spirv.translate(src)


def run(body, memory, nlanes=1):
    spv = translate(body)
    mem = dict(memory)
    si.run_workgroup(spv, nlanes, 32,
                     inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                     push_constants=[0] * 36, memory=mem)
    return mem


def buf_rsrc_lines(base, stride, num_records, word3):
    w0 = base & M32
    w1 = ((base >> 32) & 0xFFFF) | ((stride & 0x3FFF) << 16)
    return [
        f"s_mov_b32 s8, {w0:#x}",
        f"s_mov_b32 s9, {w1:#x}",
        f"s_mov_b32 s10, {num_records:#x}",
        f"s_mov_b32 s11, {word3:#x}",
    ]


def out_store_lines(reg, count=1, lane_stride=4):
    store = "buffer_store_dword" if count == 1 else f"buffer_store_dwordx{count}"
    lines = [
        "s_mov_b32 s8, 0",
        "s_mov_b32 s9, 0",
        "s_mov_b32 s10, 0x10000",
        f"s_mov_b32 s11, {RAW_W3:#x}",
        f"v_mul_lo_u32 v10, v0, {lane_stride}",
        f"v_add_nc_u32_e32 v10, {OUT:#x}, v10",
        f"{store} {reg}, v10, s[8:11], 0 offen",
    ]
    return lines


def run_buf_fminmax(kind, m, d, glc=True, base=0, stride=0, offset=0x40, num_records=0x100,
                    word3=RAW_W3, idx=None, nlanes=1, active_mask_lane0_only=False):
    mods = []
    addr_lines = []
    if idx is not None:
        addr_lines.append(f"v_mov_b32_e32 v1, {idx:#x}")
        addr_lines.append(f"v_mov_b32_e32 v2, {offset:#x}")
        vaddr = "v[1:2]"
        mods += ["idxen", "offen"]
    else:
        addr_lines.append(f"v_mov_b32_e32 v1, {offset:#x}")
        vaddr = "v1"
        mods.append("offen")
    if glc:
        mods.append("glc")
    body = buf_rsrc_lines(base, stride, num_records, word3) + addr_lines + [
        f"v_mov_b32_e32 v3, {d:#x}",
    ]
    if active_mask_lane0_only:
        body += [
            "v_cmp_eq_u32_e32 vcc_lo, 0, v0",
            "s_and_saveexec_b32 s12, vcc_lo",
        ]
    body.append(f"buffer_atomic_{kind} v3, {vaddr}, s[8:11], 0 {' '.join(mods)}")
    if active_mask_lane0_only:
        body.append("s_mov_b32 exec_lo, s12")
    body += ["s_waitcnt vmcnt(0)"] + out_store_lines("v3", count=1, lane_stride=4)
    eff = base + (idx * stride if idx is not None else 0) + offset
    mem = run(body, {eff: m, eff & ~3: m}, nlanes=nlanes)
    return mem


class BufferAtomicFminmax(unittest.TestCase):
    def test_fmin_fmax_values_glc_and_non_glc(self):
        pos0, neg0 = 0x00000000, 0x80000000
        one, neg2, half = 0x3F800000, 0xC0000000, 0x3F000000
        pinf, ninf, qnan = 0x7F800000, 0xFF800000, 0x7FC00001
        cases = [
            ("fmin", one, half, half),
            ("fmin", half, one, half),
            ("fmin", one, neg2, neg2),
            ("fmin", pos0, neg0, neg0),
            ("fmin", neg0, pos0, neg0),
            ("fmin", one, pinf, one),
            ("fmin", one, ninf, ninf),
            ("fmin", qnan, one, one),
            ("fmin", one, qnan, one),
            ("fmax", one, half, one),
            ("fmax", half, one, one),
            ("fmax", neg2, one, one),
            ("fmax", pos0, neg0, pos0),
            ("fmax", neg0, pos0, pos0),
            ("fmax", one, pinf, pinf),
            ("fmax", one, ninf, one),
            ("fmax", qnan, one, one),
            ("fmax", one, qnan, one),
        ]
        for kind, m, d, expected in cases:
            with self.subTest(op=kind, m=hex(m), d=hex(d)):
                mem_glc = run_buf_fminmax(kind, m, d, glc=True)
                self.assertEqual((mem_glc.get(OUT), mem_glc.get(0x40)), (m, expected))
                mem_noglc = run_buf_fminmax(kind, m, d, glc=False)
                self.assertEqual((mem_noglc.get(OUT), mem_noglc.get(0x40)), (d, expected))

    def test_multi_lane_cas_contention(self):
        init, one, neg2 = 0x40400000, 0x3F800000, 0xC0000000  # 3.0, 1.0, -2.0
        for kind, expected in (("fmin", neg2), ("fmax", init)):
            with self.subTest(op=kind):
                body = buf_rsrc_lines(0, 0, 0x100, RAW_W3) + [
                    "v_mov_b32_e32 v1, 0x40",
                    "v_cmp_eq_u32_e32 vcc_lo, 0, v0",
                    f"v_mov_b32_e32 v3, {neg2:#x}",
                    f"v_cndmask_b32_e32 v3, v3, {one:#x}, vcc_lo",
                    f"buffer_atomic_{kind} v3, v1, s[8:11], 0 offen glc",
                    "s_waitcnt vmcnt(0)",
                ] + out_store_lines("v3", count=1, lane_stride=4)
                mem = run(body, {0x40: init}, nlanes=2)
                self.assertEqual(mem.get(0x40), expected)
                seen = {mem.get(OUT), mem.get(OUT + 4)}
                self.assertIn(init, seen)

    def test_bounds_raw_and_structured(self):
        one, two = 0x3F800000, 0x40000000
        # RAW (OOB_SELECT=3): offset=0, num_records=4 -> in bounds; num_records=3 -> partial OOB (dropped, returns 0)
        mem_ok = run_buf_fminmax("fmax", one, two, base=0x40, offset=0, num_records=4, word3=RAW_W3)
        self.assertEqual((mem_ok.get(OUT), mem_ok.get(0x40)), (one, two))

        mem_partial = run_buf_fminmax("fmax", one, two, base=0x40, offset=0, num_records=3, word3=RAW_W3)
        self.assertEqual((mem_partial.get(OUT), mem_partial.get(0x40)), (0, one))

        mem_oob = run_buf_fminmax("fmax", one, two, base=0x40, offset=4, num_records=4, word3=RAW_W3)
        self.assertEqual((mem_oob.get(OUT), mem_oob.get(0x44)), (0, one))

        # STRUCTURED_WITH_OFFSET (OOB_SELECT=0): stride=6, off=0 ok; stride=6, off=4 partial OOB; stride=3, off=0 OOB
        mem_st_ok = run_buf_fminmax("fmin", two, one, base=0x40, stride=6, offset=0,
                                    num_records=2, word3=MODE0_W3, idx=0)
        self.assertEqual((mem_st_ok.get(OUT), mem_st_ok.get(0x40)), (two, one))

        mem_st_partial = run_buf_fminmax("fmin", two, one, base=0x40, stride=6, offset=4,
                                         num_records=2, word3=MODE0_W3, idx=0)
        self.assertEqual((mem_st_partial.get(OUT), mem_st_partial.get(0x44)), (0, two))

        mem_st_small = run_buf_fminmax("fmin", two, one, base=0x40, stride=3, offset=0,
                                       num_records=2, word3=MODE0_W3, idx=0)
        self.assertEqual((mem_st_small.get(OUT), mem_st_small.get(0x40)), (0, two))

    def test_inactive_lane_preserves_vdata(self):
        one, two = 0x3F800000, 0x40000000
        mem = run_buf_fminmax("fmax", one, two, base=0x40, offset=0, num_records=0x100,
                              word3=RAW_W3, nlanes=2, active_mask_lane0_only=True)
        self.assertEqual(mem.get(0x40), two)
        self.assertEqual(mem.get(OUT), one)      # active lane 0 gets old memory word
        self.assertEqual(mem.get(OUT + 4), two)  # inactive lane 1 preserves vdata

    def test_no_float_atomic_capability_and_rejected_forms(self):
        for kind, glsl_op in (("fmin", "NMin"), ("fmax", "NMax")):
            for body in (
                buf_rsrc_lines(0, 0, 0x100, RAW_W3) + [
                    "v_mov_b32_e32 v1, 0",
                    "v_mov_b32_e32 v2, 0x3f800000",
                    f"buffer_atomic_{kind} v2, v1, s[8:11], 0 offen glc",
                ],
                [
                    "v_mov_b32_e32 v1, 0x2000",
                    "v_mov_b32_e32 v2, 0x3f800000",
                    f"global_atomic_{kind} v5, v1, v2, s[0:1] glc",
                ],
            ):
                with self.subTest(insn=body[-1]):
                    spv = translate(body)
                    self.assertIn("OpAtomicLoad %uint", spv)
                    self.assertIn("OpAtomicCompareExchange", spv)
                    self.assertNotIn("Aligned 4", spv)  # no plain non-atomic PSB OpLoad on atomic addr
                    self.assertIn(glsl_op, spv)
                    self.assertNotIn("AtomicFloat", spv)
                    self.assertNotIn("SPV_EXT_shader_atomic_float", spv)

        for bad in ("buffer_atomic_fmin v[2:3], v1, s[8:11], 0 offen glc",
                    "buffer_atomic_fmax v2, v1, s[8:11], 0 offen lds"):
            with self.subTest(bad=bad):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate(buf_rsrc_lines(0, 0, 0x100, RAW_W3) + ["v_mov_b32_e32 v1, 0", bad])


class AtomicNaturalAlignment(unittest.TestCase):
    def test_buffer_atomic_32bit_alignment(self):
        # Measured natively on gfx1036 (wave32 & wave64): (addr & 3) != 0 drops the atomic;
        # active glc lane gets 0, inactive lane keeps vdata; base+2 + offset=2 -> aligned -> succeeds.
        for kind in ("swap", "add", "cmpswap", "fmin", "fmax"):
            for base, offset, should_run in ((0x40, 1, False), (0x40, 2, False),
                                             (0x42, 0, False), (0x42, 2, True)):
                with self.subTest(op=kind, base=hex(base), offset=offset):
                    init_m = 0x3F800000  # 1.0f
                    data_d = 0x40000000  # 2.0f
                    body = buf_rsrc_lines(base, 0, 0x100, RAW_W3) + [
                        f"v_mov_b32_e32 v1, {offset:#x}",
                        f"v_mov_b32_e32 v2, {data_d:#x}",
                        f"v_mov_b32_e32 v3, {init_m:#x}",  # cmp matches init_m for cmpswap
                        "v_cmp_eq_u32_e32 vcc_lo, 0, v0",
                        "s_and_saveexec_b32 s12, vcc_lo",
                        f"buffer_atomic_{kind} {'v[2:3]' if kind == 'cmpswap' else 'v2'}, v1, s[8:11], 0 offen glc",
                        "s_mov_b32 exec_lo, s12",
                        "s_waitcnt vmcnt(0)",
                    ] + out_store_lines("v2", count=1, lane_stride=4)
                    eff = base + offset
                    init_mem = {0x40: init_m, 0x44: init_m}
                    mem = run(body, init_mem, nlanes=2)
                    # Inactive lane 1 always preserves vdata
                    self.assertEqual(mem.get(OUT + 4), data_d)
                    if should_run:
                        self.assertEqual(mem.get(OUT), init_m)
                        expected_m = {
                            "swap": data_d,
                            "add": (init_m + data_d) & M32,
                            "cmpswap": data_d,
                            "fmin": init_m,
                            "fmax": data_d,
                        }[kind]
                        self.assertEqual(mem.get(eff), expected_m)
                    else:
                        self.assertEqual(mem.get(OUT), 0)
                        self.assertEqual(mem.get(0x40), init_m)
                        self.assertEqual(mem.get(0x44), init_m)
                        self.assertNotIn(eff, mem)

    def test_buffer_atomic_x2_alignment_and_partial_bounds(self):
        # Alignment: (addr & 7) != 0 dropped even under OOB_SELECT=2 (DISABLED).
        m_lo, m_hi = 0x11112222, 0x33334444
        d_lo, d_hi = 0x55556666, 0x77778888
        for kind in ("swap", "and", "or", "xor"):
            for base, offset, word3, should_run in (
                (0x40, 2, RAW_W3, False),
                (0x40, 4, RAW_W3, False),
                (0x40, 4, MODE2_W3, False),
                (0x44, 0, MODE2_W3, False),
                (0x44, 4, RAW_W3, True),
            ):
                with self.subTest(op=kind, base=hex(base), offset=offset, word3=hex(word3)):
                    body = buf_rsrc_lines(base, 0, 0x100, word3) + [
                        f"v_mov_b32_e32 v1, {offset:#x}",
                        f"v_mov_b32_e32 v2, {d_lo:#x}",
                        f"v_mov_b32_e32 v3, {d_hi:#x}",
                        "v_cmp_eq_u32_e32 vcc_lo, 0, v0",
                        "s_and_saveexec_b32 s12, vcc_lo",
                        f"buffer_atomic_{kind}_x2 v[2:3], v1, s[8:11], 0 offen glc",
                        "s_mov_b32 exec_lo, s12",
                        "s_waitcnt vmcnt(0)",
                    ] + out_store_lines("v[2:3]", count=2, lane_stride=8)
                    eff = base + offset
                    init_mem = {0x40: m_lo, 0x44: m_hi, 0x48: m_lo, 0x4C: m_hi}
                    mem = run(body, init_mem, nlanes=2)
                    # Inactive lane 1 always preserves vdata[0:1]
                    self.assertEqual((mem.get(OUT + 8), mem.get(OUT + 12)), (d_lo, d_hi))
                    if should_run:
                        self.assertEqual((mem.get(OUT), mem.get(OUT + 4)), (m_lo, m_hi))
                        op_fn = {"swap": lambda a, b: b, "and": lambda a, b: a & b,
                                 "or": lambda a, b: a | b, "xor": lambda a, b: a ^ b}[kind]
                        self.assertEqual((mem.get(eff), mem.get(eff + 4)),
                                         (op_fn(m_lo, d_lo), op_fn(m_hi, d_hi)))
                    else:
                        self.assertEqual((mem.get(OUT), mem.get(OUT + 4)), (0, 0))
                        self.assertEqual((mem.get(0x40), mem.get(0x44)), (m_lo, m_hi))
                        self.assertEqual((mem.get(0x48), mem.get(0x4C)), (m_lo, m_hi))

        # Partial-bounds matrix matching native gfx1036 measurements across OOB_SELECT 0..3.
        bounds_cases = [
            # (word3, stride, num_records, idx, offset, should_run)
            (RAW_W3, 0, 8, None, 0, True),
            (RAW_W3, 0, 16, None, 8, True),
            (RAW_W3, 0, 7, None, 0, False),
            (RAW_W3, 0, 5, None, 0, False),
            (RAW_W3, 0, 4, None, 0, False),
            (RAW_W3, 0, 15, None, 8, False),
            (RAW_W3, 0, 12, None, 8, False),
            (MODE0_W3, 8, 1, 0, 0, True),
            (MODE0_W3, 16, 1, 0, 8, True),
            (MODE0_W3, 12, 1, 0, 0, True),
            (MODE0_W3, 7, 1, 0, 0, False),
            (MODE0_W3, 12, 1, 0, 8, False),
            (MODE0_W3, 8, 1, 1, 0, False),
            (MODE1_W3, 0, 1, 0, 0, True),
            (MODE1_W3, 0, 1, 0, 8, True),
            (MODE1_W3, 0, 1, 1, 0, False),
            (MODE2_W3, 0, 8, None, 8, True),
            (MODE2_W3, 0, 0, None, 0, False),
        ]
        for word3, stride, num_records, idx, offset, should_run in bounds_cases:
            with self.subTest(word3=hex(word3), stride=stride, num_records=num_records, idx=idx, offset=offset):
                base = 0x40
                if idx is not None:
                    addr_setup = [f"v_mov_b32_e32 v1, {idx:#x}", f"v_mov_b32_e32 v2, {offset:#x}"]
                    vaddr, mods = "v[1:2]", "idxen offen glc"
                else:
                    addr_setup = [f"v_mov_b32_e32 v1, {offset:#x}"]
                    vaddr, mods = "v1", "offen glc"
                body = buf_rsrc_lines(base, stride, num_records, word3) + addr_setup + [
                    f"v_mov_b32_e32 v4, {d_lo:#x}",
                    f"v_mov_b32_e32 v5, {d_hi:#x}",
                    f"buffer_atomic_swap_x2 v[4:5], {vaddr}, s[8:11], 0 {mods}",
                    "s_waitcnt vmcnt(0)",
                ] + out_store_lines("v[4:5]", count=2, lane_stride=8)
                eff = base + (idx * stride if idx is not None else 0) + offset
                mem = run(body, {eff: m_lo, eff + 4: m_hi})
                if should_run:
                    self.assertEqual((mem.get(OUT), mem.get(OUT + 4)), (m_lo, m_hi))
                    self.assertEqual((mem.get(eff), mem.get(eff + 4)), (d_lo, d_hi))
                else:
                    self.assertEqual((mem.get(OUT), mem.get(OUT + 4)), (0, 0))
                    self.assertEqual((mem.get(eff), mem.get(eff + 4)), (m_lo, m_hi))

    def test_global_atomic_32bit_alignment(self):
        init_m, data_d = 0x3F800000, 0x40000000
        for kind in ("add", "swap", "cmpswap", "fmin", "fmax"):
            for vaddr, inst_off, should_run in ((TARGET + 1, 0, False),
                                                (TARGET + 2, 0, False),
                                                (TARGET, 2, False),
                                                (TARGET + 2, -2, True)):
                with self.subTest(op=kind, vaddr=hex(vaddr), inst_off=inst_off):
                    off_str = f" offset:{inst_off}" if inst_off else ""
                    vdata = "v[2:3]" if kind == "cmpswap" else "v2"
                    body = [
                        f"v_mov_b32_e32 v1, {vaddr:#x}",
                        f"v_mov_b32_e32 v2, {data_d:#x}",
                        f"v_mov_b32_e32 v3, {init_m:#x}",
                        f"v_mov_b32_e32 v5, {SENTINEL:#x}",
                        "v_cmp_eq_u32_e32 vcc_lo, 0, v0",
                        "s_and_saveexec_b32 s4, vcc_lo",
                        f"global_atomic_{kind} v5, v1, {vdata}, s[0:1]{off_str} glc",
                        "s_mov_b32 exec_lo, s4",
                        "s_waitcnt vmcnt(0)",
                        "v_lshlrev_b32_e32 v6, 2, v0",
                        f"v_add_nc_u32_e32 v6, {OUT:#x}, v6",
                        "global_store_dword v6, v5, s[0:1]",
                    ]
                    eff = vaddr + inst_off
                    mem = run(body, {TARGET: init_m}, nlanes=2)
                    # Inactive lane 1 always preserves vdst sentinel
                    self.assertEqual(mem.get(OUT + 4), SENTINEL)
                    if should_run:
                        self.assertEqual(mem.get(OUT), init_m)
                        expected_m = {
                            "add": (init_m + data_d) & M32,
                            "swap": data_d,
                            "cmpswap": data_d,
                            "fmin": init_m,
                            "fmax": data_d,
                        }[kind]
                        self.assertEqual(mem.get(eff), expected_m)
                    else:
                        # Active unaligned lane receives 0 in vdst (not SENTINEL), memory untouched
                        self.assertEqual(mem.get(OUT), 0)
                        self.assertEqual(mem.get(TARGET), init_m)
                        self.assertNotIn(eff, mem)

    def test_global_atomic_x2_alignment(self):
        m_lo, m_hi = 0x11112222, 0x33334444
        d_lo, d_hi = 0x55556666, 0x77778888
        for kind in ("swap", "and", "or", "xor"):
            for vaddr, inst_off, should_run in ((TARGET + 2, 0, False),
                                                (TARGET + 4, 0, False),
                                                (TARGET, 4, False),
                                                (TARGET + 4, 4, True)):
                with self.subTest(op=kind, vaddr=hex(vaddr), inst_off=inst_off):
                    off_str = f" offset:{inst_off}" if inst_off else ""
                    body = [
                        f"v_mov_b32_e32 v1, {vaddr:#x}",
                        f"v_mov_b32_e32 v2, {d_lo:#x}",
                        f"v_mov_b32_e32 v3, {d_hi:#x}",
                        f"v_mov_b32_e32 v4, {SENTINEL:#x}",
                        f"v_mov_b32_e32 v5, {SENTINEL2:#x}",
                        "v_cmp_eq_u32_e32 vcc_lo, 0, v0",
                        "s_and_saveexec_b32 s4, vcc_lo",
                        f"global_atomic_{kind}_x2 v[4:5], v1, v[2:3], s[0:1]{off_str} glc",
                        "s_mov_b32 exec_lo, s4",
                        "s_waitcnt vmcnt(0)",
                        "v_lshlrev_b32_e32 v6, 3, v0",
                        f"v_add_nc_u32_e32 v6, {OUT:#x}, v6",
                        "global_store_dwordx2 v6, v[4:5], s[0:1]",
                    ]
                    eff = vaddr + inst_off
                    init_mem = {TARGET: m_lo, TARGET + 4: m_hi, TARGET + 8: m_lo, TARGET + 12: m_hi}
                    mem = run(body, init_mem, nlanes=2)
                    # Inactive lane 1 always preserves vdst[0:1]
                    self.assertEqual((mem.get(OUT + 8), mem.get(OUT + 12)), (SENTINEL, SENTINEL2))
                    if should_run:
                        self.assertEqual((mem.get(OUT), mem.get(OUT + 4)), (m_lo, m_hi))
                        op_fn = {"swap": lambda a, b: b, "and": lambda a, b: a & b,
                                 "or": lambda a, b: a | b, "xor": lambda a, b: a ^ b}[kind]
                        self.assertEqual((mem.get(eff), mem.get(eff + 4)),
                                         (op_fn(m_lo, d_lo), op_fn(m_hi, d_hi)))
                    else:
                        self.assertEqual((mem.get(OUT), mem.get(OUT + 4)), (0, 0))
                        self.assertEqual((mem.get(TARGET), mem.get(TARGET + 4)), (m_lo, m_hi))
                        self.assertEqual((mem.get(TARGET + 8), mem.get(TARGET + 12)), (m_lo, m_hi))

    def test_spirv_emits_alignment_guard(self):
        cases = [
            (buf_rsrc_lines(0, 0, 0x100, RAW_W3) + ["v_mov_b32_e32 v1, 0",
             "buffer_atomic_add v2, v1, s[8:11], 0 offen glc"], "%c3"),
            (buf_rsrc_lines(0, 0, 0x100, RAW_W3) + ["v_mov_b32_e32 v1, 0",
             "buffer_atomic_fmin v2, v1, s[8:11], 0 offen glc"], "%c3"),
            (buf_rsrc_lines(0, 0, 0x100, RAW_W3) + ["v_mov_b32_e32 v1, 0",
             "buffer_atomic_xor_x2 v[2:3], v1, s[8:11], 0 offen glc"], "%c7"),
            (["v_mov_b32_e32 v1, 0", "global_atomic_add v5, v1, v2, s[0:1] glc"], "%c3"),
            (["v_mov_b32_e32 v1, 0", "global_atomic_fmax v5, v1, v2, s[0:1] glc"], "%c3"),
            (["v_mov_b32_e32 v1, 0", "global_atomic_swap_x2 v[4:5], v1, v[2:3], s[0:1] glc"], "%c7"),
        ]
        for body, mask_const in cases:
            with self.subTest(insn=body[-1]):
                spv = translate(body)
                self.assertIn("OpUConvert %uint", spv)
                self.assertIn("OpBitwiseAnd %uint", spv)
                self.assertIn(mask_const, spv)


if __name__ == "__main__":
    unittest.main()
