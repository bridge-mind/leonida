"""Semantic interpreter tests for EXEC masking on VALU carry-out / sign-out lane masks.

On RDNA2 hardware, every VALU instruction that writes a scalar lane-mask destination writes 0 for
lanes that are inactive in EXEC (SDST[lane] = EXEC[lane] ? bit : 0), matching v_cmp_*. Covers:
  - v_add_co_u32, v_add_co_ci_u32, v_sub_co_u32, v_sub_co_ci_u32, v_subrev_co_u32, v_subrev_co_ci_u32
    (both e32/VCC and e64/SGPR destinations, plus carry-in-only overflow on inactive lanes)
  - v_mad_u64_u32 (carry-out bit 64) and v_mad_i64_i32 (exact-sum sign bit 64)
  - wave32, wave64 native, and wave64 --wave64-emulate
  - wave-wide consumers: s_bcnt1_i32_b32/b64, s_cbranch_vccz, and v_cndmask_b32 after EXEC restore
  - null scalar destination under partial EXEC (VGPR preserved on inactive lanes, VCC untouched)
"""
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
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]  # OOB_SELECT 3 (raw): stores are in range up to num_records (#1104)
SENTINEL = 0xDEADBEEF
ENC4 = " ; encoding: [0x00,0x00,0x00,0x00]"


def _mask_lines(lanes, dst, tmp, wave_size):
    """Build a wave32/wave64 lane mask in `dst` that is 1 for `lanes` and 0 elsewhere (v0 = lane)."""
    is64 = wave_size == 64
    cmp_op = "v_cmp_eq_u32_e64"
    or_op = "s_or_b64" if is64 else "s_or_b32"
    lines = [f"{cmp_op} {dst}, v0, {lanes[0]}"]
    for lane in lanes[1:]:
        lines.append(f"{cmp_op} {tmp}, v0, {lane}")
        lines.append(f"{or_op} {dst}, {dst}, {tmp}")
    return lines


def _run_cs(name, body_lines, wave_size=64, emulate=False, with_enc=False):
    """Translate a synthetic compute shader and run all lanes in spirv_mini_interp.run_workgroup."""
    prev_emu = rdna2spirv.WAVE64_EMULATE
    try:
        rdna2spirv.WAVE64_EMULATE = emulate
        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(os.path.join(tmp, "disasm"))
            os.mkdir(os.path.join(tmp, "fssm"))
            s_path = os.path.join(tmp, "disasm", name + ".s")
            suf = ENC4 if with_enc else ""
            with open(s_path, "w", encoding="utf-8") as f:
                f.write(f"; {name}\n")
                for ln in body_lines:
                    f.write(f"\t{ln}{suf}\n")
                f.write(f"\ts_endpgm{suf}\n")
            with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
                f.write(STUB_DXBC)
            # COMPUTE_PGM_RSRC2: 0x8088 -> 4 user SGPRs (s[0:3] = output V#), TGID_X.
            with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
                f.write(bytes(0x138) + struct.pack("<4I", 0x8088, wave_size, 1, 1))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spv = rdna2spirv.translate(s_path)
                gc.collect()
            gen_wave64 = rdna2spirv.translate.last_gen.wave64
            assert gen_wave64 == (wave_size == 64), f"{name}: expected wave64={wave_size == 64}, got {gen_wave64}"
    finally:
        rdna2spirv.WAVE64_EMULATE = prev_emu

    sg_size = 32 if emulate else wave_size
    pc = VSHARP + [0] * 28
    return si.run_workgroup(
        spv, wave_size, sg_size,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i % sg_size},
        push_constants=pc, memory={},
    )


# Each entry: (op, with_ci, (hi_x, hi_y, hi_ci), (lo_x, lo_y, lo_ci), want_hi_res, want_lo_res)
# "hi" inputs produce carry/borrow = 1; "lo" inputs produce carry/borrow = 0.
# For *_ci_* ops, "hi" inputs trigger carry/borrow ONLY via carry_in = 1 (x+y or x-y alone does not overflow).
CO_CASES = [
    ("v_add_co_u32", False, (0xFFFFFFFF, 1, 0), (10, 20, 0), 0, 30),
    ("v_add_co_ci_u32", True, (0xFFFFFFFF, 0, 1), (0xFFFFFFFF, 0, 0), 0, 0xFFFFFFFF),
    ("v_sub_co_u32", False, (0, 1, 0), (20, 10, 0), 0xFFFFFFFF, 10),
    ("v_sub_co_ci_u32", True, (0, 0, 1), (0, 0, 0), 0xFFFFFFFF, 0),
    ("v_subrev_co_u32", False, (1, 0, 0), (10, 20, 0), 0xFFFFFFFF, 10),
    ("v_subrev_co_ci_u32", True, (0, 0, 1), (0, 0, 0), 0xFFFFFFFF, 0),
]

# Each entry: (op, (hi_s0, hi_s1, hi_acc_lo, hi_acc_hi), (lo_s0, lo_s1, lo_acc_lo, lo_acc_hi), want_hi_lo, want_hi_hi, want_lo_lo, want_lo_hi)
MAD64_CASES = [
    (
        "v_mad_u64_u32",
        (0xFFFFFFFF, 1, 0xFFFFFFFF, 0xFFFFFFFF),  # 0xffffffff + (2^64 - 1) -> carry = 1, res = 0xfffffffe
        (3, 5, 7, 0),                             # 15 + 7 = 22 -> carry = 0
        0xFFFFFFFE, 0, 22, 0,
    ),
    (
        "v_mad_i64_i32",
        (0xFFFFFFFF, 1, 0, 0),                    # -1 * 1 + 0 = -1 -> sign bit64 = 1
        (3, 5, 7, 0),                             # 3 * 5 + 7 = +22 -> sign bit64 = 0
        0xFFFFFFFF, 0xFFFFFFFF, 22, 0,
    ),
]


class _CarryExecMixin:
    wave_size = 64
    emulate = False

    def _regs(self):
        if self.wave_size == 64:
            return {
                "exec": "exec", "vcc": "vcc", "mov": "s_mov_b64", "bcnt": "s_bcnt1_i32_b64",
                "m_act": "s[4:5]", "m_tmp": "s[6:7]", "m_lo": "s[8:9]", "m_sdst": "s[10:11]",
                "m_cin": "s[12:13]",
                "active_hi": [2, 35], "active_lo": [5, 51],
            }
        return {
            "exec": "exec_lo", "vcc": "vcc_lo", "mov": "s_mov_b32", "bcnt": "s_bcnt1_i32_b32",
            "m_act": "s4", "m_tmp": "s6", "m_lo": "s8", "m_sdst": "s10",
            "m_cin": "s12",
            "active_hi": [2, 19], "active_lo": [5, 27],
        }

    def test_co_ops_inactive_lanes_zero_mask(self):
        r = self._regs()
        active_all = r["active_hi"] + r["active_lo"]
        hi_set = set(r["active_hi"])
        lo_set = set(r["active_lo"])
        for op, with_ci, hi_in, lo_in, want_hi, want_lo in CO_CASES:
            for dst_kind in ("vcc", "sgpr"):
                if dst_kind == "vcc":
                    sdst = r["vcc"]
                    cin_reg = r["vcc"]
                    inst = f"{op}_e32 v6, {sdst}, v2, v3" + (f", {cin_reg}" if with_ci else "")
                else:
                    sdst = r["m_sdst"]
                    cin_reg = r["m_cin"]
                    inst = f"{op}_e64 v6, {sdst}, v2, v3" + (f", {cin_reg}" if with_ci else "")
                with self.subTest(op=op, dst=dst_kind, wave=self.wave_size, emu=self.emulate):
                    lines = [f"{r['mov']} {r['exec']}, -1"]
                    # Active mask = active_hi | active_lo; non-carrying mask = active_lo.
                    lines += _mask_lines(active_all, r["m_act"], r["m_tmp"], self.wave_size)
                    lines += _mask_lines(r["active_lo"], r["m_lo"], r["m_tmp"], self.wave_size)
                    # Give inactive lanes + active_hi the carrying inputs; active_lo gets non-carrying inputs.
                    lines += [
                        f"v_mov_b32 v2, {hi_in[0]:#x}",
                        f"v_mov_b32 v3, {hi_in[1]:#x}",
                        f"v_mov_b32 v4, {lo_in[0]:#x}",
                        f"v_mov_b32 v5, {lo_in[1]:#x}",
                        f"v_cndmask_b32_e64 v2, v2, v4, {r['m_lo']}",
                        f"v_cndmask_b32_e64 v3, v3, v5, {r['m_lo']}",
                        f"v_mov_b32 v6, {SENTINEL:#x}",
                    ]
                    if with_ci:
                        # carry_in = 1 on inactive + active_hi, 0 on active_lo.
                        lines.append(f"v_mov_b32 v7, {hi_in[2]}")
                        lines.append(f"v_mov_b32 v8, {lo_in[2]}")
                        lines.append(f"v_cndmask_b32_e64 v7, v7, v8, {r['m_lo']}")
                        lines.append(f"v_cmp_ne_u32_e64 {cin_reg}, 0, v7")
                    else:
                        # Pre-seed sdst with all 1s so stale bits also fail if not overwritten to 0.
                        lines.append(f"{r['mov']} {sdst}, -1")
                    # Switch EXEC to active_all and run the carry op.
                    lines.append(f"{r['mov']} {r['exec']}, {r['m_act']}")
                    lines.append(inst)
                    # Restore EXEC and inspect sdst via both wave-wide bcnt1 and per-lane v_cndmask_b32.
                    lines += [
                        f"{r['mov']} {r['exec']}, -1",
                        f"{r['bcnt']} s14, {sdst}",
                        f"v_cndmask_b32_e64 v9, 0, 1, {sdst}",
                        "v_mov_b32 v10, s14",
                        "v_mul_lo_u32 v11, v0, 12",
                        "buffer_store_dword v6, v11, s[0:3], 0 offen",
                        "v_add_nc_u32 v12, 4, v11",
                        "buffer_store_dword v9, v12, s[0:3], 0 offen",
                        "v_add_nc_u32 v12, 8, v11",
                        "buffer_store_dword v10, v12, s[0:3], 0 offen",
                    ]
                    mem = _run_cs(f"cs_{op}_{dst_kind}_w{self.wave_size}", lines,
                                  wave_size=self.wave_size, emulate=self.emulate)
                    for lane in range(self.wave_size):
                        base = lane * 12
                        got_vd = mem[base]
                        got_bit = mem[base + 4]
                        got_bcnt = mem[base + 8]
                        self.assertEqual(got_bcnt, len(hi_set), f"lane {lane} bcnt1")
                        if lane in hi_set:
                            self.assertEqual(got_vd, want_hi, f"lane {lane} vd")
                            self.assertEqual(got_bit, 1, f"lane {lane} carry bit")
                        elif lane in lo_set:
                            self.assertEqual(got_vd, want_lo, f"lane {lane} vd")
                            self.assertEqual(got_bit, 0, f"lane {lane} carry bit")
                        else:
                            self.assertEqual(got_vd, SENTINEL, f"inactive lane {lane} vd")
                            self.assertEqual(got_bit, 0, f"inactive lane {lane} carry bit leaked")

    def test_mad64_inactive_lanes_zero_mask(self):
        r = self._regs()
        active_all = r["active_hi"] + r["active_lo"]
        hi_set = set(r["active_hi"])
        lo_set = set(r["active_lo"])
        for op, hi_in, lo_in, wh_lo, wh_hi, wl_lo, wl_hi in MAD64_CASES:
            for dst_kind in ("vcc", "sgpr"):
                sdst = r["vcc"] if dst_kind == "vcc" else r["m_sdst"]
                with self.subTest(op=op, dst=dst_kind, wave=self.wave_size, emu=self.emulate):
                    lines = [f"{r['mov']} {r['exec']}, -1"]
                    lines += _mask_lines(active_all, r["m_act"], r["m_tmp"], self.wave_size)
                    lines += _mask_lines(r["active_lo"], r["m_lo"], r["m_tmp"], self.wave_size)
                    lines += [
                        f"v_mov_b32 v2, {hi_in[0]:#x}",
                        f"v_mov_b32 v3, {hi_in[1]:#x}",
                        f"v_mov_b32 v12, {hi_in[2]:#x}",
                        f"v_mov_b32 v13, {hi_in[3]:#x}",
                        f"v_mov_b32 v14, {lo_in[0]:#x}",
                        f"v_mov_b32 v15, {lo_in[1]:#x}",
                        f"v_mov_b32 v16, {lo_in[2]:#x}",
                        f"v_mov_b32 v17, {lo_in[3]:#x}",
                        f"v_cndmask_b32_e64 v2, v2, v14, {r['m_lo']}",
                        f"v_cndmask_b32_e64 v3, v3, v15, {r['m_lo']}",
                        f"v_cndmask_b32_e64 v12, v12, v16, {r['m_lo']}",
                        f"v_cndmask_b32_e64 v13, v13, v17, {r['m_lo']}",
                        f"v_mov_b32 v6, {SENTINEL:#x}",
                        f"v_mov_b32 v7, {SENTINEL:#x}",
                        f"{r['mov']} {sdst}, -1",
                        f"{r['mov']} {r['exec']}, {r['m_act']}",
                        f"{op} v[6:7], {sdst}, v2, v3, v[12:13]",
                        f"{r['mov']} {r['exec']}, -1",
                        f"{r['bcnt']} s14, {sdst}",
                        f"v_cndmask_b32_e64 v8, 0, 1, {sdst}",
                        "v_mov_b32 v9, s14",
                        "v_lshlrev_b32 v10, 4, v0",
                        "buffer_store_dwordx4 v[6:9], v10, s[0:3], 0 offen",
                    ]
                    mem = _run_cs(f"cs_{op}_{dst_kind}_w{self.wave_size}", lines,
                                  wave_size=self.wave_size, emulate=self.emulate)
                    for lane in range(self.wave_size):
                        base = lane * 16
                        got_lo, got_hi, got_bit, got_bcnt = (mem[base + 4 * k] for k in range(4))
                        self.assertEqual(got_bcnt, len(hi_set), f"lane {lane} bcnt1")
                        if lane in hi_set:
                            self.assertEqual((got_lo, got_hi), (wh_lo, wh_hi), f"lane {lane} v[6:7]")
                            self.assertEqual(got_bit, 1, f"lane {lane} bit64")
                        elif lane in lo_set:
                            self.assertEqual((got_lo, got_hi), (wl_lo, wl_hi), f"lane {lane} v[6:7]")
                            self.assertEqual(got_bit, 0, f"lane {lane} bit64")
                        else:
                            self.assertEqual((got_lo, got_hi), (SENTINEL, SENTINEL), f"inactive lane {lane} v[6:7]")
                            self.assertEqual(got_bit, 0, f"inactive lane {lane} bit64 leaked")

    def test_cbranch_vccz_ignores_inactive_carry_and_sign(self):
        """When only inactive lanes carry/borrow/sign-out, VCC is 0 and s_cbranch_vccz must be taken."""
        r = self._regs()
        vcc = r["vcc"]
        cases = [
            ("v_add_co_u32", [
                "v_mov_b32 v2, 0xffffffff", "v_mov_b32 v3, 1",
                "v_mov_b32 v4, 1",
                f"v_cndmask_b32_e64 v2, v2, v4, {r['m_act']}",
                f"{r['mov']} {r['exec']}, {r['m_act']}",
                f"v_add_co_u32_e32 v6, {vcc}, v2, v3",
            ]),
            ("v_sub_co_u32", [
                "v_mov_b32 v2, 0", "v_mov_b32 v3, 1",
                "v_mov_b32 v4, 5",
                f"v_cndmask_b32_e64 v2, v2, v4, {r['m_act']}",
                f"{r['mov']} {r['exec']}, {r['m_act']}",
                f"v_sub_co_u32_e32 v6, {vcc}, v2, v3",
            ]),
            ("v_mad_u64_u32", [
                "v_mov_b32 v2, 0xffffffff", "v_mov_b32 v3, 1",
                "v_mov_b32 v12, 0xffffffff", "v_mov_b32 v13, 0xffffffff",
                "v_mov_b32 v14, 0",
                f"v_cndmask_b32_e64 v3, v3, v14, {r['m_act']}",
                f"{r['mov']} {r['exec']}, {r['m_act']}",
                f"v_mad_u64_u32 v[6:7], {vcc}, v2, v3, v[12:13]",
            ]),
            ("v_mad_i64_i32", [
                "v_mov_b32 v2, -1", "v_mov_b32 v3, 1",
                "v_mov_b32 v12, 0", "v_mov_b32 v13, 0",
                "v_mov_b32 v14, 1",
                f"v_cndmask_b32_e64 v2, v2, v14, {r['m_act']}",
                f"{r['mov']} {r['exec']}, {r['m_act']}",
                f"v_mad_i64_i32 v[6:7], {vcc}, v2, v3, v[12:13]",
            ]),
        ]
        for op, setup in cases:
            with self.subTest(op=op, wave=self.wave_size, emu=self.emulate):
                lines = [f"{r['mov']} {r['exec']}, -1"]
                lines += _mask_lines(r["active_lo"], r["m_act"], r["m_tmp"], self.wave_size)
                lines += [f"{r['mov']} {vcc}, -1", "s_mov_b32 s14, 0"]
                lines += setup
                lines += [
                    "s_cbranch_vccz 1",
                    "s_mov_b32 s14, 1",
                    f"{r['mov']} {r['exec']}, -1",
                    "v_mov_b32 v8, s14",
                    "v_lshlrev_b32 v9, 2, v0",
                    "buffer_store_dword v8, v9, s[0:3], 0 offen",
                ]
                mem = _run_cs(f"cs_vccz_{op}_w{self.wave_size}", lines,
                              wave_size=self.wave_size, emulate=self.emulate, with_enc=True)
                for lane in range(self.wave_size):
                    self.assertEqual(mem[lane * 4], 0, f"lane {lane}: s_cbranch_vccz not taken")

    def test_null_sdst_under_partial_exec(self):
        """null scalar destination writes no mask (leaves VCC untouched) and preserves inactive VGPRs."""
        r = self._regs()
        vcc = r["vcc"]
        active_set = set(r["active_hi"])
        lines = [f"{r['mov']} {r['exec']}, -1"]
        lines += _mask_lines(r["active_hi"], r["m_act"], r["m_tmp"], self.wave_size)
        lines += _mask_lines([0], vcc, r["m_tmp"], self.wave_size)  # VCC = lane 0 only
        lines += [
            "v_mov_b32 v2, 0xffffffff",
            "v_mov_b32 v3, 1",
            "v_mov_b32 v12, 0xffffffff",
            "v_mov_b32 v13, 0xffffffff",
            f"v_mov_b32 v6, {SENTINEL:#x}",
            f"v_mov_b32 v7, {SENTINEL:#x}",
            f"v_mov_b32 v8, {SENTINEL:#x}",
            f"{r['mov']} {r['exec']}, {r['m_act']}",
            "v_add_co_u32_e64 v6, null, v2, v3",
            "v_mad_u64_u32 v[7:8], null, v2, v3, v[12:13]",
            f"{r['mov']} {r['exec']}, -1",
            f"{r['bcnt']} s14, {vcc}",
            "v_mov_b32 v9, s14",
            "v_lshlrev_b32 v10, 4, v0",
            "buffer_store_dwordx4 v[6:9], v10, s[0:3], 0 offen",
        ]
        mem = _run_cs(f"cs_null_sdst_w{self.wave_size}", lines,
                      wave_size=self.wave_size, emulate=self.emulate)
        for lane in range(self.wave_size):
            base = lane * 16
            v6, v7, v8, bcnt = (mem[base + 4 * k] for k in range(4))
            self.assertEqual(bcnt, 1, f"lane {lane}: VCC modified despite null sdst")
            if lane in active_set:
                self.assertEqual(v6, 0, f"active lane {lane} v6")
                self.assertEqual((v7, v8), (0xFFFFFFFE, 0), f"active lane {lane} v[7:8]")
            else:
                self.assertEqual(v6, SENTINEL, f"inactive lane {lane} v6")
                self.assertEqual((v7, v8), (SENTINEL, SENTINEL), f"inactive lane {lane} v[7:8]")


class CarryExecWave32Tests(_CarryExecMixin, unittest.TestCase):
    wave_size = 32
    emulate = False


class CarryExecWave64Tests(_CarryExecMixin, unittest.TestCase):
    wave_size = 64
    emulate = False


class CarryExecWave64EmulateTests(unittest.TestCase):
    """Verify --wave64-emulate translates carry/mad64 ops and masks their lane-mask outputs by %active."""

    def _translate_emu(self, name, body_lines):
        prev_emu = rdna2spirv.WAVE64_EMULATE
        try:
            rdna2spirv.WAVE64_EMULATE = True
            with tempfile.TemporaryDirectory() as tmp:
                os.mkdir(os.path.join(tmp, "disasm"))
                os.mkdir(os.path.join(tmp, "fssm"))
                s_path = os.path.join(tmp, "disasm", name + ".s")
                with open(s_path, "w", encoding="utf-8") as f:
                    f.write(f"; {name}\n")
                    for ln in body_lines:
                        f.write(f"\t{ln}\n")
                    f.write("\ts_endpgm\n")
                with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
                    f.write(STUB_DXBC)
                with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
                    f.write(bytes(0x138) + struct.pack("<4I", 0x8088, 64, 1, 1))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ResourceWarning)
                    spv = rdna2spirv.translate(s_path)
                    gc.collect()
                return spv
        finally:
            rdna2spirv.WAVE64_EMULATE = prev_emu

    def _assert_mask_anded_with_active(self, spv, dst_var):
        """Confirm `OpStore <dst_var> <m>` selects on `OpLogicalAnd %bool <cond> <act>` where <act> is %active."""
        defs = {}
        active_vals = set()
        stored = None
        for ln in spv.splitlines():
            s = ln.strip()
            if " = " in s:
                lhs, rhs = s.split(" = ", 1)
                defs[lhs] = rhs
                if rhs == "OpLoad %bool %active":
                    active_vals.add(lhs)
            elif s.startswith("OpStore %active "):
                active_vals.add(s.split()[2])
            elif s.startswith(f"OpStore {dst_var} "):
                stored = s.split()[2]
        self.assertIsNotNone(stored, f"missing OpStore {dst_var}")
        sel = defs.get(stored, "")
        self.assertTrue(sel.startswith("OpSelect %uint "), f"unexpected store rhs: {sel}")
        cond = sel.split()[2]
        and_rhs = defs.get(cond, "")
        self.assertTrue(and_rhs.startswith("OpLogicalAnd %bool "),
                        f"mask select condition {cond} not ANDed with %active: {and_rhs}")
        operands = and_rhs.split()[2:4]
        self.assertTrue(any(op in active_vals for op in operands),
                        f"OpLogicalAnd operands {operands} do not read %active")

    def test_emulate_co_and_mad64_mask_by_active(self):
        cases = [
            ("v_add_co_u32", "v_add_co_u32_e32 v6, vcc, v2, v3", "%vcc"),
            ("v_add_co_ci_u32", "v_add_co_ci_u32_e32 v6, vcc, v2, v3, vcc", "%vcc"),
            ("v_sub_co_u32", "v_sub_co_u32_e32 v6, vcc, v2, v3", "%vcc"),
            ("v_sub_co_ci_u32", "v_sub_co_ci_u32_e32 v6, vcc, v2, v3, vcc", "%vcc"),
            ("v_subrev_co_u32", "v_subrev_co_u32_e32 v6, vcc, v2, v3", "%vcc"),
            ("v_subrev_co_ci_u32", "v_subrev_co_ci_u32_e32 v6, vcc, v2, v3, vcc", "%vcc"),
            ("v_mad_u64_u32", "v_mad_u64_u32 v[6:7], vcc, v2, v3, v[12:13]", "%vcc"),
            ("v_mad_i64_i32", "v_mad_i64_i32 v[6:7], vcc, v2, v3, v[12:13]", "%vcc"),
        ]
        for op, inst, dst_var in cases:
            with self.subTest(op=op):
                spv = self._translate_emu(f"cs_emu_{op}", [
                    "s_mov_b64 exec, -1",
                    "v_mov_b32 v2, 0xffffffff",
                    "v_mov_b32 v3, 1",
                    "v_mov_b32 v12, 0xffffffff",
                    "v_mov_b32 v13, 0xffffffff",
                    inst,
                    "s_bcnt1_i32_b64 s14, vcc",
                    "v_mov_b32 v8, s14",
                    "v_lshlrev_b32 v9, 2, v0",
                    "buffer_store_dword v8, v9, s[0:3], 0 offen",
                ])
                self.assertIn("OpStore %x_it", spv)
                self._assert_mask_anded_with_active(spv, dst_var)


if __name__ == "__main__":
    unittest.main()
