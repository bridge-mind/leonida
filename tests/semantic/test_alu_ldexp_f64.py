"""Semantic interpreter, capability, and spirv-val tests for RDNA2 v_ldexp_f64.

v_ldexp_f64 computes D.f64 = S0.f64 * 2^(S1.i32) on the (lo, hi) dword pair using pure 32-bit
integer operations (no OpCapability Float64 or OpCapability Int64, even with clamp).

Hardware-measured on AMD Radeon RDNA2 iGPU (gfx1036, wave32 & wave64, game MODE: ieee=0,
dx10_clamp=1, denorm32=0, denorm16_64=3):
- +-0, +-inf, and both quiet and signalling NaNs pass through bit-for-bit unchanged (after S0 abs/neg).
- Subnormal inputs are normalized before scaling; subnormal outputs shift right by 1 - e2 (1..53)
  and round to nearest even (carry into bit 52 yields +-min_norm; sh >= 54 flushes to +-0).
- VOP3 clamp applies in both denorm16_64=3 and 0 (dx10_clamp=1: NaN, -0.0, <0 -> +0.0; >1.0 -> 1.0).
- VOP3 omod (mul:2, mul:4, div:2) is ignored on hardware when denorm16_64=3 and active when 0,
  so the translator rejects it as Unsupported.
- RDNA2 places 32-bit literal S0 constants (outside inline -16..64) in bits [63:32] with [31:0] = 0.
"""
import os
import random
import shutil
import struct
import subprocess
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
import spirv_asm  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
POS_ZERO = 0x0000000000000000
NEG_ZERO = 0x8000000000000000
POS_INF = 0x7FF0000000000000
NEG_INF = 0xFFF0000000000000
QNAN = 0x7FF8000000000123
NQNAN = 0xFFF8000000000456
SNAN = 0x7FF0000000000001
NSNAN = 0xFFF0000000000789
MIN_SUB = 0x0000000000000001
MAX_SUB = 0x000FFFFFFFFFFFFF
MIN_NORM = 0x0010000000000000
MAX_FIN = 0x7FEFFFFFFFFFFFFF
ONE = 0x3FF0000000000000

SENT_LO = 0x13572468
SENT_HI = 0x24681357
SENT_E = 0x35792468


def d2u(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def ref_ldexp_f64(u: int, s1: int, mod: str = "", clamp: bool = False) -> int:
    """Exact bit-level reference matching gfx1036 game MODE (ieee=0, dx10_clamp=1, denorm16_64=3)."""
    u &= 0xFFFFFFFFFFFFFFFF
    if mod in ("abs", "negabs"):
        u &= 0x7FFFFFFFFFFFFFFF
    if mod in ("neg", "negabs"):
        u ^= 0x8000000000000000
    s1 &= 0xFFFFFFFF
    if s1 >= 0x80000000:
        s1 -= 0x100000000
    sgn = u & (1 << 63)
    e = (u >> 52) & 0x7FF
    mant = u & ((1 << 52) - 1)
    if e == 0x7FF or (e == 0 and mant == 0):
        res = u
    else:
        if e == 0:
            msb = mant.bit_length() - 1
            lsh = 52 - msb
            sig = mant << lsh
            ef = 1 - lsh
        else:
            sig = (1 << 52) | mant
            ef = e
        nc = max(-4096, min(4096, s1))
        e2 = ef + nc
        if e2 >= 2047:
            res = sgn | POS_INF
        elif e2 >= 1:
            res = sgn | (e2 << 52) | (sig & ((1 << 52) - 1))
        else:
            sh = 1 - e2
            if sh >= 54:
                res = sgn
            else:
                q = sig >> sh
                rem = sig & ((1 << sh) - 1)
                half = 1 << (sh - 1)
                if rem > half or (rem == half and (q & 1)):
                    q += 1
                res = sgn | q
    if clamp:
        re = (res >> 52) & 0x7FF
        rm = res & ((1 << 52) - 1)
        is_nan = (re == 0x7FF) and (rm != 0)
        if is_nan or (res >> 63):
            res = 0
        elif res > ONE:
            res = ONE
    return res


_TMP_DIR = None
_SRC_PATH = None
_TEMPLATES = {}


def setUpModule():
    global _TMP_DIR, _SRC_PATH, _TEMPLATES
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_ldexp_f64.s")
    with open(os.path.join(_TMP_DIR.name, "ps_ldexp_f64.dxbc"), "wb") as f:
        f.write(STUB_DXBC)
    _TEMPLATES = {}


def tearDownModule():
    global _TMP_DIR, _TEMPLATES
    _TEMPLATES = {}
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def translate_lines(lines):
    if _TMP_DIR is None:
        setUpModule()
    asm = (
        "; test_ldexp_f64\n"
        + "".join(f"\t{ln}\n" for ln in lines)
        + "\texp mrt0 v0, v1, v2, v3 done vm\n\ts_endpgm\n"
    )
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_SRC_PATH)


def validate_spirv(spv_text: str):
    spirv_asm.assemble(spv_text)
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if not spirv_as or not spirv_val:
        return
    with tempfile.TemporaryDirectory() as d:
        asm_p = os.path.join(d, "s.spvasm")
        bin_p = os.path.join(d, "s.spv")
        with open(asm_p, "w", encoding="utf-8") as f:
            f.write(spv_text)
        r1 = subprocess.run([spirv_as, "--target-env", "vulkan1.2", asm_p, "-o", bin_p], capture_output=True, text=True)
        if r1.returncode != 0:
            raise AssertionError(f"spirv-as failed: {r1.stderr}")
        r2 = subprocess.run([spirv_val, "--target-env", "vulkan1.2", bin_p], capture_output=True, text=True)
        if r2.returncode != 0:
            raise AssertionError(f"spirv-val failed: {r2.stderr}")


def fmt_src0(reg: str, mod: str) -> str:
    if mod == "neg":
        return f"-{reg}"
    if mod == "abs":
        return f"|{reg}|"
    if mod == "negabs":
        return f"-|{reg}|"
    return reg


def get_template(mod: str, clamp: bool) -> str:
    key = (mod, clamp)
    if key not in _TEMPLATES:
        s0 = fmt_src0("v[2:3]", mod)
        cl = " clamp" if clamp else ""
        spv = translate_lines([
            f"v_mov_b32 v2, {SENT_LO:#x}",
            f"v_mov_b32 v3, {SENT_HI:#x}",
            f"v_mov_b32 v4, {SENT_E:#x}",
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 0",
            f"v_ldexp_f64 v[0:1], {s0}, v4{cl}",
        ])
        _TEMPLATES[key] = spv
    return _TEMPLATES[key]


def eval_ldexp(u: int, s1: int, mod: str = "", clamp: bool = False) -> int:
    lo = u & 0xFFFFFFFF
    hi = (u >> 32) & 0xFFFFFFFF
    eu = s1 & 0xFFFFFFFF
    spv = (
        get_template(mod, clamp)
        .replace(f"OpConstant %uint {SENT_LO}\n", f"OpConstant %uint {lo}\n")
        .replace(f"OpConstant %uint {SENT_HI}\n", f"OpConstant %uint {hi}\n")
        .replace(f"OpConstant %uint {SENT_E}\n", f"OpConstant %uint {eu}\n")
    )
    out = si.run_fragment(spv)["%out_mrt0"]
    return (out[1] << 32) | out[0]


def run_lines(lines) -> int:
    spv = translate_lines(lines)
    validate_spirv(spv)
    out = si.run_fragment(spv)["%out_mrt0"]
    return (out[1] << 32) | out[0]


class LdexpF64Tests(unittest.TestCase):
    def test_capabilities_and_spirv_val(self):
        """Pure 32-bit integer lowering must emit neither Float64 nor Int64, even with clamp."""
        for mod in ("", "neg", "abs", "negabs"):
            for cl in (False, True):
                spv = get_template(mod, cl)
                self.assertNotIn("OpCapability Float64", spv)
                self.assertNotIn("OpCapability Int64", spv)
                validate_spirv(spv)

    def test_hw_measured_specials(self):
        """+-0, +-inf, and quiet/signalling NaNs pass through bit-for-bit unchanged (gfx1036, ieee=0)."""
        for u in (POS_ZERO, NEG_ZERO, POS_INF, NEG_INF, QNAN, NQNAN, SNAN, NSNAN):
            for e in (0, 1, -1, 1024, -1024, 0x7FFFFFFF, -0x80000000):
                with self.subTest(u=hex(u), e=e):
                    self.assertEqual(eval_ldexp(u, e), u)

    def test_hw_measured_subnormal_inputs(self):
        """Subnormal inputs at every bit 0..51 normalize before scaling (gfx1036, denorm16_64=3)."""
        for bit in range(52):
            u = 1 << bit
            self.assertEqual(eval_ldexp(u, 52 - bit), MIN_NORM)
            self.assertEqual(eval_ldexp(u, 1074 - bit), ONE)
            self.assertEqual(eval_ldexp(u | (1 << 63), 52 - bit), MIN_NORM | (1 << 63))
            self.assertEqual(eval_ldexp(u, 2097 - bit), 0x7FE0000000000000)
            self.assertEqual(eval_ldexp(u, 2098 - bit), POS_INF)

    def test_hw_measured_subnormal_outputs_and_ties(self):
        """Subnormal outputs round to nearest even across shifts 1..54, including carry into min_norm."""
        # Pinned hardware-measured cases from gfx1036:
        pinned = [
            (MIN_SUB, -1, POS_ZERO),
            (3 * MIN_SUB, -1, 2 * MIN_SUB),       # exact tie -> even
            (5 * MIN_SUB, -2, 1 * MIN_SUB),
            (6 * MIN_SUB, -2, 2 * MIN_SUB),       # exact tie -> even
            (7 * MIN_SUB, -2, 2 * MIN_SUB),
            (ONE, -1075, POS_ZERO),               # sh=53 exact tie -> 0 (even)
            (ONE | 1, -1075, MIN_SUB),            # sh=53 > half -> min_sub
            (ONE | (1 << 63) | 1, -1075, MIN_SUB | (1 << 63)),
            (MAX_FIN, -2098, MIN_SUB),            # sh=53 rounds up to min_sub
            (MAX_FIN, -2099, POS_ZERO),           # sh=54 flushes to +0.0
            (MAX_FIN | (1 << 63), -2099, NEG_ZERO),
            (0x3FFFFFFFFFFFFFFF, -1023, MIN_NORM),  # sh=1 carry into min_norm
            (0xBFFFFFFFFFFFFFFF, -1023, MIN_NORM | (1 << 63)),
        ]
        for u, e, want in pinned:
            with self.subTest(u=hex(u), e=e):
                self.assertEqual(eval_ldexp(u, e), want)

        # Sweep every shift boundary sh = 1..54 (ties to even, ties to odd, above half).
        for sh in range(1, 55):
            e = -1022 - sh
            test_vals = [ONE, 0x3FFFFFFFFFFFFFFF]
            if sh <= 52:
                half_bit = 1 << (sh - 1)
                lsb_bit = 1 << sh
                base_even = half_bit
                base_odd = lsb_bit | half_bit
                test_vals.extend([
                    ONE | base_even,
                    ONE | base_odd,
                    ONE | base_even | 1,
                    ONE | (1 << 63) | base_odd,
                ])
            for u in test_vals:
                self.assertEqual(eval_ldexp(u, e), ref_ldexp_f64(u, e))

    def test_hw_measured_abs_neg_clamp(self):
        """VOP3 abs, neg, and DX10 clamp match gfx1036 bit-for-bit."""
        inputs = [
            (d2u(0.5), 0), (d2u(-0.5), 0), (ONE, 0), (ONE, 1),
            (POS_ZERO, 0), (NEG_ZERO, 0), (POS_INF, 0), (NEG_INF, 0),
            (QNAN, 0), (NQNAN, 0), (SNAN, 0), (NSNAN, 0),
            (MIN_SUB, 0), (MIN_SUB | (1 << 63), 0),
            (0x3FEFFFFFFFFFFFFF, 0), (0x3FF0000000000001, 0),
        ]
        for u, e in inputs:
            for mod in ("", "neg", "abs", "negabs"):
                for cl in (False, True):
                    with self.subTest(u=hex(u), e=e, mod=mod, clamp=cl):
                        self.assertEqual(eval_ldexp(u, e, mod, cl), ref_ldexp_f64(u, e, mod, cl))

    def test_operand_forms_and_aliasing(self):
        """SGPR pairs, inline floats/ints, 32-bit RDNA2 high-dword literals, and dst aliasing."""
        # Inline float constants
        self.assertEqual(run_lines(["v_ldexp_f64 v[0:1], 1.0, 3"]), d2u(8.0))
        self.assertEqual(run_lines(["v_ldexp_f64 v[0:1], -0.5, 1"]), d2u(-1.0))
        # Inline integer constants (-16..64 are sign-extended 64-bit on gfx1036)
        self.assertEqual(run_lines(["v_ldexp_f64 v[0:1], 0, 5"]), POS_ZERO)
        self.assertEqual(run_lines(["v_ldexp_f64 v[0:1], 1, 1"]), 0x0000000000000002)
        self.assertEqual(run_lines(["v_ldexp_f64 v[0:1], 64, 1"]), 0x0000000000000080)
        self.assertEqual(run_lines(["v_ldexp_f64 v[0:1], -1, 1"]), 0xFFFFFFFFFFFFFFFF)
        self.assertEqual(run_lines(["v_ldexp_f64 v[0:1], -16, 1"]), 0xFFFFFFFFFFFFFFF0)
        # 32-bit RDNA2 literals (outside -16..64) occupy bits [63:32] with [31:0] = 0 (measured on gfx1036)
        self.assertEqual(run_lines(["v_ldexp_f64 v[0:1], 0x3ff00000, 1"]), d2u(2.0))
        self.assertEqual(run_lines(["v_ldexp_f64 v[0:1], 65, 1"]), 0x0000008200000000)
        self.assertEqual(run_lines(["v_ldexp_f64 v[0:1], -17, 1"]), 0xFFFFFFEF00000000)
        # 64-bit hex constant
        self.assertEqual(run_lines(["v_ldexp_f64 v[0:1], 0x3ff0000000000000, -2"]), d2u(0.25))
        # SGPR pair S0 and SGPR S1
        self.assertEqual(run_lines([
            "s_mov_b32 s4, 0",
            "s_mov_b32 s5, 0x3ff00000",
            "s_mov_b32 s6, -3",
            "v_ldexp_f64 v[0:1], s[4:5], s6",
        ]), d2u(0.125))
        # Destination aliases S0 and S1
        self.assertEqual(run_lines([
            "v_mov_b32 v0, 3",
            "v_mov_b32 v1, 0x3ff00000",
            "v_ldexp_f64 v[0:1], v[0:1], v0",
        ]), 0x4020000000000003)
        self.assertEqual(run_lines([
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 0x3ff00000",
            "v_ldexp_f64 v[0:1], v[0:1], v1",
        ]), POS_INF)

    def test_seeded_random_sweep(self):
        rng = random.Random(20261122)
        for _ in range(120):
            u = rng.getrandbits(64)
            e = rng.choice([rng.randint(-2150, 2150), rng.randint(-60, 60), rng.randint(-0x80000000, 0x7FFFFFFF)])
            mod = rng.choice(["", "neg", "abs", "negabs"])
            cl = rng.choice([False, False, True])
            self.assertEqual(eval_ldexp(u, e, mod, cl), ref_ldexp_f64(u, e, mod, cl))

    def test_rejected_forms(self):
        for omod in ("mul:2", "mul:4", "div:2"):
            with self.assertRaises(rdna2spirv.Unsupported):
                translate_lines([f"v_ldexp_f64 v[0:1], v[2:3], v4 {omod}"])
        with self.assertRaises(rdna2spirv.Unsupported):
            translate_lines(["v_ldexp_f64 v0, v[2:3], v4"])
        with self.assertRaises(rdna2spirv.Unsupported):
            translate_lines(["v_ldexp_f64 s[0:1], v[2:3], v4"])
        with self.assertRaises(rdna2spirv.Unsupported):
            translate_lines(["v_ldexp_f64 v[0:1], v2, v4"])


# --- wave64 coverage (SHADER-1039 follow-up) ------------------------------------------------------
# run_fragment above always runs one invocation, so nothing here could see the upper 32 bits of EXEC
# or VCC. The fixture declares its wave size in the FSSM header (0xD4 bit 5) and runs one full
# subgroup, so a regression that drops the high half of either mask shows up.

def run_cs_wave(lines, wave_size):
    fssm = bytearray(0x148)
    fssm[:4] = b"FSSM"
    struct.pack_into("<I", fssm, 0xD4, 0x20 if wave_size == 32 else 0)
    struct.pack_into("<4I", fssm, 0x138, 0x8088, wave_size, 1, 1)
    from support import translate_lines as translate_with_fssm
    spv = translate_with_fssm("cs_ldexp_f64_wave", lines + ["s_endpgm"], fssm=bytes(fssm))
    return si.run_workgroup(spv, wave_size, wave_size,
                            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                            push_constants=[0, 0, 0x1000, 0x31016FAC] + [0] * 28, memory={})


class LdexpF64WaveTests(unittest.TestCase):
    def check_vcc_and_inactive_lanes(self, wave_size, active_lanes, vcc_lanes):
        vcc = "vcc_lo" if wave_size == 32 else "vcc"
        restore_exec = "s_mov_b32 exec_lo, -1" if wave_size == 32 else "s_mov_b64 exec, -1"
        lines = ["v_mov_b32_e32 v4, 0", "v_mov_b32_e32 v5, 0x3ff00000",
                 f"v_mov_b32_e32 v6, {SENT_LO:#x}", f"v_mov_b32_e32 v7, {SENT_HI:#x}",
                 f"v_cmp_gt_u32_e32 {vcc}, {vcc_lanes}, v0",
                 f"v_cmpx_gt_u32_e32 {active_lanes}, v0",
                 "v_ldexp_f64 v[6:7], v[4:5], v0",
                 restore_exec,
                 f"v_cndmask_b32_e64 v8, 0, 1, {vcc}",
                 "v_lshlrev_b32_e32 v9, 4, v0",
                 "buffer_store_dword v6, v9, s[0:3], 0 offen",
                 "buffer_store_dword v7, v9, s[0:3], 0 offen offset:4",
                 "buffer_store_dword v8, v9, s[0:3], 0 offen offset:8"]
        mem = run_cs_wave(lines, wave_size)
        for lane in range(wave_size):
            lo, hi, vcc_bit = mem[16 * lane], mem[16 * lane + 4], mem[16 * lane + 8]
            if lane < active_lanes:
                want = d2u(2.0 ** lane)
                self.assertEqual((lo, hi), (want & 0xFFFFFFFF, want >> 32), f"lane {lane}")
            else:
                self.assertEqual((lo, hi), (SENT_LO, SENT_HI), f"inactive lane {lane}")
            self.assertEqual(vcc_bit, int(lane < vcc_lanes), f"vcc lane {lane}")

    def test_vcc_and_inactive_lanes_wave32(self):
        self.check_vcc_and_inactive_lanes(wave_size=32, active_lanes=16, vcc_lanes=3)

    def test_vcc_and_inactive_lanes_wave64(self):
        # Both masks change inside the high half, so dropping either high word cannot pass:
        # without the EXEC mask lanes 43-63 change, and clearing vcc_hi changes lanes 32-34.
        self.check_vcc_and_inactive_lanes(wave_size=64, active_lanes=43, vcc_lanes=35)


if __name__ == "__main__":
    unittest.main()
