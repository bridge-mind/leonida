"""Value tests for RDNA2 global_atomic_* (32-bit), run through spirv_mini_interp on the CPU.

RDNA2 ISA, FLAT/GLOBAL atomics: the address is the global_* one (saddr pair + zero-extended vaddr, or a 64-bit vaddr
with `off`, plus the signed 12-bit offset). M = op(M, DATA); with glc the pre-op value lands in vdst, the first
operand, and without glc there is no vdst. cmpswap takes vdata as {DATA, CMP}: M = (M == CMP) ? DATA : M.
Each case is a compute shader run as one invocation: memory holds a seeded word, the atomic runs, and the returned
value is stored to a second address. Standard library only; no GPU, no LLVM, no game files.
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
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

M32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x88, 32, 1, 1)     # 4 user SGPRs, workgroup id in s4, 32x1x1
FSSM = bytes(FSSM)
TARGET, OUT = 0x2000, 0x3000                           # Xbox VAs (heap base 0)


def s32(x):
    x &= M32
    return x - (1 << 32) if x >> 31 else x


MODEL = {
    "add": lambda m, d: (m + d) & M32, "sub": lambda m, d: (m - d) & M32,
    "umin": min, "umax": max,
    "smin": lambda m, d: m if s32(m) <= s32(d) else d, "smax": lambda m, d: m if s32(m) >= s32(d) else d,
    "and": lambda m, d: m & d, "or": lambda m, d: m | d, "xor": lambda m, d: m ^ d, "swap": lambda m, d: d,
}
VALUES = [0, 1, 0x7FFFFFFF, 0x80000000, M32, 0x12345678]


def translate(body):
    with tempfile.TemporaryDirectory() as tmp:
        for d in ("disasm", "fssm"):
            os.mkdir(os.path.join(tmp, d))
        with open(os.path.join(tmp, "cs_gatomic.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", "cs_gatomic.fssm"), "wb") as f:
            f.write(FSSM)
        src = os.path.join(tmp, "disasm", "cs_gatomic.s")
        with open(src, "w", encoding="utf-8") as f:
            f.write("; cs_gatomic\n" + "".join(f"\t{ln}\n" for ln in body) + "\ts_endpgm\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            return rdna2spirv.translate(src)


def run(body, memory, nlanes=1):
    """s[0:1] = 0 (the saddr base, from push constants), v0 = lane index."""
    spv = translate(body)
    mem = dict(memory)
    si.run_workgroup(spv, nlanes, 32,
                     inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                     push_constants=[0] * 36, memory=mem)
    return mem


def atomic(kind, m, data, cmp=None, glc=True, offset=0, vaddr=TARGET):
    body = [f"v_mov_b32_e32 v1, {vaddr - offset:#x}", f"v_mov_b32_e32 v2, {data:#x}", "v_mov_b32_e32 v5, 0xdead"]
    vdata = "v2"
    if cmp is not None:
        body.append(f"v_mov_b32_e32 v3, {cmp:#x}")
        vdata = "v[2:3]"
    off = f" offset:{offset}" if offset else ""
    body.append(f"global_atomic_{kind} v5, v1, {vdata}, s[0:1]{off} glc" if glc else
                f"global_atomic_{kind} v1, {vdata}, s[0:1]{off}")
    body += ["s_waitcnt vmcnt(0)", f"v_mov_b32_e32 v6, {OUT:#x}", "global_store_dword v6, v5, s[0:1]"]
    mem = run(body, {TARGET: m})
    return mem.get(OUT), mem.get(TARGET)


class GlobalAtomics(unittest.TestCase):
    def test_returning_forms(self):
        for kind, fn in MODEL.items():
            for m in VALUES:
                for d in VALUES[::2]:
                    with self.subTest(op=kind, m=hex(m), d=hex(d)):
                        self.assertEqual(atomic(kind, m, d), (m, fn(m, d)))

    def test_without_glc_vdst_is_kept(self):
        for kind, fn in MODEL.items():
            with self.subTest(op=kind):
                self.assertEqual(atomic(kind, 0x80000005, 7, glc=False), (0xDEAD, fn(0x80000005, 7)))

    def test_cmpswap(self):
        for m, cmp, new in ((5, 5, 9), (5, 6, 9), (M32, M32, 0), (0, 1, 2)):
            with self.subTest(m=m, cmp=cmp, new=new):
                self.assertEqual(atomic("cmpswap", m, new, cmp=cmp), (m, new if m == cmp else m))
        self.assertEqual(atomic("cmpswap", 5, 9, cmp=5, glc=False), (0xDEAD, 9))

    def test_offsets(self):
        for offset in (-2048, -4, 4, 2044):
            with self.subTest(offset=offset):
                self.assertEqual(atomic("add", 40, 2, offset=offset), (40, 42))

    def test_off_addressing(self):
        body = [f"v_mov_b32_e32 v8, {TARGET:#x}", "v_mov_b32_e32 v9, 0", "v_mov_b32_e32 v2, 3",
                "global_atomic_or v5, v[8:9], v2, off glc", "s_waitcnt vmcnt(0)",
                f"v_mov_b32_e32 v6, {OUT:#x}", "global_store_dword v6, v5, s[0:1]"]
        mem = run(body, {TARGET: 0x10})
        self.assertEqual((mem.get(OUT), mem.get(TARGET)), (0x10, 0x13))

    def test_seeded(self):
        rng = random.Random(0x6A70)
        for _ in range(20):
            kind, m, d = rng.choice(list(MODEL)), rng.getrandbits(32), rng.getrandbits(32)
            with self.subTest(op=kind, m=hex(m), d=hex(d)):
                self.assertEqual(atomic(kind, m, d), (m, MODEL[kind](m, d)))

    def test_rejected_forms(self):
        for line in ("global_atomic_add_x2 v[4:5], v1, v[2:3], s[0:1] glc", "global_atomic_inc_x2 v[4:5], v1, v[2:3], s[0:1] glc",
                     "global_atomic_fmin_x2 v[4:5], v1, v[2:3], s[0:1] glc",
                     "global_atomic_add v5, v1, v2, s[0:1] glc slc",
                     "global_atomic_cmpswap v5, v1, v2, s[0:1] glc"):
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate(["v_mov_b32_e32 v1, 0", line])

    def test_float_min_max(self):
        # IEEE 754 single-precision fmin/fmax via CAS + GLSL NMin/NMax: NaN returns the other operand, -0.0 < +0.0.
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
                self.assertEqual(atomic(kind, m, d), (m, expected))
                self.assertEqual(atomic(kind, m, d, glc=False), (0xDEAD, expected))

    def test_float_min_max_inactive_and_off_addressing(self):
        # Inactive lane (v0 != 0) must leave memory untouched and preserve vdst; lane 0 runs global_atomic_fmax via off.
        one, two = 0x3F800000, 0x40000000
        body = [
            f"v_mov_b32_e32 v8, {TARGET:#x}", "v_mov_b32_e32 v9, 0",
            f"v_mov_b32_e32 v2, {two:#x}", "v_mov_b32_e32 v5, 0xdead",
            "v_cmp_eq_u32_e32 vcc_lo, 0, v0",
            "s_and_saveexec_b32 s4, vcc_lo",
            "global_atomic_fmax v5, v[8:9], v2, off glc",
            "s_mov_b32 exec_lo, s4",
            "s_waitcnt vmcnt(0)",
            f"v_lshlrev_b32_e32 v6, 2, v0",
            f"v_add_nc_u32_e32 v6, {OUT:#x}, v6",
            "global_store_dword v6, v5, s[0:1]",
        ]
        mem = run(body, {TARGET: one}, nlanes=2)
        self.assertEqual(mem.get(TARGET), two)
        self.assertEqual(mem.get(OUT), one)
        self.assertEqual(mem.get(OUT + 4), 0xDEAD)

    def test_float_min_max_multi_lane_cas_contention(self):
        # Two active lanes race on TARGET: lane 0 proposes 1.0f, lane 1 proposes -2.0f.
        init, one, neg2 = 0x40400000, 0x3F800000, 0xC0000000    # 3.0, 1.0, -2.0
        for kind, expected in (("fmin", neg2), ("fmax", init)):
            with self.subTest(op=kind):
                body = [
                    f"v_mov_b32_e32 v1, {TARGET:#x}",
                    "v_cmp_eq_u32_e32 vcc_lo, 0, v0",
                    f"v_mov_b32_e32 v2, {neg2:#x}",
                    f"v_cndmask_b32_e32 v2, v2, {one:#x}, vcc_lo",
                    f"global_atomic_{kind} v5, v1, v2, s[0:1] glc",
                    "s_waitcnt vmcnt(0)",
                    "v_lshlrev_b32_e32 v6, 2, v0",
                    f"v_add_nc_u32_e32 v6, {OUT:#x}, v6",
                    "global_store_dword v6, v5, s[0:1]",
                ]
                mem = run(body, {TARGET: init}, nlanes=2)
                self.assertEqual(mem.get(TARGET), expected)
                # One lane saw the initial 3.0f; the second saw the first lane's post-op value.
                seen = {mem.get(OUT), mem.get(OUT + 4)}
                self.assertIn(init, seen)

    def test_float_min_max_offsets_and_first_read(self):
        one, two = 0x3F800000, 0x40000000
        for offset in (-2048, -4, 4, 2044):
            with self.subTest(offset=offset):
                self.assertEqual(atomic("fmin", two, one, offset=offset), (two, one))
                self.assertEqual(atomic("fmax", one, two, offset=offset), (one, two))
        from shader_ir import parse_line as parse
        self.assertEqual(rdna2spirv.read_before_write_vgprs([parse("\tglobal_atomic_fmax v1, v2, s[0:1]")]), {1, 2})
        self.assertEqual(rdna2spirv.read_before_write_vgprs([parse("\tglobal_atomic_fmax v5, v1, v2, s[0:1] glc")]),
                         {1, 2})

    def test_first_read_analysis(self):
        # Without glc operand 0 is the address (read); with glc it is vdst (written).
        from shader_ir import parse_line as parse
        no_glc = [parse("\tglobal_atomic_add v1, v2, s[0:1]")]
        glc = [parse("\tglobal_atomic_add v5, v1, v2, s[0:1] glc")]
        self.assertEqual(rdna2spirv.read_before_write_vgprs(no_glc), {1, 2})
        self.assertEqual(rdna2spirv.read_before_write_vgprs(glc), {1, 2})


if __name__ == "__main__":
    unittest.main()
