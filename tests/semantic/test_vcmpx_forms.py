"""The 14 v_cmpx forms that had no test: v_cmpx_{eq,gt,le,lt,ngt,nle}_f32, v_cmpx_f_i32, v_cmpx_t_u32,
v_cmpx_{eq,gt,le}_i64 and v_cmpx_{ge,lt,ne}_u64.

RDNA2 ISA (VOPC): EXEC = EXEC & compare(S0, S1), per lane, and nothing else is written, so VCC is kept.
  EQ = S0 == S1, GT = S0 > S1, LE = S0 <= S1, LT = S0 < S1: ordered, false when a source is NaN; -0 equals +0.
  NGT = !(S0 > S1), NLE = !(S0 <= S1): the negations, so true when a source is NaN.
  F = 0, T = 1, whatever the sources hold.
  _I64 / _U64 compare the whole 64-bit values signed / unsigned (the low word is never signed).

Three kinds of test, all through tools/rdna2spirv.py and tests/semantic/spirv_mini_interp.py:
  * ps_vcmpx_forms (runtime/test/shaders, also run on the GPU by check.py and run_gpu_tests.py): the four words
    its header derives by hand, compared with a reference written here from the definitions above.
  * every form in both encodings (e32, e64) on a table of operands: the pixel survives exactly when Python's own
    comparison of the operands holds.
  * a wave of 16 lanes with different operands, one of them switched off beforehand: the lanes left running.
"""
import itertools
import struct
import unittest

import support
import spirv_mini_interp as si

EXPECTED = [0xF05B2A48, 0xF05E8A12, 0xB174BE18, 0x15550000]      # derivation in ps_vcmpx_forms.asm
# NaN, 1.0, 2.0, -0.0, +0.0, -infinity, two unused words; then four pairs (A, B) as A.lo A.hi B.lo B.hi.
HEAP = (0x7FC10001, 0x3F800000, 0x40000000, 0x80000000, 0x00000000, 0xFF800000, 0, 0,
        0x80000000, 0x00000000, 0x00000001, 0x00000000,
        0x00000007, 0x80000000, 0x00000007, 0x00000000,
        0x00000001, 0x00000005, 0x80000000, 0x00000005,
        0x00000007, 0x00000005, 0x00000007, 0x00000005)

F32_OPS = ("eq", "gt", "le", "lt", "ngt", "nle")
I64_OPS = ("eq_i64", "gt_i64", "le_i64", "ge_u64", "lt_u64", "ne_u64")
ALL_OPS = tuple(f"{c}_f32" for c in F32_OPS) + ("f_i32", "t_u32") + I64_OPS


def f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def signed64(x):
    return x - (1 << 64) if x >> 63 else x


def relation(op, a, b):
    """The ISA's compare for `op` (the mnemonic without v_cmpx_). a, b: the sources' bits (32 or 64 wide)."""
    cond, ty = op.rsplit("_", 1)
    if cond in ("f", "t"):
        return cond == "t"
    if ty == "f32":
        x, y = f32(a), f32(b)                  # Python's float compares are IEEE: false on NaN, -0.0 == 0.0
        return {"eq": x == y, "gt": x > y, "le": x <= y, "lt": x < y, "ngt": not x > y, "nle": not x <= y}[cond]
    if ty == "i64":
        a, b = signed64(a), signed64(b)
    return {"eq": a == b, "ne": a != b, "gt": a > b, "ge": a >= b, "lt": a < b, "le": a <= b}[cond]


def reference_words():
    """The four mrt0 words of ps_vcmpx_forms, from `relation` and the heap (layout in the .asm header)."""
    nan, one, two, nzero, pzero, ninf = HEAP[:6]
    q = [((HEAP[9 + 4 * i] << 32) | HEAP[8 + 4 * i], (HEAP[11 + 4 * i] << 32) | HEAP[10 + 4 * i]) for i in range(4)]
    first8 = ALL_OPS[:8]

    def nibbles(ops, pairs):
        return sum(int(relation(op, *p)) << (4 * k + i) for k, op in enumerate(ops) for i, p in enumerate(pairs))

    word0 = nibbles(first8, ((nan, one), (one, two), (two, one), (one, one)))
    word1 = nibbles(first8, ((two, one), (nzero, pzero), (nan, nan), (ninf, one)))
    word2 = nibbles(I64_OPS + ("gt_i64", "ge_u64"), q)
    # Operands that make op k true (bits 0..13 and the odd VCC bits) and false (the even VCC bits).
    true = {"eq_f32": (one, one), "gt_f32": (two, one), "le_f32": (one, two), "lt_f32": (one, two),
            "ngt_f32": (nan, one), "nle_f32": (nan, one), "f_i32": (one, one), "t_u32": (one, two),
            "eq_i64": q[3], "gt_i64": q[0], "le_i64": q[1], "ge_u64": q[1], "lt_u64": q[2], "ne_u64": q[1]}
    false = {"eq_f32": (nan, nan), "le_f32": (two, one), "ngt_f32": (two, one), "f_i32": (one, one),
             "eq_i64": q[1], "le_i64": q[0], "lt_u64": q[1]}
    word3 = 0
    for k, op in enumerate(ALL_OPS):
        exec_ = False                                        # emptied by v_cmpx_ne_u32 of equal values
        exec_ = exec_ and relation(op, *true[op])
        word3 |= int(exec_) << k
        vcc = k % 2 == 0                                     # even k: VCC = 1 and a false compare; odd: 0 and true
        assert relation(op, *(false[op] if vcc else true[op])) != vcc, op
        word3 |= int(vcc) << (16 + k)                        # v_cmpx leaves VCC alone
    return [word0, word1, word2, word3]


F32_VALUES = {"0": 0x00000000, "-0": 0x80000000, "1": 0x3F800000, "-1": 0xBF800000, "2.5": 0x40200000,
              "inf": 0x7F800000, "-inf": 0xFF800000, "nan": 0x7FC00000, "-nan": 0xFFC00001}
# 64-bit values that only the high word, only the low word, or the sign tell apart.
I64_VALUES = (0x0000000000000000, 0x0000000000000001, 0x0000000080000000, 0x00000000FFFFFFFF,
              0x0000000100000000, 0x0000000500000007, 0x7FFFFFFFFFFFFFFF, 0x8000000000000000,
              0x8000000000000007, 0xFFFFFFFF00000000, 0xFFFFFFFFFFFFFFFF)


def survives(op, form, a, b):
    """Run `v_cmpx_<op>_<form> a, b` in a one-pixel shader; True when the pixel still exports afterwards."""
    if op.endswith("64"):
        lines = [f"v_mov_b32 v2, {a & 0xFFFFFFFF:#x}", f"v_mov_b32 v3, {a >> 32:#x}",
                 f"v_mov_b32 v4, {b & 0xFFFFFFFF:#x}", f"v_mov_b32 v5, {b >> 32:#x}",
                 f"v_cmpx_{op}_{form} v[2:3], v[4:5]"]
    else:
        lines = [f"v_mov_b32 v2, {a:#x}", f"v_mov_b32 v3, {b:#x}", f"v_cmpx_{op}_{form} v2, v3"]
    spv = support.translate_lines("ps_vcmpx_case", [*lines, "exp mrt0 v2, v2, v2, v2 done vm", "s_endpgm"])
    return "%out_mrt0" in si.run_fragment(spv)


class OracleTests(unittest.TestCase):
    def test_reference_matches_the_hand_derived_words(self):
        self.assertEqual([f"{w:#010x}" for w in reference_words()], [f"{w:#010x}" for w in EXPECTED])

    def test_relation_on_the_cases_that_tell_the_forms_apart(self):
        nan, one, two = HEAP[:3]
        self.assertFalse(relation("eq_f32", nan, nan))
        self.assertTrue(relation("eq_f32", 0x80000000, 0))
        self.assertTrue(relation("ngt_f32", nan, one) and relation("nle_f32", nan, one))
        self.assertFalse(relation("le_f32", nan, one) or relation("gt_f32", nan, one))
        self.assertTrue(relation("ngt_f32", one, two) and not relation("nle_f32", one, two))
        self.assertTrue(relation("ge_u64", 1 << 63, 1) and relation("le_i64", 1 << 63, 1))
        self.assertTrue(relation("gt_i64", 0x80000000, 1) and relation("lt_u64", 1, 0x80000000))


class FixtureTests(unittest.TestCase):
    def test_ps_vcmpx_forms(self):
        spv = support.translate_fixture("ps_vcmpx_forms")
        words = support.mrt0_words(spv, push_constants=[0x1000, 0, 96, 0] + [0] * 28,
                                   memory={0x1000 + 4 * i: w for i, w in enumerate(HEAP)})
        self.assertEqual([f"{w:#010x}" for w in words], [f"{w:#010x}" for w in EXPECTED])


class EveryFormOnATable(unittest.TestCase):
    def test_f32_forms(self):
        for cond, form in itertools.product(F32_OPS, ("e32", "e64")):
            op = f"{cond}_f32"
            for (xn, xb), (yn, yb) in itertools.product(F32_VALUES.items(), repeat=2):
                with self.subTest(op=op, form=form, x=xn, y=yn):
                    self.assertEqual(survives(op, form, xb, yb), relation(op, xb, yb))

    def test_f_i32_and_t_u32_ignore_their_sources(self):
        for op, form in itertools.product(("f_i32", "t_u32"), ("e32", "e64")):
            for a, b in ((0, 0), (1, 2), (2, 1), (0xFFFFFFFF, 0), (0x80000000, 0x7FFFFFFF)):
                with self.subTest(op=op, form=form, a=hex(a), b=hex(b)):
                    self.assertEqual(survives(op, form, a, b), op == "t_u32")

    def test_64_bit_forms(self):
        for op, form in itertools.product(I64_OPS, ("e32", "e64")):
            for a, b in itertools.product(I64_VALUES, repeat=2):
                with self.subTest(op=op, form=form, a=hex(a), b=hex(b)):
                    self.assertEqual(survives(op, form, a, b), relation(op, a, b))


class LanesOfAWave(unittest.TestCase):
    """One 16-thread wave32: lane i compares operand X[i] with one shared operand; lane 5 is off beforehand.
    Each lane left running stores its index, so the stored words are the EXEC the compare left."""
    LANES = 16
    OFF = 5
    F32_X = (0x7FC00000, 0x3F800000, 0x40000000, 0x80000000, 0x00000000, 0x3F800000, 0xFF800000, 0x7F800000,
             0xBF800000, 0x3F800001, 0x3F7FFFFF, 0xFFC00000, 0x00000001, 0x80000001, 0x40000000, 0x3F800000)
    F32_Y = 0x3F800000                                       # 1.0
    I64_X = (0, 1, 0x80000000, 0xFFFFFFFF, 0x100000000, 0x500000007, 0x7FFFFFFFFFFFFFFF, 0x8000000000000000,
             0x8000000000000007, 0xFFFFFFFF00000000, 0xFFFFFFFFFFFFFFFF, 0x500000006, 0x500000008,
             0x400000007, 0x600000007, 0x8000000500000007)
    I64_Y = 0x500000007

    def run_wave(self, setup, compare):
        fssm = bytearray(0x148)                              # COMPUTE_PGM_RSRC2 (4 user SGPRs), group size
        struct.pack_into("<4I", fssm, 0x138, 0x8088, self.LANES, 1, 1)
        lines = [*setup,
                 f"v_cmpx_ne_u32_e32 {self.OFF}, v0",        # lane OFF leaves the wave
                 "v_cmp_gt_u32_e32 vcc_lo, 8, v0",           # VCC = lanes 0..7: must survive the compare
                 compare,
                 "v_cndmask_b32_e64 v20, 0x100, 0x200, vcc_lo",
                 "v_or_b32_e32 v20, v20, v0",
                 "v_lshlrev_b32_e32 v21, 2, v0",
                 "buffer_store_dword v20, v21, s[0:3], 0 offen",
                 "s_endpgm"]
        spv = support.translate_lines("cs_vcmpx_wave", lines, fssm=bytes(fssm))
        return si.run_workgroup(
            spv, self.LANES, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
            push_constants=[0, 0, 0x1000, 0x31016FAC] + [0] * 28, memory={})

    def want(self, op, xs, y):
        # lane i stores (0x200 if i < 8 else 0x100) | i at 4*i when it is still running
        return {4 * i: (0x200 if i < 8 else 0x100) | i
                for i in range(self.LANES) if i != self.OFF and relation(op, xs[i], y)}

    @staticmethod
    def per_lane(reg, values):
        """v<reg> = values[lane], built with one v_cndmask per lane from the lane index in v0."""
        lines = [f"v_mov_b32_e32 v{reg}, {values[0]:#x}"]
        for i, v in enumerate(values[1:], 1):
            lines += [f"v_mov_b32_e32 v19, {v:#x}", f"v_cmp_eq_u32_e32 vcc_lo, {i}, v0",
                      f"v_cndmask_b32_e32 v{reg}, v{reg}, v19, vcc_lo"]
        return lines

    def test_f32_forms(self):
        setup = self.per_lane(1, self.F32_X) + [f"v_mov_b32_e32 v2, {self.F32_Y:#x}"]
        for cond, form in itertools.product(F32_OPS, ("e32", "e64")):
            with self.subTest(op=cond, form=form):
                got = self.run_wave(setup, f"v_cmpx_{cond}_f32_{form} v1, v2")
                self.assertEqual(got, self.want(f"{cond}_f32", self.F32_X, self.F32_Y))

    def test_f_i32_and_t_u32(self):
        setup = self.per_lane(1, self.F32_X) + [f"v_mov_b32_e32 v2, {self.F32_Y:#x}"]
        for op, form in itertools.product(("f_i32", "t_u32"), ("e32", "e64")):
            with self.subTest(op=op, form=form):
                got = self.run_wave(setup, f"v_cmpx_{op}_{form} v1, v2")
                self.assertEqual(got, self.want(op, self.F32_X, self.F32_Y))

    def test_64_bit_forms(self):
        setup = (self.per_lane(1, [x & 0xFFFFFFFF for x in self.I64_X]) + self.per_lane(2, [x >> 32 for x in self.I64_X])
                 + [f"v_mov_b32_e32 v3, {self.I64_Y & 0xFFFFFFFF:#x}", f"v_mov_b32_e32 v4, {self.I64_Y >> 32:#x}"])
        for op, form in itertools.product(I64_OPS, ("e32", "e64")):
            with self.subTest(op=op, form=form):
                got = self.run_wave(setup, f"v_cmpx_{op}_{form} v[1:2], v[3:4]")
                self.assertEqual(got, self.want(op, self.I64_X, self.I64_Y))


if __name__ == "__main__":
    unittest.main()
