"""LDS accesses narrower and wider than a dword: ds_write_b8 / _b8_d16_hi / _b16 / _b16_d16_hi, ds_read_u8 / _i8 /
_u16 / _i16, ds_write_b64 / _b96 / _b128, ds_read_b64 / _b96 / _b128 and ds_read2st64_b32.

Standard library only; no GPU, no LLVM, no game files.

    python -m unittest discover -s tests/semantic -p "test_lds_subdword_wide.py" -v

RDNA2 ISA, "Data Share Instructions"; ADDR = vaddr + offset (bytes):
  * ds_write_b8   MEM[ADDR][7:0]  = DATA[7:0]       ds_write_b8_d16_hi   MEM[ADDR][7:0]  = DATA[23:16]
  * ds_write_b16  MEM[ADDR][15:0] = DATA[15:0]      ds_write_b16_d16_hi  MEM[ADDR][15:0] = DATA[31:16]
    (the other bytes of the dword are kept)
  * ds_read_u8 / ds_read_i8   D = zero / sign-extend 8 -> 32 of MEM[ADDR][7:0]
  * ds_read_u16 / ds_read_i16 D = zero / sign-extend 16 -> 32 of MEM[ADDR][15:0]
  * ds_{read,write}_b64/b96/b128: N consecutive dwords from ADDR, lowest register first
  * ds_read2st64_b32: MEM[ADDR + offset0 * 256] and MEM[ADDR + offset1 * 256]

The inputs are the GPU tests runtime/test/shaders/cs_w32_lds_subdword.asm and cs_w32_lds_wide.asm (fixtures from
regen_fixtures.py), translated for wave32 and run as one 32-lane subgroup in lockstep. The expected words are
computed below from a byte-level model of LDS written from the rules above; they are the words in the .asm
headers, check.py and run_gpu_tests.py.
"""
import re
import struct
import unittest

import support
import spirv_mini_interp as si

# 4 user SGPRs (the output V#), workgroup id in s4, 512 bytes of LDS, 32x1x1 threads.
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x8088, 32, 1, 1)
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0x1000, 0x31016FAC]   # word 3 as in test_lds_read_u8_d16.py (OOB_SELECT is honoured since 14887e68)


class Lds:
    """Byte-addressed LDS, little endian, as the ISA describes it."""

    def __init__(self):
        self.b = bytearray(512)

    def write(self, addr, value, nbytes):
        self.b[addr:addr + nbytes] = (value & ((1 << 8 * nbytes) - 1)).to_bytes(nbytes, "little")

    def read(self, addr, nbytes, signed=False):
        return int.from_bytes(self.b[addr:addr + nbytes], "little", signed=signed) & 0xFFFFFFFF


def expected_subdword():
    m = Lds()
    m.write(0, 0x11223344, 4)
    m.write(4, 0x55667788, 4)
    data = 0xA1B2C3D4
    m.write(1, data, 1)               # ds_write_b8 offset:1
    m.write(2, data >> 16, 1)         # ds_write_b8_d16_hi offset:2
    m.write(4, data, 2)               # ds_write_b16 offset:4
    m.write(6, data >> 16, 2)         # ds_write_b16_d16_hi offset:6
    return [m.read(0, 4), m.read(4, 4), m.read(2, 1, True), m.read(2, 1), m.read(3, 1, True),
            m.read(6, 2, True), m.read(6, 2), m.read(4, 2, True)]


A, B, C, D, E, F, G, H = (0xCAFEF00D, 0x0BADBEEF, 0x600DD00D, 0xFEEDFACE, 0x8BADF00D, 0xDEADC0DE, 0x12345678,
                          0x9ABCDEF0)


def expected_wide():
    m = Lds()
    for k, v in enumerate((A, B, C, D)):  # ds_write_b128 v[30:33] offset:16
        m.write(16 + 4 * k, v, 4)
    m.write(60, 0, 4)                     # ds_write_b32 0 offset:60
    for k, v in enumerate((A, B, C)):     # ds_write_b96 v[30:32] offset:48
        m.write(48 + 4 * k, v, 4)
    m.write(64, E, 4)                     # ds_write_b64 v[34:35] offset:64
    m.write(68, F, 4)
    m.write(0, G, 4)
    m.write(256, H, 4)
    b128 = [m.read(16 + 4 * k, 4) for k in range(4)]
    b96 = [m.read(48 + 4 * k, 4) for k in range(3)]
    b64 = [m.read(64 + 4 * k, 4) for k in range(2)]
    st64 = [m.read(0 * 256, 4), m.read(1 * 256, 4)]   # offset0:0 offset1:1
    return [b128[0], b128[3], b96[2], m.read(60, 4), b64[0], b64[1], st64[0], st64[1]]


def run(name, sabotage=None):
    spv = support.translate_fixture(name, fssm=FSSM)
    if sabotage:
        spv = sabotage(spv)
    memory = si.run_workgroup(
        spv, 32, 32,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=VSHARP + [0] * 28, memory={})
    stray = sorted(a for a in memory if not 0 <= a < 32 or a % 4)
    if stray:
        raise AssertionError(f"stores outside out[0..7]: {stray}")
    return [memory.get(4 * k) for k in range(8)]


class LdsSubdwordWide(unittest.TestCase):
    def test_reference_values(self):
        # The words in the .asm headers, check.py and run_gpu_tests.py.
        self.assertEqual(expected_subdword(), [0x11B2D444, 0xA1B2C3D4, 0xFFFFFFB2, 0x000000B2, 0x00000011,
                                               0xFFFFA1B2, 0x0000A1B2, 0xFFFFC3D4])
        self.assertEqual(expected_wide(), [A, D, C, 0, E, F, G, H])

    def check(self, name, want):
        got = run(name)
        for k, (g, w) in enumerate(zip(got, want)):
            with self.subTest(shader=name, word=k):
                self.assertEqual(g, w, f"out[{k}] = {g if g is None else hex(g)}, want {hex(w)}")

    def test_subdword(self):
        self.check("cs_w32_lds_subdword", expected_subdword())

    def test_wide(self):
        self.check("cs_w32_lds_wide", expected_wide())

    def test_negative_controls(self):
        # Each sabotage of the translated SPIR-V must change the result.
        def const(t, value):     # the scalar constant the translator's cu(value) made (not the array length)
            names = re.findall(r"(%c\d+) = OpConstant %uint " + str(value) + r"\n", t)
            self.assertEqual(len(names), 1, f"uint constant {value}: {names}")
            return names[0]

        def unsigned(t):         # ds_read_i8 / ds_read_i16 zero-extend
            self.assertIn("OpBitFieldSExtract", t)
            return t.replace("OpBitFieldSExtract", "OpBitFieldUExtract")

        def no_hi_shift(t):      # the _d16_hi writes store DATA[7:0] / [15:0] instead of the high half
            pat = r"= OpShiftRightLogical %uint (%\w+) " + const(t, 16) + r"\n"
            self.assertEqual(len(re.findall(pat, t)), 2)
            return re.sub(pat, r"= OpCopyObject %uint \1\n", t)

        def st64_unit(t):        # ds_read2st64_b32 offset1:1 lands on dword 1 instead of dword 64
            pat = r"= OpIAdd %uint (%\w+) " + const(t, 256) + r"\n"
            self.assertEqual(len(re.findall(pat, t)), 2)   # ds_write_b32 offset:256 and the read2st64 element
            last = list(re.finditer(pat, t))[-1]
            return t[:last.start()] + f"= OpIAdd %uint {last.group(1)} {const(t, 4)}\n" + t[last.end():]

        for name, sab, want in (("cs_w32_lds_subdword", unsigned, expected_subdword()),
                                ("cs_w32_lds_subdword", no_hi_shift, expected_subdword()),
                                ("cs_w32_lds_wide", st64_unit, expected_wide())):
            with self.subTest(sabotage=sab.__name__):
                self.assertNotEqual(run(name, sab), want)


if __name__ == "__main__":
    unittest.main()
