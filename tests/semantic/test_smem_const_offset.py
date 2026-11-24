"""s_buffer_load / s_load with a constant SGPR offset give what the run-time path gives (no 32-bit wrap).

The translator folds an SGPR offset it knows (s_mov_b32 s12, C) into the immediate. #1686 added the fold before the
32-bit truncation, so s12 = -4 with offset:20 summed to 16 and read in-bounds data, where hardware (0xFFFFFFFC + 20,
not reduced mod 2^32, measured by @Pierre-Kookis on gfx1036) and the run-time path read 0 (found by @Impre-visible,
reproduced on a GTX 1660). Each case runs the same load twice, once with the offset as a constant and once arriving
through v_readfirstlane_b32 (which the translator cannot fold), and checks both against the rule in the
Gen.scalar_load docstring (measured on gfx1030): off = (sOFF & ~3) + (IMM & ~3); for s_buffer_load dword i is 0
when off + 4 * i + 4 > NUM_RECORDS (stride 0). Standard library only; no GPU, no game files.
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
struct.pack_into("<4I", FSSM, 0x138, 0x88, 32, 1, 1)
FSSM = bytes(FSSM)
BASE, RECORDS, OUT = 0x100, 0x40, 0x300
SEED = {BASE + 4 * k: 0x11110000 + k for k in range(RECORDS // 4)}
SEED[BASE + 0x100000] = 0xA5A50000                                # where sOFF = 2^20 lands (s_load)


def translate(body):
    with tempfile.TemporaryDirectory() as tmp:
        for d in ("disasm", "fssm"):
            os.mkdir(os.path.join(tmp, d))
        with open(os.path.join(tmp, "cs_smemoff.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", "cs_smemoff.fssm"), "wb") as f:
            f.write(FSSM)
        src = os.path.join(tmp, "disasm", "cs_smemoff.s")
        with open(src, "w", encoding="utf-8") as f:
            f.write("; cs_smemoff\n" + "".join(f"\t{ln}\n" for ln in body) + "\ts_endpgm\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            return rdna2spirv.translate(src)


def run(load_line, sgpr_off, constant):
    set_off = [f"s_mov_b32 s12, {sgpr_off & M32:#x}"] if constant else \
        [f"v_mov_b32_e32 v20, {sgpr_off & M32:#x}", "v_readfirstlane_b32 s12, v20"]
    body = ["s_mov_b32 s8, 0x100", "s_mov_b32 s9, 0", f"s_mov_b32 s10, {RECORDS:#x}", "s_mov_b32 s11, 0x31016fac",
            "s_mov_b32 s16, 0x100", "s_mov_b32 s17, 0"] + set_off + [load_line, "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v1, s14", f"v_mov_b32_e32 v2, {OUT:#x}", "buffer_store_dword v1, v2, s[0:3], 0 offen"]
    spv = translate(body)
    mem = dict(SEED)
    si.run_workgroup(spv, 1, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                     push_constants=[0, 0, 0x1000, 0x31016FAC] + [0] * 28, memory=mem)
    return mem.get(OUT)


def sbuf_ref(sgpr_off, imm):
    off = (sgpr_off & 0xFFFFFFFC) + (imm & 0xFFFFFFFC)          # no wrap
    return 0 if off + 4 > RECORDS else SEED[BASE + off]


class SmemConstOffset(unittest.TestCase):
    def test_s_buffer_load_constant_and_run_time_offset_agree(self):
        for sgpr_off, imm in ((-4, 20), (-16, 0x20), (0xFFFFFFF0, 0x10), (-1, 4), (4, 8), (0x3C, 0), (0x40, 0),
                              (0x10, 0x2C), (0x80000000, 0x10), (0, 0x3C)):
            with self.subTest(sgpr_off=hex(sgpr_off & M32), imm=hex(imm)):
                line = f"s_buffer_load_dword s14, s[8:11], s12 offset:{imm:#x}"
                const = run(line, sgpr_off, constant=True)
                dyn = run(line, sgpr_off, constant=False)
                self.assertEqual(const, dyn)
                self.assertEqual(const, sbuf_ref(sgpr_off, imm))

    def test_the_reported_case(self):
        # Impre-visible's listing: s8 = -4, offset:20 -> out of range, 0 (it read the word at byte 16 before).
        line = "s_buffer_load_dword s14, s[8:11], s12 offset:20"
        self.assertEqual(run(line, -4, constant=True), 0)
        self.assertEqual(run(line, 12, constant=True), SEED[BASE + 32])

    def test_s_load_constant_and_run_time_offset_agree(self):
        for sgpr_off, imm in ((8, 4), (0x20, -8), (0, 0x3C), (0x10, -0x10)):
            with self.subTest(sgpr_off=hex(sgpr_off), imm=imm):
                line = f"s_load_dword s14, s[16:17], s12 offset:{imm}"
                const = run(line, sgpr_off, constant=True)
                dyn = run(line, sgpr_off, constant=False)
                self.assertEqual(const, dyn)
                self.assertEqual(const, SEED[BASE + sgpr_off + imm])


    def test_s_load_constant_offset_at_or_above_2_20(self):
        # The old fold put sOFF into the 21-bit signed immediate, so sOFF = 0x100000 became -0x100000. Hardware and
        # the run-time path add the zero-extended 32-bit sOFF; the loaded words agree (an address past the seeded
        # words reads the interpreter's unset memory, the same either way).
        for sgpr_off in (0x100000, 0x1FFFFC, 0x80000000):
            with self.subTest(sgpr_off=hex(sgpr_off)):
                line = "s_load_dword s14, s[16:17], s12 offset:0"
                self.assertEqual(run(line, sgpr_off, constant=True), run(line, sgpr_off, constant=False))
        self.assertEqual(run("s_load_dword s14, s[16:17], s12 offset:0", 0x100000, constant=True), 0xA5A50000)


if __name__ == "__main__":
    unittest.main()
