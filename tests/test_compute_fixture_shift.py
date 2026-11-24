"""runtime/test/shaders/compute_fixture.py: the user-data shift for D3D12 compute fixtures (D3D-448).

    python -I tests/test_compute_fixture_shift.py

Hand-written asm text only; no game files, no llvm-mc, no GPU.
"""
import os
import struct
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "runtime", "test", "shaders"))

import compute_fixture as cf  # noqa: E402

ASM = """; rsrc2: 0x808c
; s0 = seed, s[4:5] = table (comments keep their register names)
s_load_dwordx4 s[8:11], s[4:5], 0x0
s_lshl_b32 s7, s6, 6
v_add_nc_u32_e32 v4, s0, v3 ; s0 again
s_mov_b64 exec, vcc
s_and_b32 s105, s0, m0
"""


class ShiftTest(unittest.TestCase):
    def test_every_sgpr_moves_and_comments_stay(self):
        got = cf.shift_user_data(ASM.replace("s105", "s100"), 4).splitlines()
        self.assertEqual(got[0], "; rsrc2: 0x808c")
        self.assertEqual(got[1], "; s0 = seed, s[4:5] = table (comments keep their register names)")
        self.assertEqual(got[2], "s_load_dwordx4 s[12:15], s[8:9], 0x0")
        self.assertEqual(got[3], "s_lshl_b32 s11, s10, 6")
        self.assertEqual(got[4], "v_add_nc_u32_e32 v4, s4, v3 ; s0 again")
        self.assertEqual(got[5], "s_mov_b64 exec, vcc")          # named registers stay
        self.assertEqual(got[6], "s_and_b32 s104, s4, m0")

    def test_zero_shift_is_identity(self):
        self.assertEqual(cf.shift_user_data(ASM, 0), ASM)

    def test_past_s105_is_refused(self):
        with self.assertRaises(ValueError):
            cf.shift_user_data(ASM, 4)                            # s105 + 4
        with self.assertRaises(ValueError):
            cf.shift_user_data("s_load_dwordx4 s[100:103], s[0:1], 0x0\n", 4)

    def test_mnemonics_and_vgprs_untouched(self):
        line = "s_buffer_load_dword s1, s[4:7], 0x10 v_mov_b32 v1, v2\n"
        self.assertEqual(cf.shift_user_data(line, 4), "s_buffer_load_dword s5, s[8:11], 0x10 v_mov_b32 v1, v2\n")


class Rsrc2Test(unittest.TestCase):
    def test_user_count_moves_other_fields_stay(self):
        # 0x808c: USER_SGPR 6, TGID_X, LDS 1 unit. +4 -> USER_SGPR 10, the rest unchanged.
        got = cf.shifted_rsrc2(0x808c, 4)
        self.assertEqual((got >> 1) & 31, 10)
        self.assertEqual(got & ~(31 << 1), 0x808c & ~(31 << 1))
        self.assertEqual(cf.shifted_rsrc2(0x8088, 4), 0x8090)   # the default header: 4 -> 8 user SGPRs
        self.assertEqual(cf.shifted_rsrc2(0x8088, 0), 0x8088)

    def test_user_count_overflow_refused(self):
        with self.assertRaises(ValueError):
            cf.shifted_rsrc2(31 << 1, 1)

    def test_header_uses_the_shift(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cs_x.asm")
            with open(path, "w", encoding="utf-8") as f:
                f.write("; rsrc2: 0x808c\n; group: 32,2,1\ns_endpgm\n")
            plain = struct.unpack_from("<4I", cf.compute_header(path), 0x138)
            shifted = struct.unpack_from("<4I", cf.compute_header(path, 4), 0x138)
        self.assertEqual(plain, (0x808c, 32, 2, 1))
        self.assertEqual(shifted, (cf.shifted_rsrc2(0x808c, 4), 32, 2, 1))


class EnvTest(unittest.TestCase):
    def test_env_values(self):
        for value, want in ((None, 0), ("", 0), ("4", 4), (" 4 ", 4)):
            env = {} if value is None else {"LEONIDA_USER_DATA_SHIFT": value}
            with mock.patch.dict(os.environ, env, clear=True):
                self.assertEqual(cf.user_data_shift(), want)
        for bad in ("four", "-1", "28"):
            with mock.patch.dict(os.environ, {"LEONIDA_USER_DATA_SHIFT": bad}, clear=True):
                with self.assertRaises(ValueError):
                    cf.user_data_shift()


if __name__ == "__main__":
    unittest.main()
