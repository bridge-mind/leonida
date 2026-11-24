"""Translate ADDTID reads and writes and compare them with ordinary LDS accesses."""
import struct
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from support import rdna2spirv, translate_lines  # noqa: E402
from test_loop_ballot_barrier import run_compute_workgroup  # noqa: E402


def translate_addtid(wave, emulate=False, lines=None):
    if lines is None:
        fixture = Path(__file__).with_name("fixtures") / "lds_addtid.asm"
        lines = [line for line in fixture.read_text(encoding="ascii").splitlines()
                 if line and not line.startswith(";")]
        if wave == 64:
            lines = ["s_mov_b64 exec, -1" if line == "s_mov_b32 exec_lo, -1" else line for line in lines]
    fssm = bytearray(0x148)
    struct.pack_into("<I", fssm, 0xD4, 0x20 if wave == 32 else 0)
    # Four user SGPRs for the output V# and 4 KiB of LDS; two waves per workgroup.
    struct.pack_into("<4I", fssm, 0x138, 0x40008, 2 * wave, 1, 1)
    with patch.object(rdna2spirv, "WAVE64_EMULATE", emulate):
        return translate_lines(f"cs_w{wave}_lds_addtid", lines, fssm=bytes(fssm))


def expected_words(wave, invocation):
    enabled = invocation < 16 and invocation % 2 == 0
    return (1000 + invocation % wave, 1000 + invocation if enabled else 99,
            1000 + invocation if enabled else 3000)


class LdsAddtidTests(unittest.TestCase):
    def test_wave32_wave64_and_emulated_wave64(self):
        for wave, emulate in ((32, False), (64, False), (64, True)):
            with self.subTest(wave=wave, emulate=emulate):
                spvasm = translate_addtid(wave, emulate)
                memory = run_compute_workgroup(
                    spvasm, 2 * wave, 32 if emulate else wave,
                    push_constants=[0, 0, 0x1000, 0x31016FAC] + [0] * 28, memory={},
                )
                for i in range(2 * wave):
                    self.assertEqual(tuple(memory[16 * i + n] for n in (0, 4, 8)), expected_words(wave, i),
                                     f"invocation {i}")

    def test_rejects_gds_and_invalid_offset(self):
        for line in ("ds_read_addtid_b32 v1 gds", "ds_write_addtid_b32 v1 offset:65536"):
            with self.subTest(line=line), self.assertRaises(rdna2spirv.Unsupported):
                translate_addtid(32, lines=[line, "s_endpgm"])


if __name__ == "__main__":
    unittest.main()
