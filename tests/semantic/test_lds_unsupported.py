"""LDS operations the translator does not implement are refused, not translated as something else.

Standard library only; no GPU, no LLVM, no game files.

These are RDNA2 Data Share instructions that tools/rdna2spirv.py has no rule for:
  * ds_add_src2_u32, ds_sub_src2_u32, ds_inc_src2_u32, ds_dec_src2_u32: the second operand is LDS[B] (B = vaddr +
    offset1 * 4), not a register, and the result goes to LDS[A] (A = vaddr + offset0 * 4);
  * ds_add_u64, ds_sub_u64 and their _rtn_ forms: a 64-bit atomic on two dwords (Workgroup memory has no Int64
    atomics here; only the exchange and the bitwise 64-bit forms are lowered, dword by dword);
  * ds_condxchg32_rtn_b64 (the RDNA2 name; ds_condxchg32_rtn_b32 is not an RDNA2 mnemonic at all).
Until this file, one self-built model of each of these ran in tests/semantic, with an instruction decoder of its own,
and so tested nothing the translator does. A model that is not the translator's behaviour cannot catch a regression;
what the translator does is refuse them with `Unsupported`, and that is what is pinned here, in a compute shader (the
only stage with LDS). A separate test pins that LDS outside compute is refused, using a supported op. When a rule is added, move its entry out of this list and test its values.
"""
import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import support  # noqa: E402  (also puts tools/ on sys.path)

rdna2spirv = support.rdna2spirv

# RSRC2 0x40088: 4 user SGPRs, workgroup id in s4, 4 KiB LDS; 32x1x1 threads.
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x40088, 32, 1, 1)
FSSM = bytes(FSSM)

UNSUPPORTED = (
    "ds_add_src2_u32 v1 offset0:1 offset1:2",
    "ds_sub_src2_u32 v1 offset0:1 offset1:2",
    "ds_inc_src2_u32 v1 offset0:1 offset1:2",
    "ds_dec_src2_u32 v1 offset0:1 offset1:2",
    "ds_add_u64 v1, v[2:3]",
    "ds_sub_u64 v1, v[2:3]",
    "ds_add_rtn_u64 v[4:5], v1, v[2:3]",
    "ds_sub_rtn_u64 v[4:5], v1, v[2:3]",
    "ds_sub_u64 v1, v[2:3] offset:8",
    "ds_condxchg32_rtn_b64 v[4:5], v1, v[2:3]",
    "ds_condxchg32_rtn_b32 v4, v1, v2, v3",
)


class TestRefused(unittest.TestCase):
    def test_compute_shader_refuses_each(self):
        for line in UNSUPPORTED:
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported) as cm:
                    support.translate_lines("cs_lds_unsupported", [line, "s_endpgm"], fssm=FSSM)
                self.assertIn(line.split()[0], str(cm.exception))

    def test_lds_outside_compute_is_refused(self):
        # The translator refuses any LDS op outside a compute shader, supported ones included, so this uses a
        # supported op: it pins that rule, not the unsupported list above.
        with self.assertRaises(rdna2spirv.Unsupported) as cm:
            support.translate_lines("ps_lds_outside_compute", ["ds_add_u32 v1, v2", "s_endpgm"])
        self.assertIn("outside compute", str(cm.exception))

    def test_the_lowered_neighbours_still_translate(self):
        # Guard against a blanket refusal: the supported relatives of the refused forms translate.
        for line in ("ds_add_u32 v1, v2", "ds_sub_rtn_u32 v4, v1, v2", "ds_wrxchg_rtn_b64 v[4:5], v1, v[2:3]",
                     "ds_and_b64 v1, v[2:3]", "ds_inc_u32 v1, v2", "ds_min_f32 v1, v2"):
            with self.subTest(line=line):
                self.assertIn("OpAtomic", support.translate_lines("cs_lds_neighbour", [line, "s_endpgm"], fssm=FSSM))


if __name__ == "__main__":
    unittest.main()
