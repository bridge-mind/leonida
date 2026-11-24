"""Far branches: s_getpc_b64; s_add_u32 K / s_addc_u32 0|-1 (or s_sub_u32 / s_subb_u32); s_setpc_b64.

RDNA2 ISA: s_getpc_b64 returns the address of the next instruction, s_add_u32 / s_addc_u32 add a 64-bit K (high word 0
forward, 0xffffffff backward) with SCC as the carry, s_setpc_b64 jumps there. The shader is
runtime/test/shaders/ps_far_branch.asm (also run on the GPU by check.py and run_gpu_tests.py); its header derives the
four words by hand. The reference here runs the same loops in plain Python on the byte distances of the header.
"""
import os
import re
import unittest

import support

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FIXTURE = os.path.join(HERE, "fixtures", "disasm", "ps_far_branch.s")
ASM = os.path.join(ROOT, "runtime", "test", "shaders", "ps_far_branch.asm")

EXPECTED = [0x00000007, 0x00000013, 0x00000051, 0x0000000B]       # derivation in ps_far_branch.asm


def reference():
    x = 1 | 2 | 4                       # two forward far branches skip the poison ORs 0x100 and 0x200
    s11, back = 0, 0                    # loop with s_sub_u32 / s_subb_u32: leaves when s11 >= 3
    while True:
        s11 += 1
        if s11 >= 3:
            break
        back += 1
    y = s11 | 0x10
    s12 = 0                             # loop with s_add_u32 -180 / s_addc_u32 -1: leaves when s12 >= 0x50
    while True:
        s12 += 0x10
        if s12 >= 0x50:
            break
    z = s12 | 1
    w = 0
    for compare, bit_before, bit_target in ((1, 1, 2), (3, 4, 8)):
        if w == compare:                # scc = 1: the far branch runs and skips the bit_before OR
            w |= bit_target
        else:                           # scc = 0: the branch over the far-branch sequence is taken
            w |= bit_before
            w |= bit_target
    return [x, y, z, w]


class FarBranch(unittest.TestCase):
    def test_reference_matches_the_header_words(self):
        self.assertEqual([f"{w:#010x}" for w in reference()], [f"{w:#010x}" for w in EXPECTED])

    def test_listing_is_the_asm(self):
        """The listing has the `; encoding:` comments the branches need (regen_fixtures.py writes none), so it is
        llvm-mc output; its mnemonics must be those of the .asm, in order."""
        def mnemonics(lines):
            return [re.sub(r"_e32$", "", ln.split()[0]) for ln in lines]
        with open(ASM, encoding="utf-8") as f:
            asm = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith(";")]
        with open(FIXTURE, encoding="utf-8") as f:
            listing = [ln.strip() for ln in f.read().splitlines()[1:] if ln.strip()]
        self.assertEqual(mnemonics(listing), mnemonics(asm))

    def test_ps_far_branch(self):
        with open(FIXTURE, "rb") as f:
            spv = support.translate_listing("ps_far_branch", f.read())
        words = support.mrt0_words(spv)
        self.assertEqual([f"{w:#010x}" for w in words], [f"{w:#010x}" for w in EXPECTED])


class FarBranchWithoutEncoding(unittest.TestCase):
    def test_idiom_without_encoding_is_unsupported_not_a_crash(self):
        """A hand-written listing has no `; encoding:` bytes, so the idiom's size is unknown: Unsupported, as for a
        lone s_getpc_b64 (test_missing_encoding.py), not a TypeError from joining the missing encodings."""
        import rdna2spirv
        lines = ["s_getpc_b64 s[4:5]", "s_add_u32 s4, s4, 8", "s_addc_u32 s5, s5, 0", "s_setpc_b64 s[4:5]",
                 "v_mov_b32 v0, 1", "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
        text = "; far\n" + "".join("\t" + ln + "\n" for ln in lines)
        with self.assertRaisesRegex(rdna2spirv.Unsupported, "without encoding"):
            support.translate_listing("ps_far_noenc", text.encode("utf-8"))


if __name__ == "__main__":
    unittest.main()
