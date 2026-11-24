"""The heap base is built once per module, in the entry block, not at every memory access.

Gen.host_addr turns an Xbox GPU address into a host one by adding the heap base, push-constant words 1 and 2 of
%UserData (or of %pc_tail). Push constants can't change during an invocation, so %main's entry block builds the 64-bit
base once (HEAP_BASE_ENTRY) and every access adds to %heap_base; it used to load, widen, shift and or the two words
again at each one. The entry block dominates every block of %main, so every use is valid SSA. The addresses
themselves are covered by test_scalar_mem_smem.py and test_global_mem_offset.py. Synthetic only.

    python -I -m unittest discover -s tests/semantic -p test_heap_base_once.py -v
"""
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import support  # noqa: E402

# Scalar loads and a buffer load through a V# they loaded: four accesses, one base.
LINES = ["s_load_dwordx4 s[8:11], s[2:3], 0x10",
         "s_buffer_load_dword s12, s[4:7], 0x20",
         "s_load_dword s13, s[2:3], 0x40",
         "s_waitcnt lgkmcnt(0)",
         "v_mov_b32 v1, 16",
         "buffer_load_dword v5, v1, s[8:11], 0 offen",
         "v_add_nc_u32 v5, s13, v5",
         "exp mrt0 v5, v5, v5, v5 done vm",
         "s_endpgm"]


def function_of(spvasm, name):
    m = re.search(rf"^\s*{name} = OpFunction .*?^\s*OpFunctionEnd$", spvasm, re.M | re.S)
    return m.group(0) if m else ""


class HeapBaseOnceTest(unittest.TestCase):
    def test_one_base_for_every_access(self):
        spv = support.translate_lines("ps_heap_base_once", LINES)
        main = function_of(spv, "%main")
        self.assertTrue(main)
        self.assertEqual(spv.count("OpLoad %uint %ud_base_lo_p"), 1)
        self.assertEqual(spv.count("OpLoad %uint %ud_base_hi_p"), 1)
        uses = re.findall(r"= OpIAdd %ulong %heap_base ", spv)
        self.assertGreaterEqual(len(uses), 4, "every access adds to the one base")
        # Defined in the entry block, which dominates every block of %main: after its first OpLabel, before the second.
        blocks = main.split("OpLabel")
        self.assertIn("%heap_base = OpBitwiseOr %ulong", blocks[1])


if __name__ == "__main__":
    unittest.main()
