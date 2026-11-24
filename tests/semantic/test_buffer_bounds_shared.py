"""A multi-dword MUBUF access builds each bounds check once.

buffer_address(parts=True) computes dword 0's check as `in_b`; the raw loads and stores used to ignore it and build the
same check again through parts(0), and each check added `offset + end` once for mode 0 and once for mode 3 when they
share the offset. The values are the same expressions, so the generated module simply must not contain the same
ulong sum twice. The OOB results themselves are covered by test_buffer_oob_select.py and test_vector_mem_mubuf.py.
Synthetic only.

    python -I -m unittest discover -s tests/semantic -p test_buffer_bounds_shared.py -v
"""
import collections
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import support  # noqa: E402

# The V# comes from user SGPRs, so its OOB_SELECT is only known at run time and every mode's check is emitted.
ACCESSES = {
    "load x4": ["buffer_load_dwordx4 v[5:8], v1, s[4:7], 0 offen", "exp mrt0 v5, v6, v7, v8 done vm"],
    "load x2 with soffset": ["buffer_load_dwordx2 v[5:6], v1, s[4:7], s9 offen offset:16",
                             "exp mrt0 v5, v6, v6, v6 done vm"],
    "store x4": ["buffer_store_dwordx4 v[5:8], v1, s[4:7], 0 offen", "exp mrt0 v5, v5, v5, v5 done vm"],
}


def ulong_sums(spvasm):
    return collections.Counter(m.group(1) for m in re.finditer(r"= OpIAdd %ulong (%\S+ %\S+)$", spvasm, re.M))


class SharedBoundsChecksTest(unittest.TestCase):
    def test_no_bounds_sum_is_built_twice(self):
        for name, body in ACCESSES.items():
            with self.subTest(access=name):
                lines = ["v_mov_b32 v1, 16", "v_mov_b32 v5, 1.0", "v_mov_b32 v6, 2.0", "v_mov_b32 v7, 3.0",
                         "v_mov_b32 v8, 4.0"] + body + ["s_endpgm"]
                spv = support.translate_lines("ps_buffer_bounds_shared", lines)
                twice = {k: n for k, n in ulong_sums(spv).items() if n > 1}
                self.assertEqual(twice, {}, f"{name}: the same offset + end built more than once")


if __name__ == "__main__":
    unittest.main()
