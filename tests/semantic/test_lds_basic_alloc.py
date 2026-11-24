"""How big the translator makes a compute shader's LDS, when its accesses are bounds-checked, and the wrap.

Standard library only; no GPU, no LLVM, no game files.

COMPUTE_PGM_RSRC2 bits [23:15] (LDS_SIZE) declare the group-shared size in 512-byte units (128 dwords). Where the
translator puts the Workgroup array `%lds_arr`:
  * every constant `offset:` (bytes) or offset0/offset1 (4-byte units, x64 for st64, 8-byte units for b64) reaches
    some number of dwords from address 0 (offset / 4 + 1 for a dword access); while all of them stay inside the
    declared size, the array is a power of two of at least 16 dwords covering the declared size, capped at 16384
    (64 KiB), and a dynamic address wraps at its size (dword (A / 4) mod size);
  * a constant offset past the declared size (SHADER-309; RDNA2 LDS does not wrap, ISA 3.6.1/3.6.6) makes the array
    exactly the declared size and every access bounds-checked: out of range reads 0 and writes are dropped;
  * a declared size over 64 KiB, a thread group of more than 1024 invocations and TG_SIZE_EN (RSRC2 bit 10) are
    refused.
test_cs_shared_memory_limit.py runs the bounds-checked reads, writes and atomics; this file pins the array size and
which of the two forms is chosen.

Each case translates a hand-written listing with tools/rdna2spirv.py (support.translate_lines) and reads the array
length from the SPIR-V assembly; the wrap cases also run the shader in tests/semantic/spirv_mini_interp.py. The sizes
are worked out by hand from the rules above. Until this file, the LDS tests of this name ran a self-built
interpreter that copied the older rule (offsets grow the array) instead of calling the translator.
"""
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lds_run as L  # noqa: E402  (also puts tools/ on sys.path)
import rdna2spirv  # noqa: E402

LDS_SIZE_SHIFT = 15


def rsrc2(units, extra=0):
    """4 user SGPRs, workgroup id in s4 (0x88) with `units` x 512 bytes of LDS."""
    return 0x88 | (units << LDS_SIZE_SHIFT) | extra


def translate(line, units, size=(32, 1, 1)):
    return L.translate(["v_mov_b32_e32 v1, 0", line], rsrc2(units), size)


def lds_dwords(line, units=0, size=(32, 1, 1)):
    spv = translate(line, units, size)
    m = re.search(r"%lds_arr = OpTypeArray %uint (%\S+)", spv)
    assert m, "no LDS array declared"
    c = re.search(re.escape(m.group(1)) + r" = OpConstant %uint (\d+)", spv)
    return int(c.group(1))


def bounds_checked(line, units=0):
    """True when an `OpULessThan %bool %idx <limit>` guards an access chain into %lds at the same %idx. An unrelated
    OpULessThan (a v_cmp_lt_u32, an EXEC path) does not satisfy it. (With nothing declared the array is padded to
    one dword and the limit is 0, so the limit is not compared with the array length.)"""
    spv = translate(line, units)
    for c in re.finditer(r"OpULessThan %bool (%\S+) %\S+\s", spv):
        if re.search(r"OpAccessChain %_ptr_WG_uint %lds " + re.escape(c.group(1)) + r"\s", spv):
            return True
    return False


READ = "ds_read_b32 v2, v1"


class TestAllocationSize(unittest.TestCase):
    def test_declared_size_by_hand(self):
        # (LDS_SIZE units, dwords): 128 per unit rounded up to a power of two, 16 .. 16384.
        for units, want in ((0, 16), (1, 128), (2, 256), (3, 512), (5, 1024), (8, 1024), (9, 2048), (32, 4096),
                            (33, 8192), (64, 8192), (65, 16384), (128, 16384)):
            with self.subTest(units=units):
                self.assertEqual(lds_dwords(READ, units), want)
                self.assertFalse(bounds_checked(READ, units))

    def test_offset_inside_the_declaration_changes_nothing(self):
        # 1 unit = 128 dwords = 512 bytes: the last dword is at byte 0x1FC.
        for off in (0, 0x40, 0x1FC):
            with self.subTest(offset=hex(off)):
                self.assertEqual(lds_dwords(f"ds_read_b32 v2, v1 offset:{off:#x}", 1), 128)
                self.assertFalse(bounds_checked(f"ds_read_b32 v2, v1 offset:{off:#x}", 1))

    def test_offset_past_the_declaration_bounds_checks_at_exactly_the_declared_size(self):
        # byte 0x200 is dword 128: one past the end of a 1-unit allocation. A 3-unit one is 384 dwords (not a power of
        # two): the array is 384 then, not rounded up to 512.
        for units, off, want in ((1, 0x200, 128), (1, 0x1000, 128), (3, 0x600, 384), (2, 0x400, 256)):
            line = f"ds_read_b32 v2, v1 offset:{off:#x}"
            with self.subTest(units=units, offset=hex(off)):
                self.assertEqual(lds_dwords(line, units), want)
                self.assertTrue(bounds_checked(line, units))

    def test_nothing_declared_and_an_offset_gives_an_empty_checked_array(self):
        line = "ds_read_b32 v2, v1 offset:0x100"
        self.assertTrue(bounds_checked(line, 0))
        self.assertEqual(lds_dwords(line, 0), 1)       # a 0-length array is not legal SPIR-V; every access fails the check

    def test_two_address_and_wide_offsets_count_their_width(self):
        # 1 unit = 128 dwords. read2_b32 offset1:127 ends at dword 127 (fits); 128 does not. st64: 64 dwords per unit,
        # so offset1:1 ends at dword 64 (fits) and offset1:2 at dword 128 (does not). b64 elements are two dwords.
        for line, over in (("ds_read2_b32 v[2:3], v1 offset1:127", False), ("ds_read2_b32 v[2:3], v1 offset1:128", True),
                           ("ds_read2st64_b32 v[2:3], v1 offset1:1", False),
                           ("ds_read2st64_b32 v[2:3], v1 offset1:2", True),
                           ("ds_read2_b64 v[2:5], v1 offset1:63", False), ("ds_read2_b64 v[2:5], v1 offset1:64", True),
                           ("ds_read_b128 v[2:5], v1 offset:0x1F0", False), ("ds_read_b128 v[2:5], v1 offset:0x1F4", True),
                           ("ds_read_u8 v2, v1 offset:0x1FF", False), ("ds_read_u8 v2, v1 offset:0x200", True)):
            with self.subTest(line=line):
                self.assertEqual(bounds_checked(line, 1), over)

    def test_non_ds_listing_declares_none(self):
        self.assertNotIn("%lds_arr", L.translate(["v_mov_b32_e32 v1, 0"], rsrc2(8)))

    def test_declared_size_over_64k_is_refused(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            translate(READ, 129)           # 129 x 512 B = 66048 B
        self.assertEqual(lds_dwords(READ, 128), 16384)


class TestRefusals(unittest.TestCase):
    def test_tg_size_en_refused(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            L.translate([READ], rsrc2(8, extra=1 << 10))

    def test_group_over_1024_refused(self):
        for size in ((1025, 1, 1), (32, 33, 1), (16, 16, 5)):
            with self.subTest(size=size), self.assertRaises(rdna2spirv.Unsupported):
                L.translate([READ], rsrc2(8), size)

    def test_group_of_1024_accepted(self):
        for size in ((1024, 1, 1), (32, 32, 1), (8, 8, 16)):
            with self.subTest(size=size):
                self.assertIn("%lds_arr", L.translate([READ], rsrc2(8), size))


class TestWrap(unittest.TestCase):
    # These pin the translator's current approximation of a dynamic address (dword (A / 4) mod array size). Real
    # RDNA2 LDS does not wrap (ISA 3.6.1/3.6.6), so whoever fixes the wrap should update these, not treat the
    # failure as a regression.
    def wrap_case(self, units, write_addr, read_addr):
        """Lane i writes 0x900 + i at write_addr + 4 i, then reads read_addr + 4 i (addresses held in VGPRs)."""
        body = ["v_lshlrev_b32_e32 v3, 2, v0", "v_add_nc_u32_e32 v1, %#x, v3" % write_addr,
                "v_add_nc_u32_e32 v2, 0x900, v0", "ds_write_b32 v1, v2", "s_waitcnt lgkmcnt(0)",
                "v_add_nc_u32_e32 v4, %#x, v3" % read_addr, "ds_read_b32 v5, v4", "s_waitcnt lgkmcnt(0)"] \
               + L.store("v5", 0)
        mem = L.run(body, rsrc2=rsrc2(units))
        return [L.out(mem, i, 0) for i in range(32)]

    def test_dynamic_address_wraps_at_the_array(self):
        # 1 unit = 128 dwords = 512 bytes: byte 0x200 is byte 0, in either direction.
        self.assertEqual(self.wrap_case(1, 0x200, 0), [0x900 + i for i in range(32)])
        self.assertEqual(self.wrap_case(1, 0, 0x400), [0x900 + i for i in range(32)])
        self.assertEqual(self.wrap_case(1, 0x100, 0x100), [0x900 + i for i in range(32)])     # no wrap: the same word
        self.assertEqual(self.wrap_case(1, 0x80, 0x100), [0] * 32)                            # a different word

    def test_wraps_at_the_power_of_two_above_the_declaration(self):
        # 3 units = 384 dwords, array 512 dwords = 2 KiB: byte 0x800 is byte 0, byte 0x600 (dword 384) is not byte 0.
        self.assertEqual(self.wrap_case(3, 0x800, 0), [0x900 + i for i in range(32)])
        self.assertEqual(self.wrap_case(3, 0x600, 0), [0] * 32)

    def test_wraps_at_the_cap(self):
        self.assertEqual(self.wrap_case(128, 0x10000, 0), [0x900 + i for i in range(32)])


if __name__ == "__main__":
    unittest.main()
