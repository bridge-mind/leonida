"""SHADER-309: a compute shader's Workgroup memory fits in maxComputeSharedMemorySize (#1721).

`--wave64-emulate` adds its own Workgroup buffer, `%x_buf` (2L + 6W + 7 dwords for L invocations in W waves), on top
of the shader's LDS array. With a 64 KiB LDS that made modules over the 64 KiB device limit, which `spirv-val` accepts
and RADV dies on (#1721: 16,385 uints = 65,540 B). The translator now counts both against
`rdna2spirv.MAX_SHARED_BYTES` (`--max-shared-memory`, default 65536, the hard cap): the LDS array, a power of two at
least the program's need, shrinks to the RSRC2 allocation when that still fits; otherwise the shader is refused
(`Unsupported`, with the byte counts). The device's own limit (32768 on the AMD Windows and Intel drivers) is the
runtime's check, against `lds_bytes`, which must be the whole total.

RDNA2 LDS does not wrap: an access outside the allocation reads 0 and its write is dropped (RDNA2 ISA 3.6.1, 3.6.6).
A shrunk array, or a program whose constant ds offsets reach past its RSRC2 allocation, gets bounds-checked LDS
accesses (Gen.lds_access); every other shader keeps the wrapped power-of-two array and emits no check. The
out-of-range cases run through spirv_mini_interp.

The Workgroup bytes are counted from the emitted SPIR-V. The programs are written here; no game data.
"""
import os
import re
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)  # python -I leaves the script's folder off sys.path

import support  # noqa: E402  (support puts tools/ on sys.path, honouring LEONIDA_TOOLS_DIR)
import spirv_mini_interp as si  # noqa: E402

rdna2spirv = support.rdna2spirv

# One wave of 64: write each lane's id to LDS, read the neighbour's dword back, store it to the output V# s[0:3].
PROGRAM = ["s_mov_b64 exec, -1",
           "v_lshlrev_b32_e32 v1, 2, v0",
           "v_readfirstlane_b32 s16, v1",
           "ds_write_b32 v1, v0",
           "s_waitcnt lgkmcnt(0)",
           "ds_read_b32 v2, v1 offset:4",
           "s_waitcnt lgkmcnt(0)",
           "buffer_store_dword v2, v1, s[0:3], 0 offen",
           "s_endpgm"]
X_BUF = 4 * (2 * 64 + 6 * 1 + 7)          # %x_buf for one wave of 64: 564 bytes


def fssm(lds_units, threads=64):
    """Compute header: COMPUTE_PGM_RSRC2 with 4 user SGPRs, TGID_X and LDS_SIZE (512-byte units) at [23:15]."""
    return bytes(0x138) + struct.pack("<4I", 0x88 | (lds_units << 15), threads, 1, 1)


def translate(lds_units, emulate=True, limit=None):
    prev = rdna2spirv.WAVE64_EMULATE, rdna2spirv.MAX_SHARED_BYTES
    rdna2spirv.WAVE64_EMULATE = emulate
    if limit is not None:
        rdna2spirv.MAX_SHARED_BYTES = limit
    try:
        spv = support.translate_lines("cs_shared_limit", PROGRAM, fssm=fssm(lds_units))
    finally:
        rdna2spirv.WAVE64_EMULATE, rdna2spirv.MAX_SHARED_BYTES = prev
    assert rdna2spirv.translate.last_gen.wave64, "the test program must translate as wave64"
    return spv


def workgroup_bytes(spv):
    """{variable: bytes} of every Workgroup-storage-class OpVariable, sized from its pointee type."""
    consts, types, ptrs, out = {}, {}, {}, {}
    for ln in spv.splitlines():
        t = ln.split()
        if len(t) < 3 or t[1] != "=":
            continue
        rid, op, args = t[0], t[2], t[3:]
        if op == "OpConstant":
            consts[rid] = int(args[1], 0)
        elif op == "OpTypeInt" or op == "OpTypeFloat":
            types[rid] = int(args[0]) // 8
        elif op == "OpTypeArray":
            types[rid] = types[args[0]] * consts[args[1]]
        elif op == "OpTypeStruct":
            types[rid] = sum(types[a] for a in args)
        elif op == "OpTypePointer" and args[0] == "Workgroup":
            ptrs[rid] = args[1]
        elif op == "OpVariable" and args[1:2] == ["Workgroup"]:
            out[rid] = types[ptrs[args[0]]]
    return out


def lds_index_ops(spv):
    """The op each LDS access wraps its dword index with (OpBitwiseAnd for a power of two, OpUMod when shrunk)."""
    return set(re.findall(r"= (OpBitwiseAnd|OpUMod) %uint %\w+ %\w+\n\s*%\w+ = OpAccessChain %_ptr_WG_uint\w* %lds ",
                          spv))


def bounds_checks(spv):
    """How many LDS accesses are bounds-checked (Gen.lds_access emits one `<id>_lin` in-range block per dword)."""
    return len(re.findall(r"^\s*%\w+_lin = OpLabel$", spv, re.M))


VSHARP = [0, 0, 0x1000, 0x31016FAC]       # s[0:3]: the output buffer V#


def run_one(lines, lds_units):
    """Translate a wave32 compute listing (one invocation) and run it; returns (spirv, {byte address: word})."""
    prev = rdna2spirv.WAVE64_EMULATE
    rdna2spirv.WAVE64_EMULATE = False
    try:
        spv = support.translate_lines("cs_lds_bounds", lines + ["s_endpgm"], fssm=fssm(lds_units, threads=32))
    finally:
        rdna2spirv.WAVE64_EMULATE = prev
    mem = si.run_workgroup(spv, 1, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0),
                                                             "%sg_lane": i},
                           push_constants=VSHARP + [0] * 28, memory={})
    return spv, mem


def out(*regs):
    """Store v<reg> to output word k for each k."""
    lines = []
    for k, r in enumerate(regs):
        lines += [f"v_mov_b32_e32 v31, {4 * k}", f"buffer_store_dword v{r}, v31, s[0:3], 0 offen"]
    return lines


class SharedMemoryLimitTests(unittest.TestCase):
    def test_under_limit_is_unchanged(self):
        # 32 KiB of LDS + the exchange: well under 64 KiB, the power-of-two array as before
        wg = workgroup_bytes(translate(64))
        self.assertEqual(wg, {"%lds": 32768, "%x_buf": X_BUF})
        self.assertLessEqual(sum(wg.values()), rdna2spirv.MAX_SHARED_BYTES)

    def test_full_64k_lds_is_refused(self):
        # #1721: 64 KiB of LDS leaves no room for the exchange, and the program needs all of it
        with self.assertRaises(rdna2spirv.Unsupported) as cm:
            translate(128)
        msg = str(cm.exception)
        self.assertIn(f"{65536 + X_BUF} B", msg)
        self.assertIn("LDS 65536 B", msg)
        self.assertIn(f"wave64 exchange {X_BUF} B", msg)
        self.assertIn("65536 B", msg)

    def test_rounded_up_lds_shrinks_to_the_allocation(self):
        # LDS_SIZE 100 units = 51,200 B rounds up to a 64 KiB array, over the cap with the exchange; the allocation
        # fits, so the array is the allocation and its accesses are bounds-checked (no mask, no wrap)
        spv = translate(100)
        self.assertEqual(workgroup_bytes(spv), {"%lds": 51200, "%x_buf": X_BUF})
        self.assertEqual(lds_index_ops(spv), set())
        self.assertEqual(bounds_checks(spv), 2)              # the ds_write and the ds_read

    def test_exact_limit_is_accepted_one_dword_less_is_refused(self):
        # 32 KiB of LDS (LDS_SIZE 64, all of it needed) + the exchange, against a limit at exactly that sum ...
        exact = 32768 + X_BUF
        spv = translate(64, limit=exact)
        self.assertEqual(sum(workgroup_bytes(spv).values()), exact)
        self.assertEqual(lds_index_ops(spv), {"OpBitwiseAnd"})
        self.assertEqual(bounds_checks(spv), 0)
        # ... and one dword under it: the need no longer fits, so there is nothing to shrink to
        with self.assertRaises(rdna2spirv.Unsupported):
            translate(64, limit=exact - 4)

    def test_metadata_lds_bytes_is_every_workgroup_byte(self):
        # index.json's lds_bytes is what the runtime compares with the device's maxComputeSharedMemorySize (32768 on
        # some drivers): it has to be the module's whole Workgroup total, the emulation's %x_buf included
        for units, emulate in ((64, True), (100, True), (128, False), (64, False)):
            with self.subTest(units=units, emulate=emulate):
                prev = rdna2spirv.WAVE64_EMULATE
                spv = translate(units, emulate=emulate)
                rdna2spirv.WAVE64_EMULATE = emulate
                try:
                    meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spv)
                finally:
                    rdna2spirv.WAVE64_EMULATE = prev
                self.assertEqual(meta["lds_bytes"], sum(workgroup_bytes(spv).values()))

    def test_native_allocation_over_64k_is_refused(self):
        # RSRC2 LDS_SIZE counts 512-byte units, so the smallest allocation above 64 KiB is 129 units = 66,048 B.
        # The array is clipped to 16,384 dwords; the allocation still counts, so the shader is refused, not cut.
        with self.assertRaises(rdna2spirv.Unsupported) as cm:
            translate(129, emulate=False)
        self.assertIn("LDS 66048 B", str(cm.exception))

    def test_constant_offset_at_the_end_of_64k_is_counted_exactly(self):
        # A b32 access at offset 65,532 ends at byte 65,536: the need is exactly 16,384 dwords (the estimate used
        # to add 4 dwords of slack and overcount it), so a 64 KiB allocation is in range and emits no check
        prog = [ln.replace("ds_write_b32 v1, v0", "ds_write_b32 v1, v0 offset:65532") for ln in PROGRAM]
        prev = rdna2spirv.WAVE64_EMULATE
        rdna2spirv.WAVE64_EMULATE = False
        try:
            spv = support.translate_lines("cs_shared_limit", prog, fssm=fssm(128))
        finally:
            rdna2spirv.WAVE64_EMULATE = prev
        self.assertEqual(rdna2spirv.translate.last_gen.lds_need, 16384)
        self.assertEqual(workgroup_bytes(spv), {"%lds": 65536})
        self.assertEqual(bounds_checks(spv), 0)

    def test_constant_offset_past_the_allocation_is_bounds_checked(self):
        # 512 B allocated (LDS_SIZE 1), a write at offset 65,524: on RDNA2 it is out of range. The array is the
        # allocation, every access is checked, and nothing wraps onto a valid word.
        prog = [ln.replace("ds_write_b32 v1, v0", "ds_write_b32 v1, v0 offset:65524") for ln in PROGRAM]
        prev = rdna2spirv.WAVE64_EMULATE
        rdna2spirv.WAVE64_EMULATE = False
        try:
            spv = support.translate_lines("cs_shared_limit", prog, fssm=fssm(1))
        finally:
            rdna2spirv.WAVE64_EMULATE = prev
        self.assertEqual(rdna2spirv.translate.last_gen.lds_need, 16382)
        self.assertEqual(workgroup_bytes(spv), {"%lds": 512})
        self.assertEqual(bounds_checks(spv), 2)
        self.assertEqual(lds_index_ops(spv), set())

    def test_out_of_range_write_is_dropped_not_aliased(self):
        # 512 B allocated. Word 0 holds a sentinel. A write at byte 65,536 (vaddr 0xFC00 + offset 1024) would land
        # on word 0 if the index wrapped; RDNA2 drops it. Reading the same address back gives 0.
        _, mem = run_one(["v_mov_b32_e32 v1, 0", "v_mov_b32_e32 v2, 0x5e17", "ds_write_b32 v1, v2",
                          "v_mov_b32_e32 v3, 0xfc00", "v_mov_b32_e32 v4, 0xbad",
                          "ds_write_b32 v3, v4 offset:1024", "s_waitcnt lgkmcnt(0)",
                          "v_mov_b32_e32 v5, 0xdead", "v_mov_b32_e32 v6, 0xdead",
                          "ds_read_b32 v5, v1", "ds_read_b32 v6, v3 offset:1024", "s_waitcnt lgkmcnt(0)"]
                         + out(5, 6), 1)
        self.assertEqual((mem.get(0), mem.get(4)), (0x5E17, 0))

    def test_out_of_range_read_returns_zero(self):
        # ds_read_b32 / ds_read_u16 past the allocation (offset 1024 > 512 B): 0, not the word the index wraps to
        _, mem = run_one(["v_mov_b32_e32 v1, 0", "v_mov_b32_e32 v2, 0x1234abcd", "ds_write_b32 v1, v2",
                          "s_waitcnt lgkmcnt(0)", "v_mov_b32_e32 v5, 0xdead", "v_mov_b32_e32 v6, 0xdead",
                          "ds_read_b32 v5, v1 offset:1024", "ds_read_u16 v6, v1 offset:1024",
                          "s_waitcnt lgkmcnt(0)"] + out(5, 6), 1)
        self.assertEqual((mem.get(0), mem.get(4)), (0, 0))

    def test_wide_access_is_checked_per_dword(self):
        # A b64 write at byte 508 of a 512-byte allocation: dword 127 is written, dword 128 is dropped; read back
        # as b64 it is (lo, 0). (offset:1024 elsewhere makes the shader bounds-checked.)
        _, mem = run_one(["v_mov_b32_e32 v1, 0x1fc", "v_mov_b32_e32 v2, 0x1111", "v_mov_b32_e32 v3, 0x2222",
                          "ds_write_b64 v1, v[2:3]", "s_waitcnt lgkmcnt(0)",
                          "v_mov_b32_e32 v5, 0xdead", "v_mov_b32_e32 v6, 0xdead", "ds_read_b64 v[5:6], v1",
                          "v_mov_b32_e32 v7, 0xdead", "v_mov_b32_e32 v8, 0", "ds_read_b32 v7, v8 offset:1024",
                          "s_waitcnt lgkmcnt(0)"] + out(5, 6, 7), 1)
        self.assertEqual((mem.get(0), mem.get(4), mem.get(8)), (0x1111, 0, 0))

    def test_out_of_range_atomic_returns_zero_and_changes_nothing(self):
        # ds_add_rtn_u32 / ds_cmpst_rtn_b32 past the allocation: the old value is 0 and word 0 (the wrap target)
        # keeps its sentinel
        _, mem = run_one(["v_mov_b32_e32 v1, 0", "v_mov_b32_e32 v2, 7", "ds_write_b32 v1, v2",
                          "s_waitcnt lgkmcnt(0)", "v_mov_b32_e32 v3, 0xfc00", "v_mov_b32_e32 v4, 5",
                          "v_mov_b32_e32 v5, 0xdead", "ds_add_rtn_u32 v5, v3, v4 offset:1024",
                          "v_mov_b32_e32 v6, 0xdead", "ds_cmpst_rtn_b32 v6, v3, v2, v4 offset:1024",
                          "s_waitcnt lgkmcnt(0)", "ds_read_b32 v7, v1", "s_waitcnt lgkmcnt(0)"] + out(5, 6, 7), 1)
        self.assertEqual((mem.get(0), mem.get(4), mem.get(8)), (0, 0, 7))

    def test_out_of_range_mskor_returns_zero_and_changes_nothing(self):
        # ds_mskor_rtn_b32 (two atomics on one dword) past the allocation: old value 0, the wrap target untouched
        _, mem = run_one(["v_mov_b32_e32 v1, 0", "v_mov_b32_e32 v2, 7", "ds_write_b32 v1, v2",
                          "s_waitcnt lgkmcnt(0)", "v_mov_b32_e32 v3, 0xfc00", "v_mov_b32_e32 v4, 0xff",
                          "v_mov_b32_e32 v5, 0xdead", "ds_mskor_rtn_b32 v5, v3, v4, v4 offset:1024",
                          "s_waitcnt lgkmcnt(0)", "ds_read_b32 v7, v1", "s_waitcnt lgkmcnt(0)"] + out(5, 7), 1)
        self.assertEqual((mem.get(0), mem.get(4)), (0, 7))

    def test_in_range_shader_emits_no_bounds_check(self):
        # The same accesses inside a 4 KiB allocation: the wrapped power-of-two path, no check (main's SPIR-V)
        spv, mem = run_one(["v_mov_b32_e32 v1, 0", "v_mov_b32_e32 v2, 0x5e17", "ds_write_b32 v1, v2",
                            "s_waitcnt lgkmcnt(0)", "ds_read_b32 v5, v1 offset:1020", "ds_read_b32 v6, v1",
                            "s_waitcnt lgkmcnt(0)"] + out(6), 8)
        self.assertEqual(bounds_checks(spv), 0)
        self.assertIsNone(rdna2spirv.translate.last_gen.lds_bound)
        self.assertEqual(lds_index_ops(spv), {"OpBitwiseAnd"})
        self.assertEqual(mem.get(0), 0x5E17)

    def test_native_wave64_full_64k_lds_still_translates(self):
        # without the flag there is no %x_buf: a 64 KiB LDS is exactly the limit
        wg = workgroup_bytes(translate(128, emulate=False))
        self.assertEqual(wg, {"%lds": 65536})


if __name__ == "__main__":
    unittest.main()
