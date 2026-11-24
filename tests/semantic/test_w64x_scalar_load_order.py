"""--wave64-emulate: a scalar load (s_load / s_buffer_load) is ordered with the wave's vector stores (#1587).

On hardware a wave issues its instructions in order, so every lane's `buffer_store` is done (`s_waitcnt vmcnt(0)`)
before the wave's `s_load` of the same address. Under `--wave64-emulate` the two halves of a wave can run in two
32-wide subgroups, which order memory only at the barrier between rounds (docs/design.md, "Wave64 emulation"). Until
#1587 a scalar load was not one of the accesses `Gen.x_memory_order` cuts a round at, so a store and a later scalar
load of the same wave sat in one segment, and the other half could load before this half had stored:
`tools/valu_fuzz.py --family mem --emulate` found lanes 32-63 reading the buffer's fill pattern after an
`s_buffer_load` / `s_load` of the slot lane 0 had just written (the slot comes from `v_readfirstlane_b32`, which
the emulation already reads over the whole wave).

The interpreters here don't run workgroup barriers, so this checks the rule on the emitted rounds: the SPIR-V of every
segment (a case of the round loop's `OpSwitch %pcv`) may not access buffer memory (the PhysicalStorageBuffer pointers
both the vector and the scalar memory ops go through) in a conflicting order: no access after a write, no write after
a read. The programs are written here; no game data.
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

rdna2spirv = support.rdna2spirv

# s[0:3] is the output V# (4 user SGPRs); s[8:9] a 64-bit pointer to the same memory, s[12:15] a V# of it for
# s_buffer_load. Every access is one dword, so one instruction is one OpLoad or OpStore.
HEAD = ["s_mov_b64 exec, -1", "s_mov_b32 s8, 0", "s_mov_b32 s9, 0",
        "s_mov_b32 s12, 0", "s_mov_b32 s13, 0", "s_mov_b32 s14, 0x1000", "s_mov_b32 s15, 0x31016FAC",
        "v_lshlrev_b32_e32 v1, 2, v0",                   # v1 = the lane's byte offset
        # as in the fuzz program, before any memory op: s16 = the byte offset of the wave's first lane
        "v_readfirstlane_b32 s16, v1"]
STORE = ["buffer_store_dword v0, v1, s[0:3], 0 offen", "s_waitcnt vmcnt(0)"]
SLOAD = ["s_load_dword s10, s[8:9], 0x0", "s_waitcnt lgkmcnt(0)", "v_mov_b32_e32 v2, s10"]
# The fuzz program's shape: the slot of the wave's first lane, from v_readfirstlane_b32, as the soffset.
SBUF = ["s_buffer_load_dword s10, s[12:15], s16 offset:0x0", "s_waitcnt lgkmcnt(0)",
        "v_mov_b32_e32 v2, s10"]
TAIL = ["buffer_store_dword v2, v1, s[0:3], 0 offen offset:256", "s_endpgm"]


def fssm(threads=64):
    """Compute header: COMPUTE_PGM_RSRC2 0x8088 (4 user SGPRs, TGID_X), one wave of `threads`."""
    return bytes(0x138) + struct.pack("<4I", 0x8088, threads, 1, 1)


def translate(lines, emulate=True):
    prev = rdna2spirv.WAVE64_EMULATE
    rdna2spirv.WAVE64_EMULATE = emulate
    try:
        spv = support.translate_lines("cs_w64x_sload_order", lines, fssm=fssm())
    finally:
        rdna2spirv.WAVE64_EMULATE = prev
    assert rdna2spirv.translate.last_gen.wave64, "the test program must translate as wave64"
    return spv


def segments(spv):
    """{segment label: ["r"/"w", ...]}: the buffer accesses of each case of the round loop's switch, in order."""
    lines = spv.splitlines()
    sw = next((ln.split() for ln in lines if ln.split()[:2] == ["OpSwitch", "%pcv"]), None)
    assert sw is not None, "no round-loop switch: not an emulated translation"
    labels = set(sw[4::2])
    psb = set()
    out, cur = {}, None
    for ln in lines:
        t = ln.split()
        if len(t) >= 3 and t[1] == "=" and t[2] == "OpLabel":
            if t[0] in labels:
                cur = out.setdefault(t[0], [])
            elif t[0] == sw[3]:                        # the default label ends the cases
                cur = None
            continue
        if cur is None:
            continue
        if len(t) >= 4 and t[2] == "OpAccessChain" and t[3] == "%_ptr_PSB_uint":
            psb.add(t[0])
        elif len(t) >= 5 and t[2] == "OpLoad" and t[4] in psb:
            cur.append("r")
        elif len(t) >= 2 and t[0] == "OpStore" and t[1] in psb:
            cur.append("w")
    return out


def conflicts(segs):
    """The segments where an access follows a write, or a write follows a read (x_memory_order's rule)."""
    bad = []
    for lab, acc in segs.items():
        s = "".join(acc)
        if re.search(r"w.", s) or re.search(r"r.*w", s):
            bad.append((lab, s))
    return bad


class ScalarLoadOrderTests(unittest.TestCase):
    def check_ordered(self, lines, accesses):
        segs = segments(translate(lines))
        self.assertEqual(sum(len(a) for a in segs.values()), accesses, segs)   # every access was found
        self.assertEqual(conflicts(segs), [], segs)
        return segs

    def test_store_then_s_load(self):
        # read after write: the store's round ends before the load
        self.check_ordered(HEAD + STORE + SLOAD + TAIL, 3)

    def test_store_then_s_buffer_load_at_the_first_lane_slot(self):
        # the #1587 program shape: buffer_store, then v_readfirstlane_b32 + s_buffer_load
        self.check_ordered(HEAD + STORE + SBUF + TAIL, 3)

    def test_s_load_then_store(self):
        # write after read: the other half's store may not land before this half's scalar load
        self.check_ordered(HEAD + SLOAD + STORE + ["s_endpgm"], 2)

    def test_reads_share_a_round(self):
        # loads alone never conflict: two scalar loads and a vector load stay in one segment
        lines = HEAD + SLOAD + ["s_load_dword s11, s[8:9], 0x4", "s_waitcnt lgkmcnt(0)",
                                "buffer_load_dword v3, v1, s[0:3], 0 offen", "s_waitcnt vmcnt(0)", "s_endpgm"]
        segs = self.check_ordered(lines, 3)
        self.assertIn(["r", "r", "r"], segs.values(), segs)

    def test_native_wave64_is_unchanged(self):
        # without the flag there are no rounds: the store and the load are one straight run
        spv = translate(HEAD + STORE + SLOAD + TAIL, emulate=False)
        self.assertNotIn("%pcv", spv)
        self.assertNotIn("OpControlBarrier", spv)


if __name__ == "__main__":
    unittest.main()
