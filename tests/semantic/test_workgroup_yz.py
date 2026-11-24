"""Semantic test for compute workgroup IDs Y/Z.

The contract row "Compute workgroup IDs Y/Z" in docs/design.md is checked only by the GPU
fixture ``runtime/test/shaders/cs_workgroup_yz.asm`` (plus an explicit EXEC set, which hardware
provides). This test runs the same hand-written listing through tools/rdna2spirv.py and
tests/semantic/spirv_mini_interp.py on a 2 x 3 x 4
dispatch: with COMPUTE_PGM_RSRC2 = 0x308 (4 user SGPRs, TGID_Y and TGID_Z enabled, TGID_X
disabled) the translator must put WorkgroupId.y in s4 and WorkgroupId.z in s5. The listing
clamps Y to 3 and Z to 4, packs the pair into a bin index and increments that bin with a
buffer atomic, so the expected counts come from a plain-Python model of the dispatch.

No game files, GPU, LLVM or SPIRV-Tools are needed; the listing and FSSM header are synthetic.
"""
import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
RSRC2_YZ = 0x308  # user SGPRs = 4; TGID_Y_EN (bit 8) and TGID_Z_EN (bit 9), not TGID_X_EN
FSSM_NAME = "cs_workgroup_yz"

# runtime/test/shaders/cs_workgroup_yz.asm, written out by hand so llvm-mc is not needed.
LISTING = [
    "s_mov_b64 exec, -1",  # hardware starts a wave with all lanes active
    "s_min_u32 s6, s4, 3",
    "s_min_u32 s7, s5, 4",
    "s_lshl_b32 s7, s7, 2",
    "s_add_u32 s6, s7, s6",
    "s_lshl_b32 s6, s6, 2",
    "v_mov_b32_e32 v0, s6",
    "v_mov_b32_e32 v1, 1",
    "buffer_atomic_add v1, v0, s[0:3], 0 offen",
    "s_endpgm",
]

_TMP = None
_SPV = None


def setUpModule():
    global _TMP, _SPV
    _TMP = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP.name, "disasm")
    fssm_dir = os.path.join(_TMP.name, "fssm")
    os.mkdir(disasm_dir)
    os.mkdir(fssm_dir)
    with open(os.path.join(disasm_dir, FSSM_NAME + ".s"), "w", encoding="utf-8") as f:
        f.write("; " + FSSM_NAME + "\n" + "".join("\t" + ln + "\n" for ln in LISTING))
    with open(os.path.join(_TMP.name, FSSM_NAME + ".dxbc"), "wb") as f:
        f.write(STUB_DXBC)
    with open(os.path.join(fssm_dir, FSSM_NAME + ".fssm"), "wb") as f:
        # 0x138 = COMPUTE_PGM_RSRC2, 0x13c/0x140/0x144 = thread-group size 1 x 1 x 1.
        f.write(bytes(0x138) + struct.pack("<4I", RSRC2_YZ, 1, 1, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        _SPV = rdna2spirv.translate(os.path.join(disasm_dir, FSSM_NAME + ".s"))


def tearDownModule():
    global _TMP
    if _TMP is not None:
        _TMP.cleanup()
        _TMP = None


def run_groups(groups, spv=None):
    """Run one invocation per (x, y, z) group and return the shared physical memory."""
    # V# for s[0:3]: base 0, NUM_RECORDS max, OOB_SELECT = 3 (RAW, so the byte offset is bounded by
    # NUM_RECORDS rather than by a zero stride; see docs/buffer_bounds.md).
    pc = [0, 0, 0xFFFFFFFF, 0x30000000] + [0] * 28
    return si.run_workgroup(
        _SPV if spv is None else spv,
        len(groups),
        32,
        inputs_for=lambda i: {"%wg_id": groups[i], "%local_id": (0, 0, 0), "%sg_lane": i},
        push_constants=pc,
        memory={},
    )


def bins(mem, count=20):
    return [mem.get(4 * i, 0) for i in range(count)]


class ComputeWorkgroupYZ(unittest.TestCase):
    def test_2x3x4_dispatch_counts_each_yz_bin(self):
        groups = [(x, y, z) for x in range(2) for y in range(3) for z in range(4)]
        expected = [0] * 20
        for _, y, z in groups:
            expected[y + 4 * z] += 1  # s4 = Y, s5 = Z; the listing packs Y + 4*Z
        got = bins(run_groups(groups))
        self.assertEqual(got, expected)
        # Bins the dispatch cannot reach (Y = 3, Z = 4) stay zero.
        self.assertEqual(sum(got[15:]), 0)

    def test_single_group_at_known_base(self):
        mem = run_groups([(1, 2, 3)])
        self.assertEqual(bins(mem)[14], 1, "group (y=2, z=3) must increment bin 2 + 4*3")
        self.assertEqual(sum(bins(mem)), 1)

    def test_clamped_overflow_group(self):
        # Y and Z outside the dispatch are clamped by s_min, so (7, 9) lands in bin 3 + 4*4.
        mem = run_groups([(0, 7, 9)])
        self.assertEqual(bins(mem)[19], 1)
        self.assertEqual(sum(bins(mem)), 1)

    def test_negative_control_swapped_yz_is_detected(self):
        """Swapping the emitted WorkgroupId.y/.z extracts must change the result."""
        y_line = "%wg1 = OpCompositeExtract %uint %wgi 1"
        z_line = "%wg2 = OpCompositeExtract %uint %wgi 2"
        self.assertIn(y_line, _SPV, "negative-control target must exist in the translated module")
        self.assertIn(z_line, _SPV, "negative-control target must exist in the translated module")
        bad = _SPV.replace(y_line, "%wg1 = OpCompositeExtract %uint %wgi 2")
        bad = bad.replace(z_line, "%wg2 = OpCompositeExtract %uint %wgi 1")

        groups = [(x, y, z) for x in range(2) for y in range(3) for z in range(4)]
        good_bins = bins(run_groups(groups))
        bad_bins = bins(run_groups(groups, spv=bad))
        self.assertNotEqual(bad_bins, good_bins, "swapping the Y/Z mapping went undetected")


if __name__ == "__main__":
    unittest.main()
