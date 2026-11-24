"""Translator + structured-CFG tests for the 6 remaining 32-bit RDNA2 image atomics:
  image_atomic_swap, image_atomic_inc, image_atomic_dec,
  image_atomic_fmin, image_atomic_fmax, image_atomic_fcmpswap.

Checks 2D and 3D storage image emission, a16 packed coordinates, glc return vs non-glc, OpAtomicLoad (not OpLoad)
on Image pointers, CAS-loop and selection-merge structure via spirv_mini_interp.check_structure, absence of
AtomicFloat capabilities, and bad-operand/dimension/d16 rejections. Standard library only; no GPU, no LLVM, no game
files.
"""
import os
import re
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x88, 32, 1, 1)  # 4 user SGPRs, 32x1x1
FSSM = bytes(FSSM)


def translate_cs(body):
    with tempfile.TemporaryDirectory() as tmp:
        for d in ("disasm", "fssm"):
            os.mkdir(os.path.join(tmp, d))
        name = "cs_img_atomics_32"
        with open(os.path.join(tmp, f"{name}.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", f"{name}.fssm"), "wb") as f:
            f.write(FSSM)
        src = os.path.join(tmp, "disasm", f"{name}.s")
        with open(src, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"; {name}\n" + "".join(f"\t{ln}\n" for ln in body) + "\ts_endpgm\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            return rdna2spirv.translate(src)


def check_cfg(spv):
    """Parse the SPIR-V function body and validate its structured CFG with spirv_mini_interp.check_structure."""
    body = []
    in_body = False
    for raw in spv.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = si._LINE.match(line)
        assert m, f"unparseable SPIR-V line: {line!r}"
        res, op, rest = m.groups()
        if in_body:
            body.append((res, op, rest.split(), line))
        elif op == "OpFunction":
            in_body = True
    assert body, "empty SPIR-V function body"
    si.check_structure(body)


class ImageAtomics32Tests(unittest.TestCase):
    def test_all_six_ops_2d_and_3d_glc_and_non_glc(self):
        cases = [
            ("image_atomic_swap", "v2", "0x1", ("OpAtomicExchange %uint",), ()),
            ("image_atomic_inc", "v2", "0x1",
             ("OpAtomicLoad %uint", "OpUGreaterThanEqual %bool", "OpAtomicCompareExchange %uint", "OpLoopMerge"), ()),
            ("image_atomic_dec", "v2", "0x1",
             ("OpAtomicLoad %uint", "OpUGreaterThan %bool", "OpAtomicCompareExchange %uint", "OpLoopMerge"), ()),
            ("image_atomic_fmin", "v2", "0x1",
             ("OpAtomicLoad %uint", "OpExtInst %float %glsl NMin", "OpAtomicCompareExchange %uint", "OpLoopMerge"), ()),
            ("image_atomic_fmax", "v2", "0x1",
             ("OpAtomicLoad %uint", "OpExtInst %float %glsl NMax", "OpAtomicCompareExchange %uint", "OpLoopMerge"), ()),
            ("image_atomic_fcmpswap", "v[2:3]", "0x3",
             ("OpAtomicLoad %uint", "OpUGreaterThan %bool", "OpAtomicCompareExchange %uint", "OpLoopMerge"),
             ("OpFOrdEqual",)),
        ]
        for op, vdata, dmask, must_have, must_not_have in cases:
            for dim, addr_op, vec_ty, img_var in (
                ("SQ_RSRC_IMG_2D", "v[4:5]", "%v2int", "%simages"),
                ("SQ_RSRC_IMG_3D", "v[4:6]", "%v3int", "%simages3d"),
            ):
                for glc in (False, True):
                    with self.subTest(op=op, dim=dim, glc=glc):
                        glc_s = " glc" if glc else ""
                        spv = translate_cs([
                            "v_mov_b32_e32 v2, 0x3f800000",
                            "v_mov_b32_e32 v3, 0x40000000",
                            "v_mov_b32_e32 v4, 1",
                            "v_mov_b32_e32 v5, 2",
                            "v_mov_b32_e32 v6, 3",
                            f"{op} {vdata}, {addr_op}, s[4:11] dmask:{dmask} dim:{dim} unorm{glc_s}",
                            "s_waitcnt vmcnt(0)",
                        ])
                        check_cfg(spv)
                        self.assertIn(f"OpCompositeConstruct {vec_ty}", spv)
                        self.assertIn(f"{img_var} ", spv)
                        self.assertIn("OpImageTexelPointer %_ptr_Image_uint", spv)
                        self.assertNotIn("AtomicFloat", spv)
                        # Image pointers must never be passed to plain OpLoad.
                        tp_m = re.search(r"(%\w+) = OpImageTexelPointer %_ptr_Image_uint", spv)
                        self.assertIsNotNone(tp_m)
                        self.assertNotIn(f"OpLoad %uint {tp_m.group(1)}", spv)
                        for tok in must_have:
                            self.assertIn(tok, spv)
                        for tok in must_not_have:
                            self.assertNotIn(tok, spv)
                        # Verify glc writes back to v2 while non-glc does not overwrite v2 after the atomic.
                        after_tp = spv[tp_m.start():]
                        # an inline chain to v2, or a store through its pointer hoisted into %entry (#4440)
                        vgpr2_stores = re.findall(r"OpAccessChain %_ptr_Function_uint %vgpr %c2\b|OpStore %reg_v2\b",
                                                  after_tp)
                        if glc:
                            self.assertGreaterEqual(len(vgpr2_stores), 1)
                        else:
                            self.assertEqual(len(vgpr2_stores), 0)

    def test_a16_packed_coordinates_2d_and_3d(self):
        spv_2d = translate_cs([
            "v_mov_b32_e32 v1, 0x00050003",  # (3, 5)
            "v_mov_b32_e32 v2, 7",
            "image_atomic_inc v2, v1, s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm a16 glc",
        ])
        check_cfg(spv_2d)
        self.assertIn("OpShiftRightLogical %uint", spv_2d)
        self.assertIn("OpCompositeConstruct %v2int", spv_2d)

        spv_3d = translate_cs([
            "v_mov_b32_e32 v1, 0x00050003",
            "v_mov_b32_e32 v2, 0x00000007",
            "v_mov_b32_e32 v4, 0x3f800000",
            "v_mov_b32_e32 v5, 0x40000000",
            "image_atomic_fcmpswap v[4:5], v[1:2], s[4:11] dmask:0x3 dim:SQ_RSRC_IMG_3D unorm a16 glc",
        ])
        check_cfg(spv_3d)
        self.assertIn("OpCompositeConstruct %v3int", spv_3d)
        self.assertIn("OpUGreaterThan %bool", spv_3d)
        self.assertNotIn("OpFOrdEqual", spv_3d)

    def test_rejected_forms(self):
        for line in (
            "image_atomic_swap v[2:3], v[4:5], s[4:11] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm",       # 64-bit swap
            "image_atomic_inc v[2:3], v[4:5], s[4:11] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm",        # 64-bit inc
            "image_atomic_fcmpswap v2, v[4:5], s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm",       # 1-VGPR fcmpswap
            "image_atomic_fmin v2, v4, s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm",               # too few 2D coords
            "image_atomic_fmax v2, v[4:5], s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_1D unorm",           # unsupported 1D dim
            "image_atomic_swap v2, v[4:5], s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm d16",       # unsupported d16
        ):
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate_cs([
                        "v_mov_b32_e32 v2, 0", "v_mov_b32_e32 v3, 0",
                        "v_mov_b32_e32 v4, 0", "v_mov_b32_e32 v5, 0",
                        line,
                    ])


if __name__ == "__main__":
    unittest.main()
