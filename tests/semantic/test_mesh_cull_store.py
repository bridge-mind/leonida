"""SHADER-575: a mesh shader's CullPrimitiveEXT is written as a constant on each side of a branch.

llvmpipe (Mesa 25.2.8, LLVM 20) aborts compiling the wave64-emulated mesh epilogue when the per-primitive cull bool is
stored as a computed value ("Cannot emit physreg copy instruction"), and Mesa 26.3.1 drew no primitives for the same
shaders. Storing %true or %false behind a branch on the same bool means the same on every driver and compiles there.

The program is runtime/test/shaders/ms_tri.asm, the mesh GPU test (128 threads, 64/64 outputs, triangles), translated
natively and with --wave64-emulate. Standard library only; the spirv-val check needs SPIRV-Tools.
"""
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import support  # noqa: E402  (path set above)
import rdna2spirv  # noqa: E402

MS_TRI = os.path.join(support.ROOT, "runtime", "test", "shaders", "ms_tri.asm")


def _dxbc(chunks):
    """A DXBC container with the given (fourcc, payload) parts."""
    hdr = 32 + 4 * len(chunks)
    offsets, body = [], bytearray()
    for tag, payload in chunks:
        offsets.append(hdr + len(body))
        body += tag + struct.pack("<I", len(payload)) + payload
    out = bytearray(b"DXBC") + bytes(16) + struct.pack("<III", 1, hdr + len(body), len(chunks))
    for off in offsets:
        out += struct.pack("<I", off)
    return bytes(out) + bytes(body)


def _mesh_psv0(threads, max_vertices, max_primitives):
    """PSV0 of a triangle mesh shader (stage 13)."""
    r = bytearray(48)
    struct.pack_into("<HH", r, 12, max_vertices, max_primitives)
    r[24] = 13
    r[27] = 2
    struct.pack_into("<III", r, 36, threads, 1, 1)
    return struct.pack("<I", len(r)) + bytes(r)


def _translate_ms_tri(emulate):
    with open(MS_TRI, encoding="utf-8") as f:
        lines = [ln.split(";")[0].strip() for ln in f]
    lines = [ln for ln in lines if ln]
    old = rdna2spirv.WAVE64_EMULATE
    rdna2spirv.WAVE64_EMULATE = emulate
    try:
        spv = support.translate_lines("ms_tri_cull", lines, dxbc=_dxbc([(b"PSV0", _mesh_psv0(128, 64, 64))]))
        flow = rdna2spirv.translate.last_gen.control_flow
    finally:
        rdna2spirv.WAVE64_EMULATE = old
    return spv, flow


def _cull_stores(spv):
    """The values stored through pointers into %ms_cull."""
    ptrs = set(re.findall(r"(%\S+) = OpAccessChain %\S+ %ms_cull ", spv))
    return [m.group(2) for m in re.finditer(r"OpStore (%\S+) (%\S+)", spv) if m.group(1) in ptrs]


class MeshCullStoreTests(unittest.TestCase):
    def test_cull_flag_stored_as_constants(self):
        for emulate, flow in ((False, "straight line"), (True, "emulated wave64 rounds")):
            with self.subTest(wave64_emulate=emulate):
                spv, got_flow = _translate_ms_tri(emulate)
                self.assertEqual(got_flow, flow)
                self.assertIn("OpMemberDecorate %ms_cull_blk 0 BuiltIn CullPrimitiveEXT", spv)
                self.assertEqual(sorted(_cull_stores(spv)), ["%false", "%true"])
                self.assertIn("OpBranchConditional %mnull %ms_cull1 %ms_cull0", spv)

    @unittest.skipUnless(shutil.which("spirv-as") and shutil.which("spirv-val"), "needs spirv-as and spirv-val")
    def test_assembles_and_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            for emulate in (False, True):
                with self.subTest(wave64_emulate=emulate):
                    spv, _ = _translate_ms_tri(emulate)
                    asm, out = os.path.join(tmp, "ms.spvasm"), os.path.join(tmp, "ms.spv")
                    with open(asm, "w", encoding="utf-8", newline="\n") as f:
                        f.write(spv)
                    for cmd in (["spirv-as", "--target-env", "vulkan1.2", asm, "-o", out],
                                ["spirv-val", "--target-env", "vulkan1.2", out]):
                        r = subprocess.run(cmd, capture_output=True, text=True)
                        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
