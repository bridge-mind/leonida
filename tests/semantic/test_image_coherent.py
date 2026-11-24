"""Coherent image accesses: glc on image_load / image_store (docs/design.md "Coherent image accesses").

On GFX10 a device-scope (coherent) load is glc + dlc and a device-scope store glc: Mesa ac_get_hw_cache_flags
(src/amd/common/ac_shader_util.c) sets exactly these bits for ACCESS_COHERENT. The game's POPS pixel shaders read
and write their ordered images that way, a read-modify-write that relies on seeing the previous fragment's store.
A read through the sampled array (SHADER-151's path for every other image_load) has no coherent form in SPIR-V, and
the fragment-shader interlock only makes Coherent memory visible between critical sections (Vulkan memory model:
non-atomic accesses are private unless Coherent). So a glc image_load reads the storage view the stores write
(OpImageRead), and a shader with coherent image accesses decorates its storage views Coherent.

These tests translate small pixel shaders with the real translator (tests/semantic/support.py) and check the SPIR-V:
which variable and component type every read and write uses, the decorations, capabilities and index features,
that plain accesses keep the sampled path, and that a coherent read the translator cannot model is refused.
"""
import os
import re
import shutil
import tempfile
import unittest

import support
import rdna2spirv

EPILOGUE = ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"]
# dim -> (address registers, STORE_DIMS index)
DIMS = {"2D": ("v[4:5]", 0), "2D_ARRAY": ("v[4:6]", 1), "CUBE": ("v[4:6]", 1),
        "3D": ("v[4:6]", 2), "1D": ("v4", 3)}
CLASS_TYPES = {0: "%v4float", 1: "%v4uint_st", 2: "%v4int_st"}       # storage view class -> texel type


def translate(lines):
    return support.translate_lines("ps_coherent", ["v_mov_b32 v4, 1", "v_mov_b32 v5, 2", "v_mov_b32 v6, 0"]
                                   + lines + EPILOGUE)


def rmw(dim="2D", load_mods="glc dlc", store_mods="glc", dmask=0xF):
    """The POPS pattern: read a texel, add 1.0 to red, write it back to the same T#."""
    addr = DIMS[dim][0]
    return translate([f"image_load v[0:3], {addr}, s[0:7] dmask:0x{dmask:x} dim:SQ_RSRC_IMG_{dim} unorm {load_mods}",
                      "s_waitcnt vmcnt(0)", "v_add_f32_e32 v0, 1.0, v0",
                      f"image_store v[0:3], {addr}, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_{dim} unorm {store_mods}"])


class Module:
    def __init__(self, text):
        self.text = text
        self.defs = {}
        self.ops = []
        for line in text.splitlines():
            m = re.match(r"\s*(%\w+) = (Op\w+)\s*(.*)$", line)
            if m:
                self.defs[m.group(1)] = (m.group(2), m.group(3).split())
                self.ops.append((m.group(1), m.group(2), m.group(3).split()))
            else:
                w = line.split()
                if w:
                    self.ops.append((None, w[0], w[1:]))
        self.caps = {a[0] for _, op, a in self.ops if op == "OpCapability"}
        self.coherent = {a[0] for _, op, a in self.ops if op == "OpDecorate" and a[1:] == ["Coherent"]}

    def variable(self, image):
        """The bindless array an image id was loaded from (OpLoad of an OpAccessChain into it)."""
        op, a = self.defs[image]
        assert op == "OpLoad", (image, op)
        op, a = self.defs[a[1]]
        assert op == "OpAccessChain", (image, op)
        return a[1]

    def reads(self):
        """(result type, variable) of every OpImageRead, in program order."""
        return [(a[0], self.variable(a[1])) for _, op, a in self.ops if op == "OpImageRead"]

    def writes(self):
        """(texel type, variable) of every OpImageWrite."""
        out = []
        for _, op, a in self.ops:
            if op == "OpImageWrite":
                texel_op, texel = self.defs[a[2]]
                assert texel_op == "OpCompositeConstruct", texel_op
                out.append((texel[0], self.variable(a[0])))
        return out

    def chains_into(self, prefix):
        return [a[1] for _, op, a in self.ops if op == "OpAccessChain" and a[1].startswith(prefix)]


class CoherentReadModifyWrite(unittest.TestCase):
    def test_coherent_load_reads_the_storage_views_the_store_writes(self):
        m = Module(rmw())
        reads, writes = m.reads(), m.writes()
        # One read per numeric class, each from that class's storage view, with that class's texel type.
        self.assertEqual(sorted(reads), sorted((CLASS_TYPES[c], f"%stv{c}_0") for c in range(3)))
        self.assertEqual(sorted(writes), sorted((CLASS_TYPES[c], f"%stv{c}_0") for c in range(3)))
        self.assertNotIn("OpImageFetch", m.text)                 # nothing reads through the sampled arrays
        self.assertEqual(m.chains_into("%images") + m.chains_into("%itx"), [])
        # Every read comes before every write (the read-modify-write order).
        order = [op for _, op, _ in m.ops if op in ("OpImageRead", "OpImageWrite")]
        self.assertEqual(order, ["OpImageRead"] * 3 + ["OpImageWrite"] * 3)

    def test_storage_views_are_coherent_and_the_read_feature_is_declared(self):
        text = rmw()
        m = Module(text)
        self.assertEqual(m.coherent, {"%stv0_0", "%stv1_0", "%stv2_0"})
        self.assertLessEqual({"StorageImageReadWithoutFormat", "StorageImageWriteWithoutFormat"}, m.caps)
        features = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, text)["features"]
        self.assertIn("shaderStorageImageReadWithoutFormat", features)

    def test_every_dimension_and_d16(self):
        for dim, (_, di) in DIMS.items():
            with self.subTest(dim=dim):
                m = Module(rmw(dim))
                self.assertEqual({v for _, v in m.reads()}, {f"%stv{c}_{di}" for c in range(3)})
                self.assertEqual(m.coherent, {f"%stv{c}_{di}" for c in range(3)})
        # d16: the same storage reads, the texel packed two halves per VGPR (as on the sampled path)
        m = Module(translate(["image_load v[0:1], v[4:5], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D unorm glc dlc d16"]))
        self.assertEqual({v for _, v in m.reads()}, {"%stv0_0", "%stv1_0", "%stv2_0"})
        self.assertNotIn("OpImageFetch", m.text)

    def test_glc_alone_is_coherent(self):
        # GFX10: glc without dlc is shader-array scope for a load; Coherent (device scope) covers it.
        m = Module(rmw(load_mods="glc"))
        self.assertEqual(len(m.reads()), 3)
        self.assertEqual(m.coherent, {"%stv0_0", "%stv1_0", "%stv2_0"})


class OnlyCoherentAccessesChange(unittest.TestCase):
    def test_a_coherent_store_alone_needs_no_read_feature(self):
        m = Module(translate(["image_store v[0:3], v[4:5], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D unorm glc"]))
        self.assertEqual(m.coherent, {"%stv0_0", "%stv1_0", "%stv2_0"})
        self.assertNotIn("StorageImageReadWithoutFormat", m.caps)
        self.assertEqual(m.reads(), [])

    def test_plain_load_and_store_keep_the_sampled_path(self):
        # SHADER-151: a load without glc reads the sampled arrays, so it needs no storage read feature.
        m = Module(rmw(load_mods="", store_mods=""))
        self.assertEqual(m.reads(), [])
        self.assertIn("OpImageFetch", m.text)
        self.assertEqual(m.coherent, set())
        self.assertNotIn("StorageImageReadWithoutFormat", m.caps)

    def test_dlc_alone_is_not_coherent(self):
        # GFX10: dlc without glc is a CU-scope load that bypasses GL1 (ac_get_hw_cache_flags), not device scope.
        m = Module(rmw(load_mods="dlc", store_mods=""))
        self.assertEqual(m.reads(), [])
        self.assertEqual(m.coherent, set())


class UnmodelledCoherentReads(unittest.TestCase):
    def check_refused(self, line):
        with self.assertRaises(rdna2spirv.Unsupported):
            translate([line])

    def test_a_coherent_sampled_read_is_refused(self):
        self.check_refused("image_sample v[0:3], v[4:5], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D glc")
        self.check_refused("image_gather4_lz v[0:3], v[4:5], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_2D glc")

    def test_a_coherent_load_without_a_storage_view_is_refused(self):
        self.check_refused("image_load v[0:3], v[4:6], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA unorm glc dlc")
        self.check_refused("image_load_mip v[0:3], v[4:6], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D unorm glc dlc")


@unittest.skipUnless(shutil.which("spirv-as") and shutil.which("spirv-val"), "needs spirv-as and spirv-val")
class Validation(unittest.TestCase):
    def test_spirv_val_accepts_every_dimension(self):
        with tempfile.TemporaryDirectory() as tmp:
            for dim in DIMS:
                with self.subTest(dim=dim):
                    sa, sv = os.path.join(tmp, f"{dim}.spvasm"), os.path.join(tmp, f"{dim}.spv")
                    with open(sa, "w", encoding="utf-8") as f:
                        f.write(rmw(dim))
                    result = rdna2spirv.assemble_spirv(sa, sv)
                    self.assertEqual(result.returncode, 0, result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
