"""image_store and coherent image_load (glc) on dim:SQ_RSRC_IMG_CUBE (RDNA2 ISA).

Cube storage images are bound as 6-layer 2D array views (runtime/src/vulkan_bindless.h, STORE_DIMS index 1:
bindings 11/12/13 for float/uint/sint, OpTypeImage ... 2D 0 1 0 2 Unknown, %v3int coordinates (x, y, face)),
matching non-coherent image_load on dim:SQ_RSRC_IMG_CUBE.

These tests verify:
- image_store and coherent image_load (glc / glc dlc) on dim:SQ_RSRC_IMG_CUBE across dmask subsets, a16, d16,
  unorm, and both CS and PS stages;
- llvm-mc -triple=amdgcn -mcpu=gfx1030 binary assemble -> disassemble round-trip into rdna2spirv;
- spirv-val --target-env vulkan1.2 validation of every generated SPIR-V module.
"""
import os
import re
import shlex
import shutil
import struct
import subprocess
import tempfile
import unittest

import support
import rdna2spirv

PS_EPILOGUE = ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"]
CS_EPILOGUE = ["s_endpgm"]
CLASS_TYPES = {0: "%v4float", 1: "%v4uint_st", 2: "%v4int_st"}
CUBE_BINDINGS = {"%stv0_1": 11, "%stv1_1": 12, "%stv2_1": 13}


def cs_fssm(threads=64):
    h = bytearray(0x148)
    struct.pack_into("<4I", h, 0x138, 0x8088, threads, 1, 1)
    return bytes(h)


def translate_stage(stage, lines):
    preamble = ["v_mov_b32 v0, 0x3f800000", "v_mov_b32 v1, 0x40000000",
                "v_mov_b32 v2, 0x40400000", "v_mov_b32 v3, 0x40800000",
                "v_mov_b32 v4, 1", "v_mov_b32 v5, 2", "v_mov_b32 v6, 5"]
    if stage == "cs":
        return support.translate_lines("cs_cube_store_glc", preamble + lines + CS_EPILOGUE, fssm=cs_fssm())
    return support.translate_lines("ps_cube_store_glc", preamble + lines + PS_EPILOGUE)


def vdata_reg(dmask, d16=False):
    n = bin(dmask).count("1")
    regs = (n + 1) // 2 if d16 else n
    return "v0" if regs == 1 else f"v[0:{regs - 1}]"


def llvm_mc_cmd():
    env = os.environ.get("LLVM_MC")
    if env:
        return shlex.split(env)
    exe = shutil.which("llvm-mc")
    return [exe] if exe else None


def validate_spirv(spvasm):
    with tempfile.TemporaryDirectory() as tmp:
        sa = os.path.join(tmp, "cube.spvasm")
        sv = os.path.join(tmp, "cube.spv")
        with open(sa, "w", encoding="utf-8", newline="\n") as f:
            f.write(spvasm)
        as_res = subprocess.run(["spirv-as", "--target-env", "vulkan1.2", sa, "-o", sv],
                                capture_output=True, text=True, check=False)
        if as_res.returncode != 0:
            raise AssertionError(f"spirv-as failed:\n{as_res.stderr}{as_res.stdout}")
        val_res = subprocess.run(["spirv-val", "--target-env", "vulkan1.2", sv],
                                 capture_output=True, text=True, check=False)
        if val_res.returncode != 0:
            raise AssertionError(f"spirv-val failed:\n{val_res.stderr}{val_res.stdout}")


class Module:
    def __init__(self, text):
        self.text = text
        self.defs = {}
        self.ops = []
        self.bindings = {}
        for line in text.splitlines():
            m = re.match(r"\s*(%\w+) = (Op\w+)\s*(.*)$", line)
            if m:
                self.defs[m.group(1)] = (m.group(2), m.group(3).split())
                self.ops.append((m.group(1), m.group(2), m.group(3).split()))
            else:
                w = line.split()
                if w:
                    self.ops.append((None, w[0], w[1:]))
                    if w[0] == "OpDecorate" and len(w) == 4 and w[2] == "Binding":
                        self.bindings[w[1]] = int(w[3])
        self.caps = {a[0] for _, op, a in self.ops if op == "OpCapability"}
        self.coherent = {a[0] for _, op, a in self.ops if op == "OpDecorate" and a[1:] == ["Coherent"]}

    def variable(self, image):
        op, a = self.defs[image]
        assert op == "OpLoad", (image, op)
        op, a = self.defs[a[1]]
        assert op == "OpAccessChain", (image, op)
        return a[1]

    def reads(self):
        return [(a[0], self.variable(a[1]), self.defs[a[2]][1][0]) for _, op, a in self.ops if op == "OpImageRead"]

    def writes(self):
        out = []
        for _, op, a in self.ops:
            if op == "OpImageWrite":
                texel_op, texel = self.defs[a[2]]
                assert texel_op == "OpCompositeConstruct", texel_op
                coord_op, coord = self.defs[a[1]]
                assert coord_op == "OpCompositeConstruct", coord_op
                out.append((texel[0], self.variable(a[0]), coord[0]))
        return out


class CubeImageStore(unittest.TestCase):
    def test_store_uses_2d_array_storage_views_and_v3int_coords(self):
        for stage in ("ps", "cs"):
            with self.subTest(stage=stage):
                m = Module(translate_stage(stage, [
                    "image_store v[0:3], v[4:6], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_CUBE unorm"
                ]))
                self.assertEqual(sorted(m.writes()), sorted((CLASS_TYPES[c], f"%stv{c}_1", "%v3int") for c in range(3)))
                for var, binding in CUBE_BINDINGS.items():
                    self.assertEqual(m.bindings.get(var), binding)
                for c, st in enumerate(("%float", "%uint", "%int")):
                    self.assertEqual(m.defs[f"%stimg{c}_1"], ("OpTypeImage", [st, "2D", "0", "1", "0", "2", "Unknown"]))
                self.assertIn("StorageImageWriteWithoutFormat", m.caps)
                self.assertNotIn("StorageImageReadWithoutFormat", m.caps)
                self.assertEqual(m.coherent, set())

    def test_store_glc_marks_cube_views_coherent(self):
        for stage in ("ps", "cs"):
            with self.subTest(stage=stage):
                m = Module(translate_stage(stage, [
                    "image_store v[0:3], v[4:6], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_CUBE unorm glc"
                ]))
                self.assertEqual(m.coherent, set(CUBE_BINDINGS))
                self.assertNotIn("StorageImageReadWithoutFormat", m.caps)

    def test_store_dmask_a16_d16_combinations(self):
        cases = [
            (0x1, False, False, "unorm"),
            (0x3, False, False, "unorm"),
            (0x7, False, False, "unorm"),
            (0x9, False, False, "unorm"),
            (0xF, True, False, "unorm"),
            (0xF, False, True, "unorm"),
            (0x5, True, True, "unorm"),
            (0xF, True, True, "unorm glc"),
            (0xF, False, False, ""),
        ]
        for dmask, a16, d16, extra in cases:
            addr = "v[4:5]" if a16 else "v[4:6]"
            mods = " ".join(x for x in (f"dmask:0x{dmask:x}", "dim:SQ_RSRC_IMG_CUBE", extra,
                                        "a16" if a16 else "", "d16" if d16 else "") if x)
            with self.subTest(dmask=hex(dmask), a16=a16, d16=d16, extra=extra):
                m = Module(translate_stage("cs", [f"image_store {vdata_reg(dmask, d16)}, {addr}, s[0:7] {mods}"]))
                self.assertEqual(sorted(m.writes()), sorted((CLASS_TYPES[c], f"%stv{c}_1", "%v3int") for c in range(3)))
                if d16:
                    self.assertIn("UnpackHalf2x16", m.text)
                    self.assertIn("OpBitFieldUExtract", m.text)
                    self.assertIn("OpBitFieldSExtract", m.text)

    def test_store_refuses_tfe_lwe_and_short_coords(self):
        bad = [
            "image_store v[0:3], v[4:6], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_CUBE unorm tfe",
            "image_store v[0:3], v[4:6], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_CUBE unorm lwe",
            "image_store v[0:3], v[4:5], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_CUBE unorm",
            "image_store v[0:3], v4, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_CUBE unorm a16",
        ]
        for line in bad:
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate_stage("cs", [line])


class CubeImageLoadCoherent(unittest.TestCase):
    def test_coherent_load_uses_2d_array_storage_views_and_v3int_coords(self):
        for stage in ("ps", "cs"):
            for mods in ("unorm glc", "unorm glc dlc"):
                with self.subTest(stage=stage, mods=mods):
                    text = translate_stage(stage, [
                        f"image_load v[0:3], v[4:6], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_CUBE {mods}",
                        "s_waitcnt vmcnt(0)",
                    ])
                    m = Module(text)
                    self.assertEqual(sorted(m.reads()),
                                     sorted((CLASS_TYPES[c], f"%stv{c}_1", "%v3int") for c in range(3)))
                    self.assertNotIn("OpImageFetch", m.text)
                    self.assertEqual(m.coherent, set(CUBE_BINDINGS))
                    self.assertIn("StorageImageReadWithoutFormat", m.caps)
                    features = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, text)["features"]
                    self.assertIn("shaderStorageImageReadWithoutFormat", features)

    def test_coherent_load_dmask_a16_d16_combinations(self):
        cases = [
            (0x1, False, False),
            (0x3, False, False),
            (0x7, False, False),
            (0x9, False, False),
            (0xF, True, False),
            (0xF, False, True),
            (0x5, True, True),
            (0xF, True, True),
        ]
        for dmask, a16, d16 in cases:
            addr = "v[4:5]" if a16 else "v[4:6]"
            mods = " ".join(x for x in (f"dmask:0x{dmask:x}", "dim:SQ_RSRC_IMG_CUBE", "unorm", "glc", "dlc",
                                        "a16" if a16 else "", "d16" if d16 else "") if x)
            with self.subTest(dmask=hex(dmask), a16=a16, d16=d16):
                m = Module(translate_stage("ps", [
                    f"image_load {vdata_reg(dmask, d16)}, {addr}, s[0:7] {mods}",
                    "s_waitcnt vmcnt(0)",
                ]))
                self.assertEqual(sorted(m.reads()), sorted((CLASS_TYPES[c], f"%stv{c}_1", "%v3int") for c in range(3)))
                self.assertNotIn("OpImageFetch", m.text)
                self.assertEqual(m.coherent, set(CUBE_BINDINGS))

    def test_coherent_rmw_shares_storage_views_in_order(self):
        for stage in ("ps", "cs"):
            with self.subTest(stage=stage):
                m = Module(translate_stage(stage, [
                    "image_load v[0:3], v[4:6], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_CUBE unorm glc dlc",
                    "s_waitcnt vmcnt(0)",
                    "v_add_f32_e32 v0, 1.0, v0",
                    "image_store v[0:3], v[4:6], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_CUBE unorm glc",
                ]))
                order = [op for _, op, _ in m.ops if op in ("OpImageRead", "OpImageWrite")]
                self.assertEqual(order, ["OpImageRead"] * 3 + ["OpImageWrite"] * 3)
                self.assertEqual({v for _, v, _ in m.reads()}, set(CUBE_BINDINGS))
                self.assertEqual({v for _, v, _ in m.writes()}, set(CUBE_BINDINGS))
                self.assertEqual(m.coherent, set(CUBE_BINDINGS))


@unittest.skipUnless(shutil.which("spirv-as") and shutil.which("spirv-val"), "needs spirv-as and spirv-val")
class SpirvValidation(unittest.TestCase):
    def test_spirv_val_vulkan12_accepts_all_cube_store_and_glc_variants(self):
        programs = [
            ("ps_store_f", "ps", ["image_store v[0:3], v[4:6], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_CUBE unorm"]),
            ("cs_store_f", "cs", ["image_store v[0:3], v[4:6], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_CUBE unorm"]),
            ("cs_store_a16", "cs", ["image_store v[0:3], v[4:5], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_CUBE unorm a16"]),
            ("cs_store_d16", "cs", ["image_store v[0:1], v[4:6], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_CUBE unorm d16"]),
            ("cs_store_a16_d16_glc", "cs",
             ["image_store v0, v[4:5], s[0:7] dmask:0x9 dim:SQ_RSRC_IMG_CUBE unorm glc a16 d16"]),
            ("ps_load_glc", "ps",
             ["image_load v[0:3], v[4:6], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_CUBE unorm glc dlc", "s_waitcnt vmcnt(0)"]),
            ("cs_load_glc_a16_d16", "cs",
             ["image_load v[0:1], v[4:5], s[0:7] dmask:0x7 dim:SQ_RSRC_IMG_CUBE unorm glc dlc a16 d16",
              "s_waitcnt vmcnt(0)"]),
            ("ps_rmw_glc", "ps", [
                "image_load v[0:3], v[4:6], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_CUBE unorm glc dlc",
                "s_waitcnt vmcnt(0)",
                "v_add_f32_e32 v0, 1.0, v0",
                "image_store v[0:3], v[4:6], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_CUBE unorm glc",
            ]),
            ("cs_rmw_glc_a16_d16", "cs", [
                "image_load v[0:1], v[4:5], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_CUBE unorm glc dlc a16 d16",
                "s_waitcnt vmcnt(0)",
                "image_store v[0:1], v[4:5], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_CUBE unorm glc a16 d16",
            ]),
        ]
        for name, stage, lines in programs:
            with self.subTest(name=name):
                validate_spirv(translate_stage(stage, lines))


@unittest.skipUnless(llvm_mc_cmd() is not None, "needs llvm-mc or LLVM_MC")
class LlvmMcRoundTrip(unittest.TestCase):
    def test_gfx1030_encode_decode_translate_and_validate(self):
        mc = llvm_mc_cmd()
        src_insns = [
            "image_store v[0:3], v[4:6], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_CUBE unorm",
            "image_store v[0:3], v[4:5], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_CUBE unorm a16",
            "image_store v[0:1], v[4:6], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_CUBE unorm d16",
            "image_store v0, v[4:5], s[0:7] dmask:0x9 dim:SQ_RSRC_IMG_CUBE unorm glc a16 d16",
            "image_load v[0:3], v[4:6], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_CUBE unorm glc dlc",
            "image_load v[0:1], v[4:5], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_CUBE unorm glc dlc a16 d16",
        ]
        enc = subprocess.run(mc + ["-triple=amdgcn", "-mcpu=gfx1030", "-show-encoding"],
                             input="\n".join(src_insns) + "\n", capture_output=True, text=True, check=False)
        self.assertEqual(enc.returncode, 0, enc.stderr + enc.stdout)
        byte_lines = re.findall(r"encoding:\s*\[([^\]]+)\]", enc.stdout)
        self.assertEqual(len(byte_lines), len(src_insns), enc.stdout)

        hex_input = "\n".join(b.replace(",", " ") for b in byte_lines) + "\n"
        dec = subprocess.run(mc + ["-triple=amdgcn", "-mcpu=gfx1030", "--disassemble"],
                             input=hex_input, capture_output=True, text=True, check=False)
        self.assertEqual(dec.returncode, 0, dec.stderr + dec.stdout)
        decoded = [ln.strip() for ln in dec.stdout.splitlines()
                   if ln.strip() and not ln.strip().startswith(".")]
        self.assertEqual(len(decoded), len(src_insns), dec.stdout)

        for insn in decoded:
            with self.subTest(decoded=insn):
                self.assertIn("dim:SQ_RSRC_IMG_CUBE", insn)
                spvasm = translate_stage("cs", [insn, "s_waitcnt vmcnt(0)"])
                m = Module(spvasm)
                if insn.startswith("image_store"):
                    self.assertEqual(sorted(m.writes()),
                                     sorted((CLASS_TYPES[c], f"%stv{c}_1", "%v3int") for c in range(3)))
                else:
                    self.assertEqual(sorted(m.reads()),
                                     sorted((CLASS_TYPES[c], f"%stv{c}_1", "%v3int") for c in range(3)))
                    self.assertEqual(m.coherent, set(CUBE_BINDINGS))
                if shutil.which("spirv-as") and shutil.which("spirv-val"):
                    validate_spirv(spvasm)


if __name__ == "__main__":
    unittest.main()
