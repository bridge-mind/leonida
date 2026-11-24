"""Typed image_load in tools/rdna2spirv.py reads every format from the sampled array (SHADER-151; standard library only:
no GPU, LLVM or game files; SPIRV-Tools only for the optional validation test).

    python -I tests/test_image_load_typed.py

The image's numeric type lives in the T# (word 1 FORMAT [28:20]), so the translator branches on it at run time.
UINT and SINT formats are fetched through uint and sint aliases of the dimension's sampled array (same set and
binding), everything else and a lookup miss (index 0, the RGBA8_UNORM fallback) through the float array. Nothing
reads a storage image, so a shader that only loads declares no StorageImageReadWithoutFormat (which Intel Xe-LP
lacks, #619) and no StorageImageWriteWithoutFormat. These tests translate small pixel shaders and check the SPIR-V:
the capabilities, which binding and component type each fetch reads, and, with a small evaluator of the code that
picks the branch, the branch taken for all 512 FORMAT values with a hit and with a miss.
runtime/test/shaders/cs_imgload.asm checks the result on a GPU (vk_store_test load).
"""
import itertools
import os
import re
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(ROOT, "tools"))
import rdna2spirv  # noqa: E402

EMPTY_DXBC = b"DXBC" + bytes(16) + struct.pack("<III", 1, 32, 0)
EPILOGUE = ["v_mov_b32 v8, v3", "exp mrt0 v8, v8, v8, v8 done vm", "s_endpgm"]

# dim -> (address registers, sampled binding, SPIR-V Dim, arrayed)
DIMS = {"2D": ("v[4:5]", 0, "2D", "0"), "2D_ARRAY": ("v[4:6]", 3, "2D", "1"),
        "3D": ("v[4:6]", 6, "3D", "0"), "1D": ("v4", 5, "1D", "0"),
        "CUBE": ("v[4:6]", 3, "2D", "1")}
STORAGE_IMAGE_CAPS = {"StorageImageReadWithoutFormat", "StorageImageWriteWithoutFormat", "Image1D"}


def translate(lines):
    """SPIR-V assembly text of a pixel shader made of these instruction lines (and an export of v3)."""
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        with open(os.path.join(tmp, "ps_load.dxbc"), "wb") as f:
            f.write(EMPTY_DXBC)
        path = os.path.join(tmp, "disasm", "ps_load.s")
        with open(path, "w", encoding="utf-8") as f:
            f.write("; ps_load code 0x0..0x100\n" + "".join("\t" + ln + "\n" for ln in lines + EPILOGUE))
        return rdna2spirv.translate(path)


def load_shader(dim, dmask=0xF):
    addr = DIMS[dim][0]
    return translate(["v_mov_b32 v4, 1", "v_mov_b32 v5, 2", "v_mov_b32 v6, 0",
                      f"image_load v[0:3], {addr}, s[0:7] dmask:0x{dmask:x} dim:SQ_RSRC_IMG_{dim} unorm",
                      "s_waitcnt vmcnt(0)"])


class Module:
    """The instructions of a SPIR-V assembly text by result id, its decorations and capabilities."""

    def __init__(self, text: str, vgprs: dict[int, int] | None = None):
        self.vgprs = vgprs
        self.text = text
        self.defs, self.binding, self.order = {}, {}, []
        self.caps = set(re.findall(r"^\s*OpCapability (\w+)", text, re.M))
        for line in text.splitlines():
            m = re.match(r"\s*(%\w+) = (Op\w+)\s*(.*)$", line)
            if m:
                self.defs[m.group(1)] = (m.group(2), m.group(3).split())
                self.order.append((m.group(1), m.group(2), m.group(3).split()))
                continue
            m = re.match(r"\s*OpDecorate (%\w+) Binding (\d+)$", line)
            if m:
                self.binding[m.group(1)] = int(m.group(2))
            m = re.match(r"\s*(OpBranchConditional) (.*)$", line)
            if m:
                self.order.append((None, m.group(1), m.group(2).split()))

    def fetches(self):
        """(result id, image variable, OpTypeImage operands) of every OpImageFetch."""
        out = []
        for rid, op, args in self.order:
            if op != "OpImageFetch":
                continue
            load_op, load = self.defs[args[1]]
            assert load_op == "OpLoad", load_op
            chain_op, chain = self.defs[load[1]]
            assert chain_op == "OpAccessChain", chain_op
            out.append((rid, chain[1], self.defs[load[0]][1]))
        return out

    def integer_coord_vgpr(self, operand):
        op, cast = self.defs[operand]
        assert op == "OpBitcast", op
        op, load = self.defs[cast[1]]
        assert op == "OpLoad", op
        op, ptr = self.defs[load[1]]
        assert (op, ptr[1]) == ("OpAccessChain", "%vgpr"), (op, ptr)
        return self.value(ptr[2], 0, 0)

    def value(self, x, word1, index):
        """Evaluate the uint/bool id x given the T#'s word 1 and the bindless index of the lookup."""
        if x in ("%true", "%false"):
            return x == "%true"
        op, a = self.defs[x]
        if op == "OpConstant":
            return int(a[1], 0)
        if op == "OpConstantTrue":
            return True
        if op == "OpFunctionCall" and a[1] == "%desc_lookup_fn":
            return index
        if op == "OpLoad":
            ptr_op, p = self.defs[a[1]]
            assert ptr_op == "OpAccessChain", ptr_op
            if p[1] == "%vgpr" and self.vgprs is not None:
                return self.vgprs[self.value(p[2], word1, index)]
            if p[1] == "%sgpr":
                assert self.value(p[2], word1, index) == 1, "only T# word 1 is read"
                return word1
            var_op, v = self.defs[p[1]]
            assert var_op == "OpVariable" and v[1] == "Private", (var_op, v)
            words = self.defs[v[2]][1][1:]            # OpConstantComposite %fmt_mask_arr <words>
            return self.value(words[self.value(p[2], word1, index)], word1, index)
        if op == "OpGroupNonUniformBroadcastFirst":   # desc_lookup makes the descriptor word uniform: one invocation
            return self.value(a[2], word1, index)      # here, so the value itself (a[1] is the Subgroup scope)
        if op == "OpGroupNonUniformShuffle":          # the same for a wave32 pixel shader's half (SHADER-731, #4225)
            return self.value(a[2], word1, index)
        u = [self.value(y, word1, index) for y in a[1:]]
        if op == "OpBitcast":
            return u[0]
        if op == "OpBitFieldUExtract":
            return (u[0] >> u[1]) & ((1 << u[2]) - 1)
        if op == "OpShiftRightLogical":
            return u[0] >> u[1]
        if op == "OpBitwiseAnd":
            return u[0] & u[1]
        if op == "OpINotEqual":
            return u[0] != u[1]
        if op == "OpLogicalAnd":
            return u[0] and u[1]
        raise AssertionError(f"unexpected {op} in the branch condition")

    def branch(self, word1, index):
        """'sint', 'uint' or 'float': which fetch the typed load's two selections lead to."""
        conds, last = {}, None
        for rid, op, args in self.order:
            if op == "OpBranchConditional":
                last = args[0]
            elif op == "OpAccessChain" and re.fullmatch(r"%itx[12]_\d", args[1]):
                conds["sint" if args[1].startswith("%itx2") else "uint"] = last
        assert set(conds) == {"sint", "uint"}, conds
        for kind in ("sint", "uint"):     # the sint selection encloses the uint one
            if self.value(conds[kind], word1, index):
                return kind
        return "float"


def format_class(fmt):
    name = rdna2spirv.GFX10_FORMATS.get(fmt, "")
    return "uint" if name.endswith("_UINT") else "sint" if name.endswith("_SINT") else "float"


class LoadOnlyShaders(unittest.TestCase):
    def test_volume_mip_fetch_uses_xyz_level_and_descriptor_type(self):
        for dst, modifier in (("v[0:1]", ""), ("v0", " d16")):
            with self.subTest(modifier=modifier):
                m = Module(translate([f"image_load_mip {dst}, v[4:7], s[0:7] "
                                      f"dmask:0x9 dim:SQ_RSRC_IMG_3D unorm{modifier}"]))
                fetches = m.fetches()
                self.assertEqual({image[0] for _, _, image in fetches}, {"%float", "%uint", "%int"})
                for rid, var, image in fetches:
                    self.assertEqual((m.binding[var], image[1]), (6, "3D"))
                    _, _, xyz, mask, lod = m.defs[rid][1]
                    self.assertEqual(mask, "Lod")
                    op, coords = m.defs[xyz]
                    self.assertEqual((op, coords[0]), ("OpCompositeConstruct", "%v3int"))
                    self.assertEqual([m.integer_coord_vgpr(v) for v in [*coords[1:], lod]], [4, 5, 6, 7])
                for fmt, kind in ((75, "uint"), (76, "sint"), (77, "float")):  # RGBA32 formats
                    self.assertEqual(m.branch(fmt << 20, 1), kind)
                    self.assertEqual(m.branch(fmt << 20, 0), "float")

    def test_2d_explicit_mip_reaches_every_typed_fetch(self):
        m = Module(translate(["image_load_mip v[0:3], v[4:6], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D",
                              "s_waitcnt vmcnt(0)"]))
        fetches = m.fetches()
        self.assertEqual({image[0] for _, _, image in fetches}, {"%float", "%uint", "%int"})
        for rid, var, _ in fetches:
            self.assertEqual(m.binding[var], 0)
            operands = m.defs[rid][1]
            self.assertEqual(operands[-2], "Lod")
            self.assertEqual(m.integer_coord_vgpr(operands[-1]), 6)

    def test_a16_2d_mip_address(self) -> None:
        with open(os.path.join(ROOT, "tests", "semantic", "fixtures", "packed_mip_load_2d.asm"),
                  encoding="ascii") as fixture:
            forms = [line.strip() for line in fixture if line.startswith("image_load_mip")]
        regs = {8: 0xFFFF8001, 9: 0xDEAD0003, 18: 0x8000FFFE, 5: 0xCAFE0007}
        expected = ((0x8001, 0xFFFF, 3), (0xFFFE, 0x8000, 7))
        self.assertEqual(len(forms), len(expected))
        for line, address in zip(forms, expected):
            with self.subTest(line=line):
                m = Module(translate([line]), vgprs=regs)
                fetches = m.fetches()
                self.assertEqual({image[0] for _, _, image in fetches}, {"%float", "%uint", "%int"})
                for rid, _, _ in fetches:
                    _, _, coord, mode, lod = m.defs[rid][1]
                    op, (ty, x, y) = m.defs[coord]
                    self.assertEqual((op, ty, mode), ("OpCompositeConstruct", "%v2int", "Lod"))
                    self.assertEqual(tuple(m.value(v, 0, 0) for v in (x, y, lod)), address)

    def test_a16_2d_mip_status_is_unsupported(self) -> None:
        for destination, status in (("v[0:1]", "tfe"), ("v0", "lwe")):
            with self.subTest(status=status), self.assertRaises(rdna2spirv.Unsupported):
                translate([f"image_load_mip {destination}, v[8:9], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D a16 {status}"])

    def test_2d_explicit_mip_d16_packs_only_selected_components(self):
        listing = ["image_load_mip v0, v[4:6], s[0:7] dmask:0x9 dim:SQ_RSRC_IMG_2D d16",
                   "s_waitcnt vmcnt(0)"]
        m = Module(translate(listing))
        self.assertEqual({image[0] for _, _, image in m.fetches()}, {"%float", "%uint", "%int"})
        for rid, _, _ in m.fetches():
            bits, = (r for r, op, args in m.order if op == "OpBitcast" and args[1] == rid)
            components = [int(args[2]) for _, op, args in m.order if op == "OpCompositeExtract" and args[1] == bits]
            self.assertEqual(components, [0, 3])
        # All three branches write just v0: R in the low half and A in the high half.
        writes = [m.defs[ptr][1] for ptr in re.findall(r"^\s*OpStore (%\w+) ", m.text, re.MULTILINE)]
        vgprs = [m.value(pointer[2], 0, 0) for pointer in writes if len(pointer) > 2 and pointer[1] == "%vgpr"]
        self.assertEqual(vgprs.count(0), 3)
        self.assertNotIn(1, vgprs)

    def test_no_storage_image_capability_and_no_storage_binding(self):
        for dim in DIMS:
            with self.subTest(dim=dim):
                m = Module(load_shader(dim))
                self.assertFalse(m.caps & STORAGE_IMAGE_CAPS, sorted(m.caps))
                self.assertNotIn("OpImageRead", m.text)
                # Set 0 bindings 4 and 7-19 are storage images (runtime/src/vulkan_bindless.h).
                self.assertEqual(set(m.binding.values()), {0, 1, 2, DIMS[dim][1]})

    def test_each_component_type_is_fetched_from_the_sampled_binding_of_its_dimension(self):
        for dim, (_, binding, sdim, arrayed) in DIMS.items():
            with self.subTest(dim=dim):
                m = Module(load_shader(dim))
                seen = {}
                for _, var, image in m.fetches():
                    # OpTypeImage <sampled type> <Dim> <Depth> <Arrayed> <MS> <Sampled> <Format>
                    self.assertEqual(image[1:], [sdim, "0", arrayed, "0", "1", "Unknown"])
                    self.assertEqual(m.binding[var], binding)
                    seen[image[0]] = var
                self.assertEqual(set(seen), {"%int", "%uint", "%float"})
                self.assertEqual(len(set(seen.values())), 3, "one variable per component type")

    def test_the_tsharp_format_picks_the_fetch_and_a_miss_takes_the_float_one(self):
        """All 512 FORMAT values: the 30 UINT and 15 SINT GFX10 formats read as uint and sint when the T# was found
        (index 5), everything reads float on a miss (index 0). Bits outside FORMAT [28:20] don't matter."""
        m = Module(load_shader("2D"))
        counts = {"uint": 0, "sint": 0, "float": 0}
        for fmt in range(512):
            want = format_class(fmt)
            counts[want] += 1
            for other in (0, 0xE00FFFFF):
                word1 = (fmt << 20) | other
                self.assertEqual(m.branch(word1, 5), want, f"FORMAT {fmt} word1 {word1:#x}")
                self.assertEqual(m.branch(word1, 0), "float", f"FORMAT {fmt} miss")
        self.assertEqual(counts, {"uint": 30, "sint": 15, "float": 512 - 45})

    def test_only_the_dmask_components_are_written(self):
        m = Module(load_shader("2D", dmask=0x9))
        for rid, _, _ in m.fetches():
            uses = [args for _, op, args in m.order if op == "OpBitcast" and rid in args]
            self.assertEqual(len(uses), 1)
            bits = next(r for r, op, args in m.order if op == "OpBitcast" and args[1] == rid)
            comps = sorted(int(args[2]) for _, op, args in m.order if op == "OpCompositeExtract" and args[1] == bits)
            self.assertEqual(comps, [0, 3], "dmask 0x9: R into v0, A into v1")

    def test_d16_odd_dmask_preserves_trailing_vgpr_high_half(self):
        # Odd component counts (1 or 3) must keep the trailing destination VGPR's high 16 bits via write_lo16
        # in all three typed branches (sint, uint, float), while even counts overwrite the whole register.
        cases = ((0x1, 0, True), (0x7, 1, True), (0x3, 0, False), (0xF, 1, False))
        for dim, (dmask, dest_reg, expect_preserve) in itertools.product(("2D", "CUBE"), cases):
            with self.subTest(dim=dim, dmask=hex(dmask)):
                reg = "v0" if dest_reg == 0 else "v[0:1]"
                m = Module(translate([
                    "v_mov_b32 v0, 0x5a5a0000",
                    "v_mov_b32 v1, 0x5a5a0000",
                    "v_mov_b32 v4, 1",
                    "v_mov_b32 v5, 2",
                    f"image_load {reg}, {DIMS[dim][0]}, s[0:7] dmask:0x{dmask:x} dim:SQ_RSRC_IMG_{dim} unorm d16",
                    "s_waitcnt vmcnt(0)",
                ]))
                # write_lo16 emits: OpBitwiseAnd %uint <copy_of_old_vgpr> %uint_0xFFFF0000.
                hi_masks = 0
                for _, op, args in m.order:
                    if op != "OpBitwiseAnd":
                        continue
                    rhs = m.defs.get(args[2])
                    if not rhs or rhs[0] != "OpConstant" or int(rhs[1][1], 0) != 0xFFFF0000:
                        continue
                    copy = m.defs.get(args[1])
                    if not copy or copy[0] != "OpCopyObject":
                        continue
                    load = m.defs.get(copy[1][1])
                    if not load or load[0] != "OpLoad":
                        continue
                    ptr = m.defs.get(load[1][1])
                    if ptr and ptr[0] == "OpAccessChain" and ptr[1][1] == "%vgpr" and m.value(ptr[1][2], 0, 0) == dest_reg:
                        hi_masks += 1
                self.assertEqual(hi_masks, 3 if expect_preserve else 0)


class LodQueryCoordinates(unittest.TestCase):
    def test_a16_queries_with_status_are_unsupported(self) -> None:
        for status, dest in (("tfe", "v[6:7]"), ("lwe", "v6")):
            with self.subTest(status=status), self.assertRaises(rdna2spirv.Unsupported):
                translate([f"image_get_lod {dest}, v10, s[8:15], s[16:19] "
                           f"dmask:0x1 dim:SQ_RSRC_IMG_2D a16 {status}"])

    def test_a16_queries_unpack_uv_and_keep_f32_results(self) -> None:
        fixture = Path(ROOT) / "tests/semantic/fixtures/get_lod_a16.asm"
        lines = [line for line in fixture.read_text(encoding="utf-8").splitlines()
                 if line and not line.startswith(";")]
        module = Module(translate(lines))
        queries = [(rid, args) for rid, op, args in module.order if op == "OpImageQueryLod"]
        self.assertEqual(len(queries), 2)
        for (rid, (ty, image, coord)), (register, image_type, components) in zip(
                queries, [(10, "%simg2d", [0, 1]), (12, "%simg2da", [1])]):
            self.assertEqual(ty, "%v2float")
            self.assertEqual(module.defs[image][1][0], image_type)
            op, unpack = module.defs[coord]
            self.assertEqual((op, unpack[:3]), ("OpExtInst", ["%v2float", "%glsl", "UnpackHalf2x16"]))
            op, load = module.defs[unpack[3]]
            self.assertEqual(op, "OpLoad")
            op, ptr = module.defs[load[1]]
            self.assertEqual((op, ptr[1], module.value(ptr[2], 0, 0)), ("OpAccessChain", "%vgpr", register))
            self.assertEqual([int(args[2]) for _, op, args in module.order
                              if op == "OpCompositeExtract" and args[:2] == ["%float", rid]], components)


class TypedGatherShaders(unittest.TestCase):
    def test_a16_offset_gathers_preserve_offset_and_unpack_uv_lod(self) -> None:
        fixture = Path(ROOT) / "tests/semantic/fixtures/gather_offset_a16.asm"
        lines = [line for line in fixture.read_text(encoding="ascii").splitlines()
                 if line and not line.startswith(";")]
        end = lines.index("s_waitcnt vmcnt(0)")
        for explicit in (True, False):
            with self.subTest(explicit_lod=explicit):
                body = lines[:end]
                if not explicit:
                    body[-1] = body[-1].replace("gather4_l_o", "gather4_lz_o").replace("v[4:6]", "v[4:5]")
                m = Module(translate(body))

                def source_register(value: str) -> int:
                    op, load = m.defs[value]
                    self.assertEqual(op, "OpLoad")
                    op, ptr = m.defs[load[1]]
                    self.assertEqual((op, ptr[1]), ("OpAccessChain", "%vgpr"))
                    return m.value(ptr[2], 0, 0)

                fields = [(source_register(a[1]), m.value(a[2], 0, 0), m.value(a[3], 0, 0))
                          for _, op, a in m.order if op == "OpBitFieldSExtract"]
                self.assertEqual(fields, [(4, 0, 6), (4, 8, 6)])

                def source_half(value: str) -> tuple[int, int]:
                    while m.defs[value][0] == "OpBitcast":
                        value = m.defs[value][1][1]
                    extract_op, (_, pair, half) = m.defs[value]
                    self.assertEqual(extract_op, "OpCompositeExtract")
                    op, unpack = m.defs[pair]
                    self.assertEqual((op, unpack[2]), ("OpExtInst", "UnpackHalf2x16"))
                    return source_register(unpack[3]), int(half)

                if explicit:
                    uv, = [a[1] for _, op, a in m.order if op == "OpFMul" and a[0] == "%v2float"]
                    lod, = [a[3] for _, op, a in m.order if op == "OpExtInst" and a[2] == "FClamp"]
                    self.assertEqual(source_half(lod), (6, 0))
                else:
                    uv, = {a[2] for _, op, a in m.order if op == "OpImageGather"}
                op, coord = m.defs[uv]
                self.assertEqual(op, "OpCompositeConstruct")
                self.assertEqual([source_half(c) for c in coord[1:]], [(5, 0), (5, 1)])

    def test_a16_offset_gathers_with_status_stay_unsupported(self) -> None:
        for op in ("image_gather4_l_o", "image_gather4_lz_o"):
            addr = "v[4:6]" if op == "image_gather4_l_o" else "v[4:5]"
            for status in ("tfe", "lwe"):
                with self.subTest(op=op, status=status), self.assertRaises(rdna2spirv.Unsupported):
                    dest = "v[10:14]" if status == "tfe" else "v[10:13]"
                    translate([f"{op} {dest}, {addr}, s[0:7], s[8:11] dmask:0x2 dim:SQ_RSRC_IMG_2D a16 {status}"])

    def test_2d_lod_samples_use_matching_typed_views(self):
        instructions = (("image_sample_lz", "v[4:5]"), ("image_sample_l", "v[4:6]"))
        destinations = (("v[0:3]", ""), ("v[0:1]", " d16"))
        for (name, addr), (dst, modifier) in itertools.product(instructions, destinations):
            with self.subTest(name=name, modifier=modifier):
                m = Module(translate([f"{name} {dst}, {addr}, s[0:7], s[8:11] "
                                      f"dmask:0xf dim:SQ_RSRC_IMG_2D{modifier}"]))
                samples = [args for _, op, args in m.order if op == "OpImageSampleExplicitLod"]
                self.assertEqual({args[0] for args in samples}, {"%v4float", "%v4uint_st", "%v4int_st"})
                # T# word 1 for R32_UINT, R32_SINT and R32_FLOAT, respectively.
                for fmt, expected in ((20, "uint"), (21, "sint"), (22, "float")):
                    self.assertEqual(m.branch(fmt << 20, 5), expected)
                    self.assertEqual(m.branch(fmt << 20, 0), "float")
                for ty, sampled, _, mask, lod in samples:
                    self.assertEqual(mask, "Lod")
                    if name == "image_sample_lz":
                        self.assertEqual(lod, "%f0")
                    else:
                        op, cast = m.defs[lod]
                        self.assertEqual(op, "OpBitcast")
                        op, load = m.defs[cast[1]]
                        self.assertEqual(op, "OpLoad")
                        op, ptr = m.defs[load[1]]
                        self.assertEqual((op, ptr[1], m.value(ptr[2], 0, 0)), ("OpAccessChain", "%vgpr", 6))
                    op, pair = m.defs[sampled]
                    self.assertEqual(op, "OpSampledImage")
                    _, image = m.defs[pair[1]]
                    _, chain = m.defs[image[1]]
                    self.assertEqual(m.binding[chain[1]], 0)
                    self.assertEqual(m.defs[ty][1][0], m.defs[image[0]][1][0])

    def test_explicit_lod_gather_fetches_four_taps_from_each_typed_view(self):
        for suffix, coords in (("", "v[4:6]"), ("_o", "v[4:7]")):
            with self.subTest(suffix=suffix):
                instruction = f"image_gather4_l{suffix} v[0:3], {coords}, s[0:7], s[8:11] dmask:0x2 dim:SQ_RSRC_IMG_2D"
                m = Module(translate([instruction, "s_waitcnt vmcnt(0)"]))
                fetches = m.fetches()
                footprints = []
                for scalar in ("%float", "%uint", "%int"):
                    group = [(rid, var) for rid, var, image in fetches if image[0] == scalar]
                    self.assertEqual(len(group), 4)
                    footprints.append([m.defs[rid][1][2:] for rid, _ in group])
                    for rid, var in group:
                        self.assertEqual(m.binding[var], 0)
                        self.assertEqual(m.defs[rid][1][-2], "Lod")
                        taps = [a for _, op, a in m.order if op == "OpCompositeExtract" and a[1] == rid]
                        self.assertEqual([a[2] for a in taps], ["1"])
                self.assertEqual(footprints[0], footprints[1])
                self.assertEqual(footprints[0], footprints[2])

    def test_implicit_2d_sample_takes_gradients_before_typed_dispatch(self):
        m = Module(translate(["image_sample v[0:1], v[4:5], s[0:7], s[8:11] "
                              "dmask:0xf dim:SQ_RSRC_IMG_2D d16"]))
        samples = [args for _, op, args in m.order if op == "OpImageSampleExplicitLod"]
        self.assertEqual({args[0] for args in samples}, {"%v4float", "%v4uint_st", "%v4int_st"})
        dispatch = re.search(r"OpBranchConditional %\w+ %\w+_s %\w+_rest", m.text)
        self.assertIsNotNone(dispatch)
        for _, _, coords, mode, dx, dy in samples:
            self.assertEqual(mode, "Grad")
            self.assertEqual(m.defs[dx], ("OpDPdx", ["%v2float", coords]))
            self.assertEqual(m.defs[dy], ("OpDPdy", ["%v2float", coords]))
            # The gradients must exist before the image-class selection can diverge.
            self.assertLess(m.text.index(f"{dy} = OpDPdy"), dispatch.start())

    def test_2d_gather_picks_the_sampled_type_and_selected_component(self):
        for dst, modifier in (("v[0:3]", ""), ("v[0:1]", " d16")):
            with self.subTest(modifier=modifier):
                m = Module(translate([f"image_gather4_lz {dst}, v[4:5], s[0:7], s[8:11] "
                                      f"dmask:0x2 dim:SQ_RSRC_IMG_2D{modifier}", "s_waitcnt vmcnt(0)"]))
                types = set()
                for rid, op, args in m.order:
                    if op != "OpImageGather":
                        continue
                    types.add(args[0])
                    self.assertEqual(m.value(args[3], 0, 0), 1)  # green
                    sampled_op, sampled = m.defs[args[1]]
                    self.assertEqual(sampled_op, "OpSampledImage")
                    image_load, image = m.defs[sampled[1]]
                    self.assertEqual(image_load, "OpLoad")
                    _, chain = m.defs[image[1]]
                    self.assertEqual(m.binding[chain[1]], 0)
                    if modifier and args[0] != "%v4float":
                        def uses(opcode, operand):
                            return [(r, a) for r, op, a in m.order if op == opcode and operand in a[1:]]
                        ((raw, _),) = uses("OpBitcast", rid)
                        taps = {int(a[2]): r for r, a in uses("OpCompositeExtract", raw)}
                        self.assertEqual(set(taps), {0, 1, 2, 3})
                        halves = []
                        for tap in range(4):
                            if args[0] == "%v4int_st":
                                ((signed, _),) = uses("OpBitcast", taps[tap])
                                ((sat, clamp),) = uses("OpExtInst", signed)
                                self.assertEqual(clamp[2], "SClamp")
                                ((bits, _),) = uses("OpBitcast", sat)
                                ((half, mask),) = uses("OpBitwiseAnd", bits)
                                self.assertEqual(m.value(mask[2], 0, 0), 0xffff)
                            else:
                                ((half, clamp),) = uses("OpExtInst", taps[tap])
                                self.assertEqual(clamp[2], "UMin")
                                self.assertEqual(m.value(clamp[4], 0, 0), 0xffff)
                            halves.append(half)
                        for tap in (0, 2):
                            lo = halves[tap]
                            ((hi, shift),) = uses("OpShiftLeftLogical", halves[tap + 1])
                            self.assertEqual(m.value(shift[2], 0, 0), 16)
                            self.assertTrue(any(hi in a for _, a in uses("OpBitwiseOr", lo)))

                self.assertEqual(types, {"%v4uint_st", "%v4int_st", "%v4float"})


class StoreAndLoad(unittest.TestCase):
    def test_a_shader_that_also_stores_needs_write_without_format_and_both_indexing_capabilities(self):
        m = Module(translate(["v_mov_b32 v4, 1", "v_mov_b32 v5, 2",
                              "image_store v[0:3], v[4:5], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D unorm",
                              "image_load v[0:3], v[4:5], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D unorm",
                              "s_waitcnt vmcnt(0)"]))
        self.assertIn("StorageImageWriteWithoutFormat", m.caps)
        self.assertIn("StorageImageArrayDynamicIndexing", m.caps)
        self.assertIn("SampledImageArrayDynamicIndexing", m.caps)
        self.assertNotIn("StorageImageReadWithoutFormat", m.caps)
        self.assertNotIn("OpImageRead", m.text)
        self.assertEqual(set(m.binding.values()), {0, 1, 2, 8, 9, 10})


@unittest.skipUnless(shutil.which("spirv-as") and shutil.which("spirv-val"), "needs spirv-as and spirv-val")
class Validation(unittest.TestCase):
    def test_spirv_val_accepts_every_dimension(self):
        with tempfile.TemporaryDirectory() as tmp:
            for dim in DIMS:
                with self.subTest(dim=dim):
                    sa, sv = os.path.join(tmp, f"{dim}.spvasm"), os.path.join(tmp, f"{dim}.spv")
                    with open(sa, "w", encoding="utf-8") as f:
                        f.write(load_shader(dim))
                    result = rdna2spirv.assemble_spirv(sa, sv)
                    self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_spirv_val_accepts_d16_odd_dmask(self):
        with tempfile.TemporaryDirectory() as tmp:
            for dmask, reg in ((0x1, "v0"), (0x7, "v[0:1]")):
                with self.subTest(dmask=hex(dmask)):
                    sa = os.path.join(tmp, f"d16_{dmask:x}.spvasm")
                    sv = os.path.join(tmp, f"d16_{dmask:x}.spv")
                    with open(sa, "w", encoding="utf-8") as f:
                        f.write(translate([
                            "v_mov_b32 v0, 0x5a5a0000",
                            "v_mov_b32 v1, 0x5a5a0000",
                            "v_mov_b32 v4, 1",
                            "v_mov_b32 v5, 2",
                            f"image_load {reg}, v[4:5], s[0:7] dmask:0x{dmask:x} dim:SQ_RSRC_IMG_2D unorm d16",
                            "s_waitcnt vmcnt(0)",
                        ]))
                    result = rdna2spirv.assemble_spirv(sa, sv)
                    self.assertEqual(result.returncode, 0, result.stderr + result.stdout)


class VolumeGradientSamples(unittest.TestCase):
    def test_volume_gradient_status_is_unsupported(self) -> None:
        for dest, status in (("v[2:3]", "tfe"), ("v3", "lwe")):
            with self.subTest(status=status), self.assertRaises(rdna2spirv.Unsupported):
                translate([f"image_sample_d {dest}, v[10:18], s[0:7], s[8:11] "
                           f"dmask:0x1 dim:SQ_RSRC_IMG_3D {status}"])

    def test_volume_address_operands(self) -> None:
        fixture = Path(ROOT) / "tests/semantic/fixtures/volume_gradients.asm"
        for line in fixture.read_text(encoding="utf-8").splitlines():
            if not line.startswith("image_sample_"):
                continue
            with self.subTest(line=line):
                module = Module(translate([line]))
                sample, = [args for _, op, args in module.order if op == "OpImageSampleExplicitLod"]
                _, _, coords, mask, dx, dy, *clamp = sample
                self.assertEqual(mask, "Grad|MinLod" if "_cl " in line else "Grad")
                self.assertEqual(len(clamp), int("_cl " in line))

                def source(value: str) -> tuple[int, int | None]:
                    # Through the bitcasts between a VGPR word and a float (a round trip is the original id).
                    op, args = module.defs[value]
                    while op == "OpBitcast":
                        op, args = module.defs[args[1]]
                    half = None
                    if op == "OpCompositeExtract":  # f16 coord converted to f32
                        _, unpacked, selector = args
                        unpack, (_, _, name, packed) = module.defs[unpacked]
                        self.assertEqual((unpack, name), ("OpExtInst", "UnpackHalf2x16"))
                        half = int(selector)
                        op, args = module.defs[packed]
                    self.assertEqual(op, "OpLoad")
                    access, (_, bank, index) = module.defs[args[1]]
                    self.assertEqual((access, bank), ("OpAccessChain", "%vgpr"))
                    return module.value(index, 0, 0), half

                for vector, registers in ((dx, (10, 11, 12)), (dy, (13, 14, 15))):
                    op, (ty, *components) = module.defs[vector]
                    self.assertEqual((op, ty), ("OpCompositeConstruct", "%v3float"))
                    self.assertEqual([source(c) for c in components], [(reg, None) for reg in registers])
                op, (ty, *xyz) = module.defs[coords]
                self.assertEqual((op, ty), ("OpCompositeConstruct", "%v3float"))
                packed = " a16" in line
                self.assertEqual([source(c) for c in xyz], [(16, 0), (16, 1), (17, 0)] if packed
                                 else [(16, None), (17, None), (18, None)])
                if clamp:
                    self.assertEqual(source(clamp[0]), (17, 1) if packed else (19, None))


if __name__ == "__main__":
    unittest.main()
