"""Unit tests for tools/shader_meta.py and rdna2spirv.shader_meta() (standard library only, no game data).

    python -I tests/test_shader_meta.py
"""
import gc
import hashlib
import json
import os
import struct
import sys
import tempfile
import unittest
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import rdna2spirv  # noqa: E402
import shader_meta as M  # noqa: E402


def make_dxbc(chunks):
    head = 32 + 4 * len(chunks)
    offsets, body = [], b""
    for fourcc, payload in chunks:
        offsets.append(head + len(body))
        body += fourcc + struct.pack("<I", len(payload)) + payload
    return (b"DXBC" + bytes(16) + struct.pack("<III", 1, head + len(body), len(chunks))
            + b"".join(struct.pack("<I", o) for o in offsets) + body)


def isg1_chunk(elements):
    """Build an ISG1 chunk with real NUL-terminated semantic names.

    elements = [(semantic_name, semantic_index, register, mask, rw_mask, system_value, component_type)]
    """
    cnt = len(elements)
    str_off = 8 + 32 * cnt
    table = bytearray(struct.pack("<II", cnt, 8))
    strings = bytearray()
    for name, sidx, reg, mask, rw_mask, sysv, ctype in elements:
        if name:
            noff = str_off + len(strings)
            strings.extend(name.encode("ascii") + b"\x00")
        else:
            noff = 0
        table.extend(struct.pack("<6I", 0, noff, sidx, sysv, ctype, reg))
        table.extend(bytes([mask, rw_mask, 0, 0]))
        table.extend(struct.pack("<I", 0))
    return b"ISG1", bytes(table + strings)


def sfi0_chunk(flags):
    return b"SFI0", struct.pack("<Q", flags)


def translate_synthetic(name, lines, dxbc=None, fssm=None):
    """Translate a hand-written .s listing in a temp dir and return (spvasm, shader_meta dict)."""
    with tempfile.TemporaryDirectory() as tmp:
        disasm = os.path.join(tmp, "disasm")
        os.mkdir(disasm)
        s_path = os.path.join(disasm, name + ".s")
        header = f"; {name} code 0x100..0x200\n"
        with open(s_path, "w", encoding="utf-8") as f:
            f.write(header + "".join("\t" + ln + "\n" for ln in lines))
        if dxbc is None:
            dxbc = make_dxbc([])
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(dxbc)
        if fssm is not None:
            fdir = os.path.join(tmp, "fssm")
            os.mkdir(fdir)
            with open(os.path.join(fdir, name + ".fssm"), "wb") as f:
                f.write(fssm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            asm = rdna2spirv.translate(s_path)
            meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, asm)
            gc.collect()
        return asm, meta


def vk_pipeline_check_read_index(text):
    """Mirror of runtime/test/vk_pipeline_check.cpp:read_index to verify index.json compatibility."""
    out = {}
    i, depth, key = 0, 0, ""
    n = len(text)
    while i < n:
        c = text[i]
        if c == "{":
            depth += 1
            i += 1
            continue
        if c == "}":
            depth -= 1
            i += 1
            continue
        if c == '"':
            j = text.find('"', i + 1)
            tok = text[i + 1:j]
            i = j + 1
            if depth == 1:
                key = tok
                continue
            if depth == 2 and tok == "stage":
                a = text.find('"', i)
                b = text.find('"', a + 1)
                out.setdefault(key, {})["stage"] = text[a + 1:b]
                i = b + 1
            elif depth == 2 and tok == "subgroup_size":
                a = text.find(":", i)
                if a != -1:
                    a += 1
                    while a < n and text[a] in " \t":
                        a += 1
                    val = 0
                    while a < n and "0" <= text[a] <= "9":
                        val = val * 10 + (ord(text[a]) - 48)
                        a += 1
                    out.setdefault(key, {})["subgroup_size"] = val
                    i = a
            continue
        i += 1
    return out


class DxbcMetaTests(unittest.TestCase):
    def test_isg1_semantics_and_sfi0_flags(self):
        dxbc = make_dxbc([
            isg1_chunk([
                ("POSITION", 0, 0, 0x7, 0x7, 0, 3),
                ("TEXCOORD", 1, 1, 0x3, 0x3, 0, 3),
                ("SV_InstanceID", 0, 2, 0x1, 0x1, 8, 1),
            ]),
            sfi0_chunk(M.SFI0_RESOURCE_DESCRIPTOR_HEAP_INDEXING),
        ])
        got = M.read_dxbc_meta(dxbc)
        self.assertEqual(got["input_signature"], [
            {"semantic": "POSITION", "semantic_index": 0, "register": 0, "mask": 7, "rw_mask": 7,
             "system_value": 0, "component_type": "float"},
            {"semantic": "TEXCOORD", "semantic_index": 1, "register": 1, "mask": 3, "rw_mask": 3,
             "system_value": 0, "component_type": "float"},
            {"semantic": "SV_InstanceID", "semantic_index": 0, "register": 2, "mask": 1, "rw_mask": 1,
             "system_value": 8, "component_type": "uint"},
        ])
        self.assertTrue(got["sfi0_resource_heap_directly_indexed"])
        self.assertFalse(got["sfi0_sampler_heap_directly_indexed"])

    def test_invalid_isg1_table_does_not_publish_partial_signature(self):
        fourcc, payload = isg1_chunk([("POSITION", 0, 0, 7, 7, 0, 3)])
        for count, offset in ((2, 8), (1, 0)):
            with self.subTest(count=count, offset=offset):
                malformed = struct.pack("<II", count, offset) + payload[8:]
                dxbc = make_dxbc([(fourcc, malformed), sfi0_chunk(M.SFI0_RESOURCE_DESCRIPTOR_HEAP_INDEXING)])
                meta = M.read_dxbc_meta(dxbc)
                self.assertNotIn("input_signature", meta)
                self.assertTrue(meta["sfi0_resource_heap_directly_indexed"])

    def test_invalid_second_name_drops_whole_signature(self):
        tag, payload = isg1_chunk([("POSITION", 0, 0, 7, 7, 0, 3), ("TEXCOORD", 0, 1, 3, 3, 0, 3)])
        second_name = struct.unpack_from("<I", payload, 8 + 32 + 4)[0]
        for offset, data in ((8, payload), (len(payload), payload), (second_name, payload[:-1])):
            with self.subTest(offset=offset):
                malformed = bytearray(data)
                struct.pack_into("<I", malformed, 8 + 32 + 4, offset)
                meta = M.read_dxbc_meta(make_dxbc([(tag, malformed), sfi0_chunk(M.SFI0_RESOURCE_DESCRIPTOR_HEAP_INDEXING)]))
                self.assertNotIn("input_signature", meta)
                self.assertTrue(meta["sfi0_resource_heap_directly_indexed"])

    def test_sfi0_both_heaps_and_empty_dxbc(self):
        dxbc = make_dxbc([sfi0_chunk(M.SFI0_RESOURCE_DESCRIPTOR_HEAP_INDEXING | M.SFI0_SAMPLER_DESCRIPTOR_HEAP_INDEXING)])
        got = M.read_dxbc_meta(dxbc)
        self.assertTrue(got["sfi0_resource_heap_directly_indexed"])
        self.assertTrue(got["sfi0_sampler_heap_directly_indexed"])
        self.assertNotIn("input_signature", got)
        self.assertEqual(M.read_dxbc_meta(b""), {})
        self.assertEqual(M.read_dxbc_meta(b"NOT_DXBC" + bytes(32)), {})


class ShaderMetaIntegrationTests(unittest.TestCase):
    def test_ptr64_const32_stride_literal_sampler_and_fssm_hash(self):
        # User SGPRs:
        #   s[0:1] -> 64-bit descriptor-table pointer, offset by (s2 << 5) (32-byte T# stride)
        #   s2     -> 32-bit descriptor index constant
        #   s3     -> 32-bit scalar constant copied to v4
        #   s[12:15] -> literal S# built from immediates (example from docs/design.md)
        lines = [
            "s_lshl_b32 s4, s2, 5",
            "s_add_u32 s6, s0, s4",
            "s_addc_u32 s7, s1, 0",
            "s_load_dwordx8 s[16:23], s[6:7], 0x0",
            "s_mov_b32 s12, 0x92",
            "s_mov_b32 s13, 0x00c00000",
            "s_mov_b32 s14, 0x14500000",
            "s_mov_b32 s15, 0",
            "v_mov_b32_e32 v0, 0.5",
            "v_mov_b32_e32 v1, 0.5",
            "image_sample v[4:7], v[0:1], s[16:23], s[12:15] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "v_add_nc_u32_e32 v4, s3, v4",
            "exp mrt0 v4, v5, v6, v7 done vm",
            "s_endpgm",
        ]
        fssm_bytes = b"FSSM" + b"\x01\x02\x03\x04" * 16
        dxbc = make_dxbc([
            isg1_chunk([("SV_Position", 0, 0, 0xF, 0x3, 1, 3)]),
            sfi0_chunk(M.SFI0_RESOURCE_DESCRIPTOR_HEAP_INDEXING),
        ])
        _, meta = translate_synthetic("ps_meta_test", lines, dxbc=dxbc, fssm=fssm_bytes)

        self.assertEqual(meta["fssm_sha256"], hashlib.sha256(fssm_bytes).hexdigest())
        self.assertEqual(meta["user_sgpr_reads"], [
            {"sgpr": 0, "kind": "ptr64_lo", "pair": [0, 1]},
            {"sgpr": 1, "kind": "ptr64_hi", "pair": [0, 1]},
            {"sgpr": 2, "kind": "const32"},
            {"sgpr": 3, "kind": "const32"},
        ])
        self.assertEqual(meta["descriptor_strides"], [32])
        self.assertEqual(meta["literal_samplers"], [[0x92, 0x00C00000, 0x14500000, 0]])
        self.assertEqual(meta["descriptor_access"], {
            "table_loads": True,
            "truncates_table_ptr_high": False,
            "resource_heap_directly_indexed": True,
            "sampler_heap_directly_indexed": False,
        })
        self.assertEqual(meta["input_signature"][0]["semantic"], "SV_Position")
        self.assertNotIn("uses_code_base", meta)

    def test_truncated_32bit_table_pointer_and_vscale_stride(self):
        # Descriptor-table pointer in s2 with high dword forced to 0 (below 4 GB), stride 16 via VGPR shift.
        lines = [
            "s_mov_b32 s3, 0",
            "v_mov_b32_e32 v2, 1",
            "v_lshlrev_b32_e32 v3, 4, v2",
            "v_readfirstlane_b32 s4, v3",
            "s_load_dwordx4 s[8:11], s[2:3], s4",
            "v_mov_b32_e32 v0, s8",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        _, meta = translate_synthetic("ps_trunc_ptr", lines)
        # s3 was written before any read, so only s2 is read-before-write (and classified ptr32_lo).
        self.assertEqual(meta["user_sgpr_reads"], [{"sgpr": 2, "kind": "ptr32_lo"}])
        self.assertEqual(meta["descriptor_strides"], [16])
        self.assertEqual(meta["descriptor_access"], {
            "table_loads": True,
            "truncates_table_ptr_high": True,
        })

    def test_uses_code_base_without_false_user_ptr(self):
        # s_getpc_b64 + s_load_dwordx4 reads code-embedded data; must set uses_code_base=True and NOT
        # claim any user SGPR was read or that a descriptor table was loaded.
        lines = [
            "s_getpc_b64 s[4:5] ; encoding: [0x04,0x1c,0x84,0xbe]",
            "s_add_u32 s4, s4, 64 ; encoding: [0x04,0xff,0x04,0x80,0x40,0x00,0x00,0x00]",
            "s_addc_u32 s5, s5, 0 ; encoding: [0x05,0x80,0x05,0x82]",
            "s_load_dwordx4 s[8:11], s[4:5], 0x0 ; encoding: [0x02,0x02,0x08,0xf4,0x00,0x00,0x00,0x00]",
            "v_mov_b32_e32 v0, s8 ; encoding: [0x08,0x02,0x00,0x7e]",
            "exp mrt0 v0, v0, v0, v0 done vm ; encoding: [0x0f,0x08,0x00,0xf8,0x00,0x00,0x00,0x00]",
            "s_endpgm ; encoding: [0x00,0x00,0x81,0xbf]",
        ]
        _, meta = translate_synthetic("ps_codebase", lines)
        self.assertTrue(meta["uses_code_base"])
        self.assertNotIn("user_sgpr_reads", meta)
        self.assertNotIn("descriptor_access", meta)

    def test_compute_excludes_workgroup_id_sgprs_and_detects_vsharp(self):
        # RSRC2 0x8088: 4 user SGPRs (s0..s3 = V#), TGID_X enabled in s4 (system SGPR, NOT user data).
        fssm = bytearray(0x148)
        struct.pack_into("<4I", fssm, 0x138, 0x8088, 64, 1, 1)
        lines = [
            "v_lshlrev_b32_e32 v1, 2, v0",
            "v_mov_b32_e32 v2, s4",
            "buffer_store_dword v2, v1, s[0:3], 0 offen",
            "s_endpgm",
        ]
        _, meta = translate_synthetic("cs_vsharp", lines, fssm=bytes(fssm))
        self.assertEqual(meta["user_sgprs"], 4)
        # s4 (workgroup id) is excluded; s0..s3 are classified as vsharp.
        self.assertEqual(meta["user_sgpr_reads"], [
            {"sgpr": 0, "kind": "vsharp"},
            {"sgpr": 1, "kind": "vsharp"},
            {"sgpr": 2, "kind": "vsharp"},
            {"sgpr": 3, "kind": "vsharp"},
        ])
        self.assertEqual(meta["fssm_sha256"], hashlib.sha256(bytes(fssm)).hexdigest())

    def test_branch_control_flow_must_write_and_literal_sampler_disagreement(self):
        # Block 0 branches to block 1 (writes s5, s6 and sets s12=0x92) or block 2 (writes s6 only and sets s12=0x11).
        # Block 3 reads s5 (unread on path 0->2 -> MUST appear in user_sgpr_reads), reads s6 (written on BOTH arms ->
        # must NOT appear in user_sgpr_reads), and samples with s[12:15] (s12 differs across arms -> must NOT be
        # emitted as a literal sampler).
        lines = [
            "s_cmp_eq_u32 s0, 0 ; encoding: [0x00,0x80,0x00,0xbf]",
            "s_cbranch_scc1 5 ; encoding: [0x05,0x00,0x85,0xbf]",
            "s_mov_b32 s5, 1 ; encoding: [0x81,0x03,0x85,0xbe]",
            "s_mov_b32 s6, 2 ; encoding: [0x82,0x03,0x86,0xbe]",
            "s_mov_b32 s12, 0x92 ; encoding: [0xff,0x03,0x8c,0xbe,0x92,0x00,0x00,0x00]",
            "s_branch 2 ; encoding: [0x02,0x00,0x82,0xbf]",
            "s_mov_b32 s6, 3 ; encoding: [0x83,0x03,0x86,0xbe]",
            "s_mov_b32 s12, 0x11 ; encoding: [0x91,0x03,0x8c,0xbe]",
            "s_mov_b32 s13, 0 ; encoding: [0x80,0x03,0x8d,0xbe]",
            "s_mov_b32 s14, 0 ; encoding: [0x80,0x03,0x8e,0xbe]",
            "s_mov_b32 s15, 0 ; encoding: [0x80,0x03,0x8f,0xbe]",
            "s_load_dwordx8 s[16:23], s[2:3], 0x0 ; encoding: [0x01,0x04,0x0c,0xf4,0x00,0x00,0x00,0x00]",
            "v_mov_b32_e32 v0, 0.5 ; encoding: [0xf0,0x02,0x00,0x7e]",
            "v_mov_b32_e32 v1, 0.5 ; encoding: [0xf0,0x02,0x02,0x7e]",
            "image_sample v[4:7], v[0:1], s[16:23], s[12:15] dmask:0xf dim:SQ_RSRC_IMG_2D ; encoding: [0x00,0x0f,0x80,0xf0,0x00,0x04,0x64,0x00]",
            "v_add_nc_u32_e32 v4, s5, v4 ; encoding: [0x05,0x08,0x08,0x4a]",
            "v_add_nc_u32_e32 v5, s6, v5 ; encoding: [0x06,0x0a,0x0a,0x4a]",
            "exp mrt0 v4, v5, v6, v7 done vm ; encoding: [0x0f,0x08,0x00,0xf8,0x04,0x05,0x06,0x07]",
            "s_endpgm ; encoding: [0x00,0x00,0x81,0xbf]",
        ]
        _, meta = translate_synthetic("ps_cfg_meta", lines)
        sgprs = {r["sgpr"]: r["kind"] for r in meta["user_sgpr_reads"]}
        self.assertEqual(sgprs, {0: "const32", 2: "ptr64_lo", 3: "ptr64_hi", 5: "const32"})
        self.assertNotIn("literal_samplers", meta)

    def test_global_ptr64_and_mul_stride(self):
        # global_load_dword via user-SGPR pair s[4:5], plus s_mul_i32 48-byte stride into s_load_dwordx8.
        lines = [
            "s_mul_i32 s8, s2, 48",
            "s_load_dwordx8 s[16:23], s[0:1], s8",
            "v_mov_b32_e32 v0, 0",
            "global_load_dword v1, v0, s[4:5]",
            "v_add_nc_u32_e32 v1, s16, v1",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        _, meta = translate_synthetic("ps_global_stride", lines)
        self.assertEqual(meta["user_sgpr_reads"], [
            {"sgpr": 0, "kind": "ptr64_lo", "pair": [0, 1]},
            {"sgpr": 1, "kind": "ptr64_hi", "pair": [0, 1]},
            {"sgpr": 2, "kind": "const32"},
            {"sgpr": 4, "kind": "ptr64_lo", "pair": [4, 5]},
            {"sgpr": 5, "kind": "ptr64_hi", "pair": [4, 5]},
        ])
        self.assertEqual(meta["descriptor_strides"], [48])

    def test_index_json_compatible_with_vk_pipeline_check(self):
        lines = [
            "s_lshl_b32 s4, s2, 5",
            "s_load_dwordx8 s[16:23], s[0:1], s4",
            "s_mov_b32 s12, 0x92",
            "s_mov_b32 s13, 0",
            "s_mov_b32 s14, 0",
            "s_mov_b32 s15, 0",
            "v_mov_b32_e32 v0, 0.5",
            "v_mov_b32_e32 v1, 0.5",
            "image_sample v[4:7], v[0:1], s[16:23], s[12:15] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "exp mrt0 v4, v5, v6, v7 done vm",
            "s_endpgm",
        ]
        dxbc = make_dxbc([
            isg1_chunk([("POSITION", 0, 0, 0x7, 0x7, 0, 3)]),
            sfi0_chunk(M.SFI0_RESOURCE_DESCRIPTOR_HEAP_INDEXING),
        ])
        _, ps_meta = translate_synthetic("ps_one", lines, dxbc=dxbc, fssm=b"FSSM1234")
        fssm_cs = bytearray(0x148)
        struct.pack_into("<4I", fssm_cs, 0x138, 0x8088, 64, 1, 1)
        _, cs_meta = translate_synthetic("cs_two", ["s_mov_b64 exec, -1", "s_endpgm"], fssm=bytes(fssm_cs))

        dumped = json.dumps({"ps_one": ps_meta, "cs_two": cs_meta}, indent=0, sort_keys=True)
        parsed = vk_pipeline_check_read_index(dumped)
        self.assertEqual(parsed, {
            "ps_one": {"stage": "ps", "subgroup_size": 32},
            "cs_two": {"stage": "cs", "subgroup_size": 64},
        })


    def test_imm_u32_applies_vop3_abs_before_neg(self):
        import shader_ir
        self.assertEqual(M._imm_u32(shader_ir.operand("-|1.0|")), 0xBF800000)
        self.assertEqual(M._imm_u32(shader_ir.operand("-|-2.0|")), 0xC0000000)
        self.assertEqual(M._imm_u32(shader_ir.operand("neg(0x3f800000)")), 0xBF800000)
        self.assertEqual(M._imm_u32(shader_ir.operand("|0xbf800000|")), 0x3F800000)
        self.assertEqual(M._imm_u32(shader_ir.operand("-|0|")), 0x80000000)
        self.assertEqual(M._imm_u32(shader_ir.operand("neg(1)")), 0x80000001)


class OperandClassificationTests(unittest.TestCase):
    @staticmethod
    def _parse(lines):
        import shader_ir
        return [shader_ir.parse_line(ln) for ln in lines]

    def test_split_operands_scalar_rmw(self):
        import shader_ir
        cases = [
            ("s_cmov_b32 s2, s4", [2, 4], [2]),
            ("s_cmov_b64 s[2:3], s[4:5]", [2, 3, 4, 5], [2, 3]),
            ("s_bitset0_b32 s3, 7", [3], [3]),
            ("s_bitset1_b32 s3, s5", [3, 5], [3]),
            ("s_bitset0_b64 s[4:5], 33", [4, 5], [4, 5]),
            ("s_bitset1_b64 s[4:5], s6", [4, 5, 6], [4, 5]),
        ]
        for asm, want_reads, want_writes in cases:
            with self.subTest(asm=asm):
                reads, writes = M._split_operands(shader_ir.parse_line(asm))
                self.assertEqual(reads, want_reads)
                self.assertEqual(writes, want_writes)

    def test_split_operands_v_div_scale_dual_dst(self):
        import shader_ir
        cases = [
            ("v_div_scale_f32 v0, s4, v1, v2, s6", [6], [4]),
            ("v_div_scale_f32_e64 v0, s4, v1, v2, v1", [], [4]),
            ("v_div_scale_f32 v0, s[4:5], s7, v1, v2", [7], [4, 5]),
            ("v_div_scale_f64 v[0:1], s[4:5], v[2:3], v[4:5], s[6:7]", [6, 7], [4, 5]),
            ("v_div_scale_f64 v[0:1], vcc, v[2:3], v[4:5], v[2:3]", [], []),
        ]
        for asm, want_reads, want_writes in cases:
            with self.subTest(asm=asm):
                reads, writes = M._split_operands(shader_ir.parse_line(asm))
                self.assertEqual(reads, want_reads)
                self.assertEqual(writes, want_writes)

    def test_analyse_scalar_rmw_keeps_live_user_sgprs_and_invalidates_tags(self):
        # Each RMW op on an unwritten user SGPR (or pair) must record a user_sgpr_read.
        # s10 is written first, so s_bitset1_b32 on s10 must NOT mark s10 as a user SGPR.
        # s12 is initialized to a constant then conditionally overwritten by s_cmov_b32,
        # invalidating the constant tag so s[12:15] is NOT emitted as a literal sampler.
        insns = self._parse([
            "s_mov_b32 s10, 0",
            "s_bitset1_b32 s10, 1",
            "s_cmov_b32 s0, 1",
            "s_cmov_b64 s[2:3], s[20:21]",
            "s_bitset0_b32 s4, 2",
            "s_bitset1_b32 s5, 3",
            "s_bitset0_b64 s[6:7], 4",
            "s_bitset1_b64 s[8:9], 35",
            "s_mov_b32 s12, 0x92",
            "s_mov_b32 s13, 0",
            "s_mov_b32 s14, 0",
            "s_mov_b32 s15, 0",
            "s_cmov_b32 s12, s11",
            "image_sample v[4:7], v[0:1], s[24:31], s[12:15] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "s_endpgm",
        ])
        got = M.analyse_instructions([insns], [[]])
        sgprs = {r["sgpr"]: r["kind"] for r in got["user_sgpr_reads"]}
        for r in (0, 2, 3, 4, 5, 6, 7, 8, 9, 11, 20, 21):
            self.assertIn(r, sgprs, f"expected s{r} in user_sgpr_reads")
        self.assertNotIn(10, sgprs)
        self.assertNotIn(12, sgprs)
        self.assertEqual(got["literal_samplers"], [])

    def test_analyse_v_div_scale_sdst_not_user_sgpr_even_when_read_after(self):
        # s4 (f32 wave32 sdst) and s[6:7] (f64 wave64 sdst) are scratch outputs written by
        # v_div_scale_* and then read later; neither may fabricate a user SGPR.
        # s2 and s[8:9] are source operands and MUST be reported as user SGPRs.
        insns = self._parse([
            "v_div_scale_f32 v0, s4, s2, v1, s2",
            "v_mov_b32_e32 v2, s4",
            "v_div_scale_f64 v[4:5], s[6:7], v[2:3], v[6:7], s[8:9]",
            "s_or_b64 s[10:11], s[6:7], s[6:7]",
            "s_endpgm",
        ])
        got = M.analyse_instructions([insns], [[]])
        sgprs = [r["sgpr"] for r in got["user_sgpr_reads"]]
        self.assertEqual(sgprs, [2, 8, 9])

    def test_translate_synthetic_scalar_rmw_user_sgprs(self):
        lines = [
            "s_cmp_eq_u32 s0, 0",
            "s_cmov_b32 s1, 42",
            "s_bitset0_b32 s2, 1",
            "s_bitset1_b32 s3, 2",
            "s_bitset0_b64 s[4:5], 5",
            "s_bitset1_b64 s[6:7], 36",
            "v_mov_b32_e32 v0, s1",
            "v_add_nc_u32_e32 v0, s2, v0",
            "v_add_nc_u32_e32 v0, s3, v0",
            "v_add_nc_u32_e32 v0, s4, v0",
            "v_add_nc_u32_e32 v0, s5, v0",
            "v_add_nc_u32_e32 v0, s6, v0",
            "v_add_nc_u32_e32 v0, s7, v0",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        _, meta = translate_synthetic("ps_rmw_user", lines)
        sgprs = [r["sgpr"] for r in meta["user_sgpr_reads"]]
        self.assertEqual(sgprs, [0, 1, 2, 3, 4, 5, 6, 7])

    def test_global_atomic_ptr64_and_truncated_ptr32(self):
        # global_atomic_add without glc (saddr at a[2]) and with glc (saddr at a[3]) must classify
        # 64-bit user-SGPR pointers and 32-bit high-zeroed pointers the same way as global_load/store.
        lines = [
            "v_mov_b32_e32 v0, 0",
            "v_mov_b32_e32 v1, 1",
            "global_atomic_add v0, v1, s[0:1]",
            "global_atomic_add v2, v0, v1, s[2:3] glc",
            "s_mov_b32 s5, 0",
            "global_atomic_add v3, v0, v1, s[4:5] glc",
            "exp mrt0 v2, v3, v2, v3 done vm",
            "s_endpgm",
        ]
        _, meta = translate_synthetic("ps_global_atomic_meta", lines)
        self.assertEqual(meta["user_sgpr_reads"], [
            {"sgpr": 0, "kind": "ptr64_lo", "pair": [0, 1]},
            {"sgpr": 1, "kind": "ptr64_hi", "pair": [0, 1]},
            {"sgpr": 2, "kind": "ptr64_lo", "pair": [2, 3]},
            {"sgpr": 3, "kind": "ptr64_hi", "pair": [2, 3]},
            {"sgpr": 4, "kind": "ptr32_lo"},
        ])
        self.assertEqual(meta["descriptor_access"], {
            "table_loads": False,
            "truncates_table_ptr_high": True,
        })

    def test_s_mulk_i32_stride_and_constant_folding(self):
        # s_mulk_i32 scales a descriptor index (32-byte stride) and folds signed 16-bit constants.
        lines = [
            "s_mov_b32 s4, s2",
            "s_mulk_i32 s4, 32",
            "s_load_dwordx8 s[16:23], s[0:1], s4",
            "s_mov_b32 s12, 0x49",
            "s_mulk_i32 s12, 2",
            "s_mov_b32 s13, 1",
            "s_mulk_i32 s13, -1",
            "s_mov_b32 s14, 0",
            "s_mov_b32 s15, 0",
            "v_mov_b32_e32 v0, 0.5",
            "v_mov_b32_e32 v1, 0.5",
            "image_sample v[4:7], v[0:1], s[16:23], s[12:15] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "exp mrt0 v4, v5, v6, v7 done vm",
            "s_endpgm",
        ]
        _, meta = translate_synthetic("ps_mulk_meta", lines)
        self.assertEqual(meta["descriptor_strides"], [32])
        self.assertEqual(meta["literal_samplers"], [[0x92, 0xFFFFFFFF, 0, 0]])

    def test_s_brev_b32_constant_folding(self):
        # A literal S# with one dword bit-reversed by s_brev_b32 from an immediate or a known SGPR.
        lines = [
            "s_mov_b32 s12, 0x92",
            "s_brev_b32 s13, 1",
            "s_mov_b32 s20, 0x3",
            "s_brev_b32 s14, s20",
            "s_mov_b32 s15, 0",
            "v_mov_b32_e32 v0, 0.5",
            "v_mov_b32_e32 v1, 0.5",
            "image_sample v[4:7], v[0:1], s[0:7], s[12:15] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "exp mrt0 v4, v5, v6, v7 done vm",
            "s_endpgm",
        ]
        _, meta = translate_synthetic("ps_brev_meta", lines)
        self.assertEqual(meta["literal_samplers"], [[0x92, 0x80000000, 0xC0000000, 0]])

    def test_s_brev_b32_of_unknown_source_is_not_a_literal(self):
        # s_brev_b32 of a user SGPR leaves its destination unknown, so no literal S# is recorded.
        lines = [
            "s_mov_b32 s12, 0x92",
            "s_brev_b32 s13, s2",
            "s_mov_b32 s14, 0",
            "s_mov_b32 s15, 0",
            "v_mov_b32_e32 v0, 0.5",
            "v_mov_b32_e32 v1, 0.5",
            "image_sample v[4:7], v[0:1], s[0:7], s[12:15] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "exp mrt0 v4, v5, v6, v7 done vm",
            "s_endpgm",
        ]
        _, meta = translate_synthetic("ps_brev_user_meta", lines)
        self.assertNotIn("literal_samplers", meta)

    def test_scalar_store_pointer_and_vsharp(self):
        # s_store_dword* classifies its 2-SGPR pointer pair without marking table_loads,
        # and s_buffer_store_dword* classifies its 4-SGPR resource descriptor as vsharp.
        insns = self._parse([
            "s_store_dword s8, s[0:1], 0x0",
            "s_mov_b32 s3, 0",
            "s_store_dwordx4 s[8:11], s[2:3], 0x0",
            "s_buffer_store_dword s8, s[4:7], 0x0",
            "s_endpgm",
        ])
        got = M.analyse_instructions([insns], [[]])
        self.assertEqual(got["user_sgpr_reads"], [
            {"sgpr": 0, "kind": "ptr64_lo", "pair": [0, 1]},
            {"sgpr": 1, "kind": "ptr64_hi", "pair": [0, 1]},
            {"sgpr": 2, "kind": "ptr32_lo"},
            {"sgpr": 4, "kind": "vsharp"},
            {"sgpr": 5, "kind": "vsharp"},
            {"sgpr": 6, "kind": "vsharp"},
            {"sgpr": 7, "kind": "vsharp"},
            {"sgpr": 8, "kind": "const32"},
            {"sgpr": 9, "kind": "const32"},
            {"sgpr": 10, "kind": "const32"},
            {"sgpr": 11, "kind": "const32"},
        ])
        self.assertFalse(got["table_loads"])
        self.assertTrue(got["truncates_high"])

    @staticmethod
    def _literal_samplers_with(build_s13):
        # A literal S# in s[12:15]: s12 = 0x92, s14 = s15 = 0, and s13 built by the given hand-written lines.
        # s8 = 0x80000000 and s9 = 0xf0 are known constants; s3 is a user SGPR (not a known constant).
        import shader_ir
        lines = ["s_mov_b32 s8, 0x80000000", "s_mov_b32 s9, 0xf0", "s_mov_b32 s12, 0x92", *build_s13,
                 "s_mov_b32 s14, 0", "s_mov_b32 s15, 0",
                 "image_sample v[4:7], v[0:1], s[16:23], s[12:15] dmask:0xf dim:SQ_RSRC_IMG_2D", "s_endpgm"]
        return M.analyse_instructions([[shader_ir.parse_line(ln) for ln in lines]], [[]])["literal_samplers"]

    def test_inverted_scalar_logic_folds_into_literal_sampler(self):
        # s_not_b32, s_nand_b32, s_nor_b32 and s_xnor_b32 of known constants give a known constant, so a literal
        # S# with a dword built by one of them is reported in literal_samplers.
        cases = [
            (["s_not_b32 s13, 0"], 0xFFFFFFFF),
            (["s_not_b32 s13, s9"], 0xFFFFFF0F),
            (["s_not_b32 s13, s8"], 0x7FFFFFFF),
            (["s_mov_b32 s13, 0xf0", "s_not_b32 s13, s13"], 0xFFFFFF0F),  # source and destination are one register
            (["s_nand_b32 s13, s9, 0x30"], 0xFFFFFFCF),
            (["s_nand_b32 s13, s8, s8"], 0x7FFFFFFF),
            (["s_nor_b32 s13, s9, 0x0f"], 0xFFFFFF00),
            (["s_nor_b32 s13, 0, 0"], 0xFFFFFFFF),
            (["s_xnor_b32 s13, s9, 0xff"], 0xFFFFFFF0),
            (["s_xnor_b32 s13, s9, s9"], 0xFFFFFFFF),
            # The case found with #4289 open: the complement of 0, then a second fold of the result.
            (["s_not_b32 s13, 0", "s_and_b32 s13, s13, 0xffff00ff"], 0xFFFF00FF),
        ]
        for build, want in cases:
            with self.subTest(build=build):
                self.assertEqual(self._literal_samplers_with(build), [[0x92, want, 0, 0]])

    def test_shift_bitfield_min_max_abs_fold_into_literal_sampler(self):
        # s_ashr_i32, s_bfm_b32, s_bfe_u32, s_min_u32, s_max_u32 and s_abs_i32 of known constants give a known
        # constant. The expected values are the ones the translator computes for immediates (rdna2spirv.py).
        cases = [
            (["s_ashr_i32 s13, s8, 4"], 0xF8000000),          # the sign bit comes in from the left
            (["s_ashr_i32 s13, s9, 4"], 0x0000000F),
            (["s_ashr_i32 s13, s8, 36"], 0xF8000000),         # the shift count is taken modulo 32
            (["s_bfm_b32 s13, 4, 8"], 0x00000F00),            # width 4, offset 8
            (["s_bfm_b32 s13, 0, 8"], 0x00000000),
            (["s_bfm_b32 s13, 31, 1"], 0xFFFFFFFE),
            (["s_bfe_u32 s13, s9, 0x40004"], 0x0000000F),     # offset 4, width 4
            (["s_bfe_u32 s13, s8, 0x20001f"], 0x00000001),    # offset 31, width 32: the field stops at bit 31
            (["s_bfe_u32 s13, s9, 0x4"], 0x00000000),         # width 0
            (["s_min_u32 s13, s9, 0x10"], 0x00000010),
            (["s_min_u32 s13, s8, s9"], 0x000000F0),          # unsigned: 0x80000000 is the larger value
            (["s_max_u32 s13, s9, 0x10"], 0x000000F0),
            (["s_max_u32 s13, s8, s9"], 0x80000000),
            (["s_abs_i32 s13, s9"], 0x000000F0),
            (["s_abs_i32 s13, -16"], 0x00000010),
            (["s_abs_i32 s13, s8"], 0x80000000),              # the most negative value has no positive form
        ]
        for build, want in cases:
            with self.subTest(build=build):
                self.assertEqual(self._literal_samplers_with(build), [[0x92, want, 0, 0]])

    def test_shift_bitfield_min_max_abs_of_unknown_source_is_not_a_literal(self):
        for build in (["s_ashr_i32 s13, s3, 4"],
                      ["s_mov_b32 s13, 0", "s_ashr_i32 s13, s9, s3"],
                      ["s_bfm_b32 s13, s3, 8"],
                      ["s_bfe_u32 s13, s3, 0x40004"],
                      ["s_bfe_u32 s13, s9, s3"],
                      ["s_min_u32 s13, s9, s3"],
                      ["s_max_u32 s13, s3, s9"],
                      ["s_mov_b32 s13, 0", "s_abs_i32 s13, s3"]):
            with self.subTest(build=build):
                self.assertEqual(self._literal_samplers_with(build), [])

    def test_inverted_scalar_logic_of_unknown_source_is_not_a_literal(self):
        # A source that is not a known constant (the user SGPR s3) must not give a literal sampler, also when the
        # destination held a known constant before the operation.
        for build in (["s_not_b32 s13, s3"],
                      ["s_mov_b32 s13, 0", "s_not_b32 s13, s3"],
                      ["s_nand_b32 s13, s3, 0x30"],
                      ["s_nor_b32 s13, s9, s3"],
                      ["s_mov_b32 s13, 0", "s_xnor_b32 s13, s3, s9"]):
            with self.subTest(build=build):
                self.assertEqual(self._literal_samplers_with(build), [])

    def test_slist_operands_classify_pointers_and_descriptors(self):
        # LLVM VCCTuple slist syntax ([s0, s1], [s4, s5, s6, s7], ...) must be classified
        # identically to contiguous s[a:b] ranges.
        lines = [
            "s_load_dwordx4 s[16:19], [s0, s1], 0x0",
            "v_mov_b32_e32 v0, 0",
            "global_load_dword v1, v0, [s2, s3]",
            "buffer_load_dword v2, v0, [s4, s5, s6, s7], 0 offen",
            "v_add_nc_u32_e32 v1, v1, v2",
            "v_add_nc_u32_e32 v1, s16, v1",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        _, meta = translate_synthetic("ps_slist_meta", lines)
        self.assertEqual(meta["user_sgpr_reads"], [
            {"sgpr": 0, "kind": "ptr64_lo", "pair": [0, 1]},
            {"sgpr": 1, "kind": "ptr64_hi", "pair": [0, 1]},
            {"sgpr": 2, "kind": "ptr64_lo", "pair": [2, 3]},
            {"sgpr": 3, "kind": "ptr64_hi", "pair": [2, 3]},
            {"sgpr": 4, "kind": "vsharp"},
            {"sgpr": 5, "kind": "vsharp"},
            {"sgpr": 6, "kind": "vsharp"},
            {"sgpr": 7, "kind": "vsharp"},
        ])
        self.assertTrue(meta["descriptor_access"]["table_loads"])

        # Also verify slist tsharp, ssharp, and literal sampler via analyse_instructions.
        insns = self._parse([
            "s_mov_b32 s20, 0x92",
            "s_mov_b32 s21, 0",
            "s_mov_b32 s22, 0",
            "s_mov_b32 s23, 0",
            "image_sample v[0:3], v[0:1], [s0, s1, s2, s3, s4, s5, s6, s7], [s20, s21, s22, s23] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "image_sample v[4:7], v[0:1], [s0, s1, s2, s3, s4, s5, s6, s7], [s8, s9, s10, s11] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "s_endpgm",
        ])
        got = M.analyse_instructions([insns], [[]])
        kinds = {r["sgpr"]: r["kind"] for r in got["user_sgpr_reads"]}
        for r in range(8):
            self.assertEqual(kinds[r], "tsharp")
        for r in range(8, 12):
            self.assertEqual(kinds[r], "ssharp")
        self.assertEqual(got["literal_samplers"], [[0x92, 0, 0, 0]])

        # A VCCTuple slist mixing a user SGPR with vcc_lo must classify the SGPR without crashing.
        insns_vcc = self._parse([
            "global_load_dword v0, v1, [s0, vcc_lo]",
            "s_endpgm",
        ])
        got_vcc = M.analyse_instructions([insns_vcc], [[]])
        self.assertEqual(got_vcc["user_sgpr_reads"], [{"sgpr": 0, "kind": "ptr64_lo"}])


if __name__ == "__main__":
    unittest.main()
