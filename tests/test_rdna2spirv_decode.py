from __future__ import annotations

import os
import struct
import sys
import tempfile
import unittest
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(ROOT, "tools"))

import rdna2spirv  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


class TestSameEncodingBytecodeBoundary(unittest.TestCase):
    """Exercise same_encoding() with synthetic short buffers, exact matches, and opcode bitfield tolerances."""

    def test_exact_matches_any_length(self):
        self.assertTrue(rdna2spirv.same_encoding(b"", b"", "v_mov_b32", 2))
        self.assertTrue(rdna2spirv.same_encoding(b"\x00", b"\x00", "v_mov_b32", 2))
        self.assertTrue(rdna2spirv.same_encoding(b"\x12\x34\x56\x78", b"\x12\x34\x56\x78", "v_mov_b32", 2))
        self.assertTrue(rdna2spirv.same_encoding(bytes(7), bytes(7), "v_mov_b32", 2))
        self.assertTrue(rdna2spirv.same_encoding(bytes(8), bytes(8), "v_mov_b32", 2))
        self.assertTrue(rdna2spirv.same_encoding(bytes(16), bytes(16), "v_mov_b32", 2))

    def test_short_buffers_rejected(self):
        # Different buffers under 8 bytes must be rejected immediately (returns False).
        self.assertFalse(rdna2spirv.same_encoding(b"", b"\x00", "v_mov_b32", 2))
        self.assertFalse(rdna2spirv.same_encoding(b"\x00", b"\x01", "v_mov_b32", 2))
        self.assertFalse(rdna2spirv.same_encoding(b"\x00\x01", b"\x00\x02", "v_mov_b32", 2))
        self.assertFalse(rdna2spirv.same_encoding(b"\x00" * 4, b"\x00" * 3 + b"\x01", "v_mov_b32", 2))
        self.assertFalse(rdna2spirv.same_encoding(b"\x00" * 7, b"\x00" * 6 + b"\x01", "v_mov_b32", 2))

        # Unequal buffer lengths must be rejected.
        self.assertFalse(rdna2spirv.same_encoding(bytes(4), bytes(8), "v_mov_b32", 2))
        self.assertFalse(rdna2spirv.same_encoding(bytes(8), bytes(4), "v_mov_b32", 2))
        self.assertFalse(rdna2spirv.same_encoding(bytes(7), bytes(8), "v_mov_b32", 2))
        self.assertFalse(rdna2spirv.same_encoding(bytes(8), bytes(7), "v_mov_b32", 2))
        self.assertFalse(rdna2spirv.same_encoding(bytes(8), bytes(12), "v_mov_b32", 2))
        self.assertFalse(rdna2spirv.same_encoding(bytes(12), bytes(8), "v_mov_b32", 2))

    def test_non_vop3_mismatched_opcodes_rejected(self):
        # 8-byte buffers that are neither VOP3 nor VOP3P: any bit differences must return False.
        raw = bytearray(8)
        printed = bytearray(8)
        printed[3] = 0x00  # not VOP3 ((0x00 >> 2) != 0x35) and not VOP3P (0x00 != 0xCC)
        raw[3] = 0x00

        # Diff at bit 41 (SRC1)
        val_printed = int.from_bytes(printed, "little") | (0x80 << 41)
        p_bytes = val_printed.to_bytes(8, "little")
        self.assertFalse(rdna2spirv.same_encoding(bytes(raw), p_bytes, "v_mov_b32", 2))

        # Diff at bit 14
        val_printed = int.from_bytes(printed, "little") | (1 << 14)
        p_bytes = val_printed.to_bytes(8, "little")
        self.assertFalse(rdna2spirv.same_encoding(bytes(raw), p_bytes, "v_pk_add_f16", 3))

    def test_vop3_boundary_and_opcode_checks(self):
        # VOP3 byte 3 has (byte >> 2) == 0x35, e.g. 0xD4 (0xD4 >> 2 == 0x35)
        raw = bytearray(8)
        raw[3] = 0xD4
        val_raw = int.from_bytes(raw, "little")

        # 1. Unused SRC1 (bits 49:41): raw stored 0, printed 0x80 -> tolerated
        val_printed = val_raw | (0x80 << 41)
        printed = val_printed.to_bytes(8, "little")
        self.assertTrue(rdna2spirv.same_encoding(bytes(raw), printed, "v_rcp_f32", 2))

        # Raw has non-zero at SRC1: rejected
        raw_nonzero = (val_raw | (1 << 41)).to_bytes(8, "little")
        self.assertFalse(rdna2spirv.same_encoding(raw_nonzero, printed, "v_rcp_f32", 2))

        # Printed has invalid diff value (not 0x80): rejected
        val_printed_bad = val_raw | (0x81 << 41)
        self.assertFalse(rdna2spirv.same_encoding(bytes(raw), val_printed_bad.to_bytes(8, "little"), "v_rcp_f32", 2))
        val_printed_bad2 = val_raw | (0x40 << 41)
        self.assertFalse(rdna2spirv.same_encoding(bytes(raw), val_printed_bad2.to_bytes(8, "little"), "v_rcp_f32", 2))

        # 2. Unused SRC2 (bits 58:50): raw stored 0, printed 0x80 -> tolerated
        val_printed2 = val_raw | (0x80 << 50)
        printed2 = val_printed2.to_bytes(8, "little")
        self.assertTrue(rdna2spirv.same_encoding(bytes(raw), printed2, "v_sub_f32", 3))

        # Raw has non-zero at SRC2: rejected
        raw_nonzero2 = (val_raw | (1 << 50)).to_bytes(8, "little")
        self.assertFalse(rdna2spirv.same_encoding(raw_nonzero2, printed2, "v_rcp_f32", 2))

        # 3. Both SRC1 and SRC2 unused fields stored 0, printed 0x80 -> tolerated
        val_both = val_raw | (0x80 << 41) | (0x80 << 50)
        printed_both = val_both.to_bytes(8, "little")
        self.assertTrue(rdna2spirv.same_encoding(bytes(raw), printed_both, "v_rcp_f32", 2))

        # 4. Other bits modified (e.g. bit 0) -> rejected
        val_other = val_both | 1
        self.assertFalse(rdna2spirv.same_encoding(bytes(raw), val_other.to_bytes(8, "little"), "v_rcp_f32", 2))

    def test_vop3_implicit_and_scalar_destinations(self):
        raw = (0xD4 << 24).to_bytes(8, "little")
        val_raw = int.from_bytes(raw, "little")
        src1 = (val_raw | (0x80 << 41)).to_bytes(8, "little")
        src2 = (val_raw | (0x80 << 50)).to_bytes(8, "little")
        # Both comparison inputs are used even when the EXEC destination is implicit.
        self.assertFalse(rdna2spirv.same_encoding(raw, src1, "v_cmpx_lt_f16_e64", 2))
        self.assertTrue(rdna2spirv.same_encoding(raw, src2, "v_cmpx_lt_f16_e64", 2))
        # The carry-out destination is not a source. Only the carry-in form uses SRC2.
        self.assertTrue(rdna2spirv.same_encoding(raw, src2, "v_add_co_u32", 4))
        self.assertFalse(rdna2spirv.same_encoding(raw, src2, "v_add_co_ci_u32", 5))

    def test_vop3p_boundary_and_packed_opcodes(self):
        # VOP3P byte 3 is 0xCC
        raw = bytearray(8)
        raw[3] = 0xCC
        val_raw = int.from_bytes(raw, "little")

        # Two-source packed ops: bit 14 (op_sel_hi[2]) diff is tolerated
        val_p14 = val_raw | (1 << 14)
        printed14 = val_p14.to_bytes(8, "little")

        two_src_ops = ("v_pk_add_f16", "v_pk_mul_f16", "v_pk_sub_f16", "v_pk_min_f16", "v_pk_max_f16")
        for op in two_src_ops:
            with self.subTest(op=op):
                self.assertTrue(rdna2spirv.same_encoding(bytes(raw), printed14, op, 3))

        # Three-source packed ops (fma/mad): bit 14 diff is NOT tolerated
        three_src_ops = ("v_pk_fma_f16", "v_pk_mad_i16", "v_pk_fma_i16", "v_pk_mad_u16")
        for op in three_src_ops:
            with self.subTest(op=op):
                self.assertFalse(rdna2spirv.same_encoding(bytes(raw), printed14, op, 4))

        # Non-packed ops with VOP3P encoding: bit 14 diff is NOT tolerated
        for op in ("v_mov_b32", "v_add_f32", "s_mov_b32"):
            with self.subTest(op=op):
                self.assertFalse(rdna2spirv.same_encoding(bytes(raw), printed14, op, 3))

        # A two-source packed op tolerates both defaults, but has a live SRC1.
        val_defaults = val_raw | (1 << 14) | (0x80 << 50)
        printed_defaults = val_defaults.to_bytes(8, "little")
        self.assertTrue(rdna2spirv.same_encoding(bytes(raw), printed_defaults, "v_pk_add_f16", 3))
        changed_src1 = (val_defaults | (0x80 << 41)).to_bytes(8, "little")
        self.assertFalse(rdna2spirv.same_encoding(bytes(raw), changed_src1, "v_pk_add_f16", 3))

        # Extra bit modified -> rejected
        bad = (val_defaults | 2).to_bytes(8, "little")
        self.assertFalse(rdna2spirv.same_encoding(bytes(raw), bad, "v_pk_add_f16", 3))


class TestBaseOpDecoder(unittest.TestCase):
    """Exercise base_op() opcode suffix stripping."""

    def test_strip_standard_suffixes(self):
        self.assertEqual(rdna2spirv.base_op("v_add_f32_e32"), "v_add_f32")
        self.assertEqual(rdna2spirv.base_op("v_fma_f32_e64"), "v_fma_f32")
        self.assertEqual(rdna2spirv.base_op("v_mov_b32_sdwa"), "v_mov_b32")
        self.assertEqual(rdna2spirv.base_op("v_mov_b32_dpp"), "v_mov_b32")

    def test_base_op_boundary_cases(self):
        self.assertEqual(rdna2spirv.base_op("s_endpgm"), "s_endpgm")
        self.assertEqual(rdna2spirv.base_op("exp"), "exp")
        self.assertEqual(rdna2spirv.base_op("v_mov_b32"), "v_mov_b32")
        # Suffix embedded in name but not at the end
        self.assertEqual(rdna2spirv.base_op("v_add_e32_co"), "v_add_e32_co")
        self.assertEqual(rdna2spirv.base_op("v_sub_e64_co"), "v_sub_e64_co")
        # Exact suffix string
        self.assertEqual(rdna2spirv.base_op("_e32"), "")
        self.assertEqual(rdna2spirv.base_op("_e64"), "")
        self.assertEqual(rdna2spirv.base_op("_sdwa"), "")
        self.assertEqual(rdna2spirv.base_op("_dpp"), "")
        # Empty string and short identifiers
        self.assertEqual(rdna2spirv.base_op(""), "")
        self.assertEqual(rdna2spirv.base_op("v"), "v")


class TestFormatDecoderBoundary(unittest.TestCase):
    """Exercise typed-buffer format decoding functions and boundaries in Gen."""

    def setUp(self):
        self.gen = rdna2spirv.Gen("ps")

    def test_invalid_format_codes_raise_unsupported(self):
        for code in (0, -1, 999, 255, 100, 127):   # 42 (10_11_11 SINT) is a format since #2022 and #2432
            with self.subTest(code=code):
                with self.assertRaisesRegex(rdna2spirv.Unsupported, r"format .* \(not in FORMATS\)"):
                    self.gen.decode_format(["%w0", "%w1", "%w2", "%w3"], code)

                with self.assertRaisesRegex(rdna2spirv.Unsupported, r"format .* \(not in FORMATS\)"):
                    self.gen.encode_format(["%f0"], ["%i0"], code)

    def test_decode_format_all_kinds_produce_4_components(self):
        words = [f"%w{i}" for i in range(4)]
        for code, (size, kind, widths) in rdna2spirv.Gen.FORMATS.items():
            with self.subTest(code=code, kind=kind):
                out = self.gen.decode_format(words, code)
                self.assertEqual(len(out), 4, f"format {code} ({kind}) must yield exactly 4 components")

    def test_decode_format_short_words_channel_replication(self):
        # 1-channel format 22 (f32, (32,)): replicates word 0 across all 4 components
        out1 = self.gen.decode_format(["%w0"], 22)
        self.assertEqual(out1, ["%w0", "%w0", "%w0", "%w0"])

        # 2-channel format 64 (f32, (32, 32)): replicates words [0, 1] as [0, 1, 0, 1]
        out2 = self.gen.decode_format(["%w0", "%w1"], 64)
        self.assertEqual(out2, ["%w0", "%w1", "%w0", "%w1"])

        # 3-channel format 74 (f32, (32, 32, 32)): replicates words [0, 1, 2] as [0, 1, 2, 0]
        out3 = self.gen.decode_format(["%w0", "%w1", "%w2"], 74)
        self.assertEqual(out3, ["%w0", "%w1", "%w2", "%w0"])

        # 4-channel format 77 (f32, (32, 32, 32, 32)): keeps words [0, 1, 2, 3]
        out4 = self.gen.decode_format(["%w0", "%w1", "%w2", "%w3"], 77)
        self.assertEqual(out4, ["%w0", "%w1", "%w2", "%w3"])

        # u32 format 20 (1-comp) and i32 format 21 (1-comp)
        self.assertEqual(self.gen.decode_format(["%u0"], 20), ["%u0", "%u0", "%u0", "%u0"])
        self.assertEqual(self.gen.decode_format(["%i0"], 21), ["%i0", "%i0", "%i0", "%i0"])

    def test_decode_format_fixed_vs_dynamic_normalization(self):
        words = ["%w0", "%w1", "%w2", "%w3"]
        # UNORM format 1 (8,)
        self.assertEqual(len(self.gen.tbuffer_norm_fns), 0)
        self.gen.decode_format(words, 1, fixed=True)
        self.assertIn(8, self.gen.tbuffer_norm_fns)

        # Dynamic UNORM format 7 (16,) without fixed does not add to tbuffer_norm_fns
        fns_count = len(self.gen.tbuffer_norm_fns)
        self.gen.decode_format(words, 7, fixed=False)
        self.assertEqual(len(self.gen.tbuffer_norm_fns), fns_count)

    def test_decode_any_format_selection(self):
        words = ["%w0", "%w1", "%w2", "%w3"]
        initial_lines = len(self.gen.body)
        comps = self.gen.decode_any_format("%run_fmt", words)
        self.assertEqual(len(comps), 4)
        emitted = "\n".join(self.gen.body[initial_lines:])
        self.assertIn("OpIEqual %bool %run_fmt", emitted)
        self.assertIn("OpSelect %uint", emitted)

    def test_format_decode_call_builds_and_caches_function(self):
        words = ["%w0", "%w1", "%w2", "%w3"]
        self.assertIsNone(self.gen.fmt_fn)
        comps1 = self.gen.format_decode_call("%fmt_var", words)
        self.assertEqual(len(comps1), 4)
        self.assertIsNotNone(self.gen.fmt_fn)
        first_fn = self.gen.fmt_fn

        # Subsequent call reuses the cached function
        comps2 = self.gen.format_decode_call("%fmt_var2", words)
        self.assertEqual(len(comps2), 4)
        self.assertIs(self.gen.fmt_fn, first_fn)

    def test_encode_format_saturation_and_boundary(self):
        fcomps = ["%f0", "%f1", "%f2", "%f3"]
        icomps = ["%i0", "%i1", "%i2", "%i3"]

        # Narrow uint format 60 (8, 8, 8, 8): must saturate each component using UMin
        initial_lines = len(self.gen.body)
        res60 = self.gen.encode_format(fcomps, icomps, 60)
        self.assertEqual(len(res60), 1)
        emitted = "\n".join(self.gen.body[initial_lines:])
        self.assertIn("OpExtInst %uint %glsl UMin", emitted)

        # f32 format 64 (2 comps): passes through first 2 fcomps
        self.assertEqual(self.gen.encode_format(fcomps, icomps, 64), ["%f0", "%f1"])

        # u32 format 72 (3 comps): passes through first 3 icomps
        self.assertEqual(self.gen.encode_format(fcomps, icomps, 72), ["%i0", "%i1", "%i2"])

    def test_int_format_masks_structure(self):
        u_mask, si_mask = rdna2spirv.Gen.int_format_masks()
        self.assertEqual(len(u_mask), 16)
        self.assertEqual(len(si_mask), 16)
        self.assertTrue(all(isinstance(x, int) for x in u_mask))
        self.assertTrue(all(isinstance(x, int) for x in si_mask))

        flags = self.gen.fmt_int_flags("%t_word1")
        self.assertEqual(len(flags), 2)
        emitted = "\n".join(self.gen.body)
        self.assertIn("OpBitFieldUExtract %uint %t_word1", emitted)


class TestInstructionDecoderBoundary(unittest.TestCase):
    """Exercise opcode decoder failure modes and boundary parameter checks in Gen."""

    def setUp(self):
        self.gen = rdna2spirv.Gen("ps")

    def test_invalid_opcodes_raise_unsupported(self):
        invalid_ops = ("v_bogus_fake_op", "s_not_an_instruction", "", "12345", "v_!@#$%")
        for op in invalid_ops:
            with self.subTest(op=op):
                with self.assertRaises(rdna2spirv.Unsupported) as cm:
                    self.gen._insn({"op": op, "args": [], "mods": []})
                self.assertEqual(str(cm.exception), op)

    def test_mode_opcodes_without_immediates(self):
        for op, error in (("s_denorm_mode", "denormal mode"), ("s_round_mode", "requires an immediate")):
            with self.subTest(op=op), self.assertRaisesRegex(rdna2spirv.Unsupported, error):
                self.gen._insn({"op": op, "args": [], "mods": []})

    def test_dpp_invalid_opcodes_and_boundary_modifiers(self):
        # DPP combined with SDWA
        with self.assertRaisesRegex(rdna2spirv.Unsupported, "DPP with SDWA"):
            self.gen.insn({"op": "v_mov_b32_sdwa_dpp", "args": [], "mods": []})

        # Missing DPP control (none of dpp8, quad_perm, or row_* supplied)
        with self.assertRaisesRegex(rdna2spirv.Unsupported, "DPP control"):
            self.gen.insn({"op": "v_mov_b32_dpp", "args": [], "mods": []})

        # Conflicting DPP controls (quad_perm and row_mirror simultaneously)
        with self.assertRaisesRegex(rdna2spirv.Unsupported, "DPP controls"):
            self.gen.insn({
                "op": "v_mov_b32_dpp",
                "args": [{"k": "vgpr", "r": 0}, {"k": "vgpr", "r": 1}],
                "mods": ["quad_perm:[0,1,2,3]", "row_mirror"],
            })

        # DPP invalid flag (bound_ctrl out of range 0..1)
        with self.assertRaisesRegex(rdna2spirv.Unsupported, "DPP flag"):
            self.gen.insn({
                "op": "v_mov_b32_dpp",
                "args": [{"k": "vgpr", "r": 0}, {"k": "vgpr", "r": 1}],
                "mods": ["quad_perm:[0,1,2,3]", "bound_ctrl:2"],
            })

        # DPP out-of-range row_mask (> 0xF)
        with self.assertRaisesRegex(rdna2spirv.Unsupported, "DPP mask"):
            self.gen.insn({
                "op": "v_mov_b32_dpp",
                "args": [{"k": "vgpr", "r": 0}, {"k": "vgpr", "r": 1}],
                "mods": ["quad_perm:[0,1,2,3]", "row_mask:0x10"],
            })

        # DPP invalid non-integer modifier
        with self.assertRaisesRegex(rdna2spirv.Unsupported, "DPP row_mask"):
            self.gen.insn({
                "op": "v_mov_b32_dpp",
                "args": [{"k": "vgpr", "r": 0}, {"k": "vgpr", "r": 1}],
                "mods": ["quad_perm:[0,1,2,3]", "row_mask:invalid_hex"],
            })

        # DPP operands boundary (fewer than 2 operands)
        with self.assertRaisesRegex(rdna2spirv.Unsupported, "DPP operands"):
            self.gen.insn({
                "op": "v_mov_b32_dpp",
                "args": [{"k": "vgpr", "r": 0}],
                "mods": ["quad_perm:[0,1,2,3]"],
            })


class TestTranslateIntegrationBoundaryChecks(unittest.TestCase):
    """End-to-end translation boundary checks on synthetic shaders with truncated blobs or invalid opcodes."""

    def test_short_and_truncated_native_blob(self):
        with tempfile.TemporaryDirectory() as tmp:
            disasm = os.path.join(tmp, "disasm")
            fssm = os.path.join(tmp, "fssm")
            os.makedirs(disasm)
            os.makedirs(fssm)

            with open(os.path.join(tmp, "ps_trunc.dxbc"), "wb") as f:
                f.write(STUB_DXBC)

            # Native blob is only 4 bytes (shorter than 8-byte instruction encoding)
            with open(os.path.join(fssm, "ps_trunc.fssm"), "wb") as f:
                f.write(bytes(4))

            shader_text = (
                "ps_trunc.s:\n"
                "\tv_mov_b32 v0, 1.0 ; encoding: [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00]\n"
                "; invalid instruction encoding at 0x10\n"
                "\texp mrt0 v0, v0, v0, v0 done vm ; encoding: [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00]\n"
                "\ts_endpgm ; encoding: [0x00,0x00,0x81,0xbf]\n"
            )
            src = os.path.join(disasm, "ps_trunc.s")
            with open(src, "w", encoding="utf-8") as f:
                f.write(shader_text)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                with self.assertRaisesRegex(rdna2spirv.Unsupported, "undecodable words \\(cannot re-anchor\\)"):
                    rdna2spirv.translate(src)

    def test_fall_through_into_undecodable_words_is_refused(self):
        # SHADER-813: llvm-mc skips words it cannot decode (e.g. some v_dot4_i32_i8 op_sel forms). When a reachable
        # instruction runs into them the path used to end there, so the export after the gap was dropped and the
        # module translated, valid and silent. It must be refused instead.
        mov0 = [0xF2, 0x02, 0x00, 0x7E]
        mov1 = [0xF2, 0x02, 0x02, 0x7E]
        export = [0x0F, 0x18, 0x00, 0xF8, 0x00, 0x01, 0x00, 0x01]
        endpgm = [0x00, 0x00, 0x81, 0xBF]
        gap = [0x00, 0x50, 0x16, 0xCC, 0x01, 0x05, 0x0E, 0x0C]   # the word from the finding, as llvm-mc leaves it

        def enc(words):
            return ",".join(f"0x{b:02x}" for b in words)

        with tempfile.TemporaryDirectory() as tmp:
            disasm = os.path.join(tmp, "disasm")
            fssm = os.path.join(tmp, "fssm")
            os.makedirs(disasm)
            os.makedirs(fssm)
            with open(os.path.join(tmp, "ps_gap.dxbc"), "wb") as f:
                f.write(STUB_DXBC)
            with open(os.path.join(fssm, "ps_gap.fssm"), "wb") as f:
                f.write(bytes(mov0 + mov1 + gap + export + endpgm))
            shader_text = (
                "ps_gap.s:\n"
                f"\tv_mov_b32_e32 v0, 1.0 ; encoding: [{enc(mov0)}]\n"
                f"\tv_mov_b32_e32 v1, 1.0 ; encoding: [{enc(mov1)}]\n"
                "; invalid instruction encoding at 0x8\n"
                f"\texp mrt0 v0, v1, v0, v1 done vm ; encoding: [{enc(export)}]\n"
                f"\ts_endpgm ; encoding: [{enc(endpgm)}]\n"
            )
            src = os.path.join(disasm, "ps_gap.s")
            with open(src, "w", encoding="utf-8") as f:
                f.write(shader_text)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                with self.assertRaisesRegex(rdna2spirv.Unsupported, "fall-through into undecodable words"):
                    rdna2spirv.translate(src)

    def test_fall_through_into_undecodable_words_at_the_end_of_the_range_is_refused(self):
        # SHADER-813 follow-up (@Flyander): the skipped words come after the last printed instruction but inside the
        # code range the header declares, so the path runs into them as well
        mov0 = [0xF2, 0x02, 0x00, 0x7E]
        mov1 = [0xF2, 0x02, 0x02, 0x7E]
        gap = [0x00, 0x50, 0x16, 0xCC, 0x01, 0x05, 0x0E, 0x0C]

        def enc(words):
            return ",".join(f"0x{b:02x}" for b in words)

        with tempfile.TemporaryDirectory() as tmp:
            disasm = os.path.join(tmp, "disasm")
            fssm = os.path.join(tmp, "fssm")
            os.makedirs(disasm)
            os.makedirs(fssm)
            with open(os.path.join(tmp, "ps_tail.dxbc"), "wb") as f:
                f.write(STUB_DXBC)
            with open(os.path.join(fssm, "ps_tail.fssm"), "wb") as f:
                f.write(bytes(mov0 + mov1 + gap))
            shader_text = (
                "; ps_tail  code 0x0..0x10 (header), gfx1013, 2 invalid words\n"
                f"\tv_mov_b32_e32 v0, 1.0 ; encoding: [{enc(mov0)}]\n"
                f"\tv_mov_b32_e32 v1, 1.0 ; encoding: [{enc(mov1)}]\n"
                "; invalid instruction encoding at 0x8\n"
            )
            src = os.path.join(disasm, "ps_tail.s")
            with open(src, "w", encoding="utf-8") as f:
                f.write(shader_text)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                with self.assertRaisesRegex(rdna2spirv.Unsupported, "fall-through into undecodable words"):
                    rdna2spirv.translate(src)

    def test_native_blob_mismatched_encoding_reachable_normalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            disasm = os.path.join(tmp, "disasm")
            fssm = os.path.join(tmp, "fssm")
            os.makedirs(disasm)
            os.makedirs(fssm)

            with open(os.path.join(tmp, "ps_norm.dxbc"), "wb") as f:
                f.write(STUB_DXBC)

            # Blob has sufficient length (64 bytes of zeros), but encoding bytes differ completely.
            # Re-anchoring fails to match, marks instruction normalized, and reachable_code rejects it.
            with open(os.path.join(fssm, "ps_norm.fssm"), "wb") as f:
                f.write(bytes(64))

            shader_text = (
                "ps_norm.s:\n"
                "\tv_mov_b32 v0, 1.0 ; encoding: [0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88]\n"
                "; invalid instruction encoding at 0x10\n"
                "\texp mrt0 v0, v0, v0, v0 done vm ; encoding: [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00]\n"
                "\ts_endpgm ; encoding: [0x00,0x00,0x81,0xbf]\n"
            )
            src = os.path.join(disasm, "ps_norm.s")
            with open(src, "w", encoding="utf-8") as f:
                f.write(shader_text)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                with self.assertRaisesRegex(rdna2spirv.Unsupported, "reachable instruction with normalized encoding"):
                    rdna2spirv.translate(src)

    def test_invalid_opcode_in_shader_listing(self):
        with tempfile.TemporaryDirectory() as tmp:
            disasm = os.path.join(tmp, "disasm")
            os.makedirs(disasm)

            with open(os.path.join(tmp, "ps_inv.dxbc"), "wb") as f:
                f.write(STUB_DXBC)

            shader_text = (
                "ps_inv.s:\n"
                "\tv_bogus_unsupported_opcode v0, 1.0\n"
                "\texp mrt0 v0, v0, v0, v0 done vm\n"
                "\ts_endpgm\n"
            )
            src = os.path.join(disasm, "ps_inv.s")
            with open(src, "w", encoding="utf-8") as f:
                f.write(shader_text)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                with self.assertRaisesRegex(rdna2spirv.Unsupported, "v_bogus_unsupported_opcode"):
                    rdna2spirv.translate(src)

    def test_empty_or_no_tab_indented_instructions(self):
        with tempfile.TemporaryDirectory() as tmp:
            disasm = os.path.join(tmp, "disasm")
            os.makedirs(disasm)

            with open(os.path.join(tmp, "ps_empty.dxbc"), "wb") as f:
                f.write(STUB_DXBC)

            # Not tab-indented: parse_file ignores non-tabbed lines
            shader_text = (
                "ps_empty.s:\n"
                "v_mov_b32 v0, 1.0\n"
                "s_endpgm\n"
            )
            src = os.path.join(disasm, "ps_empty.s")
            with open(src, "w", encoding="utf-8") as f:
                f.write(shader_text)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                with self.assertRaisesRegex(rdna2spirv.Unsupported, "no instructions \\(nothing tab-indented\\?\\)"):
                    rdna2spirv.translate(src)

    def test_invalid_shader_stage_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            disasm = os.path.join(tmp, "disasm")
            os.makedirs(disasm)

            with open(os.path.join(tmp, "xx_unknown.dxbc"), "wb") as f:
                f.write(STUB_DXBC)

            shader_text = (
                "xx_unknown.s:\n"
                "\tv_mov_b32 v0, 1.0\n"
                "\ts_endpgm\n"
            )
            src = os.path.join(disasm, "xx_unknown.s")
            with open(src, "w", encoding="utf-8") as f:
                f.write(shader_text)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                with self.assertRaisesRegex(rdna2spirv.Unsupported, "stage xx"):
                    rdna2spirv.translate(src)


if __name__ == "__main__":
    unittest.main()
