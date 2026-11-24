"""Unit tests for read-before-write operand classification in scripts/vs_input_survey.py.

    python -I tests/test_vs_input_survey_rbw.py

Uses small synthetic RDNA2 instruction listings only; no game data.
"""
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import shader_ir  # noqa: E402
import vs_input_survey as V  # noqa: E402


def parse(lines):
    return [shader_ir.parse_line(line) for line in lines]


class ReadBeforeWriteTests(unittest.TestCase):
    def test_carry_in_carry_out_writes_both_vdst_and_sdst(self):
        # v_add_co_ci_u32_e64 writes v1 and s[2:3], and reads v0, v2, s[4:5]; s[2:3] is not RBW.
        insns = parse([
            "\tv_add_co_ci_u32_e64 v1, s[2:3], v0, v2, s[4:5]",
            "\ts_or_b64 s[6:7], s[2:3], s[2:3]",
            "\tv_sub_co_ci_u32_e64 v3, s[8:9], v1, v2, s[2:3]",
            "\tv_subrev_co_ci_u32_e64 v4, s[10:11], v3, v2, s[8:9]",
        ])
        read_s, read_v = V.read_before_write(insns)
        self.assertEqual(read_s, {4, 5})
        self.assertEqual(read_v, {0, 2})

    def test_dual_dest_valu_ops_write_scalar_carry_or_scale(self):
        insns = parse([
            "\tv_mad_u64_u32 v[0:1], s[2:3], v4, v5, v[6:7]",
            "\tv_mad_i64_i32 v[8:9], s[10:11], v0, v1, v[6:7]",
            "\tv_div_scale_f32 v12, s[12:13], v4, v5, v6",
            "\tv_div_scale_f64 v[14:15], s[14:15], v[8:9], v[0:1], v[6:7]",
            "\ts_and_b64 s[16:17], s[2:3], s[10:11]",
            "\ts_or_b64 s[18:19], s[12:13], s[14:15]",
        ])
        read_s, read_v = V.read_before_write(insns)
        self.assertEqual(read_s, set())
        self.assertEqual(read_v, {4, 5, 6, 7})

    def test_v_cmp_two_arg_implicit_vcc_reads_both_operands(self):
        # Implicit-VCC 2-operand form has no explicit destination: v0 is a source, not written.
        insns = parse([
            "\tv_cmp_lt_f32_e32 v0, v1",
            "\tv_mov_b32_e32 v2, v0",
        ])
        read_s, read_v = V.read_before_write(insns)
        self.assertEqual(read_s, set())
        self.assertEqual(read_v, {0, 1})

    def test_v_cmp_three_arg_writes_explicit_scalar_dest(self):
        insns = parse([
            "\tv_cmp_eq_u32_e64 s[2:3], v0, v1",
            "\ts_mov_b64 s[4:5], s[2:3]",
        ])
        read_s, read_v = V.read_before_write(insns)
        self.assertEqual(read_s, set())
        self.assertEqual(read_v, {0, 1})

    def test_nsa_vlist_and_vcctuple_slist_operands(self):
        # NSA image_sample reads every VGPR in [v0, v3]; VCCTuple slist writes s100..s105 and reads s2..s3
        # as well as unwritten slist sources [s12, s13, s14, s15].
        insns = parse([
            "\ts_load_dwordx8 [s100, s101, s102, s103, s104, s105, vcc_lo, vcc_hi], s[2:3], 0x0",
            "\ts_buffer_load_dword s20, [s12, s13, s14, s15], 0x0",
            "\timage_sample v[4:7], [v0, v3], [s100, s101, s102, s103, s104, s105, vcc_lo, vcc_hi], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D",
        ])
        read_s, read_v = V.read_before_write(insns)
        self.assertEqual(read_s, {2, 3, 8, 9, 10, 11, 12, 13, 14, 15})
        self.assertEqual(read_v, {0, 3})

    def test_tbuffer_and_scalar_stores_have_no_destination(self):
        insns = parse([
            "\ttbuffer_store_format_xyzw v[0:3], v4, s[4:7], 0 format:[BUF_FMT_32_32_32_32_FLOAT] offen",
            "\ts_store_dword s8, s[0:1], 0x0",
            "\ts_buffer_store_dword s9, s[12:15], 0x0",
            "\tv_mov_b32_e32 v5, v0",
            "\ts_mov_b32 s10, s8",
        ])
        read_s, read_v = V.read_before_write(insns)
        self.assertEqual(read_s, {0, 1, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15})
        self.assertEqual(read_v, {0, 1, 2, 3, 4})

    def test_atomics_distinguish_glc_and_global_vs_buffer(self):
        # Non-glc buffer/global atomics write nothing; glc global_atomic writes vdst at args[0] without reading it;
        # glc buffer_atomic both reads and writes vdata at args[0].
        insns = parse([
            "\tbuffer_atomic_add v0, v1, s[0:3], 0 offen",
            "\tglobal_atomic_add v2, v3, s[4:5]",
            "\tglobal_atomic_add v4, v2, v3, s[4:5] glc",
            "\tbuffer_atomic_add v5, v4, s[0:3], 0 offen glc",
            "\tv_add_nc_u32_e32 v6, v4, v5",
        ])
        read_s, read_v = V.read_before_write(insns)
        self.assertEqual(read_s, {0, 1, 2, 3, 4, 5})
        self.assertEqual(read_v, {0, 1, 2, 3, 5})

    def test_ds_non_returning_vs_returning_ops(self):
        # Non-returning ds_add_u32 / ds_store_b32 / ds_gws_init do not write args[0];
        # ds_add_rtn_u32, ds_read_b32, ds_ordered_count, ds_swizzle/permute/bpermute, and ds_append/consume do.
        insns = parse([
            "\tds_add_u32 v0, v1",
            "\tds_store_b32 v2, v3",
            "\tds_gws_init v14 gds",
            "\tds_ordered_count v5, v1 offset:772 gds",
            "\tds_add_rtn_u32 v4, v0, v1",
            "\tds_read_b32 v6, v4",
            "\tds_swizzle_b32 v7, v5 offset:0x0",
            "\tds_permute_b32 v8, v6, v7",
            "\tds_bpermute_b32 v9, v7, v8",
            "\tds_append v10 gds",
            "\tds_consume v11 gds",
            "\tv_add_nc_u32_e32 v12, v5, v10",
            "\tv_add_nc_u32_e32 v13, v9, v11",
        ])
        read_s, read_v = V.read_before_write(insns)
        self.assertEqual(read_s, set())
        self.assertEqual(read_v, {0, 1, 2, 3, 14})

    def test_flat_and_image_atomics_with_and_without_glc(self):
        # Non-glc flat/image atomics write no destination; glc flat_atomic writes vdst without reading it;
        # glc image_atomic reads and writes vdata at args[0].
        insns = parse([
            "\tflat_atomic_add v[0:1], v2",
            "\timage_atomic_add v3, v4, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D",
            "\tflat_atomic_add v5, v[0:1], v2 glc",
            "\timage_atomic_add v6, v5, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D glc",
            "\tv_add_nc_u32_e32 v7, v5, v6",
        ])
        read_s, read_v = V.read_before_write(insns)
        self.assertEqual(read_s, {0, 1, 2, 3, 4, 5, 6, 7})
        self.assertEqual(read_v, {0, 1, 2, 3, 4, 6})

    def test_sgpr_rmw_and_dot_accumulate_and_v_swap(self):
        insns = parse([
            "\ts_addk_i32 s4, 16",
            "\ts_bitset1_b32 s5, 3",
            "\tv_fmac_f32_e32 v10, v11, v12",
            "\tv_mac_f32_e32 v13, v11, v12",
            "\tv_dot2c_f32_f16 v14, v11, v12",
            "\tv_dot4c_i32_i8 v1, v2, v3",
            "\tv_dot8c_i32_i4 v15, v2, v3",
            "\tv_writelane_b32 v16, s4, 0",
            "\tv_swap_b32 v6, v7",
            "\tv_mov_b32_e32 v8, v7",
        ])
        read_s, read_v = V.read_before_write(insns)
        self.assertEqual(read_s, {4, 5})
        self.assertEqual(read_v, {1, 2, 3, 6, 7, 10, 11, 12, 13, 14, 15, 16})

    def test_vcc_carry_out_forms(self):
        insns = parse([
            "\tv_add_co_u32_e32 v1, vcc_lo, v0, v2",
            "\tv_add_co_ci_u32_e32 v3, vcc_lo, v1, v4, vcc_lo",
        ])
        read_s, read_v = V.read_before_write(insns)
        self.assertEqual(read_s, set())
        self.assertEqual(read_v, {0, 2, 4})

    def test_survey_counts_nsa_vertex_and_instance_ids(self):
        # End-to-end check: a VS that reads v0 and v3 only through an NSA vlist lands in the v0+v3 family.
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "disasm"))
            listing = (
                "; nsa_vs\n"
                "\timage_load v[4:7], [v0, v3], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D\n"
                "\texp pos0 v4, v5, v6, v7 done\n"
                "\ts_endpgm\n"
            )
            with open(os.path.join(tmp, "disasm", "nsa_vs.s"), "w", encoding="utf-8", newline="\n") as f:
                f.write(listing)
            with open(os.path.join(tmp, "index.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump([{"name": "nsa_vs", "file": "nsa_vs.dxbc", "stage": "vs"}], f)
            result = V.survey(tmp)
        self.assertEqual(result["input_vgprs_read"], {"v0": 1, "v3": 1})
        self.assertEqual(result["families"], {"reads v0 (vertex id) and v3 (instance id)": 1})


if __name__ == "__main__":
    unittest.main()
