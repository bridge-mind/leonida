"""Unit tests for the pure-Python RDNA2 binary ISA decoder (tools/rdna2_decode.py, #1901).

Covers:
  - All major RDNA2 encoding families (SOP1, SOP2, SOPC, SOPK, SOPP, SMEM, VOP1, VOP2, VOPC,
    VINTRP, VOP3, VOP3B, VOP3P, DPP16, DPP8, SDWA, DS, MUBUF, MTBUF, MIMG + NSA + BVH, EXP,
    FLAT/GLOBAL) from hand-encoded instruction words (no LLVM, no GPU, no game files).
  - Special cases: VOP3P `v_dot4_i32_i8` with `op_sel` 2 / `op_sel_hi` 5 (`0xCC165008 0x0C0E0501`,
    which `llvm-mc` rejects), LLVM `VCCTuple` scalar ranges ending in `vcc_lo,vcc_hi`, `s_waitcnt`,
    `s_setreg_imm32_b32`, `ds_swizzle_b32`, and `shader_disasm.disasm(..., backend="python")`.
  - Round-trip translation: raw RDNA2 machine code -> `rdna2_decode.disasm_bytes` -> `rdna2spirv.translate`
    -> `spirv_mini_interp` value verification.
  - Optional full-suite parity test against `llvm-mc` across all `runtime/test/shaders/*.asm` when
    `LLVM_MC` (or `llvm-mc` on PATH) is available.
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOLS = os.path.join(ROOT, "tools")
SEMANTIC = os.path.join(HERE, "semantic")

sys.dont_write_bytecode = True
for p in (TOOLS, SEMANTIC):
    if p not in sys.path:
        sys.path.insert(0, p)

import rdna2_decode  # noqa: E402
import rdna2spirv  # noqa: E402
import shader_disasm  # noqa: E402
import shader_ir  # noqa: E402
import spirv_mini_interp as si  # noqa: E402


def words(*ws: int) -> bytes:
    return b"".join(struct.pack("<I", w & 0xFFFFFFFF) for w in ws)


class Rdna2DecodeUnitTests(unittest.TestCase):
    def test_image_atomic_data_width(self):
        for opcode, name, mask, data in (
            (0x17, "umax", 1, "v16"), (0x17, "umax", 3, "v[16:17]"),
            (0x10, "cmpswap", 3, "v[16:17]"), (0x10, "cmpswap", 15, "v[16:19]"),
        ):
            for glc in (0, 1):
                with self.subTest(name=name, mask=mask, glc=glc):
                    code = words(0xf0001008 | opcode << 18 | mask << 8 | glc << 13, 0x00021004)
                    ins, = rdna2_decode.decode_bytes(code)["insns"]
                    asm = (f"image_atomic_{name} {data}, v[4:5], s[8:15] "
                           f"dmask:0x{mask:x} dim:SQ_RSRC_IMG_2D unorm" + (" glc" if glc else ""))
                    expected = shader_ir.parse_line(asm)
                    for field in ("op", "args", "mods"):
                        self.assertEqual(ins[field], expected[field])

    def test_image_slc_is_independent_of_sampler(self):
        for sampler in (0, 64):
            for slc in (0, 1):
                with self.subTest(sampler=sampler, slc=slc):
                    code = words(0xF0800108 | (slc << 25), 0x0004090A | ((sampler // 4) << 21))
                    ins, = rdna2_decode.decode_bytes(code)["insns"]
                    self.assertEqual(ins["args"][-1]["r"], sampler)
                    self.assertEqual("slc" in ins["mods"], bool(slc))

    def test_mix_source_modifiers(self):
        # LLVM gfx1013: v17, -v19, |v21|, -|v23| across all three mix variants.
        for opcode, name in enumerate(("v_fma_mix_f32", "v_fma_mixlo_f16", "v_fma_mixhi_f16")):
            with self.subTest(name=name):
                code = words(0xCC200611 + (opcode << 16), 0xA45E2B13)
                ins, = rdna2_decode.decode_bytes(code)["insns"]
                expected = shader_ir.parse_line(f"{name} v17, -v19, |v21|, -|v23|")
                self.assertEqual((ins["op"], ins["args"], ins["mods"]),
                                 (name, expected["args"], expected["mods"]))

    def test_vop3_legacy_mad_fma_target(self):
        code = words(0xD5400001, 0x04120702)
        self.assertEqual(rdna2_decode.decode_bytes(code)["insns"][0]["op"], "v_mad_legacy_f32")
        for cpu, op in (("gfx1013", "v_mad_legacy_f32"), ("gfx1030", "v_fma_legacy_f32")):
            with self.subTest(cpu=cpu):
                asm, invalid, _ = shader_disasm.disasm(code, cpu, backend="python")
                self.assertEqual(invalid, 0)
                self.assertIn(op + " v1, v2, v3, v4", asm)

    def test_sopp_endpgm_branch_and_waitcnt(self):
        # s_waitcnt vmcnt(0) lgkmcnt(0) = 0xBF8C0070; s_cbranch_scc1 -4 = 0xBF85FFFC;
        # s_branch -20 = 0xBF82FFEC; s_endpgm = 0xBF810000
        code = words(0xBF8C0070, 0xBF85FFFC, 0xBF82FFEC, 0xBF810000)
        info = rdna2_decode.decode_bytes(code)
        self.assertEqual(info["invalid"], 0)
        self.assertEqual([i["op"] for i in info["insns"]], ["s_waitcnt", "s_cbranch_scc1", "s_branch", "s_endpgm"])
        self.assertEqual([a["t"] for a in info["insns"][0]["args"]], ["vmcnt(0)", "lgkmcnt(0)"])
        self.assertEqual(info["insns"][1]["args"], [{"k": "imm", "v": -4}])
        self.assertEqual(info["insns"][2]["args"], [{"k": "imm", "v": -20}])
        self.assertEqual([i["pc"] for i in info["insns"]], [0, 4, 8, 12])
        text, inv = rdna2_decode.disasm_bytes(code)
        self.assertEqual(inv, 0)
        self.assertIn("s_cbranch_scc1 -4", text)
        self.assertIn("s_branch -20", text)

    def test_s_ttracedata_imm_keeps_simm16(self):
        for word, value in ((0xBFA80000, 0x0), (0xBFA80001, 0x1),
                            (0xBFA80063, 0x63), (0xBFA80064, 0x64),
                            (0xBFA81234, 0x1234), (0xBFA8FFFF, 0xFFFF)):
            with self.subTest(word=hex(word)):
                code = words(word)
                info = rdna2_decode.decode_bytes(code)
                self.assertEqual(info["invalid"], 0)
                self.assertEqual(info["insns"][0]["op"], "s_ttracedata_imm")
                self.assertEqual(info["insns"][0]["args"],
                                 [{"k": "imm", "v": value}])
                text, invalid = rdna2_decode.disasm_bytes(code)
                self.assertEqual(invalid, 0)
                self.assertIn(f"s_ttracedata_imm 0x{value:x}", text)

    def test_disasm_formats_each_instruction_before_decoding_the_next(self):
        code = words(0xBF800000, 0xBF810000)  # s_nop 0; s_endpgm
        events = []
        decode_one = rdna2_decode._decode_one
        format_insn = rdna2_decode.format_insn

        def record_decode(code, offset, *args, **kwargs):
            events.append(("decode", offset))
            return decode_one(code, offset, *args, **kwargs)

        def record_format(ins):
            events.append(("format", ins["pc"]))
            return format_insn(ins)

        with (
            patch.object(rdna2_decode, "_decode_one", record_decode),
            patch.object(rdna2_decode, "format_insn", record_format),
        ):
            text, invalid = rdna2_decode.disasm_bytes(code, code_start=0x100, name="synthetic")

        expected_text = (
            f"\t{'s_nop 0':<40}; encoding: [0x00,0x00,0x80,0xbf]\n"
            f"\t{'s_endpgm':<40}; encoding: [0x00,0x00,0x81,0xbf]\n"
        )
        self.assertEqual((text, invalid), (expected_text, 0))
        self.assertEqual(events, [("decode", 0), ("format", 0x100), ("decode", 4), ("format", 0x104)])

    def test_sop1_sop2_sopc_sopk_with_literals(self):
        # s_mov_b32 s4, 0x12345678 (SOP1: op=3, sdst=4, ssrc0=255 + lit)
        w_mov = (0x17D << 23) | (4 << 16) | (3 << 8) | 0xFF
        # s_add_u32 s5, s4, 1 (SOP2: op=0, sdst=5, ssrc1=129 (imm 1), ssrc0=4)
        w_add = (0b10 << 30) | (0 << 23) | (5 << 16) | (129 << 8) | 4
        # s_cmp_eq_u32 s5, s4 (SOPC: op=6, ssrc1=4, ssrc0=5)
        w_cmp = (0x17E << 23) | (6 << 16) | (4 << 8) | 5
        # s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 3 (SOPK op=0x15, simm16 = 1 | (2<<6) | (1<<11))
        simm16 = 1 | (2 << 6) | (1 << 11)
        w_setreg = (0xB << 28) | (0x15 << 23) | simm16
        code = words(w_mov, 0x12345678, w_add, w_cmp, w_setreg, 3)
        info = rdna2_decode.decode_bytes(code)
        self.assertEqual(info["invalid"], 0)
        self.assertEqual([i["op"] for i in info["insns"]],
                         ["s_mov_b32", "s_add_u32", "s_cmp_eq_u32", "s_setreg_imm32_b32"])
        self.assertEqual(info["insns"][0]["args"][1], {"k": "imm", "v": 0x12345678})
        self.assertEqual(info["insns"][3]["args"], [{"k": "special", "t": "hwreg(HW_REG_MODE, 2, 2)"},
                                                    {"k": "imm", "v": 3}])

    def test_vop1_vop2_vopc_and_readfirstlane(self):
        # v_mov_b32_e32 v2, 1.0 (VOP1 op=1, vdst=2, src0=242)
        w_mov = (0x3F << 25) | (2 << 17) | (1 << 9) | 242
        # v_readfirstlane_b32 s7, v2 (VOP1 op=2, vdst=7 -> sgpr 7, src0=256+2)
        w_rfl = (0x3F << 25) | (7 << 17) | (2 << 9) | (256 + 2)
        # v_fmaak_f32 v8, v2, v3, 0x3f800000 (VOP2 op=0x2d, vdst=8, vsrc1=3, src0=256+2)
        w_fmaak = (0x2D << 25) | (8 << 17) | (3 << 9) | (256 + 2)
        # v_cmp_eq_u32_e32 vcc_lo, 0, v2 (VOPC op=0xc2, vsrc1=2, src0=128)
        w_cmp = (0x3E << 25) | (0xC2 << 17) | (2 << 9) | 128
        code = words(w_mov, w_rfl, w_fmaak, 0x3F800000, w_cmp)
        info = rdna2_decode.decode_bytes(code, wave32=True)
        self.assertEqual(info["invalid"], 0)
        self.assertEqual([i["op"] for i in info["insns"]],
                         ["v_mov_b32_e32", "v_readfirstlane_b32", "v_fmaak_f32", "v_cmp_eq_u32_e32"])
        self.assertEqual(info["insns"][0]["args"][1]["v"], 1.0)
        self.assertEqual(info["insns"][1]["args"][0]["k"], "sgpr")
        self.assertEqual(info["insns"][1]["args"][0]["r"], 7)
        self.assertEqual(info["insns"][2]["args"][3], {"k": "imm", "v": 0x3F800000})
        self.assertEqual(info["insns"][3]["args"][0]["t"], "vcc_lo")
        # Wave64 uses `vcc` instead of `vcc_lo`.
        info64 = rdna2_decode.decode_bytes(words(w_cmp), wave32=False)
        self.assertEqual(info64["insns"][0]["args"][0]["t"], "vcc")

    def test_vop3p_dot4_i8_with_opsel_that_llvm_mc_rejects(self):
        # Raw RDNA2 hardware-measured word from docs/fspx.md & tests/semantic/test_alu_dot4_i8.py:
        # 0xCC165008 0x0C0E0501 = v_dot4_i32_i8 v8, v1, v2, v3 op_sel:[0,1,0] op_sel_hi:[1,0,1]
        code = words(0xCC165008, 0x0C0E0501)
        info = rdna2_decode.decode_bytes(code)
        self.assertEqual(info["invalid"], 0)
        ins = info["insns"][0]
        self.assertEqual(ins["op"], "v_dot4_i32_i8")
        self.assertEqual([a["r"] for a in ins["args"]], [8, 1, 2, 3])
        self.assertEqual(ins["mods"], ["op_sel:[0,1,0]", "op_sel_hi:[1,0,1]"])
        # Formatting + shader_ir re-parse round-trips identically.
        line = rdna2_decode.format_insn(ins)
        reparsed = shader_ir.parse_line(line)
        self.assertEqual(reparsed["op"], ins["op"])
        self.assertEqual(reparsed["mods"], ins["mods"])

    def test_smem_vcc_tuple_and_offsets(self):
        # s_buffer_load_dwordx8 [s100..s105,vcc_lo,vcc_hi], s[0:3], null (SMEM op=0x0b, sdata=100, sbase=0, soff=125)
        w0 = (0x3D << 26) | (0x0B << 18) | (100 << 6) | 0
        w1 = (125 << 25)
        info = rdna2_decode.decode_bytes(words(w0, w1))
        self.assertEqual(info["invalid"], 0)
        ins = info["insns"][0]
        self.assertEqual(ins["op"], "s_buffer_load_dwordx8")
        self.assertEqual(ins["args"][0], {
            "k": "slist", "r": [100, 101, 102, 103, 104, 105, "vcc_lo", "vcc_hi"], "n": 8,
        })
        self.assertEqual(ins["args"][2], {"k": "special", "t": "null", "neg": False, "abs": False})

    def test_mubuf_mtbuf_mimg_global_ds_exp_vintrp(self):
        # MUBUF: buffer_load_format_d16_xyzw v[8:9], v5, s[4:7], 0 offen (encoding verified against llvm-mc)
        b_mubuf = bytes([0x00, 0x10, 0x0C, 0xE2, 0x05, 0x08, 0x01, 0x80])
        # MIMG BVH: image_bvh_intersect_ray v[20:23], [v2..v12], s[4:7]
        b_bvh = bytes([0x07, 0x9F, 0x98, 0xF1, 0x02, 0x14, 0x01, 0x00,
                       0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x00, 0x00])
        # EXP: exp mrt0 v0, v1, v2, v3 done vm
        w_exp0 = (0x3E << 26) | (1 << 12) | (1 << 11) | (0 << 4) | 0xF
        w_exp1 = 0 | (1 << 8) | (2 << 16) | (3 << 24)
        # VINTRP: v_interp_p1_f32 v2, v0, attr1.y (op=0, vdst=2, attr=1, chan=1, vsrc=0)
        w_vint = (0x32 << 26) | (2 << 18) | (0 << 16) | (1 << 10) | (1 << 8) | 0
        # GLOBAL: global_store_dword v[4:5], v1, off offset:-4 (seg=2, op=0x1c, saddr=125)
        w_glb0 = (0x37 << 26) | (0x1C << 18) | (2 << 14) | (0xFFC)
        w_glb1 = (125 << 16) | (1 << 8) | 4
        code = b_mubuf + b_bvh + words(w_exp0, w_exp1, w_vint, w_glb0, w_glb1)
        info = rdna2_decode.decode_bytes(code)
        self.assertEqual(info["invalid"], 0)
        self.assertEqual([i["op"] for i in info["insns"]], [
            "buffer_load_format_d16_xyzw",
            "image_bvh_intersect_ray",
            "exp",
            "v_interp_p1_f32",
            "global_store_dword",
        ])
        self.assertEqual(info["insns"][1]["args"][1], {"k": "vlist", "r": list(range(2, 13))})
        self.assertEqual(info["insns"][1]["mods"], [])
        self.assertEqual(info["insns"][2]["mods"], ["done", "vm"])
        self.assertEqual(info["insns"][3]["args"][2], {"k": "attr", "a": 1, "c": 1})
        self.assertEqual(info["insns"][4]["args"][2]["t"], "off")
        self.assertEqual(info["insns"][4]["mods"], ["offset:-4"])

    def test_sdwa_inline_sources(self):
        values = list(range(65)) + list(range(-1, -17, -1))
        floats = [0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 0.15915494]
        constants = [(128 + i, "imm", v) for i, v in enumerate(values)]
        constants += [(240 + i, "fimm", v) for i, v in enumerate(floats)]
        # gfx1013 v_mov_b32, v_add_f32 and v_cmp_eq_f32 SDWA encodings.
        for opcode in (0x7E0002F9, 0x060000F9, 0x7C0400F9):
            slots = (0,) if opcode == 0x7E0002F9 else (0, 1)
            for slot in slots:
                for code, kind, value in constants:
                    with self.subTest(opcode=hex(opcode), slot=slot, code=code):
                        first = opcode
                        second = code | (1 << 23)
                        if slot == 1:
                            first |= code << 9
                            second = 1 | (1 << 31)
                        ins = rdna2_decode.decode_bytes(words(first, second))["insns"][0]
                        operand = ins["args"][1 + slot]
                        self.assertEqual((operand["k"], operand["v"]), (kind, value))

    def test_roundtrip_binary_to_spirv_and_interpreter(self):
        # Synthetic pixel shader binary: v_mov_b32 v0, 1.0; v_add_f32 v1, v0, 2.0; exp mrt0 v0, v1, v0, v1 done vm; s_endpgm
        w_mov = (0x3F << 25) | (0 << 17) | (1 << 9) | 242        # v0 = 1.0
        w_add = (0x03 << 25) | (1 << 17) | (0 << 9) | 244        # v1 = 2.0 + v0 = 3.0
        w_exp0 = (0x3E << 26) | (1 << 12) | (1 << 11) | 0xF
        w_exp1 = 0 | (1 << 8) | (0 << 16) | (1 << 24)
        w_end = 0xBF810000
        code = words(w_mov, w_add, w_exp0, w_exp1, w_end)

        asm, bad, err = shader_disasm.disasm(code, "gfx1013", backend="python")
        self.assertEqual(bad, 0)
        self.assertEqual(err, "")
        with tempfile.TemporaryDirectory() as tmp:
            disasm_dir = os.path.join(tmp, "disasm")
            os.mkdir(disasm_dir)
            with open(os.path.join(tmp, "ps_dec.dxbc"), "wb") as f:
                f.write(b"DXBC" + bytes(28))
            src = os.path.join(disasm_dir, "ps_dec.s")
            with open(src, "w", encoding="utf-8") as f:
                f.write("; ps_dec  code 0x0..0x14 (python), gfx1013, 0 invalid words\n" + asm)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spv = rdna2spirv.translate(src)
        out = si.run_fragment(spv)["%out_mrt0"]
        self.assertEqual(list(out), [0x3F800000, 0x40400000, 0x3F800000, 0x40400000])

    def test_parity_with_llvm_mc_on_shader_fixtures(self):
        mc = os.environ.get("LLVM_MC") or shutil.which("llvm-mc") or shutil.which("llvm-mc-18")
        if not mc:
            self.skipTest("llvm-mc not available")
        asms = sorted(glob.glob(os.path.join(ROOT, "runtime", "test", "shaders", "*.asm")))
        self.assertGreater(len(asms), 50)

        def norm_op(op):
            for suf in ("_e32", "_e64", "_sdwa", "_dpp"):
                if op.endswith(suf):
                    op = op[:-len(suf)]
            return op

        def norm_arg(a, op=""):
            k = a["k"]
            if k in ("vgpr", "sgpr"):
                return (k, a["r"], a.get("n", 1), bool(a.get("neg")), bool(a.get("abs")), bool(a.get("sext")))
            if k == "imm":
                # LLVM 18 disassembles SOPP branch offsets unsigned (0xfff0), LLVM 21/22 signed (-16).
                mask = 0xFFFF if op in rdna2_decode._SOPP_BRANCHES else 0xFFFFFFFF
                return ("imm", a["v"] & mask, bool(a.get("neg")), bool(a.get("abs")))
            if k == "fimm":
                return ("fimm", round(a["v"], 6), bool(a.get("neg")), bool(a.get("abs")))
            if k in ("vlist", "slist"):
                return (k, tuple(a["r"]))
            if k == "attr":
                return ("attr", a["a"], a["c"])
            return (k, a.get("t", ""), bool(a.get("neg")), bool(a.get("abs")))

        checked = 0
        for path in asms:
            with open(path, encoding="utf-8") as f:
                txt = f.read()
            m = re.search(r"^;\s*mattr:\s*(\S+)", txt, re.M)
            mattr = m.group(1) if m else ""
            wave32 = "-wavefrontsize32" not in mattr and "+wavefrontsize64" not in mattr
            cmd = [mc, "-triple=amdgcn--amdpal", "-mcpu=gfx1013", "-show-encoding"]
            if mattr:
                cmd.append("-mattr=" + mattr)
            if os.path.basename(path) == "vs_clip.asm":
                cmd.append("-defsym=PARAM_SLOT=1")
            r = subprocess.run(cmd, input=txt, text=True, capture_output=True)
            if r.returncode != 0 or "error:" in r.stderr:
                continue
            raw = bytearray()
            for ln in r.stdout.splitlines():
                if "; encoding:" not in ln or ln.lstrip().startswith("."):
                    continue
                ins = shader_ir.parse_line(ln)
                if ins and ins["enc"]:
                    raw.extend(bytes.fromhex(ins["enc"]))
            dcmd = [mc, "-disassemble", "-triple=amdgcn--amdpal", "-mcpu=gfx1013", "-show-encoding"]
            if mattr:
                dcmd.append("-mattr=" + mattr)
            dr = subprocess.run(dcmd, input=" ".join(f"0x{b:02x}" for b in raw), text=True, capture_output=True)
            llvm_insns = []
            pc = 0
            for ln in dr.stdout.splitlines():
                if "; encoding:" not in ln or ln.lstrip().startswith("."):
                    continue
                ins = shader_ir.parse_line(ln)
                if ins and ins["enc"]:
                    ins["pc"] = pc
                    pc += len(ins["enc"]) // 2
                    llvm_insns.append(ins)
            dec = rdna2_decode.decode_bytes(bytes(raw), wave32=wave32)
            self.assertEqual(dec["invalid"], 0, path)
            self.assertEqual(len(dec["insns"]), len(llvm_insns), path)
            for li, di in zip(llvm_insns, dec["insns"]):
                self.assertEqual(di["enc"], li["enc"], path)
                self.assertEqual(norm_op(di["op"]), norm_op(li["op"]), path)
                self.assertEqual([norm_arg(x, di["op"]) for x in di["args"]],
                                 [norm_arg(x, li["op"]) for x in li["args"]],
                                 f"{path}: {li['op']}")
                self.assertEqual(sorted(di["mods"]), sorted(li["mods"]), f"{path}: {li['op']}")
                checked += 1
        self.assertGreater(checked, 5000)

    def test_sdwa_inline_constants_and_sgpr(self):
        # Exact encodings from runtime/test/shaders/ps_sdwa_f16_const.asm:
        # 1) v_add_f16_sdwa v2, 1.0, v1 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:WORD_0
        # 2) v_mul_f16_sdwa v3, 2.0, v1 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:BYTE_1 src1_sel:WORD_1
        # 3) v_add_f16_sdwa v4, s8, v1  dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:WORD_0
        # 4) v_cvt_f16_u16_sdwa v5, 1.0 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 (VOP1 opc 0x50)
        # 5) v_cmp_eq_u32_sdwa vcc_lo, 5, -1 src0_sel:WORD_0 src1_sel:WORD_0 (VOPC opc 0xc2, s0=133, s1=193)
        raw = bytes([
            0xF9, 0x02, 0x04, 0x64, 0xF2, 0x14, 0x85, 0x04,
            0xF9, 0x02, 0x06, 0x6A, 0xF4, 0x14, 0x81, 0x05,
            0xF9, 0x02, 0x08, 0x64, 0x08, 0x14, 0x85, 0x04,
            0xF9, 0xA0, 0x0A, 0x7E, 0xF2, 0x14, 0x85, 0x00,
            0xF9, 0x82, 0x85, 0x7D, 0x85, 0x00, 0x84, 0x84,
        ])
        dec = rdna2_decode.decode_bytes(raw, wave32=True)
        self.assertEqual(dec["invalid"], 0)
        insns = dec["insns"]
        self.assertEqual(len(insns), 5)

        self.assertEqual(insns[0]["op"], "v_add_f16_sdwa")
        self.assertEqual(insns[0]["args"][1]["k"], "fimm")
        self.assertEqual(insns[0]["args"][1]["v"], 1.0)
        self.assertEqual(insns[0]["args"][2]["k"], "vgpr")
        self.assertEqual(insns[0]["args"][2]["r"], 1)

        self.assertEqual(insns[1]["op"], "v_mul_f16_sdwa")
        self.assertEqual(insns[1]["args"][1]["k"], "fimm")
        self.assertEqual(insns[1]["args"][1]["v"], 2.0)

        self.assertEqual(insns[2]["op"], "v_add_f16_sdwa")
        self.assertEqual(insns[2]["args"][1]["k"], "sgpr")
        self.assertEqual(insns[2]["args"][1]["r"], 8)

        self.assertEqual(insns[3]["op"], "v_cvt_f16_u16_sdwa")
        self.assertEqual(insns[3]["args"][1]["k"], "fimm")
        self.assertEqual(insns[3]["args"][1]["v"], 1.0)

        self.assertEqual(insns[4]["op"], "v_cmp_eq_u32_sdwa")
        self.assertEqual(insns[4]["args"][1]["k"], "imm")
        self.assertEqual(insns[4]["args"][1]["v"], 5)
        self.assertEqual(insns[4]["args"][2]["k"], "imm")
        self.assertEqual(insns[4]["args"][2]["v"], -1)

        # Verify formatted text round-trips cleanly through shader_ir.parse_line.
        text, bad = rdna2_decode.disasm_bytes(raw, wave32=True)
        self.assertEqual(bad, 0)
        parsed = [shader_ir.parse_line(ln) for ln in text.splitlines() if ln.strip()]
        self.assertEqual(len(parsed), 5)
        self.assertEqual(parsed[0]["args"][1]["k"], "fimm")
        self.assertEqual(parsed[0]["args"][1]["v"], 1.0)
        self.assertEqual(parsed[3]["args"][1]["k"], "fimm")
        self.assertEqual(parsed[3]["args"][1]["v"], 1.0)


if __name__ == "__main__":
    unittest.main()
