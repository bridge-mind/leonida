"""Tests for entry-block hoisting of constant-index %vgpr/%sgpr OpAccessChain pointers."""
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import support  # noqa: E402
from support import rdna2spirv  # noqa: E402
from spirv_wave_interp import run_wave  # noqa: E402


def _make_fssm_cs(user_sgprs=4, tgid_mask=1, tid_cnt=0, threads=(64, 1, 1), wave32=True):
    """Minimal FSSM blob carrying COMPUTE_PGM_RSRC2, workgroup size, and wave flag."""
    buf = bytearray(0x160)
    buf[0xD4] = 0x20 if wave32 else 0x00
    rsrc2 = ((user_sgprs & 31) << 1) | ((tgid_mask & 7) << 7) | ((tid_cnt & 3) << 11)
    struct.pack_into("<IIII", buf, 0x138, rsrc2, threads[0], threads[1], threads[2])
    return bytes(buf)


def _encode_listing(name, body_lines):
    """Build a .s listing with valid relative branch encodings for translate_listing()."""
    lbl_idx = {}
    ins_count = 0
    for ln in body_lines:
        if ln.endswith(":"):
            lbl_idx[ln[:-1]] = ins_count
        else:
            ins_count += 1
    out = [f"; {name}  synthetic hoisting test"]
    idx = 0
    for ln in body_lines:
        if ln.endswith(":"):
            continue
        enc = 0
        if ln.startswith(("s_cbranch_", "s_branch ")):
            op, target = ln.split()
            rel = (lbl_idx[target] - idx - 1) & 0xFFFF
            enc = 0xBF880000 | rel
            signed = rel if rel < 0x8000 else rel - 0x10000
            ln = f"{op} {signed}"
        b = enc.to_bytes(4, "little")
        enc_s = ",".join(f"0x{x:02x}" for x in b)
        out.append(f"\t{ln:<44} ; encoding: [{enc_s}]")
        idx += 1
    return ("\n".join(out) + "\n").encode("utf-8")


def _validate_spirv(spvasm):
    sa = shutil.which("spirv-as")
    sv = shutil.which("spirv-val")
    if not (sa and sv):
        if os.environ.get("LEONIDA_REQUIRE_SPIRV_TOOLS"):
            raise AssertionError("spirv-as / spirv-val required but not found on PATH")
        return
    with tempfile.TemporaryDirectory() as td:
        asm_p = os.path.join(td, "mod.spvasm")
        spv_p = os.path.join(td, "mod.spv")
        with open(asm_p, "w", encoding="utf-8", newline="\n") as f:
            f.write(spvasm)
        subprocess.run([sa, "--target-env", "vulkan1.2", asm_p, "-o", spv_p], check=True)
        subprocess.run([sv, "--target-env", "vulkan1.2", spv_p], check=True)


def _split_main_blocks(spvasm):
    """Return list of (label, lines) for every basic block inside %main."""
    lines = spvasm.splitlines()
    in_main = False
    blocks = []
    cur_label = None
    cur_lines = []
    for ln in lines:
        s = ln.strip()
        if s.startswith("%main = OpFunction "):
            in_main = True
            continue
        if not in_main:
            continue
        if s == "OpFunctionEnd":
            if cur_label is not None:
                blocks.append((cur_label, cur_lines))
            break
        if s.endswith("= OpLabel"):
            if cur_label is not None:
                blocks.append((cur_label, cur_lines))
            cur_label = s.split(" = ")[0]
            cur_lines = []
        elif cur_label is not None:
            cur_lines.append(s)
    return blocks


def _f32_bits(val):
    return struct.unpack("<I", struct.pack("<f", val))[0]


class RegPtrHoistTests(unittest.TestCase):
    def test_entry_hoisting_order_and_dedup_in_structured_cfg(self):
        """Constant-index %vgpr/%sgpr OpAccessChain pointers are emitted once in %entry after all OpVariables."""
        body = [
            "s_mov_b32 s4, 0",
            "s_mov_b32 s5, 4",
            "v_mov_b32 v0, 10",
            "v_mov_b32 v1, 3",
            "v_mov_b32 v2, 7",
            "v_mov_b32 v3, 0",
            "v_writelane_b32 v3, 1, 5",      # diverge lane 5 only (also tests hoisted v_writelane_b32 ptr)
            "BB_loop:",
            "v_cmp_eq_u32 vcc_lo, v3, 0",
            "s_and_saveexec_b32 s6, vcc_lo",
            "s_cbranch_execz BB_skip",
            "v_add_nc_u32 v0, v0, v1",
            "v_mul_lo_u32 v1, v1, 2",
            "v_add_nc_u32 v2, v2, s4",
            "BB_skip:",
            "s_or_b32 exec_lo, exec_lo, s6",
            "s_add_u32 s4, s4, 1",
            "s_cmp_lt_u32 s4, s5",
            "s_cbranch_scc1 BB_loop",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_listing("ps_hoist_cfg", _encode_listing("ps_hoist_cfg", body))
        _validate_spirv(spvasm)

        blocks = _split_main_blocks(spvasm)
        self.assertGreater(len(blocks), 1, "expected multi-block structured CFG")
        entry_label, entry_lines = blocks[0]
        self.assertEqual(entry_label, "%entry")

        # Every OpVariable in %entry must strictly precede every %reg_* OpAccessChain.
        last_var = max(i for i, ln in enumerate(entry_lines) if "= OpVariable " in ln)
        hoisted = [(i, ln) for i, ln in enumerate(entry_lines)
                   if re.match(r"^%reg_[vs]\d+ = OpAccessChain %_ptr_Function_uint %[vs]gpr %c[0-9a-f]+$", ln)]
        self.assertGreater(len(hoisted), 0, "expected hoisted %reg_v*/%reg_s* pointers in %entry")
        self.assertTrue(all(i > last_var for i, _ in hoisted))

        # No duplicate hoisted pointer names, and zero constant-index %vgpr/%sgpr OpAccessChain in non-entry blocks.
        names = [ln.split()[0] for _, ln in hoisted]
        self.assertEqual(len(names), len(set(names)))
        for lbl, blines in blocks[1:]:
            for ln in blines:
                if "OpAccessChain %_ptr_Function_uint %vgpr " in ln or "OpAccessChain %_ptr_Function_uint %sgpr " in ln:
                    self.fail(f"unexpected non-entry constant-index register OpAccessChain in {lbl}: {ln}")

        # Semantic check across 32 lanes via run_wave: lane 5 stays masked out; other 31 lanes run 4 iterations.
        res = run_wave(spvasm, wave=32)
        v0, v1, v2 = 10, 3, 7
        for s4 in range(4):
            v0 = (v0 + v1) & 0xFFFFFFFF
            v1 = (v1 * 2) & 0xFFFFFFFF
            v2 = (v2 + s4) & 0xFFFFFFFF
        active_expected = (v0, v1, v2, 0)
        masked_expected = (10, 3, 7, 1)
        for lane in range(32):
            self.assertEqual(
                tuple(res["outputs"][lane]["%out_mrt0"]),
                masked_expected if lane == 5 else active_expected,
            )

    def test_heavy_alu_deduplication_and_exact_results(self):
        """Repeated accesses to the same VGPRs reuse a single hoisted pointer and match exact math."""
        lines = [
            "v_mov_b32 v0, 1.0",
            "v_mov_b32 v1, 0.5",
            "v_mov_b32 v2, 0.25",
            "v_mov_b32 v3, 2.0",
        ]
        for _ in range(30):
            lines.append("v_fma_f32 v0, v0, v1, v2")
        lines += ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"]
        spvasm = support.translate_lines("ps_hoist_alu", lines)
        _validate_spirv(spvasm)

        vgpr_ac = [ln.strip() for ln in spvasm.splitlines()
                   if "OpAccessChain %_ptr_Function_uint %vgpr " in ln]
        self.assertEqual(len(vgpr_ac), 4, f"expected 4 VGPR OpAccessChain (v0..v3), got {len(vgpr_ac)}")

        v0 = 1.0
        for _ in range(30):
            v0 = v0 * 0.5 + 0.25
        self.assertEqual(
            support.mrt0_words(spvasm),
            [_f32_bits(v0), _f32_bits(0.5), _f32_bits(0.25), _f32_bits(2.0)],
        )

    def test_dynamic_movrel_coexists_with_hoisted_ptrs(self):
        """v_movrels_b32 keeps its dynamic OpAccessChain while static accesses use hoisted pointers."""
        lines = [
            "v_mov_b32 v4, 0x11111111",
            "v_mov_b32 v5, 0x22222222",
            "v_mov_b32 v6, 0x33333333",
            "s_mov_b32 m0, 2",
            "v_movrels_b32 v0, v4",
            "v_mov_b32 v1, v5",
            "v_mov_b32 v2, v6",
            "v_mov_b32 v3, v0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("ps_hoist_movrel", lines)
        _validate_spirv(spvasm)
        self.assertEqual(
            support.mrt0_words(spvasm),
            [0x33333333, 0x22222222, 0x33333333, 0x33333333],
        )

    def test_compute_wave64_emulate_validates(self):
        """Hoisted pointers dominate all blocks in --wave64-emulate compute shaders too."""
        body = [
            "s_mov_b32 s4, 0",
            "s_mov_b32 s5, 2",
            "v_mov_b32 v0, 1.0",
            "v_mov_b32 v1, 2.0",
            "BB_loop:",
            "v_cmp_gt_f32 vcc, v0, 0.5",
            "s_and_saveexec_b64 s[6:7], vcc",
            "s_cbranch_execz BB_skip",
            "v_add_f32 v0, v0, v1",
            "v_mul_f32 v1, v1, v0",
            "BB_skip:",
            "s_or_b64 exec, exec, s[6:7]",
            "s_add_u32 s4, s4, 1",
            "s_cmp_lt_u32 s4, s5",
            "s_cbranch_scc1 BB_loop",
            "s_endpgm",
        ]
        fssm = _make_fssm_cs(user_sgprs=4, tgid_mask=1, tid_cnt=0, threads=(64, 1, 1), wave32=False)
        old_emu = rdna2spirv.WAVE64_EMULATE
        try:
            rdna2spirv.WAVE64_EMULATE = True
            spvasm = support.translate_listing("cs_hoist_w64", _encode_listing("cs_hoist_w64", body), fssm=fssm)
        finally:
            rdna2spirv.WAVE64_EMULATE = old_emu
        _validate_spirv(spvasm)
        self.assertIn("%reg_v0 = OpAccessChain %_ptr_Function_uint %vgpr %c0", spvasm)
        self.assertIn("%reg_s4 = OpAccessChain %_ptr_Function_uint %sgpr %c4", spvasm)

    def test_direct_gen_keeps_inline_access_chains(self):
        """Direct Gen() unit-test callers default to inline OpAccessChain inside g.body."""
        g = rdna2spirv.Gen("cs")
        val = g.load_reg("vgpr", 7)
        g.store_reg("vgpr", 8, val)
        body = "\n".join(g.body)
        self.assertIn("OpAccessChain %_ptr_Function_uint %vgpr %c7", body)
        self.assertIn("OpAccessChain %_ptr_Function_uint %vgpr %c8", body)
        self.assertEqual(g.reg_ptrs, {})


if __name__ == "__main__":
    unittest.main()
