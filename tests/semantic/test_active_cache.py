"""Tests for per-block %active caching and static no-EXEC-write dst() fast path."""
import os
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
    buf = bytearray(0x160)
    buf[0xD4] = 0x20 if wave32 else 0x00
    rsrc2 = ((user_sgprs & 31) << 1) | ((tgid_mask & 7) << 7) | ((tid_cnt & 3) << 11)
    struct.pack_into("<IIII", buf, 0x138, rsrc2, threads[0], threads[1], threads[2])
    return bytes(buf)


def _encode_listing(name, body_lines):
    lbl_idx = {}
    ins_count = 0
    for ln in body_lines:
        if ln.endswith(":"):
            lbl_idx[ln[:-1]] = ins_count
        else:
            ins_count += 1
    out = [f"; {name}  synthetic active-cache test"]
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


def _active_loads_in_main(spvasm):
    """Return list of (block_label, count_of_active_loads) for every block inside %main."""
    in_main = False
    cur = None
    counts = []
    for raw in spvasm.splitlines():
        s = raw.strip()
        if s.startswith("%main = OpFunction "):
            in_main = True
            continue
        if not in_main:
            continue
        if s == "OpFunctionEnd":
            break
        if s.endswith("= OpLabel"):
            cur = [s.split(" = ")[0], 0]
            counts.append(cur)
        elif cur is not None and "= OpLoad %bool %active" in s:
            cur[1] += 1
    return counts


class ActiveCacheTests(unittest.TestCase):
    def test_divergent_block_reuses_active_load_and_tracks_exec_writes(self):
        """Mid-block EXEC save/restore updates the cached %active SSA value without stale reuse."""
        body = [
            "v_mov_b32 v0, 100",
            "v_mov_b32 v1, 200",
            "v_mov_b32 v2, 300",
            "v_mov_b32 v3, 0",
            "v_writelane_b32 v3, 1, 7",       # lane 7 gets v3=1, all other lanes v3=0
            # First region: all 32 lanes active
            "v_add_nc_u32 v0, v0, 1",
            "v_add_nc_u32 v0, v0, 2",
            # Mask off lane 7 in the SAME basic block (no branch!)
            "v_cmp_eq_u32 vcc_lo, v3, 0",
            "s_and_saveexec_b32 s4, vcc_lo",
            # Multiple VALU ops under masked EXEC: must NOT use stale pre-saveexec %active!
            "v_add_nc_u32 v0, v0, 10",
            "v_add_nc_u32 v1, v1, 20",
            "v_add_nc_u32 v2, v2, 30",
            # Restore EXEC in the SAME basic block
            "s_or_b32 exec_lo, exec_lo, s4",
            # All 32 lanes active again: lane 7 must now update!
            "v_add_nc_u32 v1, v1, 5",
            "v_add_nc_u32 v2, v2, 6",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("ps_active_midblock", body)
        _validate_spirv(spvasm)

        # Every basic block in %main loads %active at most once (and %act_end in epilogue is in its own check).
        for lbl, n in _active_loads_in_main(spvasm):
            # In a single straight-line PS with `exp ... done vm`, %entry has at most 1 body load + %act_end.
            self.assertLessEqual(n, 2, f"unexpected redundant OpLoad %active in {lbl}: {n}")

        res = run_wave(spvasm, wave=32)
        for lane in range(32):
            if lane == 7:
                self.assertEqual(tuple(res["outputs"][lane]["%out_mrt0"]), (103, 205, 306, 1))
            else:
                self.assertEqual(tuple(res["outputs"][lane]["%out_mrt0"]), (113, 225, 336, 0))

    def test_divergent_loop_loads_active_once_per_block(self):
        """A divergent loop with many VALU ops per block caches %active per block and stays exact."""
        body = [
            "s_mov_b32 s4, 0",
            "s_mov_b32 s5, 3",
            "v_mov_b32 v0, 1",
            "v_mov_b32 v1, 2",
            "v_mov_b32 v2, 3",
            "v_mov_b32 v3, 0",
            "v_writelane_b32 v3, 1, 3",
            "v_writelane_b32 v3, 1, 19",
            "BB_loop:",
            "v_cmp_eq_u32 vcc_lo, v3, 0",
            "s_and_saveexec_b32 s6, vcc_lo",
            "s_cbranch_execz BB_skip",
            "v_add_nc_u32 v0, v0, 1",
            "v_add_nc_u32 v0, v0, 2",
            "v_add_nc_u32 v0, v0, 3",
            "v_add_nc_u32 v1, v1, 4",
            "v_add_nc_u32 v1, v1, 5",
            "v_add_nc_u32 v2, v2, 6",
            "BB_skip:",
            "s_or_b32 exec_lo, exec_lo, s6",
            "s_add_u32 s4, s4, 1",
            "s_cmp_lt_u32 s4, s5",
            "s_cbranch_scc1 BB_loop",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_listing("ps_active_loop", _encode_listing("ps_active_loop", body))
        _validate_spirv(spvasm)

        # The body block has 6 VALU writes but should load %active at most once (not 6 times).
        total_loads = sum(n for _, n in _active_loads_in_main(spvasm))
        self.assertLessEqual(total_loads, 6, f"expected cached %active loads, got {total_loads}")

        res = run_wave(spvasm, wave=32)
        for lane in range(32):
            if lane in (3, 19):
                self.assertEqual(tuple(res["outputs"][lane]["%out_mrt0"]), (1, 2, 3, 1))
            else:
                self.assertEqual(tuple(res["outputs"][lane]["%out_mrt0"]), (1 + 3 * 6, 2 + 3 * 9, 3 + 3 * 6, 0))

    def test_no_exec_write_fast_path_elides_active_loads_and_selects(self):
        """A shader that never writes EXEC elides %active loads and dst() OpSelects while keeping DPP guards."""
        body = [
            "s_mov_b32 s4, 0",
            "s_mov_b32 s5, 4",
            "v_mov_b32 v0, 10",
            "v_mov_b32 v1, 20",
            "v_mov_b32 v2, 30",
            "v_mov_b32 v3, 40",
            "BB_loop:",
            "v_add_nc_u32 v0, v0, 1",
            "v_add_nc_u32 v1, v1, 2",
            "v_add_nc_u32 v2, v2, 3",
            "v_add_nc_u32 v3, v3, s4",
            "s_add_u32 s4, s4, 1",
            "s_cmp_lt_u32 s4, s5",
            "s_cbranch_scc1 BB_loop",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_listing("ps_no_exec", _encode_listing("ps_no_exec", body))
        _validate_spirv(spvasm)

        self.assertNotIn("OpLoad %bool %active", spvasm)
        # Only the branch pc select should exist; no dst() OpSelect %uint %...
        res = run_wave(spvasm, wave=32)
        for lane in range(32):
            self.assertEqual(tuple(res["outputs"][lane]["%out_mrt0"]), (14, 28, 42, 46))

    def test_no_exec_write_still_preserves_dpp_row_mask(self):
        """Even when exec_static_true is set, DPP row_mask/bank_mask still preserves masked lanes."""
        lines = [
            "v_mov_b32 v0, 0x11111111",
            "v_mov_b32 v1, 0x22222222",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "v_mov_b32_dpp v0, v1 quad_perm:[0,0,0,0] row_mask:0x1 bank_mask:0xf",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("ps_no_exec_dpp", lines)
        _validate_spirv(spvasm)
        self.assertNotIn("OpLoad %bool %active", spvasm)

        res = run_wave(spvasm, wave=32)
        for lane in range(32):
            # row_mask:0x1 enables only row 0 (lanes 0..15); lanes 16..31 must keep 0x11111111.
            want_v0 = 0x22222222 if lane < 16 else 0x11111111
            self.assertEqual(res["outputs"][lane]["%out_mrt0"][0], want_v0)

    def test_wave64_emulate_validates_with_active_cache(self):
        """Cached %active never leaks across x_split segments under --wave64-emulate."""
        body = [
            "s_mov_b32 s4, 0",
            "s_mov_b32 s5, 2",
            "v_mov_b32 v0, 1",
            "v_mov_b32 v1, 2",
            "BB_loop:",
            "v_cmp_gt_u32 vcc, v0, 0",
            "s_and_saveexec_b64 s[6:7], vcc",
            "s_cbranch_execz BB_skip",
            "v_add_nc_u32 v0, v0, v1",
            "v_add_nc_u32 v1, v1, v0",
            "v_add_nc_u32 v0, v0, 3",
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
            spvasm = support.translate_listing("cs_active_w64", _encode_listing("cs_active_w64", body), fssm=fssm)
        finally:
            rdna2spirv.WAVE64_EMULATE = old_emu
        _validate_spirv(spvasm)

    def test_direct_gen_defaults_to_uncached_active(self):
        """Direct Gen() unit-test callers keep uncached OpLoad %bool %active for backwards compatibility."""
        g = rdna2spirv.Gen("ps")
        b1 = g.active_b()
        b2 = g.active_b()
        self.assertNotEqual(b1, b2)
        self.assertEqual(sum(1 for ln in g.body if "= OpLoad %bool %active" in ln), 2)

    def test_vop3b_carry_out_to_exec_disables_static_true(self):
        """VOP3b two-destination ops (e.g. v_add_co_u32) whose second operand is EXEC must disable exec_static_true."""
        lines = [
            "v_mov_b32 v0, 5",
            "v_mov_b32 v1, -1",
            "v_add_co_u32 v2, exec_lo, v1, v0",
            "exp mrt0 v0, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("ps_vop3b_exec_carry", lines)
        _validate_spirv(spvasm)
        self.assertIn("OpStore %active ", spvasm)

        # Also check writes_exec directly across VOP3_TWO_DESTS vs read-only EXEC sources.
        vgpr = lambda r: {"k": "vgpr", "r": r, "n": 1}
        sgpr = lambda r: {"k": "sgpr", "r": r, "n": 1}
        ex = {"k": "special", "t": "exec_lo"}
        for op in ("v_add_co_u32", "v_sub_co_u32_e64", "v_add_co_ci_u32_e64", "v_mad_u64_u32", "v_div_scale_f32"):
            self.assertTrue(rdna2spirv.writes_exec({"op": op, "args": [vgpr(2), ex, vgpr(1), vgpr(0)]}), op)
        # Reading exec_lo as a source operand in a normal op must NOT be treated as an EXEC write.
        self.assertFalse(rdna2spirv.writes_exec({"op": "s_mov_b32", "args": [sgpr(0), ex]}))
        self.assertFalse(rdna2spirv.writes_exec({"op": "s_and_b32", "args": [sgpr(0), ex, sgpr(1)]}))

    def test_no_exec_write_reads_exec_as_data(self):
        """A shader that never writes EXEC still returns all-ones when reading exec_lo as data."""
        lines = [
            "v_mov_b32 v0, 0x11",
            "v_mov_b32 v1, 0x22",
            "s_mov_b32 s4, exec_lo",
            "v_mov_b32 v3, s4",
            "v_cndmask_b32_e64 v2, v0, v1, exec_lo",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("ps_no_exec_read", lines)
        _validate_spirv(spvasm)
        self.assertNotIn("OpLoad %bool %active", spvasm)
        res = run_wave(spvasm, wave=32)
        for lane in range(32):
            self.assertEqual(tuple(res["outputs"][lane]["%out_mrt0"]), (0x11, 0x22, 0x22, 0xFFFFFFFF))

    def test_vcmpx_only_and_saveexec_only_disable_static_true(self):
        """Shaders whose ONLY EXEC writer is v_cmpx or s_*_saveexec must disable exec_static_true."""
        # Case 1: only v_cmpx writes EXEC (no s_or/s_mov restore).
        lines_vcmpx = [
            "v_mov_b32 v0, 10",
            "v_mov_b32 v1, 0",
            "v_writelane_b32 v1, 1, 4",
            "v_cmpx_eq_u32 v1, 0",
            "v_add_nc_u32 v0, v0, 5",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv_vcmpx = support.translate_lines("ps_vcmpx_only", lines_vcmpx)
        _validate_spirv(spv_vcmpx)
        self.assertIn("OpStore %active ", spv_vcmpx)
        res_vcmpx = run_wave(spv_vcmpx, wave=32)
        for lane in range(32):
            if lane != 4:
                self.assertEqual(tuple(res_vcmpx["outputs"][lane]["%out_mrt0"]), (15, 0, 15, 0))

        # Case 2: only s_and_saveexec_b32 writes EXEC (no s_or/s_mov restore).
        lines_save = [
            "v_mov_b32 v0, 20",
            "v_mov_b32 v1, 0",
            "v_writelane_b32 v1, 1, 6",
            "v_cmp_eq_u32 vcc_lo, v1, 0",
            "s_and_saveexec_b32 s4, vcc_lo",
            "v_add_nc_u32 v0, v0, 7",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv_save = support.translate_lines("ps_saveexec_only", lines_save)
        _validate_spirv(spv_save)
        self.assertIn("OpStore %active ", spv_save)
        res_save = run_wave(spv_save, wave=32)
        for lane in range(32):
            if lane != 6:
                self.assertEqual(tuple(res_save["outputs"][lane]["%out_mrt0"]), (27, 0, 27, 0))


if __name__ == "__main__":
    unittest.main()
