"""Semantic interpreter test for s_or_saveexec_b32 (and its wave64 form, s_or_saveexec_b64).

AMD "RDNA 2" Instruction Set Architecture, section 12.3 SOP1 Instructions (opcodes 37 and 61), with S0 the source
mask and EXEC the mask before the instruction:

    D = EXEC; EXEC = S0 | EXEC; SCC = (EXEC != 0)

Each case translates a synthetic compute shader with tools/rdna2spirv.py and runs every lane of the wave in lockstep
with spirv_mini_interp.run_workgroup, in wave32 (_b32) and native wave64 (_b64); --wave64-emulate is checked to
translate only, since the interpreter does not run its workgroup barriers. Per lane the shader stores whether the
lane is on in the new EXEC, its bit of D, and SCC; the expected values come from a plain-Python model of the
pseudocode above. Before the op, D is set to all ones and SCC to the opposite of the expected value, so a result the
translator never writes cannot pass. The cases: S0 and EXEC overlapping in part; EXEC empty, so the op turns S0's
lanes back on and D = 0; and both empty, so EXEC stays 0 and SCC = 0.
"""
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)  # python -I leaves the script's folder off sys.path

import support  # noqa: E402  (support puts tools/ on sys.path, honouring LEONIDA_TOOLS_DIR)
import spirv_mini_interp as si  # noqa: E402

rdna2spirv = support.rdna2spirv
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]  # raw buffer, every store in range (as in test_carry_exec_mask.py)


def ref(s0, exec_):
    """(new EXEC, D) from the ISA pseudocode; masks are Python sets of lanes."""
    return s0 | exec_, exec_


# (S0 lanes, EXEC lanes) per wave size: partial overlap; EXEC empty; both empty. The wave64 masks have lanes in both
# halves, so a result taken from one half only shows.
MASKS = {
    32: [({1, 3, 5, 20, 31}, {3, 4, 5, 17, 31}), ({2, 30}, set()), (set(), set())],
    64: [({1, 3, 33, 40, 63}, {3, 4, 33, 50, 63}), ({2, 45}, set()), (set(), set())],
}


def _regs(wave_size):
    if wave_size == 64:
        return {"op": "s_or_saveexec_b64", "exec": "exec", "mov": "s_mov_b64", "or": "s_or_b64",
                "s0": "s[4:5]", "tmp": "s[6:7]", "old": "s[8:9]", "d": "s[10:11]"}
    return {"op": "s_or_saveexec_b32", "exec": "exec_lo", "mov": "s_mov_b32", "or": "s_or_b32",
            "s0": "s4", "tmp": "s6", "old": "s8", "d": "s10"}


def _mask_lines(lanes, dst, r):
    """Build a lane mask in `dst` that is 1 for `lanes` and 0 elsewhere (v0 = lane index), under a full EXEC.

    Masks come from v_cmp; the only constants moved into a mask anywhere in this file are 0 and -1, the same for
    every lane, so the test does not depend on how the translator splits a constant mask into lane bits (SHADER-198).
    """
    if not lanes:
        return [f"{r['mov']} {dst}, 0"]
    lanes = sorted(lanes)
    lines = [f"v_cmp_eq_u32_e64 {dst}, v0, {lanes[0]}"]
    for lane in lanes[1:]:
        lines.append(f"v_cmp_eq_u32_e64 {r['tmp']}, v0, {lane}")
        lines.append(f"{r['or']} {dst}, {dst}, {r['tmp']}")
    return lines


def _program(s0_lanes, exec_lanes, wave_size, scc_before):
    """Lines that run the op with S0 = s0_lanes and EXEC = exec_lanes, then store per lane (12 bytes): the new EXEC
    bit, the D bit and SCC. D starts as all ones and SCC as `scc_before`, so a value the op does not write shows."""
    r = _regs(wave_size)
    lines = [f"{r['mov']} {r['exec']}, -1"]
    lines += _mask_lines(s0_lanes, r["s0"], r)
    lines += _mask_lines(exec_lanes, r["old"], r)
    lines += [
        "v_mov_b32 v6, 0",                      # 1 below only on lanes the new EXEC keeps on
        f"{r['mov']} {r['d']}, -1",
        f"{r['mov']} {r['exec']}, {r['old']}",  # EXEC before the op (s_mov does not write SCC)
        "s_cmp_eq_u32 0, 0" if scc_before else "s_cmp_lg_u32 0, 0",
        f"{r['op']} {r['d']}, {r['s0']}",
        "v_mov_b32 v6, 1",
        "s_cselect_b32 s14, 1, 0",              # SCC of the op
        f"{r['mov']} {r['exec']}, -1",
        f"v_cndmask_b32_e64 v7, 0, 1, {r['d']}",
        "v_mov_b32 v8, s14",
        "v_mul_lo_u32 v11, v0, 12",
        "buffer_store_dword v6, v11, s[0:3], 0 offen",
        "v_add_nc_u32 v12, 4, v11",
        "buffer_store_dword v7, v12, s[0:3], 0 offen",
        "v_add_nc_u32 v12, 8, v11",
        "buffer_store_dword v8, v12, s[0:3], 0 offen",
    ]
    return lines


def _fssm(wave_size):
    """Compute-shader header: COMPUTE_PGM_RSRC2 0x8088 (4 user SGPRs, s[0:3] = output V#; TGID_X), then the
    thread-group size: one wave of `wave_size` threads."""
    return bytes(0x138) + struct.pack("<4I", 0x8088, wave_size, 1, 1)


def _translate_cs(name, body_lines, wave_size, emulate):
    """Translate a synthetic compute shader of one wave; returns the SPIR-V text."""
    prev_emu = rdna2spirv.WAVE64_EMULATE
    rdna2spirv.WAVE64_EMULATE = emulate
    try:
        spv = support.translate_lines(name, [*body_lines, "s_endpgm"], fssm=_fssm(wave_size))
    finally:
        rdna2spirv.WAVE64_EMULATE = prev_emu
    got64 = rdna2spirv.translate.last_gen.wave64
    if got64 != (wave_size == 64):
        raise AssertionError(f"{name}: expected wave64={wave_size == 64}, got {got64}")
    return spv


def _run_cs(name, body_lines, wave_size):
    """Translate a synthetic compute shader and run all lanes of one wave on one subgroup; returns the memory."""
    spv = _translate_cs(name, body_lines, wave_size, emulate=False)
    return si.run_workgroup(
        spv, wave_size, wave_size,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=VSHARP + [0] * 28, memory={},
    )


class ReferenceTests(unittest.TestCase):
    def test_reference(self):
        # The model itself, on hand-checked pairs.
        self.assertEqual(ref({1, 3}, {3, 4}), ({1, 3, 4}, {3, 4}))
        self.assertEqual(ref({2}, set()), ({2}, set()))
        self.assertEqual(ref(set(), set()), (set(), set()))


class _OrSaveExecMixin:
    wave_size = 32

    def test_exec_d_and_scc(self):
        for case, (s0, exec_) in enumerate(MASKS[self.wave_size]):
            with self.subTest(case=case, wave=self.wave_size):
                want_exec, want_d = ref(s0, exec_)
                want_scc = 1 if want_exec else 0
                lines = _program(s0, exec_, self.wave_size, scc_before=1 - want_scc)
                mem = _run_cs(f"cs_or_saveexec_w{self.wave_size}_{case}", lines, self.wave_size)
                for lane in range(self.wave_size):
                    base = lane * 12
                    self.assertEqual(mem[base], int(lane in want_exec), f"lane {lane}: new EXEC bit")
                    self.assertEqual(mem[base + 4], int(lane in want_d), f"lane {lane}: D bit")
                    self.assertEqual(mem[base + 8], want_scc, f"lane {lane}: SCC")


class Wave32Tests(_OrSaveExecMixin, unittest.TestCase):
    wave_size = 32


class Wave64Tests(_OrSaveExecMixin, unittest.TestCase):
    wave_size = 64


class Wave64EmulateTranslationTests(unittest.TestCase):
    """--wave64-emulate: the emulated wave synchronises through workgroup barriers, which spirv_mini_interp does not
    run (as in test_carry_exec_mask.py and test_salu64_bitwise_data.py), so only the translation is checked here."""

    def test_translates(self):
        s0, exec_ = MASKS[64][0]
        spv = _translate_cs("cs_emu_or_saveexec", _program(s0, exec_, 64, scc_before=0), 64, emulate=True)
        self.assertIn("OpStore %x_it", spv)  # the exchange-round counter, declared only by the emulated translation


if __name__ == "__main__":
    unittest.main()
