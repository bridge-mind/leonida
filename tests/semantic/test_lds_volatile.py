"""Plain LDS loads and stores are emitted Volatile (docs/design.md, "LDS access order within a wave").

RDNA2 runs a wave in lockstep: every lane's ds_write has happened when the next instruction starts, so a ds_read
issued after it sees the other lanes' writes with no s_barrier in between. Translated, each lane is one invocation
and the Vulkan memory model orders nothing between invocations without a barrier, so a driver's compiler may move
a lane's read ahead of the wave's writes, or merge two reads of one address. AMD's Windows driver 24.30.46 does
both on a Radeon 880M: the GPU tests cs_w64x_lds_raw and cs_w64x_lds_war read uninitialized Workgroup memory there
on every run, and pass once the accesses are Volatile.

The first class checks the emitted SPIR-V of every plain access form, on the native wave32 and wave64 paths and
the wave64 emulation: every OpLoad / OpStore through a pointer into %lds carries the Volatile memory operand, and
nothing else does. The second executes Volatile accesses on a whole wave with the CPU interpreter. Standard library
only; no GPU, no LLVM, no game files.
"""
import re
import struct
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from support import rdna2spirv, translate_lines  # noqa: E402
from test_loop_ballot_barrier import run_compute_workgroup  # noqa: E402

VSHARP = [0, 0, 0x1000, 0x31016FAC]

# (name, instruction lines, plain LDS loads, plain LDS stores). v1 holds an LDS byte address.
FORMS = (
    ("read_b32", ["ds_read_b32 v2, v1"], 1, 0),
    ("read_b64", ["ds_read_b64 v[2:3], v1 offset:8"], 2, 0),
    ("read_b96", ["ds_read_b96 v[2:4], v1"], 3, 0),
    ("read_b128", ["ds_read_b128 v[2:5], v1"], 4, 0),
    ("write_b32", ["ds_write_b32 v1, v2"], 0, 1),
    ("write_b64", ["ds_write_b64 v1, v[2:3] offset:16"], 0, 2),
    ("write_b96", ["ds_write_b96 v1, v[2:4]"], 0, 3),
    ("write_b128", ["ds_write_b128 v1, v[2:5]"], 0, 4),
    ("read2_b32", ["ds_read2_b32 v[2:3], v1 offset0:1 offset1:3"], 2, 0),
    ("write2_b32", ["ds_write2_b32 v1, v2, v3 offset0:1 offset1:3"], 0, 2),
    ("read2st64_b32", ["ds_read2st64_b32 v[2:3], v1 offset0:0 offset1:1"], 2, 0),
    ("read2_b64", ["ds_read2_b64 v[2:5], v1 offset0:0 offset1:2"], 4, 0),
    ("write2_b64", ["ds_write2_b64 v1, v[2:3], v[4:5] offset0:0 offset1:2"], 0, 4),
    ("read_u8", ["ds_read_u8 v2, v1"], 1, 0),
    ("read_i16", ["ds_read_i16 v2, v1"], 2, 0),      # the dword, and the next one for a halfword at byte 3
    ("read_addtid", ["ds_read_addtid_b32 v2"], 1, 0),
    ("write_addtid", ["ds_write_addtid_b32 v2"], 0, 1),
    ("write_then_read", ["ds_write_b32 v1, v2", "ds_read_b32 v3, v1 offset:4"], 1, 1),
)


def fssm(wave, threads):
    h = bytearray(0x148)
    struct.pack_into("<I", h, 0xD4, 0x20 if wave == 32 else 0)
    # Four user SGPRs for the output V# and 4 KiB of LDS, as tests/semantic/test_lds_addtid.py.
    struct.pack_into("<4I", h, 0x138, 0x40008, threads, 1, 1)
    return bytes(h)


def translate(name, lines, wave, emulate=False, threads=None):
    with patch.object(rdna2spirv, "WAVE64_EMULATE", emulate):
        return translate_lines(f"cs_w{wave}_{name}", lines, fssm=fssm(wave, threads or wave))


def lds_accesses(spvasm):
    """(loads, stores, volatile loads, volatile stores) through pointers into %lds, and every other Volatile line."""
    pointers = set(re.findall(r"(%\S+) = OpAccessChain %_ptr_WG_uint %lds ", spvasm))
    loads = stores = vol_loads = vol_stores = 0
    other = []
    for line in spvasm.splitlines():
        words = line.split()
        volatile = "Volatile" in words
        if len(words) >= 5 and words[2] == "OpLoad" and words[4] in pointers:
            loads += 1
            vol_loads += volatile
        elif len(words) >= 3 and words[0] == "OpStore" and words[1] in pointers:
            stores += 1
            vol_stores += volatile
        elif volatile:
            other.append(line.strip())
    return loads, stores, vol_loads, vol_stores, other


class EveryPlainAccessIsVolatile(unittest.TestCase):
    def test_forms(self):
        for wave, emulate in ((32, False), (64, False), (64, True)):
            for name, body, want_loads, want_stores in FORMS:
                with self.subTest(form=name, wave=wave, emulate=emulate):
                    lines = (["v_mov_b32_e32 v1, 64"] + [f"v_mov_b32_e32 v{r}, {r}" for r in range(2, 6)]
                             + body + ["s_waitcnt lgkmcnt(0)", "s_endpgm"])
                    spvasm = translate(name, lines, wave, emulate)
                    loads, stores, vol_loads, vol_stores, other = lds_accesses(spvasm)
                    self.assertEqual((loads, stores), (want_loads, want_stores), "plain LDS accesses emitted")
                    self.assertEqual((vol_loads, vol_stores), (loads, stores), "each one is Volatile")
                    self.assertEqual(other, [], "nothing but an LDS access is Volatile")

    def test_a_shader_without_lds_has_no_volatile_access(self):
        spvasm = translate("no_lds", ["v_mov_b32_e32 v1, 7", "v_mov_b32_e32 v2, 0",
                                      "buffer_store_dword v1, v2, s[0:3], 0 offen", "s_endpgm"], 32)
        self.assertNotIn("Volatile", spvasm)
        self.assertNotIn("%lds", spvasm)


def store_lane_word(reg, base):
    """Every lane stores `reg` to out[base + lane]."""
    return ["v_lshlrev_b32_e32 v21, 2, v1", f"v_add_nc_u32_e32 v21, {4 * base}, v21",
            f"buffer_store_dword {reg}, v21, s[0:3], 0 offen"]


def lane_id(wave):
    return ["v_mbcnt_lo_u32_b32 v1, -1, 0"] + (["v_mbcnt_hi_u32_b32 v1, -1, v1"] if wave == 64 else [])


class VolatileAccessesExecute(unittest.TestCase):
    """The Volatile accesses run. The CPU interpreter runs the invocations of a subgroup one after another between
    subgroup operations and barriers, so it cannot show the lockstep exchange of cs_w64x_lds_raw (a GPU test); these
    two programs have the same answer in any order: a lane reads back its own slot, and the other half's slot only
    after an s_barrier."""

    def run_wave(self, name, lines, wave):
        spvasm = translate(name, lines, wave)
        loads, stores, vol_loads, vol_stores, _ = lds_accesses(spvasm)
        self.assertEqual((vol_loads, vol_stores), (loads, stores))
        self.assertGreater(loads * stores, 0)
        return run_compute_workgroup(spvasm, wave, wave, push_constants=VSHARP + [0] * 28, memory={})

    def test_own_slot_round_trip(self):
        for wave in (32, 64):
            with self.subTest(wave=wave):
                lines = (lane_id(wave)
                         + ["v_add_nc_u32_e32 v2, 100, v1", "v_lshlrev_b32_e32 v3, 2, v1", "ds_write_b32 v3, v2",
                            "ds_read_b32 v5, v3", "v_add_nc_u32_e32 v6, 2000, v1", "ds_write_b32 v3, v6",
                            "ds_read_b32 v8, v3", "s_waitcnt lgkmcnt(0)"]
                         + store_lane_word("v5", 0) + store_lane_word("v8", wave) + ["s_endpgm"])
                memory = self.run_wave("lds_own_slot", lines, wave)
                for lane in range(wave):
                    self.assertEqual((memory[4 * lane], memory[4 * (wave + lane)]), (100 + lane, 2000 + lane),
                                     f"lane {lane}")

    def test_other_half_after_a_barrier(self):
        for wave in (32, 64):
            with self.subTest(wave=wave):
                half = wave // 2
                lines = (lane_id(wave)
                         + ["v_add_nc_u32_e32 v2, 100, v1", "v_lshlrev_b32_e32 v3, 2, v1", "ds_write_b32 v3, v2",
                            "s_waitcnt lgkmcnt(0)", "s_barrier",
                            f"v_xor_b32_e32 v4, {half}, v1", "v_lshlrev_b32_e32 v4, 2, v4", "ds_read_b32 v5, v4",
                            "s_waitcnt lgkmcnt(0)"]
                         + store_lane_word("v5", 0) + ["s_endpgm"])
                memory = self.run_wave("lds_barrier", lines, wave)
                for lane in range(wave):
                    self.assertEqual(memory[4 * lane], 100 + (lane ^ half), f"lane {lane}")


if __name__ == "__main__":
    unittest.main()
