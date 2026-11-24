"""ds_permute_b32 (SHADER-90), the forward (scatter) cross-lane permute, run across every lane of the translated SPIR-V.

Programs are written here (no game data) and translated by tools/rdna2spirv.py; spirv_mini_interp.run_workgroup runs one
workgroup of 32 or 64 lanes. Each lane stores its result to out[lane]. The expected vectors come from a model
written from the RDNA2 ISA (DS_PERMUTE_B32, "Forward permute"), not from the translator:
  * every source lane that is on in EXEC writes its data to lane ((addr + offset) >> 2) mod 32 of its own half
    (wave64 runs as two wave32s, ISA 10.4.4, as ds_bpermute_b32 here does);
  * a destination lane that no source writes gets 0 (the ISA's tmp[0..63] = 0);
  * a destination lane that is off in EXEC keeps its VDST;
  * several sources to one lane: the value of one of them (the ISA's "not predictable"; this translator takes the
    highest-numbered). The tests that have such a lane accept any of the writers.
The GPU test is runtime/test/shaders/cs_w32_permute.asm; its fixture also runs here. The --wave64-emulate output
needs Workgroup memory, which the interpreter rejects, so it is covered by the GPU test only.
"""
import gc
import os
import struct
import tempfile
import unittest
import warnings

import support
import spirv_mini_interp as si

VSHARP = [0, 0, 0x1000, 0x31016FAC]
OLD = 99
ALL = 0xFFFFFFFF
ALL64 = (1 << 64) - 1


def fssm(threads):
    h = bytearray(0x148)
    struct.pack_into("<4I", h, 0x138, 0x8088, threads, 1, 1)
    return bytes(h)


def translate_text(name, insns, threads):
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        os.mkdir(os.path.join(tmp, "fssm"))
        path = os.path.join(tmp, "disasm", name + ".s")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"; {name}\n" + "".join("\t" + i + "\n" for i in insns))
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(support.STUB_DXBC)
        with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
            f.write(fssm(threads))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = support.rdna2spirv.translate(path)
            gc.collect()
        return spv


def program(addr_insns, exec_mask, offset=0, wave=32):
    """v2 = 7 * lane + 1 is the data, v3 the address; v10 starts at OLD; the result of lane l is stored at out[l]."""
    on = [f"s_mov_b32 exec_lo, {exec_mask & 0xFFFFFFFF:#x}"]
    off = ["s_mov_b32 exec_lo, -1"]
    if wave == 64:
        on.append(f"s_mov_b32 exec_hi, {exec_mask >> 32:#x}")
        off.append("s_mov_b32 exec_hi, -1")
    mod = f" offset:{offset}" if offset else ""
    return (["v_mad_u32_u24 v2, v0, 7, 1"] + addr_insns + [f"v_mov_b32_e32 v10, {OLD}"] + on
            + [f"ds_permute_b32 v10, v3, v2{mod}", "s_waitcnt lgkmcnt(0)"] + off
            + ["v_lshlrev_b32_e32 v21, 2, v0", "buffer_store_dword v10, v21, s[0:3], 0 offen", "s_endpgm"])


def run(spv, lanes, subgroup):
    memory = si.run_workgroup(
        spv, lanes, subgroup,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i % subgroup},
        push_constants=VSHARP + [0] * 28, memory={})
    stray = sorted(a for a in memory if not 0 <= a < 4 * lanes or a % 4)
    assert not stray, f"stores outside out[0..{lanes - 1}]: {stray}"
    return [memory.get(4 * k) for k in range(lanes)]


class SwizzleExecTests(unittest.TestCase):
    def test_swizzled_fragment_color_with_disabled_neighbors(self) -> None:
        fixture = os.path.join(support.HERE, "fixtures", "fragment_swizzle_exec.asm")
        with open(fixture, encoding="ascii") as source:
            instructions = [line.strip() for line in source if line.strip() and not line.startswith(";")]
        for pattern in ("QUAD_PERM,1,0,3,2", "SWAP,1"):
            lines = [line.replace("QUAD_PERM,1,0,3,2", pattern) for line in instructions]
            spv = support.translate_lines("ps_swizzle_exec", lines)
            for lanes in (32, 7):
                with self.subTest(pattern=pattern, lanes=lanes):
                    # Lane 0 reads disabled lane 1. Lane 1 keeps its destination. Lanes 2/3 exchange active data.
                    expected = [0, 0x3f000000, 0x3f400000, 0x3f400000] * 8
                    if lanes == 7:
                        expected[6] = 0  # translator's model for absent lane 7; unverified on hardware
                    self.assertEqual(run(spv, lanes, 32), expected[:lanes])


def model(addrs, exec_mask, offset, lanes=32):
    """ISA model. Per lane, the set of values the ISA allows (one value unless several sources collide)."""
    out = []
    for base in range(0, lanes, 32):
        writers = {d: [] for d in range(32)}
        for i in range(32):
            lane = base + i
            if (exec_mask >> lane) & 1:
                writers[((addrs[lane] + offset) >> 2) % 32].append(7 * lane + 1)
        for i in range(32):
            out.append(set(writers[i]) or {0} if (exec_mask >> (base + i)) & 1 else {OLD})
    return out


# name -> (instructions computing v3 from v0, python address function, offset, exec mask)
CASES = {
    "identity": (["v_lshlrev_b32_e32 v3, 2, v0"], lambda l: 4 * l, 0, ALL),
    "rotate_by_one": (["v_add_nc_u32_e32 v3, 1, v0", "v_lshlrev_b32_e32 v3, 2, v3"], lambda l: 4 * (l + 1), 0, ALL),
    "reversal": (["v_sub_nc_u32_e32 v3, 31, v0", "v_lshlrev_b32_e32 v3, 2, v3"], lambda l: 4 * (31 - l), 0, ALL),
    "reversal_offset_8": (["v_sub_nc_u32_e32 v3, 31, v0", "v_lshlrev_b32_e32 v3, 2, v3"],
                          lambda l: 4 * (31 - l), 8, ALL),
    "offset_only": (["v_lshlrev_b32_e32 v3, 2, v0"], lambda l: 4 * l, 12, ALL),
    # EXEC = lanes 0-15, lane l writes lane 2l: odd lanes are written by nobody (0), lanes 16-31 are off and keep OLD
    # although sources 8-15 target them
    "unwritten_lanes_are_zero": (["v_lshlrev_b32_e32 v3, 3, v0"], lambda l: 8 * l, 0, 0xFFFF),
    # the off odd lanes send nothing: every even destination lane gets 0
    "inactive_sources_send_nothing": (["v_add_nc_u32_e32 v3, 1, v0", "v_lshlrev_b32_e32 v3, 2, v3"],
                                      lambda l: 4 * (l + 1), 0, 0x55555555),
    # an even lane l writes l + 2, an odd lane l writes l + 1; EXEC = even lanes: the off lane 3 must not win lane 4
    "inactive_source_does_not_win": (["v_and_b32_e32 v3, 1, v0", "v_sub_nc_u32_e32 v3, 2, v3",
                                      "v_add_nc_u32_e32 v3, v0, v3", "v_lshlrev_b32_e32 v3, 2, v3"],
                                     lambda l: 4 * (l + 2 - (l & 1)), 0, 0x55555555),
    # every lane to lane 5: a collision, any writer is allowed
    "all_to_one_lane": (["v_mov_b32_e32 v3, 20"], lambda l: 20, 0, ALL),
    # index 32 + l wraps to l
    "index_wraps_mod_32": (["v_add_nc_u32_e32 v3, 32, v0", "v_lshlrev_b32_e32 v3, 2, v3"],
                           lambda l: 4 * (l + 32), 0, ALL),
    "high_address_bits_ignored": (["v_lshlrev_b32_e32 v3, 2, v0", "v_or_b32_e32 v3, 0x100, v3"],
                                  lambda l: 4 * l + 0x100, 0, ALL),
}


class Wave32(unittest.TestCase):
    def test_cases(self):
        for name, (insns, addr, off, mask) in CASES.items():
            with self.subTest(case=name):
                got = run(translate_text("cs_" + name, program(insns, mask, off), 32), 32, 32)
                want = model([addr(l) for l in range(32)], mask, off)
                for lane in range(32):
                    self.assertIn(got[lane], want[lane], f"{name}: lane {lane} got {got[lane]}, ISA allows {want[lane]}")
                if name != "all_to_one_lane":
                    self.assertTrue(all(len(w) == 1 for w in want), f"{name} has a collision")

    def test_known_values(self):
        # Hand-derived (data[l] = 7l + 1), independent of the model above.
        rot = run(translate_text("cs_rot", program(CASES["rotate_by_one"][0], ALL), 32), 32, 32)
        self.assertEqual((rot[0], rot[5], rot[31]), (218, 29, 211))
        rev = run(translate_text("cs_rev", program(CASES["reversal_offset_8"][0], ALL, 8), 32), 32, 32)
        self.assertEqual((rev[0], rev[3]), (8, 211))
        c = run(translate_text("cs_c", program(CASES["unwritten_lanes_are_zero"][0], 0xFFFF), 32), 32, 32)
        self.assertEqual((c[3], c[4], c[20]), (0, 15, OLD))
        d = run(translate_text("cs_d", program(CASES["inactive_source_does_not_win"][0], 0x55555555), 32), 32, 32)
        self.assertEqual((d[4], d[3]), (15, OLD))

    def test_gpu_test_fixture(self):
        """cs_w32_permute.asm: the words check.py expects (lane 0 and a few more store)."""
        spv = support.translate_fixture("cs_w32_permute", fssm=fssm(32))
        memory = si.run_workgroup(
            spv, 32, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
            push_constants=VSHARP + [0] * 28, memory={})
        self.assertEqual([memory.get(4 * k) for k in range(8)], [218, 29, 8, 211, 0, 15, 99, 15])


class Wave64Halves(unittest.TestCase):
    """Wave64 without emulation: two independent wave32 halves, on a 64-wide subgroup and on two 32-wide ones."""

    def test_halves(self):
        for subgroup in (64, 32):
            for name, mask in (("rotate_by_one", ALL64), ("reversal_offset_8", ALL64),
                               ("unwritten_lanes_are_zero", 0x0000FFFF0000FFFF),
                               ("inactive_source_does_not_win", 0x5555555555555555)):
                with self.subTest(case=name, subgroup=subgroup):
                    insns, addr, off, _ = CASES[name]
                    spv = translate_text("cs_h_" + name, program(insns, mask, off, wave=64), 64)
                    got = run(spv, 64, subgroup)
                    want = model([addr(l) for l in range(64)], mask, off, lanes=64)
                    for lane in range(64):
                        self.assertIn(got[lane], want[lane], f"lane {lane} got {got[lane]}, ISA allows {want[lane]}")
                    if name == "rotate_by_one":
                        # the upper half rotates inside itself: lane 32 gets lane 63's data, not lane 31's
                        self.assertEqual((got[32], got[63]), (7 * 63 + 1, 7 * 62 + 1))

    def test_gpu_test_fixture(self):
        """cs_w64x_permute.asm: the words check.py expects, on one 64-wide subgroup and on two 32-wide ones."""
        for subgroup in (64, 32):
            with self.subTest(subgroup=subgroup):
                spv = support.translate_fixture("cs_w64x_permute", fssm=fssm(64))
                memory = si.run_workgroup(
                    spv, 64, subgroup,
                    inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i % subgroup},
                    push_constants=VSHARP + [0] * 28, memory={})
                self.assertEqual([memory.get(4 * k) for k in range(8)], [218, 442, 435, 239, 0, 99, 239, 232])


if __name__ == "__main__":
    unittest.main()
