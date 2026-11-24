"""Value tests for RDNA2 32-bit LDS CAS atomics via spirv_mini_interp:
  ds_inc_u32 / ds_inc_rtn_u32, ds_dec_u32 / ds_dec_rtn_u32, ds_rsub_u32 / ds_rsub_rtn_u32,
  ds_min_f32 / ds_min_rtn_f32, ds_max_f32 / ds_max_rtn_f32.

Each lowers to a Workgroup-scope OpAtomicCompareExchange CAS loop inside Gen.lds_op(). Standard library only;
no GPU, no LLVM, no game files.
"""
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_asm  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

M32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
OUT = 0x3000

INT_VALUES = [0, 1, 2, 3, 4, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, M32]
INT_MODELS = {
    "inc_u32": lambda m, d: 0 if m >= d else (m + 1) & M32,
    "dec_u32": lambda m, d: d if (m == 0 or m > d) else (m - 1) & M32,
    "rsub_u32": lambda m, d: (d - m) & M32,
}


def make_fssm(lds_units=1, threads=32):
    fssm = bytearray(0x148)
    struct.pack_into("<4I", fssm, 0x138, 0x88 | (lds_units << 15), threads, 1, 1)
    return bytes(fssm)


def translate(body, stage="cs", lds_units=1, threads=32, emulate=False):
    with tempfile.TemporaryDirectory() as tmp:
        for d in ("disasm", "fssm"):
            os.mkdir(os.path.join(tmp, d))
        name = f"{stage}_lds_cas"
        with open(os.path.join(tmp, f"{name}.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        if stage == "cs":
            with open(os.path.join(tmp, "fssm", f"{name}.fssm"), "wb") as f:
                f.write(make_fssm(lds_units=lds_units, threads=threads))
        src = os.path.join(tmp, "disasm", f"{name}.s")
        with open(src, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"; {name}\n" + "".join(f"\t{ln}\n" for ln in body) + "\ts_endpgm\n")
        prev = rdna2spirv.WAVE64_EMULATE
        rdna2spirv.WAVE64_EMULATE = emulate
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                return rdna2spirv.translate(src)
        finally:
            rdna2spirv.WAVE64_EMULATE = prev


def run_cs(body, nlanes=1, lds_units=1, threads=32):
    spv = translate(body, stage="cs", lds_units=lds_units, threads=threads)
    mem = {}
    si.run_workgroup(spv, nlanes, threads,
                     inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                     push_constants=[0] * 36, memory=mem)
    return mem


def run_single(op_stem, m, d, rtn=True, offset=0, alias_dst=False):
    addr = 16
    op = f"ds_{op_stem.replace('_', '_rtn_', 1)}" if rtn else f"ds_{op_stem}"
    off = f" offset:{offset}" if offset else ""
    body = [
        f"v_mov_b32_e32 v1, {addr}",
        f"v_mov_b32_e32 v2, {m:#x}",
        "ds_write_b32 v1, v2",
        "s_waitcnt lgkmcnt(0)",
        f"v_mov_b32_e32 v1, {addr - offset}",
        f"v_mov_b32_e32 v3, {d:#x}",
        "v_mov_b32_e32 v5, 0xdead",
    ]
    if rtn:
        dst_reg = "v3" if alias_dst else "v5"
        body.append(f"{op} {dst_reg}, v1, v3{off}")
    else:
        body.append(f"{op} v1, v3{off}")
    body += [
        "s_waitcnt lgkmcnt(0)",
        f"v_mov_b32_e32 v1, {addr}",
        "ds_read_b32 v4, v1",
        "s_waitcnt lgkmcnt(0)",
        f"v_mov_b32_e32 v6, {OUT:#x}",
        f"global_store_dword v6, {'v3' if (rtn and alias_dst) else 'v5'}, s[0:1]",
        f"v_mov_b32_e32 v7, {OUT + 4:#x}",
        "global_store_dword v7, v4, s[0:1]",
    ]
    mem = run_cs(body)
    return mem.get(OUT), mem.get(OUT + 4)


class LdsCasAtomics(unittest.TestCase):
    def test_int_edge_grid_rtn_and_non_rtn(self):
        for stem, fn in INT_MODELS.items():
            for m in INT_VALUES:
                for d in INT_VALUES[::2]:
                    with self.subTest(op=stem, m=hex(m), d=hex(d)):
                        self.assertEqual(run_single(stem, m, d, rtn=True), (m, fn(m, d)))
                        self.assertEqual(run_single(stem, m, d, rtn=False), (0xDEAD, fn(m, d)))

    def test_int_seeded_and_offsets_and_aliasing(self):
        rng = random.Random(0x1D5C)
        for _ in range(18):
            stem = rng.choice(list(INT_MODELS))
            m, d = rng.getrandbits(32), rng.getrandbits(32)
            with self.subTest(op=stem, m=hex(m), d=hex(d)):
                self.assertEqual(run_single(stem, m, d, rtn=True, offset=8, alias_dst=True),
                                 (m, INT_MODELS[stem](m, d)))

    def test_float_min_max_edge_cases(self):
        pos0, neg0 = 0x00000000, 0x80000000
        one, neg2, half = 0x3F800000, 0xC0000000, 0x3F000000
        pinf, ninf, qnan = 0x7F800000, 0xFF800000, 0x7FC00001
        cases = [
            ("min_f32", one, half, half),
            ("min_f32", half, one, half),
            ("min_f32", one, neg2, neg2),
            ("min_f32", pos0, neg0, neg0),
            ("min_f32", neg0, pos0, neg0),
            ("min_f32", one, pinf, one),
            ("min_f32", one, ninf, ninf),
            ("min_f32", qnan, one, one),
            ("min_f32", one, qnan, one),
            ("max_f32", one, half, one),
            ("max_f32", half, one, one),
            ("max_f32", neg2, one, one),
            ("max_f32", pos0, neg0, pos0),
            ("max_f32", neg0, pos0, pos0),
            ("max_f32", one, pinf, pinf),
            ("max_f32", one, ninf, one),
            ("max_f32", qnan, one, one),
            ("max_f32", one, qnan, one),
        ]
        for stem, m, d, expected in cases:
            with self.subTest(op=stem, m=hex(m), d=hex(d)):
                self.assertEqual(run_single(stem, m, d, rtn=True), (m, expected))
                self.assertEqual(run_single(stem, m, d, rtn=False), (0xDEAD, expected))

    def test_inactive_lane_leaves_lds_and_vdst_alone(self):
        # Under exec_lo = 0, every op must skip the atomic and leave both LDS and vdst intact.
        for op, rtn in (
            ("ds_inc_rtn_u32", True), ("ds_inc_u32", False),
            ("ds_dec_rtn_u32", True), ("ds_dec_u32", False),
            ("ds_rsub_rtn_u32", True), ("ds_rsub_u32", False),
            ("ds_min_rtn_f32", True), ("ds_min_f32", False),
            ("ds_max_rtn_f32", True), ("ds_max_f32", False),
        ):
            with self.subTest(op=op):
                body = [
                    "v_mov_b32_e32 v1, 16",
                    "v_mov_b32_e32 v2, 0x3f800000",  # 1.0f
                    "ds_write_b32 v1, v2",
                    "s_waitcnt lgkmcnt(0)",
                    "v_mov_b32_e32 v3, 0xc0000000",  # -2.0f / large u32
                    "v_mov_b32_e32 v5, 0xdead",
                    "s_mov_b32 s4, exec_lo",
                    "s_mov_b32 exec_lo, 0",
                    f"{op} v5, v1, v3" if rtn else f"{op} v1, v3",
                    "s_mov_b32 exec_lo, s4",
                    "s_waitcnt lgkmcnt(0)",
                    "ds_read_b32 v4, v1",
                    "s_waitcnt lgkmcnt(0)",
                    f"v_mov_b32_e32 v6, {OUT:#x}",
                    "global_store_dword v6, v5, s[0:1]",
                    f"v_mov_b32_e32 v7, {OUT + 4:#x}",
                    "global_store_dword v7, v4, s[0:1]",
                ]
                mem = run_cs(body)
                self.assertEqual((mem.get(OUT), mem.get(OUT + 4)), (0xDEAD, 0x3F800000))

    def test_sequential_wrap_and_cas_spirv_structure(self):
        # 4 consecutive ds_inc_rtn_u32 with DATA=2 starting at 0 return 0, 1, 2, 0 and leave 1.
        body = [
            "v_mov_b32_e32 v1, 16", "v_mov_b32_e32 v2, 0",
            "ds_write_b32 v1, v2", "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v3, 2",
            "ds_inc_rtn_u32 v10, v1, v3",
            "ds_inc_rtn_u32 v11, v1, v3",
            "ds_inc_rtn_u32 v12, v1, v3",
            "ds_inc_rtn_u32 v13, v1, v3",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v14, v1", "s_waitcnt lgkmcnt(0)",
        ]
        for idx, reg in enumerate(("v10", "v11", "v12", "v13", "v14")):
            body += [f"v_mov_b32_e32 v6, {OUT + 4 * idx:#x}", f"global_store_dword v6, {reg}, s[0:1]"]
        mem = run_cs(body)
        self.assertEqual([mem.get(OUT + 4 * i) for i in range(5)], [0, 1, 2, 0, 1])

        # Verify Workgroup-scope CAS loop emission and absence of float atomic capabilities.
        spv = translate(["v_mov_b32_e32 v1, 0", "v_mov_b32_e32 v3, 0x3f800000", "ds_min_rtn_f32 v5, v1, v3"])
        exchange, = (line.split("OpAtomicCompareExchange ")[1].split()
                     for line in spv.splitlines() if "OpAtomicCompareExchange " in line)
        loads = [line.split("OpAtomicLoad ")[1].split() for line in spv.splitlines() if "OpAtomicLoad " in line]
        self.assertEqual(loads, [exchange[:4]])  # type, pointer, scope, memory semantics
        self.assertIn("OpLoopMerge", spv)
        self.assertIn("OpExtInst %float %glsl NMin", spv)
        self.assertNotIn("AtomicFloat", spv)

    def test_rejected_forms_and_outside_compute(self):
        for line in (
            "ds_inc_rtn_u32 v[4:5], v1, v3",
            "ds_dec_u32 v1, v[2:3]",
            "ds_rsub_rtn_u32 v1, v3",
            "ds_min_f32 v5, v1, v3",
        ):
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate(["v_mov_b32_e32 v1, 0", line])
        with self.assertRaises(rdna2spirv.Unsupported):
            translate(["v_mov_b32_e32 v1, 0", "ds_inc_u32 v1, v1", "exp mrt0 v1, v1, v1, v1 done vm"], stage="ps")

    CAS_BOUNDS_CASES = (
        ("ds_inc_rtn_u32", True, 77, 999, 78),
        ("ds_inc_u32", False, 77, 999, 78),
        ("ds_dec_rtn_u32", True, 77, 999, 76),
        ("ds_dec_u32", False, 77, 999, 76),
        ("ds_rsub_rtn_u32", True, 77, 999, 922),
        ("ds_rsub_u32", False, 77, 999, 922),
        ("ds_min_rtn_f32", True, 0x3F800000, 0xC0000000, 0xC0000000),
        ("ds_min_f32", False, 0x3F800000, 0xC0000000, 0xC0000000),
        ("ds_max_rtn_f32", True, 0x3F800000, 0x40800000, 0x40800000),
        ("ds_max_f32", False, 0x3F800000, 0x40800000, 0x40800000),
    )

    def test_oob_access_drops_write_and_returns_zero(self):
        # With LDS_SIZE=1 (512 B = 128 dwords) and an out-of-range extent forcing lds_bound=128,
        # accessing byte 512 (dword 128) must not wrap onto LDS[0]: _rtn_ returns 0 and the write is dropped.
        for op, rtn, init_val, data_val, _ in self.CAS_BOUNDS_CASES:
            with self.subTest(op=op):
                body = [
                    "v_mov_b32_e32 v0, 0",
                    f"v_mov_b32_e32 v1, {init_val:#x}",
                    "ds_write_b32 v0, v1",
                    "s_waitcnt lgkmcnt(0)",
                    "v_mov_b32_e32 v8, 0xfc00",
                    "ds_read_b32 v9, v8 offset:1024",
                    "v_mov_b32_e32 v2, 512",
                    f"v_mov_b32_e32 v3, {data_val:#x}",
                    "v_mov_b32_e32 v5, 0xdead",
                    f"{op} v5, v2, v3" if rtn else f"{op} v2, v3",
                    "s_waitcnt lgkmcnt(0)",
                    "ds_read_b32 v4, v0",
                    "s_waitcnt lgkmcnt(0)",
                    f"v_mov_b32_e32 v6, {OUT:#x}",
                    "global_store_dword v6, v5, s[0:1]",
                    f"v_mov_b32_e32 v7, {OUT + 4:#x}",
                    "global_store_dword v7, v4, s[0:1]",
                ]
                mem = run_cs(body, lds_units=1)
                self.assertEqual(rdna2spirv.translate.last_gen.lds_bound, 128)
                self.assertEqual((mem.get(OUT), mem.get(OUT + 4)), (0 if rtn else 0xDEAD, init_val))

    def test_inactive_lane_oob_access_leaves_vdst_alone(self):
        # Under exec_lo = 0 the out-of-range path must not run either: _rtn_ keeps vdst (no zero for an inactive
        # lane) and LDS[0] is untouched, with lds_bound=128 as in test_oob_access_drops_write_and_returns_zero.
        for op, rtn, init_val, data_val, _ in self.CAS_BOUNDS_CASES:
            if not rtn:
                continue
            with self.subTest(op=op):
                body = [
                    "v_mov_b32_e32 v0, 0",
                    f"v_mov_b32_e32 v1, {init_val:#x}",
                    "ds_write_b32 v0, v1",
                    "s_waitcnt lgkmcnt(0)",
                    "v_mov_b32_e32 v8, 0xfc00",
                    "ds_read_b32 v9, v8 offset:1024",
                    "v_mov_b32_e32 v2, 512",
                    f"v_mov_b32_e32 v3, {data_val:#x}",
                    "v_mov_b32_e32 v5, 0xdead",
                    "s_mov_b32 s4, exec_lo",
                    "s_mov_b32 exec_lo, 0",
                    f"{op} v5, v2, v3",
                    "s_mov_b32 exec_lo, s4",
                    "s_waitcnt lgkmcnt(0)",
                    "ds_read_b32 v4, v0",
                    "s_waitcnt lgkmcnt(0)",
                    f"v_mov_b32_e32 v6, {OUT:#x}",
                    "global_store_dword v6, v5, s[0:1]",
                    f"v_mov_b32_e32 v7, {OUT + 4:#x}",
                    "global_store_dword v7, v4, s[0:1]",
                ]
                mem = run_cs(body, lds_units=1)
                self.assertEqual(rdna2spirv.translate.last_gen.lds_bound, 128)
                self.assertEqual((mem.get(OUT), mem.get(OUT + 4)), (0xDEAD, init_val))

    def test_non_power_of_two_bound_addresses_valid_slot(self):
        # With LDS_SIZE=3 (1536 B = 384 dwords) and lds_bound=384, wrapping with (384 - 1) = 0x17f
        # would clear bit 7 (dword 128 -> dword 0); bounds checking must hit LDS[128] and preserve LDS[0].
        guard_val = 0xCAFE1234
        for op, rtn, init_val, data_val, expected in self.CAS_BOUNDS_CASES:
            with self.subTest(op=op):
                body = [
                    "v_mov_b32_e32 v0, 0",
                    f"v_mov_b32_e32 v1, {guard_val:#x}",
                    "ds_write_b32 v0, v1",
                    "v_mov_b32_e32 v2, 512",
                    f"v_mov_b32_e32 v10, {init_val:#x}",
                    "ds_write_b32 v2, v10",
                    "s_waitcnt lgkmcnt(0)",
                    "v_mov_b32_e32 v8, 0xfc00",
                    "ds_read_b32 v9, v8 offset:1536",
                    f"v_mov_b32_e32 v3, {data_val:#x}",
                    "v_mov_b32_e32 v5, 0xdead",
                    f"{op} v5, v2, v3" if rtn else f"{op} v2, v3",
                    "s_waitcnt lgkmcnt(0)",
                    "ds_read_b32 v4, v0",
                    "ds_read_b32 v11, v2",
                    "s_waitcnt lgkmcnt(0)",
                    f"v_mov_b32_e32 v6, {OUT:#x}",
                    "global_store_dword v6, v5, s[0:1]",
                    f"v_mov_b32_e32 v7, {OUT + 4:#x}",
                    "global_store_dword v7, v4, s[0:1]",
                    f"v_mov_b32_e32 v12, {OUT + 8:#x}",
                    "global_store_dword v12, v11, s[0:1]",
                ]
                mem = run_cs(body, lds_units=3)
                gen = rdna2spirv.translate.last_gen
                self.assertEqual((gen.lds_bound, gen.lds_dwords), (384, 384))
                self.assertEqual(
                    (mem.get(OUT), mem.get(OUT + 4), mem.get(OUT + 8)),
                    (init_val if rtn else 0xDEAD, guard_val, expected),
                )

    def test_zero_lds_allocation_returns_zero_and_drops_write(self):
        # LDS_SIZE=0 with a constant extent past 0 yields lds_bound=0; _rtn_ returns 0 and non-_rtn_ preserves vdst.
        for op, rtn in (("ds_inc_rtn_u32", True), ("ds_inc_u32", False), ("ds_min_rtn_f32", True)):
            with self.subTest(op=op):
                body = [
                    "v_mov_b32_e32 v1, 0",
                    "v_mov_b32_e32 v3, 7",
                    "v_mov_b32_e32 v5, 0xdead",
                    f"{op} v5, v1, v3 offset:4" if rtn else f"{op} v1, v3 offset:4",
                    "s_waitcnt lgkmcnt(0)",
                    f"v_mov_b32_e32 v6, {OUT:#x}",
                    "global_store_dword v6, v5, s[0:1]",
                ]
                mem = run_cs(body, lds_units=0)
                self.assertEqual(rdna2spirv.translate.last_gen.lds_bound, 0)
                self.assertEqual(mem.get(OUT), 0 if rtn else 0xDEAD)

    def test_wave64_emulate_shrunk_lds_checks_bounds_and_skips_wrap_mask(self):
        # Under --wave64-emulate with LDS_SIZE=100 (51200 B = 12800 dwords) and 64 threads,
        # fit_shared_memory shrinks %lds to 12800 dwords and sets lds_bound=12800 without offset tricks.
        shrunk = translate(
            ["s_mov_b64 exec, -1", "v_mov_b32_e32 v1, 0", "v_readfirstlane_b32 s16, v1",
             "v_mov_b32_e32 v3, 2", "ds_inc_rtn_u32 v5, v1, v3"],
            lds_units=100, threads=64, emulate=True,
        )
        gen = rdna2spirv.translate.last_gen
        self.assertTrue(gen.wave64 and gen.emulate)
        self.assertEqual((gen.lds_bound, gen.lds_dwords), (12800, 12800))
        self.assertGreaterEqual(len(re.findall(r"^\s*%\w+_lin = OpLabel$", shrunk, re.M)), 1)
        self.assertIsNone(re.search(r"= OpBitwiseAnd %uint %\w+ %\w+\n\s*%\w+ = OpAccessChain %_ptr_WG_uint\w* %lds ",
                                    shrunk))

        unbounded = translate(["v_mov_b32_e32 v1, 0", "v_mov_b32_e32 v3, 2", "ds_inc_rtn_u32 v5, v1, v3"])
        self.assertIsNone(rdna2spirv.translate.last_gen.lds_bound)
        self.assertEqual(len(re.findall(r"^\s*%\w+_lin = OpLabel$", unbounded, re.M)), 0)

    def _bounds_validation_modules(self):
        return (
            ("unbounded_min_rtn", translate(
                ["v_mov_b32_e32 v1, 0", "v_mov_b32_e32 v3, 0x3f800000", "ds_min_rtn_f32 v5, v1, v3"])),
            ("bounded_inc_rtn", translate([
                "v_mov_b32_e32 v8, 0xfc00", "ds_read_b32 v9, v8 offset:1024",
                "v_mov_b32_e32 v1, 0", "v_mov_b32_e32 v3, 2", "ds_inc_rtn_u32 v5, v1, v3",
            ], lds_units=1)),
            ("bounded_inc_no_rtn", translate([
                "v_mov_b32_e32 v8, 0xfc00", "ds_read_b32 v9, v8 offset:1024",
                "v_mov_b32_e32 v1, 0", "v_mov_b32_e32 v3, 2", "ds_inc_u32 v1, v3",
            ], lds_units=1)),
            ("npot_max_rtn", translate([
                "v_mov_b32_e32 v8, 0xfc00", "ds_read_b32 v9, v8 offset:1536",
                "v_mov_b32_e32 v1, 512", "v_mov_b32_e32 v3, 0x40800000", "ds_max_rtn_f32 v5, v1, v3",
            ], lds_units=3)),
            ("zero_lds_dec_rtn", translate(
                ["v_mov_b32_e32 v1, 0", "v_mov_b32_e32 v3, 7", "ds_dec_rtn_u32 v5, v1, v3 offset:4"], lds_units=0)),
            ("emulate_shrunk_rsub_rtn", translate(
                ["s_mov_b64 exec, -1", "v_mov_b32_e32 v1, 0", "v_readfirstlane_b32 s16, v1",
                 "v_mov_b32_e32 v3, 5", "ds_rsub_rtn_u32 v5, v1, v3"],
                lds_units=100, threads=64, emulate=True)),
        )

    def test_bounds_checked_cas_spirv_assembly(self):
        for label, spv in self._bounds_validation_modules():
            with self.subTest(case=label):
                binary = spirv_asm.assemble(spv, target_env="vulkan1.2")
                self.assertEqual(struct.unpack_from("<I", binary, 0)[0], 0x07230203)

    @unittest.skipUnless(shutil.which("spirv-val"), "needs spirv-val")
    def test_bounds_checked_cas_spirv_val(self):
        with tempfile.TemporaryDirectory() as tmp:
            for label, spv in self._bounds_validation_modules():
                with self.subTest(case=label):
                    spv_path = os.path.join(tmp, f"{label}.spv")
                    with open(spv_path, "wb") as f:
                        f.write(spirv_asm.assemble(spv, target_env="vulkan1.2"))
                    res = subprocess.run(["spirv-val", "--target-env", "vulkan1.2", spv_path],
                                         capture_output=True, text=True)
                    if res.returncode != 0:
                        self.fail(f"spirv-val failed for {label}: {res.stderr}")


if __name__ == "__main__":
    unittest.main()
