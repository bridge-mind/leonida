"""Semantic tests for RDNA2 8/16-bit global sub-dword memory operations (loads and stores).

Covers AMD RDNA2 / GFX10 sub-dword memory instructions:
  - global_load_ubyte: 8-bit unsigned load, zero-extended to 32 bits
  - global_load_sbyte: 8-bit signed load, sign-extended to 32 bits
  - global_load_ushort: 16-bit unsigned load, zero-extended to 32 bits
  - global_load_sshort: 16-bit signed load, sign-extended to 32 bits
  - global_store_byte: 8-bit store from VGPR low byte into memory (preserving adjacent bytes)
  - global_store_short: 16-bit store from VGPR low 16 bits into memory (preserving adjacent bytes)
  - d16 variants: global_load_{u,s}byte_d16[_hi], global_load_short_d16[_hi],
                  global_store_{byte,short}_d16_hi

Executes compiled SPIR-V shaders through spirv_mini_interp on CPU and validates against
a deterministic reference execution oracle. Standard library only; no GPU, no game files.
"""
from __future__ import annotations

import os
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
M64 = 0xFFFFFFFFFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def make_fssm(threads: int = 32) -> bytes:
    fssm = bytearray(0x148)
    struct.pack_into("<4I", fssm, 0x138, 0x88, threads, 1, 1)
    return bytes(fssm)


FSSM = make_fssm(32)

BASE, OUT = 0x2000, 0x3000
SEED = [0x44332211, 0x88776655, 0xCCBBAA99]


def parse_offset_modifier(offset_mod: int | str) -> int:
    """Parse signed 12-bit offset modifier (-2048..2047)."""
    if isinstance(offset_mod, str):
        val = int(offset_mod, 0)
    else:
        val = int(offset_mod)
    if 0x800 <= val <= 0xFFF:
        val -= 0x1000
    if not -0x800 <= val <= 0x7FF:
        raise ValueError(f"Offset {val} out of 12-bit signed range [-2048, 2047]")
    return val


def compute_global_effective_address(base: int, voffset: int = 0, offset_mod: int | str = 0) -> int:
    """Compute 64-bit effective address for global memory instruction."""
    imm = parse_offset_modifier(offset_mod)
    return (base + (voffset & M32) + imm) & M64


class GlobalSubDwordMemoryBank:
    """Deterministic simulated memory bank for global sub-dword byte-accurate operations."""

    def __init__(self, seed_dwords: dict[int, int] | None = None):
        self._mem: dict[int, int] = {}
        if seed_dwords:
            for addr, dw in seed_dwords.items():
                self.write_dword(addr, dw)

    def read_byte(self, byte_addr: int) -> int:
        dw_addr = (byte_addr & ~3) & M64
        shift = (byte_addr & 3) * 8
        dw = self._mem.get(dw_addr, 0)
        return (dw >> shift) & 0xFF

    def write_byte(self, byte_addr: int, val: int) -> None:
        dw_addr = (byte_addr & ~3) & M64
        shift = (byte_addr & 3) * 8
        mask = ~(0xFF << shift) & M32
        dw = self._mem.get(dw_addr, 0) & mask
        new_dw = dw | ((val & 0xFF) << shift)
        self._mem[dw_addr] = new_dw & M32

    def read_short(self, byte_addr: int) -> int:
        b0 = self.read_byte(byte_addr)
        b1 = self.read_byte((byte_addr + 1) & M64)
        return (b1 << 8) | b0

    def write_short(self, byte_addr: int, val: int) -> None:
        self.write_byte(byte_addr, val & 0xFF)
        self.write_byte((byte_addr + 1) & M64, (val >> 8) & 0xFF)

    def read_dword(self, byte_addr: int) -> int:
        dw_addr = (byte_addr & ~3) & M64
        if (byte_addr & 3) == 0:
            return self._mem.get(dw_addr, 0)
        b0 = self.read_byte(byte_addr)
        b1 = self.read_byte((byte_addr + 1) & M64)
        b2 = self.read_byte((byte_addr + 2) & M64)
        b3 = self.read_byte((byte_addr + 3) & M64)
        return (b3 << 24) | (b2 << 16) | (b1 << 8) | b0

    def write_dword(self, byte_addr: int, val: int) -> None:
        if (byte_addr & 3) == 0:
            self._mem[(byte_addr & ~3) & M64] = val & M32
        else:
            self.write_byte(byte_addr, val & 0xFF)
            self.write_byte((byte_addr + 1) & M64, (val >> 8) & 0xFF)
            self.write_byte((byte_addr + 2) & M64, (val >> 16) & 0xFF)
            self.write_byte((byte_addr + 3) & M64, (val >> 24) & 0xFF)

    def snapshot(self) -> dict[int, int]:
        return dict(self._mem)


class GlobalSubDwordReferenceOracle:
    """Reference behavioral oracle for RDNA2 global sub-dword memory instructions."""

    def __init__(self, memory: GlobalSubDwordMemoryBank):
        self.memory = memory

    def execute_load(self, op: str, base: int, voffset: int = 0, offset_mod: int | str = 0,
                     old_dst: int = 0, active: bool = True) -> int:
        if not active:
            return old_dst & M32
        ea = compute_global_effective_address(base, voffset, offset_mod)
        if op == "global_load_ubyte":
            return self.memory.read_byte(ea)
        elif op == "global_load_sbyte":
            b = self.memory.read_byte(ea)
            return (b if b < 0x80 else (b | 0xFFFFFF00)) & M32
        elif op == "global_load_ushort":
            return self.memory.read_short(ea)
        elif op == "global_load_sshort":
            s = self.memory.read_short(ea)
            return (s if s < 0x8000 else (s | 0xFFFF0000)) & M32
        elif op == "global_load_ubyte_d16":
            b = self.memory.read_byte(ea)
            return ((old_dst & 0xFFFF0000) | b) & M32
        elif op == "global_load_ubyte_d16_hi":
            b = self.memory.read_byte(ea)
            return ((old_dst & 0x0000FFFF) | (b << 16)) & M32
        elif op == "global_load_sbyte_d16":
            b = self.memory.read_byte(ea)
            s = b if b < 0x80 else (b | 0xFF00)
            return ((old_dst & 0xFFFF0000) | (s & 0xFFFF)) & M32
        elif op == "global_load_sbyte_d16_hi":
            b = self.memory.read_byte(ea)
            s = b if b < 0x80 else (b | 0xFF00)
            return ((old_dst & 0x0000FFFF) | ((s & 0xFFFF) << 16)) & M32
        elif op == "global_load_short_d16":
            s = self.memory.read_short(ea)
            return ((old_dst & 0xFFFF0000) | s) & M32
        elif op == "global_load_short_d16_hi":
            s = self.memory.read_short(ea)
            return ((old_dst & 0x0000FFFF) | (s << 16)) & M32
        else:
            raise ValueError(f"Unsupported load op: {op}")

    def execute_store(self, op: str, data: int, base: int, voffset: int = 0,
                      offset_mod: int | str = 0, active: bool = True) -> None:
        if not active:
            return
        ea = compute_global_effective_address(base, voffset, offset_mod)
        if op == "global_store_byte":
            self.memory.write_byte(ea, data & 0xFF)
        elif op == "global_store_byte_d16_hi":
            self.memory.write_byte(ea, (data >> 16) & 0xFF)
        elif op == "global_store_short":
            self.memory.write_short(ea, data & 0xFFFF)
        elif op == "global_store_short_d16_hi":
            self.memory.write_short(ea, (data >> 16) & 0xFFFF)
        else:
            raise ValueError(f"Unsupported store op: {op}")


def translate(body: list[str], threads: int = 32, emulate: bool = False) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        for d in ("disasm", "fssm"):
            os.mkdir(os.path.join(tmp, d))
        with open(os.path.join(tmp, "cs_gsub.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", "cs_gsub.fssm"), "wb") as f:
            f.write(make_fssm(threads=threads))
        src = os.path.join(tmp, "disasm", "cs_gsub.s")
        with open(src, "w", encoding="utf-8", newline="\n") as f:
            f.write("; cs_gsub\n" + "".join(f"\t{ln}\n" for ln in body) + "\ts_endpgm\n")
        prev = rdna2spirv.WAVE64_EMULATE
        rdna2spirv.WAVE64_EMULATE = emulate
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                return rdna2spirv.translate(src)
        finally:
            rdna2spirv.WAVE64_EMULATE = prev


def run(body: list[str], memory: dict[int, int] | None = None) -> dict[int, int]:
    if memory is None:
        mem = {BASE + 4 * k: w for k, w in enumerate(SEED)}
    else:
        mem = dict(memory)
    si.run_workgroup(translate(body), 1, 32,
                     inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                     push_constants=[0] * 36, memory=mem)
    return mem


def seed_bytes() -> bytearray:
    return bytearray(b"".join(struct.pack("<I", w) for w in SEED))


def words(buf: bytes | bytearray) -> list[int]:
    return [struct.unpack_from("<I", buf, 4 * k)[0] for k in range(len(SEED))]


STORES = {
    "global_store_byte": (1, 0),
    "global_store_byte_d16_hi": (1, 16),
    "global_store_short": (2, 0),
    "global_store_short_d16_hi": (2, 16),
}
LOADS = {
    "global_load_ubyte_d16": (1, False, 0),
    "global_load_ubyte_d16_hi": (1, False, 16),
    "global_load_sbyte_d16": (1, True, 0),
    "global_load_sbyte_d16_hi": (1, True, 16),
    "global_load_short_d16": (2, False, 0),
    "global_load_short_d16_hi": (2, False, 16),
}


class TestGlobalSubDwordOracle(unittest.TestCase):
    """Unit tests validating reference oracle logic against known arithmetic rules."""

    def test_effective_address_calculation(self):
        self.assertEqual(compute_global_effective_address(0x1000, 0x200, 4), 0x1204)
        self.assertEqual(compute_global_effective_address(0x1000, 0x0, -4), 0x0FFC)
        self.assertEqual(compute_global_effective_address(0x1000, 0x0, "0xFFC"), 0x0FFC)
        self.assertEqual(compute_global_effective_address(0x2000, 0x0, -2048), 0x1800)
        self.assertEqual(compute_global_effective_address(0x2000, 0x0, "0x800"), 0x1800)
        self.assertEqual(compute_global_effective_address(0x2000, 0x0, 2047), 0x27FF)
        # 64-bit pointer extremes and address wrapping (M64 masking)
        self.assertEqual(compute_global_effective_address(0xFFFFFFFFFFFFFFFF, 1, 0), 0)
        self.assertEqual(compute_global_effective_address(0xFFFFFFFFFFFFFFFE, 2, 0), 0)

    def test_oracle_byte_load_and_store(self):
        bank = GlobalSubDwordMemoryBank({0x1000: 0x44332211})
        oracle = GlobalSubDwordReferenceOracle(bank)
        self.assertEqual(oracle.execute_load("global_load_ubyte", 0x1000, 0), 0x11)
        self.assertEqual(oracle.execute_load("global_load_ubyte", 0x1000, 1), 0x22)
        oracle.execute_store("global_store_byte", 0xEE, 0x1000, 1)
        self.assertEqual(bank.read_dword(0x1000), 0x4433EE11)

    def test_oracle_short_load_and_store(self):
        bank = GlobalSubDwordMemoryBank({0x1000: 0x44332211, 0x1004: 0x88776655})
        oracle = GlobalSubDwordReferenceOracle(bank)
        self.assertEqual(oracle.execute_load("global_load_ushort", 0x1000, 0), 0x2211)
        self.assertEqual(oracle.execute_load("global_load_ushort", 0x1000, 3), 0x5544)
        oracle.execute_store("global_store_short", 0xAABB, 0x1000, 3)
        self.assertEqual(bank.read_dword(0x1000), 0xBB332211)
        self.assertEqual(bank.read_dword(0x1004), 0x887766AA)

    def test_oracle_sign_extension(self):
        bank = GlobalSubDwordMemoryBank({0x1000: 0xFE807F00})
        oracle = GlobalSubDwordReferenceOracle(bank)
        self.assertEqual(oracle.execute_load("global_load_ubyte", 0x1000, 1), 0x7F)
        self.assertEqual(oracle.execute_load("global_load_sbyte", 0x1000, 1), 0x7F)
        self.assertEqual(oracle.execute_load("global_load_ubyte", 0x1000, 2), 0x80)
        self.assertEqual(oracle.execute_load("global_load_sbyte", 0x1000, 2), 0xFFFFFF80)
        self.assertEqual(oracle.execute_load("global_load_ushort", 0x1000, 1), 0x807F)
        self.assertEqual(oracle.execute_load("global_load_sshort", 0x1000, 1), 0xFFFF807F)


class TestGlobalSubDwordInterpreterExecution(unittest.TestCase):
    """End-to-end tests comparing SPIR-V execution with reference oracle predictions."""

    def test_basic_loads_against_oracle(self):
        mem_init = {BASE: 0x89ABCDEF, BASE + 4: 0x01234567, BASE + 8: 0xEFCDAB80}
        bank = GlobalSubDwordMemoryBank(mem_init)
        oracle = GlobalSubDwordReferenceOracle(bank)

        ops = [
            ("global_load_ubyte", 1),
            ("global_load_sbyte", 1),
            ("global_load_ushort", 2),
            ("global_load_sshort", 2),
        ]
        for op, size in ops:
            for offset in range(8):
                with self.subTest(op=op, offset=offset):
                    body = [
                        f"v_mov_b32_e32 v1, {BASE + offset:#x}",
                        f"{op} v5, v1, s[0:1]",
                        "s_waitcnt vmcnt(0)",
                        f"v_mov_b32_e32 v6, {OUT:#x}",
                        "global_store_dword v6, v5, s[0:1]",
                    ]
                    actual_mem = run(body, memory=mem_init)
                    expected_val = oracle.execute_load(op, BASE + offset, 0)
                    self.assertEqual(actual_mem[OUT], expected_val)

    def test_basic_stores_against_oracle(self):
        mem_init = {BASE: 0x12345678}
        bank = GlobalSubDwordMemoryBank(mem_init)
        oracle = GlobalSubDwordReferenceOracle(bank)

        data = 0xAABBCCDD
        body = [
            f"v_mov_b32_e32 v1, {BASE + 1:#x}",
            f"v_mov_b32_e32 v2, {data:#x}",
            "global_store_byte v1, v2, s[0:1]",
        ]
        actual_mem = run(body, memory=mem_init)
        oracle.execute_store("global_store_byte", data, BASE + 1, 0)
        self.assertEqual(actual_mem[BASE], bank.read_dword(BASE))

        bank = GlobalSubDwordMemoryBank(mem_init)
        oracle = GlobalSubDwordReferenceOracle(bank)
        body = [
            f"v_mov_b32_e32 v1, {BASE + 2:#x}",
            f"v_mov_b32_e32 v2, {data:#x}",
            "global_store_short v1, v2, s[0:1]",
        ]
        actual_mem = run(body, memory=mem_init)
        oracle.execute_store("global_store_short", data, BASE + 2, 0)
        self.assertEqual(actual_mem[BASE], bank.read_dword(BASE))

    def test_vgpr_off_addressing_mode(self):
        mem_init = {BASE: 0xCAFEBABE}
        body = [
            f"v_mov_b32_e32 v1, {BASE + 1:#x}",
            "v_mov_b32_e32 v2, 0",
            "global_load_ubyte v5, v[1:2], off",
            "s_waitcnt vmcnt(0)",
            f"v_mov_b32_e32 v6, {OUT:#x}",
            "v_mov_b32_e32 v7, 0",
            "global_store_dword v[6:7], v5, off",
        ]
        mem = run(body, memory=mem_init)
        self.assertEqual(mem[OUT], 0xBA)

        body = [
            f"v_mov_b32_e32 v1, {BASE + 2:#x}",
            "v_mov_b32_e32 v2, 0",
            "v_mov_b32_e32 v5, 0x55",
            "global_store_byte v[1:2], v5, off",
        ]
        mem = run(body, memory=mem_init)
        self.assertEqual(mem[BASE], 0xCA55BABE)

    def test_stores_keep_neighbouring_bytes(self):
        data = 0xF1E2D3C4
        for op, (n, shift) in STORES.items():
            for byte in range(1, 8):
                with self.subTest(op=op, byte=byte):
                    mem = run([
                        f"v_mov_b32_e32 v1, {BASE + byte:#x}",
                        f"v_mov_b32_e32 v2, {data:#x}",
                        f"{op} v1, v2, s[0:1]",
                    ])
                    want = seed_bytes()
                    want[byte:byte + n] = ((data >> shift) & ((1 << 8 * n) - 1)).to_bytes(n, "little")
                    self.assertEqual([mem[BASE + 4 * k] for k in range(3)], words(want))

    def test_d16_loads_write_one_half(self):
        keep = 0xBEEF1234
        buf = seed_bytes()
        for op, (n, signed, shift) in LOADS.items():
            for byte in range(0, 8):
                with self.subTest(op=op, byte=byte):
                    mem = run([
                        f"v_mov_b32_e32 v1, {BASE + byte:#x}",
                        f"v_mov_b32_e32 v5, {keep:#x}",
                        f"{op} v5, v1, s[0:1]",
                        "s_waitcnt vmcnt(0)",
                        f"v_mov_b32_e32 v6, {OUT:#x}",
                        "global_store_dword v6, v5, s[0:1]",
                    ])
                    v = int.from_bytes(buf[byte:byte + n], "little")
                    if signed and v >> 7:
                        v -= 0x100
                    v &= 0xFFFF
                    want = (keep & ~(0xFFFF << shift) | v << shift) & M32
                    self.assertEqual(mem[OUT], want)

    def test_inactive_lane_does_nothing(self):
        body = [
            "v_cmpx_eq_u32_e32 1, v0",
            f"v_mov_b32_e32 v1, {BASE + 1:#x}",
            "v_add_nc_u32_e32 v2, 0x70, v0",
            "global_store_byte v1, v2, s[0:1]",
        ]
        mem = {BASE + 4 * k: w for k, w in enumerate(SEED)}
        si.run_workgroup(translate(body), 2, 32,
                         inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                         push_constants=[0] * 36, memory=mem)
        self.assertEqual(mem[BASE], (SEED[0] & ~0xFF00) | 0x71 << 8)

    def test_rejected_forms(self):
        for line in ("global_load_dword_addtid v0, s[0:1]", "global_store_byte v1, v[2:3], s[0:1]",
                     "global_store_short v1, v2, s[0:1] slc"):
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate(["v_mov_b32_e32 v1, 0", line])

    def test_first_read_analysis_counts_the_kept_half(self):
        from shader_ir import parse_line
        self.assertEqual(rdna2spirv.read_before_write_vgprs([parse_line("\tglobal_load_short_d16 v5, v1, s[0:1]")]),
                         {1, 5})
        self.assertEqual(rdna2spirv.read_before_write_vgprs([parse_line("\tglobal_load_ushort v5, v1, s[0:1]")]),
                         {1})

    def test_short_loads_crossing_dword_both_modes_and_inactive_lane(self):
        mem_init = {BASE: 0x7FAABBCC, BASE + 4: 0x11223380}
        for op, want in (("global_load_ushort", 0x807F), ("global_load_sshort", 0xFFFF807F)):
            for addr_reg, off_mod in ((BASE + 3, ""), (BASE + 4, " offset:-1")):
                with self.subTest(op=op, addr=hex(addr_reg), off_mod=off_mod):
                    mem = run([
                        f"v_mov_b32_e32 v1, {addr_reg:#x}",
                        "v_mov_b32_e32 v2, 0",
                        "v_mov_b32_e32 v5, 0xdeadbeef",
                        f"{op} v5, v[1:2], off{off_mod}",
                        "s_waitcnt vmcnt(0)",
                        f"v_mov_b32_e32 v6, {OUT:#x}",
                        "global_store_dword v6, v5, s[0:1]",
                    ], memory=mem_init)
                    self.assertEqual(mem[OUT], want)

            with self.subTest(op=op, inactive=True):
                mem = run([
                    f"v_mov_b32_e32 v1, {BASE + 3:#x}",
                    "v_mov_b32_e32 v5, 0xdeadbeef",
                    "s_mov_b32 s4, exec_lo",
                    "s_mov_b32 exec_lo, 0",
                    f"{op} v5, v1, s[0:1]",
                    "s_mov_b32 exec_lo, s4",
                    "s_waitcnt vmcnt(0)",
                    f"v_mov_b32_e32 v6, {OUT:#x}",
                    "global_store_dword v6, v5, s[0:1]",
                ], memory=mem_init)
                self.assertEqual(mem[OUT], 0xDEADBEEF)

    def _load_validation_modules(self):
        return (
            ("w32_ushort_saddr", translate([
                f"v_mov_b32_e32 v1, {BASE + 3:#x}",
                "global_load_ushort v5, v1, s[0:1]",
            ])),
            ("w32_sshort_voff", translate([
                f"v_mov_b32_e32 v1, {BASE + 4:#x}",
                "v_mov_b32_e32 v2, 0",
                "global_load_sshort v5, v[1:2], off offset:-1",
            ])),
            ("w64_emulate_sshort", translate([
                "s_mov_b64 exec, -1",
                f"v_mov_b32_e32 v1, {BASE + 3:#x}",
                "v_readfirstlane_b32 s16, v1",
                "global_load_sshort v5, v1, s[0:1]",
            ], threads=64, emulate=True)),
        )

    def test_sub_dword_loads_spirv_assembly(self):
        for label, spv in self._load_validation_modules():
            with self.subTest(case=label):
                binary = spirv_asm.assemble(spv, target_env="vulkan1.2")
                self.assertEqual(struct.unpack_from("<I", binary, 0)[0], 0x07230203)

    @unittest.skipUnless(shutil.which("spirv-val"), "needs spirv-val")
    def test_sub_dword_loads_spirv_val(self):
        with tempfile.TemporaryDirectory() as tmp:
            for label, spv in self._load_validation_modules():
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
