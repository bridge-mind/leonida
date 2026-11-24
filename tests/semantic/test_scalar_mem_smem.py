"""Semantic interpreter tests for scalar memory read offsets and descriptor address calculation (s_load_dword).

Covers AMD RDNA2 / GFX10 SMEM (Scalar Memory) ISA semantics and tools/rdna2spirv.py translation:
  * 64-bit base pointer reconstruction from SGPR pair s[p:p+1]
  * Buffer Resource Descriptor (V#) 128-bit layout and base address extraction:
      - Dword 0: Base address low 32 bits [31:0]
      - Dword 1: Base address high 16 bits [47:32] in bits [15:0]; bits [31:16] are format/stride/flags
      - Proper masking (dword1 & 0xFFFF) preventing descriptor flags from polluting the 64-bit pointer
  * Read offset calculations:
      - Zero offset, immediate byte offsets, and SGPR dynamic offsets
      - 32-bit carry propagation across the 4GB boundary into the upper 32 bits
      - 64-bit address wrapping modulo 2^64
  * Multi-dword scalar loads (s_load_dword, s_load_dwordx2, x4, x8, x16) with sequential 4-byte address strides
  * Destination register targeting: SGPR ranges and special registers (vcc_lo, vcc_hi, vcc pair, m0)
  * tools/rdna2spirv.py translator validation (Gen.scalar_load instruction sequences and operand validation)
  * SMEM loads with a register offset and an immediate at once ("sOFF offset:IMM"): address = base + sOFF + IMM,
    checked by running the emitted code (SmemBodyEvaluator, SmemSoffsetImmTranslatorTests)
  * The hardware-measured SMEM rules (SmemHardwareRuleTests): base, sOFF and IMM truncated to 4 each, the
    s_buffer_load NUM_RECORDS bound per dword (no read past it), the stride limit, the signed s_load immediate
  * spirv_mini_interp strictness validation (rejecting unmodelled 64-bit types and PhysicalStorageBuffer)
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
from shader_ir import parse_line  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


def compute_smem_pointer_address(base_lo: int, base_hi: int, offset: int) -> int:
    """Compute 64-bit effective address for s_load_dword from 64-bit pointer and offset."""
    base_64 = ((base_hi & MASK32) << 32) | (base_lo & MASK32)
    return (base_64 + (offset & MASK32)) & MASK64


def compute_smem_descriptor_address(desc_dw0: int, desc_dw1: int, offset: int) -> int:
    """Compute 64-bit effective address for s_buffer_load_dword from buffer descriptor V# and offset.

    Per AMD RDNA2 ISA:
      - dword0: Base address [31:0]
      - dword1: Base address [47:32] in bits [15:0]. Bits [31:16] are format/stride metadata.
    """
    base_hi = desc_dw1 & 0xFFFF
    base_64 = (base_hi << 32) | (desc_dw0 & MASK32)
    return (base_64 + (offset & MASK32)) & MASK64


class SmemMemoryBank:
    """Deterministic simulated memory bank for SMEM load validation."""

    def __init__(self):
        self._mem = {}

    def write_dword(self, byte_addr: int, val: int) -> None:
        self._mem[byte_addr & MASK64] = val & MASK32

    def read_dword(self, byte_addr: int) -> int:
        return self._mem.get(byte_addr & MASK64, 0)


class SmemReferenceInterpreter:
    """Reference execution engine for RDNA2 scalar memory loads."""

    def __init__(self, memory: SmemMemoryBank):
        self.mem = memory
        self.sgpr = [0] * 106
        self.vcc_lo = 0
        self.vcc_hi = 0
        self.m0 = 0

    def load_register(self, name: str, reg_idx: int = 0) -> int:
        if name == "sgpr":
            return self.sgpr[reg_idx] & MASK32
        if name == "vcc_lo":
            return self.vcc_lo & MASK32
        if name == "vcc_hi":
            return self.vcc_hi & MASK32
        if name == "m0":
            return self.m0 & MASK32
        raise ValueError(f"unknown register {name}")

    def store_register(self, name: str, reg_idx: int, val: int) -> None:
        val &= MASK32
        if name == "sgpr":
            self.sgpr[reg_idx] = val
        elif name == "vcc_lo":
            self.vcc_lo = val
        elif name == "vcc_hi":
            self.vcc_hi = val
        elif name == "m0":
            self.m0 = val
        else:
            raise ValueError(f"unknown register {name}")

    def execute_s_load(self, count: int, dest_reg: str, dest_idx: int,
                       base_lo_idx: int, offset: int) -> int:
        lo = self.load_register("sgpr", base_lo_idx)
        hi = self.load_register("sgpr", base_lo_idx + 1)
        addr = compute_smem_pointer_address(lo, hi, offset)
        for i in range(count):
            dword_val = self.mem.read_dword(addr + i * 4)
            if dest_reg == "sgpr":
                self.store_register("sgpr", dest_idx + i, dword_val)
            elif dest_reg == "vcc":
                if i == 0:
                    self.store_register("vcc_lo", 0, dword_val)
                elif i == 1:
                    self.store_register("vcc_hi", 0, dword_val)
            elif dest_reg in ("vcc_lo", "vcc_hi", "m0"):
                self.store_register(dest_reg, 0, dword_val)
        return addr

    def execute_s_buffer_load(self, count: int, dest_reg: str, dest_idx: int,
                              desc_dw0_idx: int, offset: int) -> int:
        dw0 = self.load_register("sgpr", desc_dw0_idx)
        dw1 = self.load_register("sgpr", desc_dw0_idx + 1)
        addr = compute_smem_descriptor_address(dw0, dw1, offset)
        for i in range(count):
            dword_val = self.mem.read_dword(addr + i * 4)
            if dest_reg == "sgpr":
                self.store_register("sgpr", dest_idx + i, dword_val)
            elif dest_reg == "vcc":
                if i == 0:
                    self.store_register("vcc_lo", 0, dword_val)
                elif i == 1:
                    self.store_register("vcc_hi", 0, dword_val)
            elif dest_reg in ("vcc_lo", "vcc_hi", "m0"):
                self.store_register(dest_reg, 0, dword_val)
        return addr


class SmemAddressOracleTests(unittest.TestCase):
    """Verifies SMEM 64-bit effective address calculations and V# descriptor masking."""

    def test_s_load_pointer_address_basic(self):
        cases = [
            (0x1000, 0x0, 0, 0x1000),
            (0x1000, 0x0, 16, 0x1010),
            (0x1000, 0x0, 0x40, 0x1040),
            (0x2000, 0x1, 0x100, 0x100002100),
            (0x00000000, 0x00000008, 0x200, 0x800000200),
        ]
        for lo, hi, off, expected in cases:
            with self.subTest(lo=hex(lo), hi=hex(hi), off=hex(off)):
                self.assertEqual(compute_smem_pointer_address(lo, hi, off), expected)

    def test_s_load_pointer_carry_into_upper_32_bits(self):
        # Adding offset to low dword crossing 0xFFFFFFFF boundary must carry into hi dword.
        cases = [
            (0xFFFFFFFC, 0x00000001, 8, 0x0000000200000004),
            (0xFFFFFFFF, 0x00000000, 1, 0x0000000100000000),
            (0xFFFFFFF0, 0x00000005, 0x20, 0x0000000600000010),
            (0xFFFFFFFF, 0x0000000F, 0xFFFFFFFF, 0x00000010FFFFFFFE),
        ]
        for lo, hi, off, expected in cases:
            with self.subTest(lo=hex(lo), hi=hex(hi), off=hex(off)):
                self.assertEqual(compute_smem_pointer_address(lo, hi, off), expected)

    def test_s_load_pointer_wrap_modulo_64(self):
        # 64-bit overflow wraps around to 0
        self.assertEqual(compute_smem_pointer_address(0xFFFFFFF0, 0xFFFFFFFF, 0x20), 0x0000000000000010)
        self.assertEqual(compute_smem_pointer_address(0xFFFFFFFF, 0xFFFFFFFF, 1), 0)

    def test_s_buffer_load_descriptor_masking(self):
        # Bits [31:16] of dword 1 contain format/stride/cache attributes;
        # only bits [15:0] participate in base address formation.
        cases = [
            # dw0, dw1, offset, expected_addr
            (0x00001000, 0x00000002, 0x10, 0x0000000200001010),
            # dw1 has 0x8004 in bits [31:16] (typical buffer resource metadata)
            (0x00001000, 0x80040002, 0x10, 0x0000000200001010),
            # dw1 has all upper 16 bits set (0xFFFF0000)
            (0x12345678, 0xFFFF0007, 0x0, 0x0000000712345678),
            # dw1 has format bits set, address bits 0
            (0x00000040, 0x3F000000, 0x20, 0x0000000000000060),
            # Carry from dw0 into masked dw1
            (0xFFFFFFF8, 0x80000001, 0x10, 0x0000000200000008),
        ]
        for dw0, dw1, off, expected in cases:
            with self.subTest(dw0=hex(dw0), dw1=hex(dw1), off=hex(off)):
                self.assertEqual(compute_smem_descriptor_address(dw0, dw1, off), expected)


class SmemExecutionTests(unittest.TestCase):
    """Verifies multi-dword scalar loads, sequential strides, and register targets."""

    def setUp(self):
        self.bank = SmemMemoryBank()
        self.interp = SmemReferenceInterpreter(self.bank)

    def test_s_load_dword_single(self):
        # Base s[2:3] = 0x1_00000000, offset = 0x20 -> address 0x1_00000020
        self.interp.store_register("sgpr", 2, 0x00000000)
        self.interp.store_register("sgpr", 3, 0x00000001)
        self.bank.write_dword(0x100000020, 0xCAFEBABE)

        addr = self.interp.execute_s_load(1, "sgpr", 4, 2, 0x20)
        self.assertEqual(addr, 0x100000020)
        self.assertEqual(self.interp.load_register("sgpr", 4), 0xCAFEBABE)

    def test_s_load_dwordx2_sequential_stride(self):
        self.interp.store_register("sgpr", 6, 0x00001000)
        self.interp.store_register("sgpr", 7, 0x00000000)
        self.bank.write_dword(0x1000, 0x11112222)
        self.bank.write_dword(0x1004, 0x33334444)

        addr = self.interp.execute_s_load(2, "sgpr", 8, 6, 0)
        self.assertEqual(addr, 0x1000)
        self.assertEqual(self.interp.load_register("sgpr", 8), 0x11112222)
        self.assertEqual(self.interp.load_register("sgpr", 9), 0x33334444)

    def test_s_load_dwordx4_sequential_stride(self):
        self.interp.store_register("sgpr", 0, 0x00002000)
        self.interp.store_register("sgpr", 1, 0x00000000)
        expected_words = [0xA0000001, 0xA0000002, 0xA0000003, 0xA0000004]
        for i, val in enumerate(expected_words):
            self.bank.write_dword(0x2010 + i * 4, val)

        addr = self.interp.execute_s_load(4, "sgpr", 12, 0, 0x10)
        self.assertEqual(addr, 0x2010)
        for i, val in enumerate(expected_words):
            self.assertEqual(self.interp.load_register("sgpr", 12 + i), val)

    def test_s_load_dwordx8_sequential_stride(self):
        self.interp.store_register("sgpr", 0, 0x00004000)
        self.interp.store_register("sgpr", 1, 0x00000000)
        for i in range(8):
            self.bank.write_dword(0x4000 + i * 4, 0xB0000000 | i)

        addr = self.interp.execute_s_load(8, "sgpr", 16, 0, 0)
        self.assertEqual(addr, 0x4000)
        for i in range(8):
            self.assertEqual(self.interp.load_register("sgpr", 16 + i), 0xB0000000 | i)

    def test_s_load_dwordx16_sequential_stride(self):
        self.interp.store_register("sgpr", 0, 0x00008000)
        self.interp.store_register("sgpr", 1, 0x00000000)
        for i in range(16):
            self.bank.write_dword(0x8000 + i * 4, 0xC0000000 | i)

        addr = self.interp.execute_s_load(16, "sgpr", 32, 0, 0)
        self.assertEqual(addr, 0x8000)
        for i in range(16):
            self.assertEqual(self.interp.load_register("sgpr", 32 + i), 0xC0000000 | i)

    def test_s_buffer_load_dword_with_descriptor_masking(self):
        # V# descriptor: dw0 = 0x5000, dw1 = 0x80040003 (stride=0x8004, hi_addr=3)
        self.interp.store_register("sgpr", 8, 0x00005000)
        self.interp.store_register("sgpr", 9, 0x80040003)
        self.interp.store_register("sgpr", 10, 0x00010000)  # num_records
        self.interp.store_register("sgpr", 11, 0x00140000)  # flags

        expected_addr = 0x0000000300005040
        self.bank.write_dword(expected_addr, 0x99887766)

        addr = self.interp.execute_s_buffer_load(1, "sgpr", 20, 8, 0x40)
        self.assertEqual(addr, expected_addr)
        self.assertEqual(self.interp.load_register("sgpr", 20), 0x99887766)

    def test_special_destination_registers(self):
        self.interp.store_register("sgpr", 0, 0x1000)
        self.interp.store_register("sgpr", 1, 0x0)
        self.bank.write_dword(0x1000, 0x12345678)
        self.bank.write_dword(0x1004, 0x9ABCDEF0)

        # Single dword to vcc_lo
        self.interp.execute_s_load(1, "vcc_lo", 0, 0, 0)
        self.assertEqual(self.interp.load_register("vcc_lo"), 0x12345678)

        # Single dword to vcc_hi
        self.interp.execute_s_load(1, "vcc_hi", 0, 0, 4)
        self.assertEqual(self.interp.load_register("vcc_hi"), 0x9ABCDEF0)

        # Single dword to m0
        self.interp.execute_s_load(1, "m0", 0, 0, 0)
        self.assertEqual(self.interp.load_register("m0"), 0x12345678)

        # Two dwords to vcc pair (low to vcc_lo, high to vcc_hi)
        self.interp.execute_s_load(2, "vcc", 0, 0, 0)
        self.assertEqual(self.interp.load_register("vcc_lo"), 0x12345678)
        self.assertEqual(self.interp.load_register("vcc_hi"), 0x9ABCDEF0)


class SmemTranslatorGenTests(unittest.TestCase):
    """Tests tools/rdna2spirv.py Gen.scalar_load SPIR-V emission structure and validation rules."""

    def test_s_load_dword_emits_64bit_shift_or_add(self):
        g = rdna2spirv.Gen("ps")
        ins = parse_line("s_load_dword s0, s[2:3], 0x10")
        g.scalar_load(ins["op"], ins["args"], ins["mods"])
        text = "\n".join(g.body)

        self.assertIn("OpShiftLeftLogical %ulong", text)
        self.assertIn("OpBitwiseOr %ulong", text)
        self.assertIn("OpIAdd %ulong", text)
        self.assertIn("OpConvertUToPtr %_ptr_PSB_arr", text)
        self.assertIn("OpAccessChain %_ptr_PSB_uint", text)
        self.assertIn("OpLoad %uint", text)
        self.assertIn("Aligned 4", text)
        self.assertTrue(g.uses_psb)

    def test_s_buffer_load_dword_emits_cffff_mask(self):
        g = rdna2spirv.Gen("ps")
        ins = parse_line("s_buffer_load_dword s4, s[8:11], 0x20")
        g.scalar_load(ins["op"], ins["args"], ins["mods"])
        text = "\n".join(g.body)

        self.assertIn("OpBitwiseAnd %uint", text)
        self.assertIn("%cffff", text)
        self.assertIn("OpConvertUToPtr %_ptr_PSB_arr", text)
        self.assertTrue(g.uses_psb)

    def test_multi_dword_emission_counts(self):
        cases = [
            ("s_load_dword s0, s[2:3], 0", 1),
            ("s_load_dwordx2 s[0:1], s[2:3], 0", 2),
            ("s_load_dwordx4 s[0:3], s[4:5], 0", 4),
            ("s_load_dwordx8 s[0:7], s[8:9], 0", 8),
            ("s_load_dwordx16 s[0:15], s[16:17], 0", 16),
            ("s_buffer_load_dword s0, s[4:7], 0", 1),
            ("s_buffer_load_dwordx4 s[0:3], s[8:11], 0", 4),
            ("s_buffer_load_dwordx8 s[0:7], s[8:11], 0", 8),
        ]
        for line, expected_n in cases:
            with self.subTest(line=line):
                g = rdna2spirv.Gen("ps")
                ins = parse_line(line)
                g.scalar_load(ins["op"], ins["args"], ins["mods"])
                psb_chains = [ln for ln in g.body if "OpAccessChain %_ptr_PSB_uint" in ln]
                psb_loads = [ln for ln in g.body if "OpLoad %uint" in ln and "Aligned 4" in ln]
                self.assertEqual(len(psb_chains), expected_n)
                self.assertEqual(len(psb_loads), expected_n)

    def test_special_destinations_emission(self):
        for line in ("s_buffer_load_dword vcc_lo, s[0:3], 0",
                     "s_buffer_load_dword vcc_hi, s[0:3], 4",
                     "s_buffer_load_dword m0, s[0:3], 8",
                     "s_load_dwordx2 vcc, s[0:1], 0"):
            with self.subTest(line=line):
                g = rdna2spirv.Gen("ps")
                ins = parse_line(line)
                g.scalar_load(ins["op"], ins["args"], ins["mods"])
                self.assertTrue(g.uses_psb)

    def test_unsupported_operands_raise_cleanly(self):
        # Base must have n=2 for s_load
        g = rdna2spirv.Gen("ps")
        ins_bad_sload_base = parse_line("s_load_dword s0, s[0:3], 0")
        with self.assertRaises(rdna2spirv.Unsupported) as ctx:
            g.scalar_load(ins_bad_sload_base["op"], ins_bad_sload_base["args"], ins_bad_sload_base["mods"])
        self.assertIn("s_load without 64-bit pointer", str(ctx.exception))

        # Base must have n=4 for s_buffer_load
        ins_bad_sbuf_base = parse_line("s_buffer_load_dword s0, s[0:1], 0")
        with self.assertRaises(rdna2spirv.Unsupported) as ctx:
            g.scalar_load(ins_bad_sbuf_base["op"], ins_bad_sbuf_base["args"], ins_bad_sbuf_base["mods"])
        self.assertIn("s_buffer_load without V#", str(ctx.exception))

        # Non-SGPR base must raise
        ins_vgpr_base = parse_line("s_load_dword s0, v[0:1], 0")
        with self.assertRaises(rdna2spirv.Unsupported):
            g.scalar_load(ins_vgpr_base["op"], ins_vgpr_base["args"], ins_vgpr_base["mods"])


class SmemBodyEvaluator:
    """Runs the SPIR-V that Gen.scalar_load emits, against concrete SGPR values and a dword memory.

    spirv_mini_interp refuses 64-bit integers and PhysicalStorageBuffer by design, so the address a scalar load
    reads cannot be checked there. This evaluator knows only the handful of instructions scalar_load uses and
    raises on anything else, so a change in the emitted code cannot be approximated silently. It follows the
    forward selection constructs the bounds check emits (OpSelectionMerge, OpBranchConditional, OpBranch to a later
    label, OpPhi naming the block it came from). The heap base (push constant dwords 32..33) is 0 unless given,
    which makes every address read an Xbox VA.
    """

    def __init__(self, gen, sgpr, memory, heap_base=0):
        self.consts = {name: value for value, name in gen.consts.items()}
        self.consts.update({name: value for value, name in gen.lconsts.items()})
        self.sgpr = dict(sgpr)
        self.memory = memory
        self.heap_base = heap_base
        self.ids = {}
        self.reads = []          # byte address of every PhysicalStorageBuffer dword load, in order
        lines = [line.split() for line in gen.body]
        labels = {w[0]: i for i, w in enumerate(lines) if len(w) == 3 and w[1:] == ["=", "OpLabel"]}
        self.block, self.came_from = "%entry", None
        pc = 0
        while pc < len(lines):
            words = lines[pc]
            if words[0] == "OpSelectionMerge":
                pc += 1
                continue
            target = None
            if words[0] == "OpBranch":
                target = words[1]
            elif words[0] == "OpBranchConditional":
                cond = self._value(words[1])
                if not isinstance(cond, bool):
                    raise AssertionError("OpBranchConditional on a non-bool")
                target = words[2] if cond else words[3]
            if target is not None:
                if labels.get(target, -1) <= pc:
                    raise AssertionError("branch backwards or to an unknown label: " + " ".join(words))
                self.came_from, pc = self.block, labels[target]
                continue
            if len(words) == 3 and words[1:] == ["=", "OpLabel"]:
                if self.block != words[0] and pc > 0 and lines[pc - 1][0] not in ("OpBranch", "OpBranchConditional"):
                    raise AssertionError("fall-through into " + words[0])
                self.block = words[0]
                pc += 1
                continue
            self._step(words)
            pc += 1

    def _value(self, name):
        if name == "%heap_base":          # built once in the entry block, outside the body run here
            return self.heap_base
        if name == "%c64_32":
            return 32
        if name in self.ids:
            return self.ids[name]
        if name in self.consts:
            return self.consts[name]
        raise AssertionError(f"unknown id {name}")

    def _step(self, words):
        if words[0] == "OpStore":
            kind, index = self._value(words[1])
            if kind != "sgpr":
                raise AssertionError(f"store through {kind}")
            self.sgpr[index] = self._value(words[2])
            return
        if len(words) < 4 or words[1] != "=":
            raise AssertionError("unmodelled line: " + " ".join(words))
        result, op, ty, args = words[0], words[2], words[3], words[4:]
        if op == "OpAccessChain" and ty == "%_ptr_Function_uint" and args[0] == "%sgpr":
            value = ("sgpr", self._value(args[1]))
        elif op == "OpAccessChain" and ty == "%_ptr_PSB_uint":
            kind, address = self._value(args[0])
            if kind != "psb":
                raise AssertionError(f"PSB access chain on {kind}")
            value = ("mem", (address + 4 * self._value(args[1])) & MASK64)
        elif op == "OpLoad" and ty == "%uint":
            kind, where = self._value(args[0])
            if kind == "sgpr":
                value = self.sgpr.get(where, 0) & MASK32
            elif kind == "mem" and args[1:] == ["Aligned", "4"]:
                self.reads.append(where)
                value = self.memory.read_dword(where)
            else:
                raise AssertionError("unmodelled load: " + " ".join(words))
        elif op == "OpPhi" and ty == "%uint":
            pairs = dict(zip(args[1::2], args[0::2]))
            if len(args) % 2 or self.came_from not in pairs:
                raise AssertionError("OpPhi without the block it came from: " + " ".join(words))
            value = self._value(pairs[self.came_from])
        elif op == "OpBitwiseAnd" and ty == "%uint":
            value = self._value(args[0]) & self._value(args[1])
        elif op == "OpBitFieldUExtract" and ty == "%uint":
            value = (self._value(args[0]) >> self._value(args[1])) & ((1 << self._value(args[2])) - 1)
        elif op == "OpIAdd" and ty == "%uint":
            value = (self._value(args[0]) + self._value(args[1])) & MASK32
        elif op == "OpUConvert" and ty == "%ulong":
            value = self._value(args[0]) & MASK32
        elif op == "OpShiftLeftLogical" and ty == "%ulong":
            value = (self._value(args[0]) << self._value(args[1])) & MASK64
        elif op == "OpBitwiseOr" and ty == "%ulong":
            value = self._value(args[0]) | self._value(args[1])
        elif op == "OpIAdd" and ty == "%ulong":
            value = (self._value(args[0]) + self._value(args[1])) & MASK64
        elif op == "OpIMul" and ty == "%ulong":
            value = (self._value(args[0]) * self._value(args[1])) & MASK64
        elif op == "OpIEqual" and ty == "%bool":
            value = self._value(args[0]) == self._value(args[1])
        elif op == "OpUGreaterThanEqual" and ty == "%bool":
            value = self._value(args[0]) >= self._value(args[1])
        elif op == "OpSelect" and ty in ("%ulong", "%uint"):
            cond = self._value(args[0])
            if not isinstance(cond, bool):
                raise AssertionError("OpSelect on a non-bool")
            value = self._value(args[1] if cond else args[2])
        elif op == "OpConvertUToPtr" and ty == "%_ptr_PSB_arr":
            value = ("psb", self._value(args[0]))
        else:
            raise AssertionError("unmodelled instruction: " + " ".join(words))
        self.ids[result] = value


class SmemSoffsetImmTranslatorTests(unittest.TestCase):
    """SMEM loads that carry a register offset and an immediate at once: llvm-mc prints "sOFF offset:IMM".

    Rule: address = base + soffset + imm (+ 4*i for dword i).
    Measured on an AMD Radeon RX 6800 (gfx1030) by running s_buffer_load_dword/x2/x4/x8 as a raw compute kernel:
    soffset 8 + imm 4 reads the dword at +12. s_load_dword with both offsets was measured later (soffset 8 + imm 4
    reads +12 too, SmemHardwareRuleTests.test_s_load_truncates_pointer_and_offsets).
    Every case here is dword-aligned and in bounds; truncation and NUM_RECORDS are SmemHardwareRuleTests.

    The instruction lines are hand-written. Each case runs the code Gen.scalar_load emits with SmemBodyEvaluator
    and asserts both the addresses read and the values that land in the destination SGPRs.
    """

    VSHARP_BASE = 0x0000000200001000        # s24 = 0x00001000, s25 = 0x80040002 (flags above bit 15 are not address)
    POINTER = 0x0000000300004000            # s[24:25] for s_load

    def run_load(self, line, pointer=False):
        ins = parse_line(line)
        g = rdna2spirv.Gen("ps")
        g.scalar_load(ins["op"], ins["args"], ins["mods"])
        base = self.POINTER if pointer else self.VSHARP_BASE
        bank = SmemMemoryBank()
        for off in range(0, 0x40, 4):       # every dword near the base holds its own offset, tagged
            bank.write_dword(base + off, 0x5EED0000 | off)
        sgpr = {40: 8}                      # the register offset used by every case
        if pointer:
            sgpr.update({24: 0x00004000, 25: 0x00000003})
        else:
            sgpr.update({24: 0x00001000, 25: 0x80040002, 26: 0x00010000, 27: 0x00140000})
        ev = SmemBodyEvaluator(g, sgpr, bank)
        return ev, [a - base for a in ev.reads], g

    def test_s_buffer_load_dword_soffset_plus_imm(self):
        # The measured case: soffset 8 + imm 4 reads the dword at +12 (not +8, which is what dropping the imm gives).
        ev, offsets, _ = self.run_load("s_buffer_load_dword s12, s[24:27], s40 offset:0x4")
        self.assertEqual(offsets, [12])
        self.assertEqual(ev.sgpr[12], 0x5EED000C)

    def test_s_load_dword_soffset_plus_imm(self):
        # Measured on the same GPU later (soffset 8 + imm 4 reads +12).
        ev, offsets, _ = self.run_load("s_load_dword s12, s[24:25], s40 offset:0x4", pointer=True)
        self.assertEqual(offsets, [12])
        self.assertEqual(ev.sgpr[12], 0x5EED000C)

    def test_multi_dword_soffset_plus_imm(self):
        cases = [
            ("s_buffer_load_dwordx2 s[12:13], s[24:27], s40 offset:0x4", False, 2, 0x0C),
            ("s_buffer_load_dwordx4 s[12:15], s[24:27], s40 offset:0x10", False, 4, 0x18),
            ("s_buffer_load_dwordx8 s[12:19], s[24:27], s40 offset:0x14", False, 8, 0x1C),
            ("s_load_dwordx2 s[12:13], s[24:25], s40 offset:0x4", True, 2, 0x0C),
            ("s_load_dwordx4 s[12:15], s[24:25], s40 offset:0x10", True, 4, 0x18),
        ]
        for line, pointer, n, first in cases:
            with self.subTest(line=line):
                ev, offsets, _ = self.run_load(line, pointer)
                self.assertEqual(offsets, [first + 4 * i for i in range(n)])
                self.assertEqual([ev.sgpr[12 + i] for i in range(n)], [0x5EED0000 | (first + 4 * i) for i in range(n)])

    def test_imm_only_and_soffset_only_addresses_unchanged(self):
        cases = [
            ("s_buffer_load_dword s12, s[24:27], 0x4", False, [0x04]),
            ("s_buffer_load_dword s12, s[24:27], s40", False, [0x08]),
            ("s_buffer_load_dwordx2 s[12:13], s[24:27], 0x10", False, [0x10, 0x14]),
            ("s_buffer_load_dwordx4 s[12:15], s[24:27], s40", False, [0x08, 0x0C, 0x10, 0x14]),
            ("s_load_dword s12, s[24:25], 0x4", True, [0x04]),
            ("s_load_dword s12, s[24:25], s40", True, [0x08]),
            ("s_load_dwordx2 s[12:13], s[24:25], 0", True, [0x00, 0x04]),
            ("s_load_dwordx2 s[12:13], s[24:25], s40", True, [0x08, 0x0C]),
        ]
        for line, pointer, want in cases:
            with self.subTest(line=line):
                ev, offsets, _ = self.run_load(line, pointer)
                self.assertEqual(offsets, want)
                self.assertEqual([ev.sgpr[12 + i] for i in range(len(want))], [0x5EED0000 | o for o in want])

    def test_zero_imm_with_soffset_reads_the_soffset_address(self):
        ev, offsets, _ = self.run_load("s_buffer_load_dword s12, s[24:27], s40 offset:0x0")
        self.assertEqual(offsets, [8])
        self.assertEqual(ev.sgpr[12], 0x5EED0008)

    def test_non_sgpr_scalar_register_offset_plus_imm(self):
        # llvm-mc also accepts m0, vcc_lo or null as the register offset. null reads as zero; m0 is not an SGPR slot,
        # so only check that the form still translates and that the immediate changes the code.
        ev, offsets, _ = self.run_load("s_buffer_load_dword s12, s[24:27], null offset:0x14")
        self.assertEqual(offsets, [0x14])
        self.assertEqual(ev.sgpr[12], 0x5EED0014)
        bodies = []
        for line in ("s_buffer_load_dword s12, s[24:27], m0", "s_buffer_load_dword s12, s[24:27], m0 offset:0x4"):
            ins = parse_line(line)
            g = rdna2spirv.Gen("ps")
            g.scalar_load(ins["op"], ins["args"], ins["mods"])
            bodies.append(g.body)
        self.assertNotEqual(bodies[0], bodies[1])

    def test_offset_modifier_changes_the_emitted_code(self):
        for line, pointer in (("s_buffer_load_dword s12, s[24:27], s40", False),
                              ("s_load_dword s12, s[24:25], s40", True)):
            with self.subTest(line=line):
                _, _, plain = self.run_load(line, pointer)
                _, _, both = self.run_load(line + " offset:0x4", pointer)
                self.assertNotEqual(plain.body, both.body)

    def test_heap_base_is_added_once(self):
        # The V# needs a NUM_RECORDS that covers +12: with s26 unset (NUM_RECORDS 0) the measured bound makes the
        # load return 0 without reading memory (SmemHardwareRuleTests), and there would be no address to check.
        ins = parse_line("s_buffer_load_dword s12, s[24:27], s40 offset:0x4")
        g = rdna2spirv.Gen("ps")
        g.scalar_load(ins["op"], ins["args"], ins["mods"])
        ev = SmemBodyEvaluator(g, {24: 0x1000, 25: 0, 26: 0x10000, 40: 8}, SmemMemoryBank(),
                               heap_base=0x0000007F00000000)
        self.assertEqual(ev.reads, [0x0000007F00000000 + 0x1000 + 12])

    def test_offset_forms_the_assembler_rejects_raise(self):
        # llvm-mc accepts none of these; hand-written or damaged listings must not translate to a guessed address.
        for line in ("s_buffer_load_dword s12, s[24:27], 0x8 offset:0x4",       # immediate operand and offset modifier
                     "s_buffer_load_dword s12, s[24:27], s40 offset:zz",        # unparseable immediate
                     "s_buffer_load_dword s12, s[24:27], s40 offset:0x4 offset:0x8",
                     "s_buffer_load_dword s12, s[24:27], s40, 0x4",             # a fourth operand
                     "s_load_dword s12, s[24:25], 0x8 offset:0x4"):
            with self.subTest(line=line):
                ins = parse_line(line)
                g = rdna2spirv.Gen("ps")
                with self.assertRaises(rdna2spirv.Unsupported):
                    g.scalar_load(ins["op"], ins["args"], ins["mods"])


class PatternMemory:
    """Byte-addressed memory: byte n of the region at `region` holds (0x40 + n) & 0xff, for any n (also n < 0).

    The measurement filled its buffer the same way, so "data@+X" in the oracle's model is the dword this returns at
    region + X; an unaligned read returns the bytes at that exact address."""

    def __init__(self, region):
        self.region = region

    def read_dword(self, address):
        return int.from_bytes(bytes((0x40 + address + k - self.region) & 0xFF for k in range(4)), "little")

    def data(self, offset):
        return self.read_dword(self.region + offset)


class SmemHardwareRuleTests(unittest.TestCase):
    """The SMEM rules measured on an AMD Radeon RX 6800 (gfx1030), raw compute kernel, wave32 and wave64:

      off = (sOFF & ~3) + (IMM & ~3);  dword i reads (base & ~3) + off + 4*i
      s_buffer_load*: dword i is 0 and is not read when off + 4*i + 4 > limit;
                      limit = NUM_RECORDS * stride (stride = V# dword1[29:16]) or NUM_RECORDS when stride is 0
      s_load*:        IMM is sign-extended from 21 bits, the sum is 64-bit, no bound

    Each case is a row of that measurement (the row names of the oracle's model are in the comments; the rule is in
    docs/design.md, Scalar/buffer memory). Every case checks the value in the destination SGPRs and the set of
    dword addresses read: an out-of-range dword must not be read at all. In `want`, an int X is the data at
    region offset X and None is 0. The lines are hand-written; the evaluator runs the code Gen.scalar_load emits.
    """

    REGION = 0x0000000500002000

    def run_line(self, line, sgpr, *, pointer=False, base_low=0, known=None):
        ins = parse_line(line)
        g = rdna2spirv.Gen("ps")
        g.sgpr_imm.update(known or {})                  # SGPRs the translator knows as constants (an s_mov before)
        g.scalar_load(ins["op"], ins["args"], ins["mods"])
        regs = dict(sgpr)
        address = self.REGION + base_low
        regs[24] = address & MASK32
        if pointer:
            regs[25] = address >> 32
        else:
            regs[25] = (regs.pop("stride", 0) << 16) | (address >> 32)
            regs[26] = regs.pop("nr")
            regs[27] = 0x31016FAC                       # word3 of the measurement (OOB_SELECT 3)
        mem = PatternMemory(self.REGION)
        ev = SmemBodyEvaluator(g, regs, mem)
        return ev, [a - self.REGION for a in ev.reads], mem

    def check(self, line, sgpr, want, reads, **kw):
        ev, got_reads, mem = self.run_line(line, sgpr, **kw)
        values = [0 if w is None else mem.data(w) for w in want]
        self.assertEqual([f"{ev.sgpr.get(12 + i, 0):#010x}" for i in range(len(want))],
                         [f"{v:#010x}" for v in values], line)
        # The addresses read, as a set: an out-of-range dword re-reads the load's first dword (in range whenever any
        # dword is) instead of taking a branch of its own, so a dword may be read twice; what must hold is that
        # nothing outside the bound is read at all.
        self.assertEqual(sorted({hex(r) for r in got_reads}), sorted({hex(r) for r in reads}), line)

    # s_buffer_load_dword, stride 0
    def test_bound_counts_the_whole_dword(self):
        # q1imm-k0-1[1]: imm 0x10, NR 18 -> 0 (a start-only check, 0x10 < 18, would read it)
        self.check("s_buffer_load_dword s12, s[24:27], 0x10", {"nr": 18}, [None], [])
        # q1imm-k0-3[11]: imm 0x14, NR 21 -> 0
        self.check("s_buffer_load_dword s12, s[24:27], 0x14", {"nr": 21}, [None], [])
        # the last dword that fits: imm 0x10, NR 20 -> data
        self.check("s_buffer_load_dword s12, s[24:27], 0x10", {"nr": 20}, [0x10], [0x10])

    def test_bound_uses_the_truncated_offset(self):
        # q1imm-k0-3[10]: imm 0x13, NR 21 -> data@+0x10 (raw 0x13 + 4 > 21 would give 0)
        self.check("s_buffer_load_dword s12, s[24:27], 0x13", {"nr": 21}, [0x10], [0x10])
        # q1imm-k0-1[2]: imm 0x11, NR 18 -> 0
        self.check("s_buffer_load_dword s12, s[24:27], 0x11", {"nr": 18}, [None], [])

    def test_soffset_and_imm_are_truncated_before_they_are_added(self):
        # q1pair-k0-0[1]: soffset 0xe + imm 2, NR 16 -> data@+0xc (truncating the sum, 0x10, would be out)
        self.check("s_buffer_load_dword s12, s[24:27], s40 offset:0x2", {"nr": 16, 40: 0xE}, [0xC], [0xC])
        # q1pair-k0-3[6]: soffset 0x10 + imm 4, NR 20 -> 0
        self.check("s_buffer_load_dword s12, s[24:27], s40 offset:0x4", {"nr": 20, 40: 0x10}, [None], [])
        # s0-k1[6], s0-k1[7]: soffset 3 + imm 3 -> data@+0; soffset 2 + imm 6 -> data@+4
        self.check("s_buffer_load_dword s12, s[24:27], s40 offset:0x3", {"nr": 0x1000, 40: 3}, [0x0], [0x0])
        self.check("s_buffer_load_dword s12, s[24:27], s40 offset:0x6", {"nr": 0x1000, 40: 2}, [0x4], [0x4])
        # q1soff-k0-3[7], [11]: soffset alone 0x13 -> data@+0x10, 0x14 -> 0 (NR 21)
        self.check("s_buffer_load_dword s12, s[24:27], s40", {"nr": 21, 40: 0x13}, [0x10], [0x10])
        self.check("s_buffer_load_dword s12, s[24:27], s40", {"nr": 21, 40: 0x14}, [None], [])

    def test_base_low_bits_are_dropped_and_not_part_of_the_bound(self):
        # s0-k1[1]: base & 3 = 1, imm 3 -> data@+0 (the unaligned sum would read bytes 4..7)
        self.check("s_buffer_load_dword s12, s[24:27], 0x3", {"nr": 0x1000}, [0x0], [0x0], base_low=1)
        # q1imm-k3-2[10], [13]: base & 3 = 3, imm 0x10 and 0x13, NR 20 -> data@+0x10
        self.check("s_buffer_load_dword s12, s[24:27], 0x10", {"nr": 20}, [0x10], [0x10], base_low=3)
        self.check("s_buffer_load_dword s12, s[24:27], 0x13", {"nr": 20}, [0x10], [0x10], base_low=3)

    def test_multi_dword_loads_are_bounded_per_dword(self):
        # multi-k0-3: x4 at 0xc, NR 21 -> data, data, 0, 0; x8 at 0, NR 21 -> d0..d4 data, d5..d7 0
        self.check("s_buffer_load_dwordx4 s[12:15], s[24:27], 0xc", {"nr": 21},
                   [0xC, 0x10, None, None], [0xC, 0x10])
        self.check("s_buffer_load_dwordx8 s[12:19], s[24:27], 0x0", {"nr": 21},
                   [0x0, 0x4, 0x8, 0xC, 0x10, None, None, None], [0x0, 0x4, 0x8, 0xC, 0x10])
        # multi-k0-0: x4 at 0xc, NR 17 -> d0 data, d1..d3 0
        self.check("s_buffer_load_dwordx4 s[12:15], s[24:27], 0xc", {"nr": 17},
                   [0xC, None, None, None], [0xC])
        # x2 with a soffset: 0x10 + 0, NR 0x16 -> data, 0
        self.check("s_buffer_load_dwordx2 s[12:13], s[24:27], s40", {"nr": 0x16, 40: 0x10},
                   [0x10, None], [0x10])

    def test_stride_makes_the_limit_num_records_times_stride(self):
        # q2stride-k0-0[1], [3], [4]: stride 4, NR 5 -> limit 20: imm 4 and 0x10 data, 0x14 -> 0
        self.check("s_buffer_load_dword s12, s[24:27], 0x4", {"nr": 5, "stride": 4}, [0x4], [0x4])
        self.check("s_buffer_load_dword s12, s[24:27], 0x10", {"nr": 5, "stride": 4}, [0x10], [0x10])
        self.check("s_buffer_load_dword s12, s[24:27], 0x14", {"nr": 5, "stride": 4}, [None], [])
        # stride 1, NR 17, imm 0x10 -> 0 (the index rule, 16 < 17, would read it)
        self.check("s_buffer_load_dword s12, s[24:27], 0x10", {"nr": 17, "stride": 1}, [None], [])
        # stride 16, NR 2: 0x1c data, 0x20 0
        self.check("s_buffer_load_dword s12, s[24:27], 0x1c", {"nr": 2, "stride": 16}, [0x1C], [0x1C])
        self.check("s_buffer_load_dword s12, s[24:27], 0x20", {"nr": 2, "stride": 16}, [None], [])

    def test_immediate_is_unsigned(self):
        # big[0], big[1]: imm 0xffffc, NR 0x100000 -> data@+0xffffc; NR 0xfffff -> 0
        self.check("s_buffer_load_dword s12, s[24:27], 0xffffc", {"nr": 0x100000}, [0xFFFFC], [0xFFFFC])
        self.check("s_buffer_load_dword s12, s[24:27], 0xffffc", {"nr": 0xFFFFF}, [None], [])

    def test_num_records_zero_reads_nothing(self):
        self.check("s_buffer_load_dwordx4 s[12:15], s[24:27], 0x0", {"nr": 0}, [None] * 4, [])

    # SHADER-336, measured on gfx1036 (Ryzen 9 9950X3D iGPU, tools/smem_neg_probe.py --oracle hw), wave32 = wave64.
    # Each row runs twice: sOFF in a register, and sOFF known to the translator (folded into the constant).
    def both(self, line, soff, want, reads, extra=None, **kw):
        self.check(line, {40: soff, **(extra or {})}, want, reads, **kw)
        self.check(line, {40: soff, **(extra or {})}, want, reads, known={40: soff}, **kw)

    def test_s_load_negative_sum_is_a_64_bit_signed_offset(self):
        self.check("s_load_dword s12, s[24:25], -0x4", {}, [-4], [-4], pointer=True)
        self.check("s_load_dword s12, s[24:25], -0x40", {}, [-0x40], [-0x40], pointer=True)
        self.both("s_load_dword s12, s[24:25], s40 offset:-0x8", 4, [-4], [-4], pointer=True)
        self.both("s_load_dword s12, s[24:25], s40 offset:-0x1", 6, [0], [0], pointer=True)   # 4 + -4, not 5 & ~3
        self.both("s_load_dword s12, s[24:25], s40 offset:-0x20", 0xC, [-0x14], [-0x14], pointer=True)
        self.both("s_load_dwordx2 s[12:13], s[24:25], s40 offset:-0x10", 4, [-0xC, -8], [-0xC, -8], pointer=True)

    def test_s_buffer_load_soffset_sum_does_not_wrap_at_32_bits(self):
        # sOFF 0xfffffffc + IMM 8 is 0x1_0000_0004, past NUM_RECORDS: 0 on hardware, in both translator forms
        self.both("s_buffer_load_dword s12, s[24:27], s40 offset:0x8", 0xFFFFFFFC, [None], [], extra={"nr": 0x100})
        self.both("s_buffer_load_dword s12, s[24:27], s40 offset:0x20", 0xFFFFFFF0, [None], [], extra={"nr": 0x100})
        for soff in (0xFFFFFFFC, 0x80000000, 0x7FFFFFFC):
            self.both("s_buffer_load_dword s12, s[24:27], s40", soff, [None], [], extra={"nr": 0x100})
        self.both("s_buffer_load_dword s12, s[24:27], s40", 8, [8], [8], extra={"nr": 0x100})

    # Measured on gfx1036 (RDNA2, Ryzen 9 9950X3D iGPU, AMD Windows driver 32.0.21045.16010) through tools/hw_oracle,
    # wave32 = wave64, every read inside a seeded region so only the bound decides: the edges docs/design.md listed
    # as not measured.
    def test_x16_is_bounded_per_dword(self):
        # x16 at 0, NR 0x14 -> d0..d4 data, d5..d15 0; NR 0x40 -> all 16 data
        line = "s_buffer_load_dwordx16 s[40:55], s[24:27], 0x0"
        self.check_wide(line, {"nr": 0x14}, [4 * i if i < 5 else None for i in range(16)])
        self.check_wide(line, {"nr": 0x40}, [4 * i for i in range(16)])
        # x16 with imm 8 or soffset 8, NR 0x20 -> d0..d5 data (bytes 8..0x1f), d6..d15 0
        self.check_wide("s_buffer_load_dwordx16 s[40:55], s[24:27], 0x8", {"nr": 0x20},
                        [8 + 4 * i if i < 6 else None for i in range(16)])
        self.check_wide("s_buffer_load_dwordx16 s[40:55], s[24:27], s20", {"nr": 0x20, 20: 8},
                        [8 + 4 * i if i < 6 else None for i in range(16)])

    def test_num_records_times_stride_does_not_wrap_at_4gib(self):
        # stride 0x1000 x NR 0x100000 and 0x2000 x 0x80000 = 2^32, stride 8 x NR 0x20000001 = 2^32 + 8: all four
        # dwords are data. A 32-bit product (0, 0, 8) would give 0 for every dword, or for d2 and d3.
        for stride, nr in ((0x1000, 0x100000), (0x2000, 0x80000), (8, 0x20000001)):
            self.check("s_buffer_load_dwordx4 s[12:15], s[24:27], 0x0", {"nr": nr, "stride": stride},
                       [0x0, 0x4, 0x8, 0xC], [0x0, 0x4, 0x8, 0xC])
        # control from the same run: stride 8 x NR 1 = 8 bytes -> d0, d1 data, d2, d3 0
        self.check("s_buffer_load_dwordx4 s[12:15], s[24:27], 0x0", {"nr": 1, "stride": 8},
                   [0x0, 0x4, None, None], [0x0, 0x4])

    def test_num_records_at_or_above_2_31_is_unsigned(self):
        for nr in (0x80000000, 0xFFFFFFFF):
            self.check("s_buffer_load_dwordx4 s[12:15], s[24:27], 0x0", {"nr": nr},
                       [0x0, 0x4, 0x8, 0xC], [0x0, 0x4, 0x8, 0xC])

    def check_wide(self, line, sgpr, want):
        """check() for a destination starting at s40 (a 16-dword load does not fit below the V# in s[24:27])."""
        ev, got_reads, mem = self.run_line(line, sgpr)
        values = [0 if w is None else mem.data(w) for w in want]
        self.assertEqual([f"{ev.sgpr.get(40 + i, 0):#010x}" for i in range(len(want))],
                         [f"{v:#010x}" for v in values], line)
        self.assertEqual(sorted({hex(r) for r in got_reads}), sorted({hex(w) for w in want if w is not None}), line)

    # s_load: 64-bit pointer, no bound
    def test_s_load_truncates_pointer_and_offsets(self):
        # p0-k1[0]: ptr & 3 = 1, imm 1 -> data@+0; p0-k1[6]: soffset 3 + imm 3 -> data@+0; p0-k1[8]: 8 + 4 -> +0xc
        self.check("s_load_dword s12, s[24:25], 0x1", {}, [0x0], [0x0], pointer=True, base_low=1)
        self.check("s_load_dword s12, s[24:25], s40 offset:0x3", {40: 3}, [0x0], [0x0], pointer=True)
        self.check("s_load_dword s12, s[24:25], s40 offset:0x4", {40: 8}, [0xC], [0xC], pointer=True)
        # p1-0: nothing clamps
        self.check("s_load_dword s12, s[24:25], s40 offset:0xffffc", {40: 0x100000}, [0x1FFFFC], [0x1FFFFC],
                   pointer=True)

    def test_s_load_immediate_is_signed(self):
        # neg[0], neg[1]: soffset 8, imm -4 -> data@+4; imm -8 -> data@+0 (zero-extension reads 4 GiB further)
        self.check("s_load_dword s12, s[24:25], s40 offset:-0x4", {40: 8}, [0x4], [0x4], pointer=True)
        self.check("s_load_dword s12, s[24:25], s40 offset:-0x8", {40: 8}, [0x0], [0x0], pointer=True)
        # neg[3], neg[10]: soffset 0xb, imm -1 -> data@+4 (8 + -4; truncating the sum 0xa would read +8)
        self.check("s_load_dword s12, s[24:25], s40 offset:-0x1", {40: 0xB}, [0x4], [0x4], pointer=True)
        self.check("s_load_dword s12, s[24:25], s40 offset:-0x1", {40: 0xB}, [0x4], [0x4], pointer=True,
                   base_low=3)
        # neg[6]: soffset 0x100000, imm -0x100000 -> data@+0
        self.check("s_load_dword s12, s[24:25], s40 offset:-0x100000", {40: 0x100000}, [0x0], [0x0], pointer=True)
        self.check("s_load_dwordx2 s[12:13], s[24:25], s40 offset:-0x8", {40: 0x10}, [0x8, 0xC], [0x8, 0xC],
                   pointer=True)

    def test_s_load_negative_total_is_the_signed_sum(self):
        # NOT measured (it could not be probed safely): an immediate-only negative offset, i.e. soffset + imm < 0.
        # The translator uses the 64-bit signed sum, which reads below the pointer; this pins that inference.
        self.check("s_load_dword s12, s[24:25], -0x4", {}, [-0x4], [-0x4], pointer=True)


class SmemMiniInterpStrictnessTests(unittest.TestCase):
    """Verifies that spirv_mini_interp rejects storage classes and constructs it does not model.

    As documented in tests/semantic/README.md, the mini-interpreter shares no code with the translator
    and raises UnsupportedConstruct for anything not explicitly modelled (such as physical-storage-buffer
    OpVariables, odd integer widths, or pointer-to-int casts), guaranteeing that translator regressions fail
    tests immediately. 64-bit integers, OpConvertUToPtr and physical-storage-buffer loads are modelled; their
    positive paths are covered in test_interp.py and test_value_semantics.py.
    """

    def test_rejects_unsupported_integer_widths(self):
        module = """\
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
%void = OpTypeVoid
%ushort = OpTypeInt 16 0
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"""
        with self.assertRaises(si.UnsupportedConstruct) as ctx:
            si.run_fragment(module)
        self.assertIn("integer type that is not 32 or 64 bits", str(ctx.exception))

    def test_rejects_physical_storage_buffer_storage_class(self):
        module = """\
OpCapability Shader
OpCapability PhysicalStorageBufferAddresses
OpMemoryModel PhysicalStorageBuffer64 GLSL450
OpEntryPoint Fragment %main "main"
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%_ptr_PSB = OpTypePointer PhysicalStorageBuffer %uint
%var = OpVariable %_ptr_PSB PhysicalStorageBuffer
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"""
        with self.assertRaises(si.UnsupportedConstruct) as ctx:
            si.run_fragment(module)
        self.assertIn("storage class PhysicalStorageBuffer", str(ctx.exception))

    def test_rejects_convert_ptr_to_u_in_body(self):
        module = """\
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%_ptr_Func = OpTypePointer Function %uint
%fn = OpTypeFunction %void
%c1 = OpConstant %uint 1
%main = OpFunction %void None %fn
%entry = OpLabel
%p = OpConvertUToPtr %_ptr_Func %c1
%u = OpConvertPtrToU %uint %p
OpReturn
OpFunctionEnd
"""
        with self.assertRaises(si.UnsupportedConstruct) as ctx:
            si.run_fragment(module)
        self.assertIn("OpConvertPtrToU", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
