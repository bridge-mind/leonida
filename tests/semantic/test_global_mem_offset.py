"""global_load / global_store instruction offsets (tools/rdna2spirv.py Gen.global_mem).

Rule: address = base + offset, where base is the 64-bit SGPR pair plus the zero-extended 32-bit VGPR offset, or the
64-bit VGPR address with `off`, and offset is the signed 12-bit field as llvm-mc prints it: offset:-4 (field 0xffc,
encoding 0xfc,0x8f,...), range -2048..2047. llvm-mc rejects offset:0xffc, offset:2048 and offset:-2049 ("expected a
12-bit signed offset") and disassembles the field 0xffc back as offset:-4, so the signed decimal is the only spelling
the repo's disassembly (tools/shader_disasm.py) and test build (runtime/test/shaders/build.py) deliver.
Measured on an AMD Radeon RX 6800 (gfx1030) by running global_load_dword as a raw compute kernel, wave32 and wave64
alike, with the dword at byte n holding 0x10000000+n and the pointer at 0x1000: offset 0 reads 0x10001000, -4
0x10000ffc, -2048 0x10000800, 2044 0x100017fc, -1024 0x10000c00. The GPU test is
runtime/test/shaders/cs_global_neg_offset.asm.

The instruction lines are hand-written. Each case runs the code Gen.global_mem emits with GlobalBodyEvaluator and
asserts the byte address accessed and the value that lands in (or is stored from) the VGPRs.
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

import rdna2spirv  # noqa: E402
from shader_ir import parse_line  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


class GlobalBodyEvaluator:
    """Runs the code Gen.global_mem emits against concrete registers and a dword memory, one active lane.

    spirv_mini_interp refuses 64-bit integers and PhysicalStorageBuffer by design, so this knows only the
    instructions global_mem uses (64-bit address arithmetic, the active-lane selection, OpPhi, PSB loads and stores,
    the sub-dword extract) and raises on anything else, so a change in the emitted code is never approximated.
    The heap base (push constant dwords 32..33) is 0 unless given, which makes every address an Xbox VA.
    """

    def __init__(self, gen, sgpr, vgpr, memory, heap_base=0, active=True):
        self.consts = {name: value for value, name in gen.consts.items()}
        self.regs = {"%sgpr": dict(sgpr), "%vgpr": dict(vgpr)}
        self.memory = dict(memory)
        self.heap_base = heap_base
        self.active = active
        self.ids = {}
        self.reads, self.writes = [], []     # byte addresses of PhysicalStorageBuffer dword accesses, in order
        self.label, self.prev, self.skip_to = "%entry", None, None
        for line in gen.body:
            self._step(line.split())

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

    def _step(self, w):
        if len(w) == 3 and w[1] == "=" and w[2] == "OpLabel":
            if self.skip_to is not None and w[0] != self.skip_to:
                return
            self.skip_to, self.prev, self.label = None, self.label, w[0]
            return
        if self.skip_to is not None:
            return
        if w[0] == "OpSelectionMerge":
            return
        if w[0] == "OpBranch":
            self.skip_to = w[1]
            return
        if w[0] == "OpBranchConditional":
            self.skip_to = w[2] if self._value(w[1]) else w[3]
            return
        if w[0] == "OpStore":
            kind, where = self._value(w[1])
            value = self._value(w[2]) & MASK32
            if kind == "mem" and w[3:] == ["Aligned", "4"]:
                self.writes.append(where)
                self.memory[where] = value
            elif kind in ("%sgpr", "%vgpr") and len(w) == 3:
                self.regs[kind][where] = value
            else:
                raise AssertionError("unmodelled store: " + " ".join(w))
            return
        if len(w) < 4 or w[1] != "=":
            raise AssertionError("unmodelled line: " + " ".join(w))
        result, op, ty, args = w[0], w[2], w[3], w[4:]
        if op == "OpAccessChain" and ty == "%_ptr_Function_uint" and args[0] in ("%sgpr", "%vgpr"):
            value = (args[0], self._value(args[1]))
        elif op == "OpAccessChain" and ty == "%_ptr_PSB_uint":
            kind, address = self._value(args[0])
            if kind != "arr":
                raise AssertionError(f"PSB access chain on {kind}")
            value = ("mem", (address + 4 * self._value(args[1])) & MASK64)
        elif op == "OpConvertUToPtr" and ty == "%_ptr_PSB_arr":
            value = ("arr", self._value(args[0]))
        elif op == "OpConvertUToPtr" and ty == "%_ptr_PSB_uint":
            value = ("mem", self._value(args[0]))
        elif op == "OpLoad" and ty == "%bool" and args == ["%active"]:
            value = self.active
        elif op == "OpLoad" and ty == "%uint":
            kind, where = self._value(args[0])
            if kind in ("%sgpr", "%vgpr") and len(args) == 1:
                value = self.regs[kind].get(where, 0) & MASK32
            elif kind == "mem" and args[1:] == ["Aligned", "4"]:
                if where % 4:
                    raise AssertionError(f"unaligned dword load at {where:#x}")
                self.reads.append(where)
                value = self.memory.get(where, 0)
            else:
                raise AssertionError("unmodelled load: " + " ".join(w))
        elif op == "OpBitcast" and ty in ("%int", "%uint"):
            value = self._value(args[0]) & MASK32
        elif op == "OpBitcast" and ty == "%ulong":
            value = self._value(args[0]) & MASK64
        elif op == "OpSConvert" and ty == "%long":
            v = self._value(args[0]) & MASK32
            value = (v - (1 << 32) if v & 0x80000000 else v) & MASK64
        elif op == "OpUConvert" and ty == "%ulong":
            value = self._value(args[0]) & MASK32
        elif op == "OpUConvert" and ty == "%uint":
            value = self._value(args[0]) & MASK32
        elif op == "OpShiftLeftLogical" and ty == "%ulong":
            value = (self._value(args[0]) << self._value(args[1])) & MASK64
        elif op == "OpShiftLeftLogical" and ty == "%uint":
            value = (self._value(args[0]) << self._value(args[1])) & MASK32
        elif op == "OpShiftRightLogical" and ty == "%uint":
            value = (self._value(args[0]) >> self._value(args[1])) & MASK32
        elif op == "OpBitwiseOr" and ty == "%ulong":
            value = self._value(args[0]) | self._value(args[1])
        elif op == "OpBitwiseOr" and ty == "%uint":
            value = (self._value(args[0]) | self._value(args[1])) & MASK32
        elif op == "OpIEqual" and ty == "%bool":
            value = self._value(args[0]) == self._value(args[1])
        elif op == "OpBitwiseAnd" and ty == "%uint":
            value = self._value(args[0]) & self._value(args[1])
        elif op == "OpIAdd" and ty == "%ulong":
            value = (self._value(args[0]) + self._value(args[1])) & MASK64
        elif op == "OpISub" and ty == "%ulong":
            value = (self._value(args[0]) - self._value(args[1])) & MASK64
        elif op in ("OpBitFieldUExtract", "OpBitFieldSExtract") and ty == "%uint":
            v, off, count = (self._value(x) for x in args)
            if off + count > 32:
                raise AssertionError(f"bit-field extract beyond 32 bits: off={off}, count={count}")
            value = (v >> off) & ((1 << count) - 1)
            if op == "OpBitFieldSExtract" and value >> (count - 1):
                value -= 1 << count
            value &= MASK32
        elif op == "OpPhi" and ty == "%uint":
            pairs = dict(zip(args[1::2], args[0::2]))
            value = self._value(pairs[self.prev])
        else:
            raise AssertionError("unmodelled instruction: " + " ".join(w))
        self.ids[result] = value


def tagged(address):
    """The seed word at a dword address: its own offset, tagged, so a wrong address shows in the value."""
    return 0x5EED0000 | (address & 0xFFFF)


class GlobalOffsetTests(unittest.TestCase):
    """Both addressing modes, every width global_mem handles, offsets -4, -2048, 2047 and 0."""

    # saddr: s[20:21] = 0x3_00004000, v1 = 0x800 (zero-extended) -> base 0x3_00004800.
    # off:   v[4:5]   = 0x3_00004800 -> the same base.
    BASE = 0x0000000300004800
    SGPR = {20: 0x00004000, 21: 0x00000003}
    VGPR = {1: 0x00000800, 4: 0x00004800, 5: 0x00000003, 6: 0xA1B2C3D4, 7: 0x0BADC0DE}

    def run_line(self, line, memory=None, active=True, heap_base=0):
        ins = parse_line(line)
        g = rdna2spirv.Gen("cs")
        g.global_mem(ins["op"], ins["args"], ins["mods"])
        if memory is None:
            memory = {a: tagged(a) for a in range(self.BASE - 0x1010, self.BASE + 0x1010, 4)}
        ev = GlobalBodyEvaluator(g, self.SGPR, {**self.VGPR, 10: 0xEEEEEEEE, 11: 0xEEEEEEEE}, memory,
                                 heap_base=heap_base, active=active)
        return ev, g

    def test_dword_load_both_modes(self):
        for mode in ("v1, s[20:21]", "v[4:5], off"):
            for off in (-4, -2048, 2044, 0):
                line = f"global_load_dword v10, {mode}" + (f" offset:{off}" if off else "")
                with self.subTest(line=line):
                    ev, _ = self.run_line(line)
                    want = self.BASE + off
                    self.assertEqual(ev.reads, [want])
                    self.assertEqual(ev.regs["%vgpr"][10], tagged(want))

    def test_multi_dword_load(self):
        for mode in ("v1, s[20:21]", "v[4:5], off"):
            for off in (-4, -2048, 2040, 0):
                line = f"global_load_dwordx2 v[10:11], {mode}" + (f" offset:{off}" if off else "")
                with self.subTest(line=line):
                    ev, _ = self.run_line(line)
                    want = self.BASE + off
                    self.assertEqual(ev.reads, [want, want + 4])
                    self.assertEqual([ev.regs["%vgpr"][10], ev.regs["%vgpr"][11]], [tagged(want), tagged(want + 4)])
        ev, _ = self.run_line("global_load_dwordx4 v[10:13], v[4:5], off offset:-2048")
        self.assertEqual(ev.reads, [self.BASE - 2048 + 4 * i for i in range(4)])

    def test_sub_dword_loads(self):
        # Byte-distinct seeds: the dword at address a holds bytes (a+0..a+3) & 0xff, then the top byte gets 0x80
        # so sign extension shows. The containing dword is read and the byte/short at (address & 3) extracted.
        def word(a):
            return ((a & 0xFF) | ((a + 1) & 0xFF) << 8 | ((a + 2) & 0xFF) << 16 | (((a + 3) & 0x7F) | 0x80) << 24)
        mem = {a: word(a) for a in range(self.BASE - 0x1010, self.BASE + 0x1010, 4)}
        for mode in ("v1, s[20:21]", "v[4:5], off"):
            for op, bits, signed in (("global_load_ubyte", 8, False), ("global_load_sbyte", 8, True),
                                     ("global_load_ushort", 16, False), ("global_load_sshort", 16, True)):
                offs = (-4, -2048, 2047, 0, -1, -2045) if bits == 8 else (-4, -2048, 2046, 0, -2)
                for off in offs:
                    line = f"{op} v10, {mode}" + (f" offset:{off}" if off else "")
                    with self.subTest(line=line):
                        ev, _ = self.run_line(line, memory=mem)
                        addr = self.BASE + off
                        self.assertEqual(ev.reads, [addr & ~3])
                        v = (mem[addr & ~3] >> (8 * (addr & 3))) & ((1 << bits) - 1)
                        if signed and v >> (bits - 1):
                            v = (v - (1 << bits)) & MASK32
                        self.assertEqual(ev.regs["%vgpr"][10], v)
        # One spelled-out case: sshort at offset -2046 from 0x...4800 is bytes 0x4002..0x4003 of the dword at 0x4000.
        ev, _ = self.run_line("global_load_sshort v10, v[4:5], off offset:-2046", memory={self.BASE - 2048: 0x8BAD1800})
        self.assertEqual(ev.regs["%vgpr"][10], 0xFFFF8BAD)

    def test_short_loads_crossing_dword_boundary(self):
        # At (addr & 3) == 3, 16-bit global loads read byte 3 of (addr & ~3) and byte 0 of (addr & ~3) + 4.
        for mode in ("v1, s[20:21]", "v[4:5], off"):
            for op, signed in (("global_load_ushort", False), ("global_load_sshort", True)):
                for off, hi_byte, want_hi in ((-1, 0x3F, 0x3F), (-2045, 0x80, 0x80), (2047, 0xFE, 0xFE)):
                    addr = self.BASE + off
                    lo_dw, hi_dw = addr & ~3, (addr & ~3) + 4
                    mem = {lo_dw: 0x99000000, hi_dw: 0xAABBCC00 | hi_byte}
                    expected = (want_hi << 8) | 0x99
                    if signed and (expected & 0x8000):
                        expected |= 0xFFFF0000
                    line = f"{op} v10, {mode} offset:{off}"
                    with self.subTest(line=line, hi_byte=hex(hi_byte)):
                        ev, _ = self.run_line(line, memory=mem)
                        self.assertEqual(ev.reads, [lo_dw, hi_dw])
                        self.assertEqual(ev.regs["%vgpr"][10], expected)
                        ev_off, _ = self.run_line(line, memory=mem, active=False)
                        self.assertEqual((ev_off.reads, ev_off.regs["%vgpr"][10]), ([], 0xEEEEEEEE))

    def test_store_both_modes(self):
        for mode in ("v1, v6, s[20:21]", "v[4:5], v6, off"):
            for off in (-4, -2048, 2044, 0):
                line = f"global_store_dword {mode}" + (f" offset:{off}" if off else "")
                with self.subTest(line=line):
                    ev, _ = self.run_line(line, memory={})
                    self.assertEqual(ev.writes, [self.BASE + off])
                    self.assertEqual(ev.memory, {self.BASE + off: 0xA1B2C3D4})
        ev, _ = self.run_line("global_store_dwordx2 v[4:5], v[6:7], off offset:-2048", memory={})
        self.assertEqual(ev.memory, {self.BASE - 2048: 0xA1B2C3D4, self.BASE - 2044: 0x0BADC0DE})

    def test_offset_2047_on_dword_forms(self):
        # A base one past a dword boundary makes base + 2047 dword-aligned, so the largest offset is checked on the
        # dword, multi-dword and store forms without depending on unaligned behaviour (which was not measured).
        self.SGPR = {20: 0x00004000, 21: 0x00000003}
        self.VGPR = {1: 0x00000801, 4: 0x00004801, 5: 0x00000003, 6: 0xA1B2C3D4, 7: 0x0BADC0DE}
        want = 0x0000000300004801 + 2047
        for mode, smode in (("v1, s[20:21]", "v1, v[6:7], s[20:21]"), ("v[4:5], off", "v[4:5], v[6:7], off")):
            with self.subTest(mode=mode):
                ev, _ = self.run_line(f"global_load_dword v10, {mode} offset:2047")
                self.assertEqual((ev.reads, ev.regs["%vgpr"][10]), ([want], tagged(want)))
                ev, _ = self.run_line(f"global_load_dwordx2 v[10:11], {mode} offset:2047")
                self.assertEqual(ev.reads, [want, want + 4])
                ev, _ = self.run_line(f"global_store_dwordx2 {smode} offset:2047", memory={})
                self.assertEqual(ev.memory, {want: 0xA1B2C3D4, want + 4: 0x0BADC0DE})
                ev, _ = self.run_line(f"global_store_dword {smode.replace('v[6:7]', 'v6')} offset:2047", memory={})
                self.assertEqual(ev.memory, {want: 0xA1B2C3D4})

    def test_hex_spelling_of_a_signed_offset(self):
        # Hand-written listings may spell the same signed value in hex; llvm-mc accepts and reprints it as decimal.
        for spelled, off in (("0x7ff", 2047), ("-0x4", -4), ("-0x800", -2048)):
            with self.subTest(spelled=spelled):
                ev, _ = self.run_line(f"global_load_ubyte v10, v[4:5], off offset:{spelled}")
                self.assertEqual(ev.reads, [(self.BASE + off) & ~3])

    def test_inactive_lane_neither_loads_nor_stores(self):
        ev, _ = self.run_line("global_load_dword v10, v[4:5], off offset:-4", active=False)
        self.assertEqual((ev.reads, ev.regs["%vgpr"][10]), ([], 0xEEEEEEEE))
        ev, _ = self.run_line("global_store_dword v[4:5], v6, off offset:-4", memory={}, active=False)
        self.assertEqual((ev.writes, ev.memory), ([], {}))

    def test_heap_base_is_added_once(self):
        ev, _ = self.run_line("global_load_dword v10, v[4:5], off offset:-4", memory={}, heap_base=0x0000007F00000000)
        self.assertEqual(ev.reads, [0x0000007F00000000 + self.BASE - 4])

    def test_carry_and_borrow_across_the_low_dword(self):
        # A negative offset below a 4 GiB boundary borrows from the high dword (the add is 64-bit).
        self.SGPR = {20: 0x00000000, 21: 0x00000003}
        self.VGPR = {1: 0x00000000}
        ev, _ = self.run_line("global_load_dword v10, v1, s[20:21] offset:-4", memory={})
        self.assertEqual(ev.reads, [0x00000002FFFFFFFC])
        self.VGPR = {4: 0xFFFFFFFC, 5: 0x00000003}
        ev, _ = self.run_line("global_load_dword v10, v[4:5], off offset:8", memory={})
        self.assertEqual(ev.reads, [0x0000000400000004])

    def test_raw_unsigned_field_is_the_same_offset(self):
        # llvm-mc never prints the field unsigned, but a tool that did (0xffc for -4) means the same address.
        for spelled, off in (("0xffc", -4), ("4092", -4), ("0x800", -2048), ("2048", -2048), ("0xff8", -8)):
            with self.subTest(offset=spelled):
                ev, _ = self.run_line(f"global_load_dword v10, v[4:5], off offset:{spelled}", memory={})
                self.assertEqual(ev.reads, [self.BASE + off])

    def test_offsets_outside_12_bits_raise(self):
        for spelled in ("4096", "-2049", "0x1000", "zz"):
            line = f"global_load_dword v10, v[4:5], off offset:{spelled}"
            with self.subTest(line=line):
                ins = parse_line(line)
                with self.assertRaises(rdna2spirv.Unsupported):
                    rdna2spirv.Gen("cs").global_mem(ins["op"], ins["args"], ins["mods"])


if __name__ == "__main__":
    unittest.main()
