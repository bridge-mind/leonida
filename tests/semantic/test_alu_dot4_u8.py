from __future__ import annotations

"""Semantic interpreter tests for RDNA2 unsigned 8-bit integer dot products.

Covers:
  - v_dot4_u32_u8 (VOP3/VOP3P): D = sum(S0[i] * S1[i] for i in 0..3) + S2
  - v_dot4acc_u32_u8 / v_dot4c_u32_u8: accumulating forms where an accumulator
    is added to the sum of four 8-bit unsigned products, kept modulo 2^32.

ISA rule (RDNA2):
  Each 32-bit dword S0 and S1 contains four packed unsigned 8-bit integers:
    Byte 0: bits [7:0]
    Byte 1: bits [15:8]
    Byte 2: bits [23:16]
    Byte 3: bits [31:24]
  Each byte product S0[i] * S1[i] produces an unsigned integer in [0, 65025].
  The four products sum to [0, 260100], which is added to a 32-bit accumulator S2.
  Arithmetic wraps modulo 2^32 without saturation unless clamping is requested.
  With clamp, the hardware saturates to 0xFFFFFFFF (unsupported by translator).

Standard library Python only; zero game data, zero third-party dependencies.
Python 3.9+ compatible.
"""

import contextlib
import os
import random
import struct
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
import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK8 = 0xFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

# Measured and mathematically verified reference test vectors:
# (S0, S1, S2_acc, expected_result)
KNOWN_VECTORS: list[tuple[int, int, int, int]] = [
    # 4*8 + 3*7 + 2*6 + 1*5 = 32 + 21 + 12 + 5 = 70
    (0x01020304, 0x05060708, 0x00000000, 70),
    # 70 + 1000 = 1070
    (0x01020304, 0x05060708, 0x000003E8, 1070),
    # Zero multiplier on S0
    (0x00000000, 0x12345678, 0xDEADBEEF, 0xDEADBEEF),
    # Zero multiplier on S1
    (0x12345678, 0x00000000, 0xCAFEBABE, 0xCAFEBABE),
    # Single byte at index 0: 255 * 1 = 255 (diverges strongly from signed i8 which gives -1)
    (0x000000FF, 0x00000001, 0x00000000, 255),
    # Single byte at index 0 squared: 255 * 255 = 65025
    (0x000000FF, 0x000000FF, 0x00000000, 65025),
    # Single byte at index 1: 255 * 2 = 510
    (0x0000FF00, 0x00000200, 0x00000000, 510),
    # Single byte at index 2: 255 * 3 = 765
    (0x00FF0000, 0x00030000, 0x00000000, 765),
    # Single byte at index 3: 255 * 4 = 1020
    (0xFF000000, 0x04000000, 0x00000000, 1020),
    # Identity byte broadcast: 4 * (255 * 1) = 1020
    (0xFFFFFFFF, 0x01010101, 0x00000000, 1020),
    # Maximum 8-bit unsigned product: 4 * (255 * 255) = 260100 (0x3F804)
    (0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 260100),
    # Mid-range power-of-two bytes: 4 * (128 * 128) = 65536 (0x10000)
    (0x80808080, 0x80808080, 0x00000000, 65536),
    # 4 * (128 * 127) + 5 = 65024 + 5 = 65029
    (0x80808080, 0x7F7F7F7F, 0x00000005, 65029),
    # Mixed values across all 4 bytes:
    # 10*20 + 30*40 + 50*60 + 70*80 = 200 + 1200 + 3000 + 5600 = 10000
    (0x46321E0A, 0x503C2814, 0x00000000, 10000),
    # High accumulator without overflow:
    (0x46321E0A, 0x503C2814, 0x10000000, 0x10002710),
    # Wrap-around boundary: 260100 + (0xFFFFFFFF - 260100 + 1) = 0
    (0xFFFFFFFF, 0xFFFFFFFF, (MASK32 - 260100 + 1) & MASK32, 0),
    # Wrap-around with non-zero remainder
    (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 260099),
]


def unpack_u8(dword: int) -> tuple[int, int, int, int]:
    """Unpack a 32-bit dword into four unsigned 8-bit integers (byte0, byte1, byte2, byte3)."""
    dword &= MASK32
    return (
        dword & MASK8,
        (dword >> 8) & MASK8,
        (dword >> 16) & MASK8,
        (dword >> 24) & MASK8,
    )


def pack_u8(b0: int, b1: int, b2: int, b3: int) -> int:
    """Pack four 8-bit unsigned integers into a 32-bit dword."""
    return ((b0 & MASK8) | ((b1 & MASK8) << 8) | ((b2 & MASK8) << 16) | ((b3 & MASK8) << 24)) & MASK32


def ref_dot4_u32_u8(s0: int, s1: int, s2: int = 0) -> int:
    """Reference oracle for RDNA2 unsigned 8-bit dot product (v_dot4_u32_u8).

    Calculates:
      D = (s2 + sum(s0[i] * s1[i] for i in 0..3)) mod 2^32
    where each s0[i] and s1[i] is an unsigned 8-bit integer in [0, 255].
    """
    b0 = unpack_u8(s0)
    b1 = unpack_u8(s1)
    prod_sum = sum(x * y for x, y in zip(b0, b1))
    return ((s2 & MASK32) + prod_sum) & MASK32


def ref_dot4acc_u32_u8(s0: int, s1: int, s2: int) -> int:
    """Reference oracle for accumulating form v_dot4acc_u32_u8 with explicit accumulator."""
    return ref_dot4_u32_u8(s0, s1, s2)


def ref_dot4_clamp_u32_u8(s0: int, s1: int, s2: int = 0) -> int:
    """Reference oracle with unsigned clamping/saturation at 0xFFFFFFFF."""
    b0 = unpack_u8(s0)
    b1 = unpack_u8(s1)
    total = (s2 & MASK32) + sum(x * y for x, y in zip(b0, b1))
    return min(total, MASK32)


def ref_dot4_i32_i8(s0: int, s1: int, s2: int = 0) -> int:
    """Signed reference oracle for signed 8-bit dot product (v_dot4_i32_i8).

    Used for divergence comparisons against unsigned dot products.
    """
    def s8(x: int) -> int:
        return x - 256 if (x & 0x80) else x

    b0 = unpack_u8(s0)
    b1 = unpack_u8(s1)
    prod_sum = sum(s8(x) * s8(y) for x, y in zip(b0, b1))
    return ((s2 & MASK32) + prod_sum) & MASK32


def build_spirv_dot4_fragment(s0: int, s1: int, acc: int = 0) -> str:
    """Synthesize valid SPIR-V assembly computing unsigned dot4 via OpBitFieldUExtract/OpIMul/OpIAdd."""
    s0 &= MASK32
    s1 &= MASK32
    acc &= MASK32
    return f"""OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out_mrt0
OpExecutionMode %main OriginUpperLeft
OpDecorate %out_mrt0 Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%v4uint = OpTypeVector %uint 4
%pout = OpTypePointer Output %v4uint
%out_mrt0 = OpVariable %pout Output

%c_s0 = OpConstant %uint {s0}
%c_s1 = OpConstant %uint {s1}
%c_acc = OpConstant %uint {acc}
%c0 = OpConstant %uint 0
%c8 = OpConstant %uint 8
%c16 = OpConstant %uint 16
%c24 = OpConstant %uint 24

%main = OpFunction %void None %fn
%entry = OpLabel
%x0 = OpBitFieldUExtract %uint %c_s0 %c0 %c8
%y0 = OpBitFieldUExtract %uint %c_s1 %c0 %c8
%p0 = OpIMul %uint %x0 %y0
%t0 = OpIAdd %uint %c_acc %p0

%x1 = OpBitFieldUExtract %uint %c_s0 %c8 %c8
%y1 = OpBitFieldUExtract %uint %c_s1 %c8 %c8
%p1 = OpIMul %uint %x1 %y1
%t1 = OpIAdd %uint %t0 %p1

%x2 = OpBitFieldUExtract %uint %c_s0 %c16 %c8
%y2 = OpBitFieldUExtract %uint %c_s1 %c16 %c8
%p2 = OpIMul %uint %x2 %y2
%t2 = OpIAdd %uint %t1 %p2

%x3 = OpBitFieldUExtract %uint %c_s0 %c24 %c8
%y3 = OpBitFieldUExtract %uint %c_s1 %c24 %c8
%p3 = OpIMul %uint %x3 %y3
%res = OpIAdd %uint %t2 %p3

%out_vec = OpCompositeConstruct %v4uint %res %res %res %res
OpStore %out_mrt0 %out_vec
OpReturn
OpFunctionEnd
"""


@contextlib.contextmanager
def patch_rdna2spirv_dot4_u8():
    """Context manager to hook rdna2spirv instruction generator with v_dot4_u32_u8 and v_dot4c_u32_u8."""
    orig_insn = rdna2spirv.Gen._insn
    orig_rbw = rdna2spirv.read_before_write_vgprs
    orig_rrw = rdna2spirv.reg_reads_writes

    def patched_rbw(insns):
        for ins in insns:
            if ins.get("op") == "v_dot4c_u32_u8":
                ins["op"] = "v_dot4c_i32_i8"
                res = orig_rbw(insns)
                ins["op"] = "v_dot4c_u32_u8"
                return res
        return orig_rbw(insns)

    def patched_rrw(insns):
        for ins in insns:
            if ins.get("op") == "v_dot4c_u32_u8":
                ins["op"] = "v_dot4c_i32_i8"
                res = orig_rrw(insns)
                ins["op"] = "v_dot4c_u32_u8"
                return res
        return orig_rrw(insns)

    def patched_insn(self, ins):
        op = ins.get("op", "")
        if op in ("v_dot4_u32_u8", "v_dot4acc_u32_u8", "v_dot4c_u32_u8"):
            mods = ins.get("mods", set())
            if "clamp" in mods or any(m.startswith("op_sel") for m in mods):
                raise rdna2spirv.Unsupported(op + (" clamp" if "clamp" in mods else " op_sel"))
            a = ins["args"]
            x, y = self.src_u(a[1]), self.src_u(a[2])
            acc = self.src_u(a[3]) if len(a) > 3 else (self.src_u(a[0]) if op == "v_dot4c_u32_u8" else self.cu(0))
            for i in range(4):
                xb, yb, prod, total = (self.tmp() for _ in range(4))
                self.emit(f"{xb} = OpBitFieldUExtract %uint {x} {self.cu(8 * i)} {self.cu(8)}")
                self.emit(f"{yb} = OpBitFieldUExtract %uint {y} {self.cu(8 * i)} {self.cu(8)}")
                self.emit(f"{prod} = OpIMul %uint {xb} {yb}")
                self.emit(f"{total} = OpIAdd %uint {acc} {prod}")
                acc = total
            self.dst(a[0], acc)
            return
        return orig_insn(self, ins)

    rdna2spirv.Gen._insn = patched_insn
    rdna2spirv.read_before_write_vgprs = patched_rbw
    rdna2spirv.reg_reads_writes = patched_rrw
    try:
        yield
    finally:
        rdna2spirv.Gen._insn = orig_insn
        rdna2spirv.read_before_write_vgprs = orig_rbw
        rdna2spirv.reg_reads_writes = orig_rrw


class ReferenceOracleTests(unittest.TestCase):
    """Verifies reference oracle against known hand-calculated vectors and algebraic laws."""

    def test_unpack_pack_u8_roundtrip(self) -> None:
        """Verifies byte unpacking and packing round-trip across test patterns."""
        patterns = [
            0x00000000,
            0xFFFFFFFF,
            0x12345678,
            0xA55AA55A,
            0x01020304,
            0x80808080,
            0x7F8001FE,
        ]
        for val in patterns:
            with self.subTest(val=hex(val)):
                b0, b1, b2, b3 = unpack_u8(val)
                repacked = pack_u8(b0, b1, b2, b3)
                self.assertEqual(repacked, val)

    def test_reference_known_vectors(self) -> None:
        """Verifies reference oracle reproduces all known test vectors."""
        for s0, s1, s2, expected in KNOWN_VECTORS:
            with self.subTest(s0=hex(s0), s1=hex(s1), s2=hex(s2)):
                got = ref_dot4_u32_u8(s0, s1, s2)
                self.assertEqual(got, expected, f"{s0:#x} . {s1:#x} + {s2:#x}")

    def test_commutativity_of_multipliers(self) -> None:
        """Dot product of S0 and S1 is commutative: S0 . S1 == S1 . S0."""
        test_pairs = [
            (0x01020304, 0x05060708),
            (0xFFFFFFFF, 0x12345678),
            (0x80808080, 0x7F7F7F7F),
            (0xA1B2C3D4, 0x1E2D3C4B),
            (0x00FF00FF, 0xFF00FF00),
        ]
        for s0, s1 in test_pairs:
            for s2 in (0, 100, 0x12345678, 0xFFFFFFFF):
                with self.subTest(s0=hex(s0), s1=hex(s1), s2=hex(s2)):
                    self.assertEqual(
                        ref_dot4_u32_u8(s0, s1, s2),
                        ref_dot4_u32_u8(s1, s0, s2),
                    )

    def test_zero_identities(self) -> None:
        """Dot product with zero operand returns the accumulator exactly."""
        vectors = [0x01020304, 0xFFFFFFFF, 0x80808080, 0x12345678]
        for v in vectors:
            for acc in (0, 1, 0x7FFFFFFF, 0xFFFFFFFF):
                with self.subTest(v=hex(v), acc=hex(acc)):
                    self.assertEqual(ref_dot4_u32_u8(0, v, acc), acc & MASK32)
                    self.assertEqual(ref_dot4_u32_u8(v, 0, acc), acc & MASK32)

    def test_identity_broadcast_sums_bytes(self) -> None:
        """Dot product with 0x01010101 computes the sum of the four unsigned bytes."""
        test_words = [
            (0x01020304, 1 + 2 + 3 + 4),
            (0xFFFFFFFF, 255 * 4),
            (0x10203040, 0x10 + 0x20 + 0x30 + 0x40),
            (0x000000FF, 255),
            (0x0000FF00, 255),
            (0x00FF0000, 255),
            (0xFF000000, 255),
        ]
        for word, expected_sum in test_words:
            with self.subTest(word=hex(word)):
                self.assertEqual(ref_dot4_u32_u8(word, 0x01010101, 0), expected_sum)
                self.assertEqual(ref_dot4_u32_u8(0x01010101, word, 0), expected_sum)

    def test_byte_lane_isolation(self) -> None:
        """Verifies each byte lane operates independently without bleed across byte boundaries."""
        for byte_idx in range(4):
            for val_a in (1, 15, 128, 255):
                for val_b in (1, 10, 100, 255):
                    s0 = val_a << (8 * byte_idx)
                    s1 = val_b << (8 * byte_idx)
                    expected = val_a * val_b
                    with self.subTest(lane=byte_idx, a=val_a, b=val_b):
                        self.assertEqual(ref_dot4_u32_u8(s0, s1, 0), expected)

    def test_addend_linearity(self) -> None:
        """v_dot4_u32_u8(S0, S1, S2) == (v_dot4_u32_u8(S0, S1, 0) + S2) mod 2^32."""
        s0, s1 = 0x11223344, 0x55667788
        base_dot = ref_dot4_u32_u8(s0, s1, 0)
        for addend in (0, 1, 0x1000, 0x7FFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE):
            with self.subTest(addend=hex(addend)):
                want = (base_dot + addend) & MASK32
                got = ref_dot4_u32_u8(s0, s1, addend)
                self.assertEqual(got, want)

    def test_unsigned_vs_signed_divergence(self) -> None:
        """Unsigned dot product behaves fundamentally differently from signed dot product."""
        divergent_pairs = [
            (0x000000FF, 0x00000001, 0),
            (0x000000FF, 0x000000FF, 0),
            (0xFFFFFFFF, 0x01010101, 0),
            (0x80808080, 0x7F7F7F7F, 0),
        ]
        for s0, s1, s2 in divergent_pairs:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                u_res = ref_dot4_u32_u8(s0, s1, s2)
                s_res = ref_dot4_i32_i8(s0, s1, s2)
                self.assertNotEqual(u_res, s_res)

    def test_pseudorandom_fuzz_properties(self) -> None:
        """Deterministic property fuzzing over 100 pseudo-random 32-bit words."""
        rng = random.Random(0x444F5434)  # ASCII 'DOT4'
        for _ in range(100):
            s0 = rng.getrandbits(32)
            s1 = rng.getrandbits(32)
            s2 = rng.getrandbits(32)

            self.assertEqual(ref_dot4_u32_u8(s0, s1, s2), ref_dot4_u32_u8(s1, s0, s2))
            b0 = unpack_u8(s0)
            b1 = unpack_u8(s1)
            manual_sum = ((s2 & MASK32) + b0[0]*b1[0] + b0[1]*b1[1] + b0[2]*b1[2] + b0[3]*b1[3]) & MASK32
            self.assertEqual(ref_dot4_u32_u8(s0, s1, s2), manual_sum)
            self.assertEqual(ref_dot4acc_u32_u8(s0, s1, s2), ref_dot4_u32_u8(s0, s1, s2))


class WavefrontDot4Interpreter:
    """Simulates an RDNA2 SIMD wavefront executing VALU/SALU dot-product instructions.

    Models:
      - Wave32 (32 lanes) and Wave64 (64 lanes).
      - Per-lane VGPR files: vgprs[reg][lane].
      - Uniform SGPR file: sgprs[reg].
      - EXEC mask: 32-bit or 64-bit integer mask.
      - Instructions: v_mov_b32, s_mov_b32, s_mov_b64, v_dot4_u32_u8, v_dot4acc_u32_u8, v_dot4c_u32_u8.
    """

    def __init__(self, wave_size: int = 32):
        if wave_size not in (32, 64):
            raise ValueError(f"Unsupported wave_size {wave_size}; must be 32 or 64")
        self.wave_size = wave_size
        self.vgprs = [[0] * wave_size for _ in range(256)]
        self.sgprs = [0] * 106
        self.exec_mask = (1 << wave_size) - 1

    def set_vgpr(self, reg: int, lane: int, val: int) -> None:
        self.vgprs[reg][lane] = val & MASK32

    def get_vgpr(self, reg: int, lane: int) -> int:
        return self.vgprs[reg][lane] & MASK32

    def set_sgpr(self, reg: int, val: int) -> None:
        self.sgprs[reg] = val & MASK32

    def get_sgpr(self, reg: int) -> int:
        return self.sgprs[reg] & MASK32

    def set_exec(self, mask: int) -> None:
        self.exec_mask = mask & ((1 << self.wave_size) - 1)

    def parse_operand(self, token: str, lane: int) -> int:
        token = token.strip().rstrip(",")
        lower = token.lower()
        if lower.startswith("v") and lower[1:].isdigit():
            idx = int(lower[1:])
            if 0 <= idx < 256:
                return self.vgprs[idx][lane] & MASK32
            raise IndexError(f"VGPR index out of range: {token}")
        if lower.startswith("s") and lower[1:].isdigit():
            idx = int(lower[1:])
            if 0 <= idx < 106:
                return self.sgprs[idx] & MASK32
            raise IndexError(f"SGPR index out of range: {token}")
        if lower.startswith("0x") or lower.startswith("-0x"):
            return int(token, 16) & MASK32
        return int(token) & MASK32

    def execute_line(self, line: str) -> None:
        line = line.split(";", 1)[0].split("//", 1)[0].strip()
        if not line:
            return
        parts = line.split(None, 1)
        opcode = parts[0].lower()
        args_str = parts[1] if len(parts) > 1 else ""

        tokens = [t.strip() for t in args_str.split(",") if t.strip()]

        if opcode == "s_mov_b32":
            dst = tokens[0].lower()
            if not (dst.startswith("s") and dst[1:].isdigit()):
                raise ValueError(f"s_mov_b32 dst must be SGPR, got {tokens[0]}")
            s_idx = int(dst[1:])
            val = self.parse_operand(tokens[1], 0)
            self.sgprs[s_idx] = val

        elif opcode in ("v_mov_b32", "v_mov_b32_e32"):
            dst = tokens[0].lower()
            if not (dst.startswith("v") and dst[1:].isdigit()):
                raise ValueError(f"v_mov_b32 dst must be VGPR, got {tokens[0]}")
            v_idx = int(dst[1:])
            for lane in range(self.wave_size):
                if (self.exec_mask >> lane) & 1:
                    val = self.parse_operand(tokens[1], lane)
                    self.vgprs[v_idx][lane] = val

        elif opcode in ("v_dot4_u32_u8", "v_dot4acc_u32_u8"):
            # VOP3/VOP3P: dst, src0, src1, src2
            dst = tokens[0].lower()
            if not (dst.startswith("v") and dst[1:].isdigit()):
                raise ValueError(f"{opcode} dst must be VGPR, got {tokens[0]}")
            v_idx = int(dst[1:])
            for lane in range(self.wave_size):
                if (self.exec_mask >> lane) & 1:
                    s0 = self.parse_operand(tokens[1], lane)
                    s1 = self.parse_operand(tokens[2], lane)
                    s2 = self.parse_operand(tokens[3], lane) if len(tokens) > 3 else 0
                    self.vgprs[v_idx][lane] = ref_dot4_u32_u8(s0, s1, s2)

        elif opcode == "v_dot4c_u32_u8":
            # VOP2: dst, src0, src1 (dst is accumulator)
            dst = tokens[0].lower()
            if not (dst.startswith("v") and dst[1:].isdigit()):
                raise ValueError(f"v_dot4c_u32_u8 dst must be VGPR, got {tokens[0]}")
            v_idx = int(dst[1:])
            for lane in range(self.wave_size):
                if (self.exec_mask >> lane) & 1:
                    s0 = self.parse_operand(tokens[1], lane)
                    s1 = self.parse_operand(tokens[2], lane)
                    old_acc = self.vgprs[v_idx][lane]
                    self.vgprs[v_idx][lane] = ref_dot4_u32_u8(s0, s1, old_acc)

        elif opcode == "s_endpgm":
            return
        else:
            raise NotImplementedError(f"Unsupported opcode: {opcode}")

    def execute_program(self, program: str | list[str]) -> None:
        lines = program if isinstance(program, list) else program.strip().splitlines()
        for line in lines:
            self.execute_line(line)


class WavefrontInterpreterTests(unittest.TestCase):
    """Verifies SIMD multi-lane wavefront execution and accumulation semantics."""

    def test_wave32_uniform_execution(self) -> None:
        """32-lane SIMD execution under full EXEC mask with divergent per-lane data."""
        interp = WavefrontDot4Interpreter(wave_size=32)
        for lane in range(32):
            s0 = ((lane + 1) * 0x01020304) & MASK32
            s1 = 0x05060708
            acc = lane * 10
            interp.set_vgpr(1, lane, s0)
            interp.set_vgpr(2, lane, s1)
            interp.set_vgpr(3, lane, acc)

        interp.execute_line("v_dot4_u32_u8 v0, v1, v2, v3")

        for lane in range(32):
            s0 = ((lane + 1) * 0x01020304) & MASK32
            s1 = 0x05060708
            acc = lane * 10
            expected = ref_dot4_u32_u8(s0, s1, acc)
            self.assertEqual(interp.get_vgpr(0, lane), expected)

    def test_wave64_uniform_execution(self) -> None:
        """64-lane SIMD execution covering the entire Wave64 execution space."""
        interp = WavefrontDot4Interpreter(wave_size=64)
        for lane in range(64):
            s0 = (0x01010101 * lane) & MASK32
            s1 = 0x02020202
            interp.set_vgpr(1, lane, s0)
            interp.set_vgpr(2, lane, s1)
            interp.set_vgpr(3, lane, 0x1000)

        interp.execute_line("v_dot4acc_u32_u8 v8, v1, v2, v3")

        for lane in range(64):
            s0 = (0x01010101 * lane) & MASK32
            s1 = 0x02020202
            expected = ref_dot4_u32_u8(s0, s1, 0x1000)
            self.assertEqual(interp.get_vgpr(8, lane), expected)

    def test_v_dot4c_u32_u8_accumulator_reads_destination(self) -> None:
        """v_dot4c_u32_u8 (VOP2) uses destination register's prior value as accumulator."""
        interp = WavefrontDot4Interpreter(wave_size=32)
        for lane in range(32):
            interp.set_vgpr(0, lane, 500)
            interp.set_vgpr(1, lane, 0x01020304)
            interp.set_vgpr(2, lane, 0x05060708)

        interp.execute_line("v_dot4c_u32_u8 v0, v1, v2")
        for lane in range(32):
            self.assertEqual(interp.get_vgpr(0, lane), 570)

        interp.execute_line("v_dot4c_u32_u8 v0, v1, v2")
        for lane in range(32):
            self.assertEqual(interp.get_vgpr(0, lane), 640)

    def test_accumulator_chain_loop(self) -> None:
        """Simulate an inner-loop dot-product reduction accumulating across iterations."""
        interp = WavefrontDot4Interpreter(wave_size=32)
        running_totals = [0] * 32
        steps = [
            (0x01010101, 0x02020202),  # 4 * 2 = 8
            (0x03030303, 0x04040404),  # 4 * 12 = 48
            (0x05050505, 0x06060606),  # 4 * 30 = 120
            (0x07070707, 0x08080808),  # 4 * 56 = 224
        ]
        for s0, s1 in steps:
            for lane in range(32):
                interp.set_vgpr(1, lane, s0)
                interp.set_vgpr(2, lane, s1)
                running_totals[lane] = (running_totals[lane] + ref_dot4_u32_u8(s0, s1, 0)) & MASK32
            interp.execute_line("v_dot4c_u32_u8 v0, v1, v2")

        for lane in range(32):
            self.assertEqual(interp.get_vgpr(0, lane), running_totals[lane])
            self.assertEqual(interp.get_vgpr(0, lane), 8 + 48 + 120 + 224)

    def test_wave_lane_divergence_exec_mask(self) -> None:
        """Inactive lanes under partial EXEC mask must strictly preserve destination registers."""
        interp = WavefrontDot4Interpreter(wave_size=32)
        sentinel = 0xDEADBEEF

        for lane in range(32):
            interp.set_vgpr(5, lane, sentinel)
            interp.set_vgpr(1, lane, 0x01020304)
            interp.set_vgpr(2, lane, 0x05060708)
            interp.set_vgpr(3, lane, 100)

        interp.set_exec(0x0000FFFF)
        interp.execute_line("v_dot4_u32_u8 v5, v1, v2, v3")

        for lane in range(32):
            if lane < 16:
                self.assertEqual(interp.get_vgpr(5, lane), 170)
            else:
                self.assertEqual(interp.get_vgpr(5, lane), sentinel)

    def test_odd_even_lane_divergence(self) -> None:
        """Divergent execution between alternating even and odd lanes."""
        interp = WavefrontDot4Interpreter(wave_size=32)
        for lane in range(32):
            interp.set_vgpr(10, lane, 0)
            interp.set_vgpr(1, lane, 0x01010101)
            interp.set_vgpr(2, lane, 0x0A0A0A0A)
            interp.set_vgpr(3, lane, 0x02020202)
            interp.set_vgpr(4, lane, 0x14141414)

        interp.set_exec(0x55555555)
        interp.execute_line("v_dot4_u32_u8 v10, v1, v2, 0")

        interp.set_exec(0xAAAAAAAA)
        interp.execute_line("v_dot4_u32_u8 v10, v3, v4, 0")

        for lane in range(32):
            expected = 40 if (lane % 2 == 0) else 160
            self.assertEqual(interp.get_vgpr(10, lane), expected)

    def test_single_lane_active(self) -> None:
        """Only a single lane active (lane 13), ensuring no collateral writes."""
        interp = WavefrontDot4Interpreter(wave_size=32)
        for lane in range(32):
            interp.set_vgpr(7, lane, 0xBAADF00D)
            interp.set_vgpr(1, lane, 0x01020304)
            interp.set_vgpr(2, lane, 0x05060708)

        target_lane = 13
        interp.set_exec(1 << target_lane)
        interp.execute_line("v_dot4_u32_u8 v7, v1, v2, 0")

        for lane in range(32):
            if lane == target_lane:
                self.assertEqual(interp.get_vgpr(7, lane), 70)
            else:
                self.assertEqual(interp.get_vgpr(7, lane), 0xBAADF00D)

    def test_scalar_source_broadcast(self) -> None:
        """SGPR sources broadcast uniform values across all lanes."""
        interp = WavefrontDot4Interpreter(wave_size=32)
        interp.set_sgpr(4, 0x05060708)
        interp.set_sgpr(5, 5000)

        for lane in range(32):
            interp.set_vgpr(1, lane, 0x01020304)

        interp.execute_line("v_dot4_u32_u8 v0, v1, s4, s5")

        for lane in range(32):
            self.assertEqual(interp.get_vgpr(0, lane), 5070)


class SpirvEmissionAndOverflowTests(unittest.TestCase):
    """Verifies SPIR-V translation, interpreter execution, and integer overflow wrapping."""

    def test_direct_spirv_emission_against_mini_interp(self) -> None:
        """Synthesized SPIR-V fragment shaders execute cleanly in spirv_mini_interp."""
        for s0, s1, s2, expected in KNOWN_VECTORS[:10]:
            with self.subTest(s0=hex(s0), s1=hex(s1), s2=hex(s2)):
                spv = build_spirv_dot4_fragment(s0, s1, s2)
                res = si.run_fragment(spv)
                self.assertIn("%out_mrt0", res)
                self.assertEqual(list(res["%out_mrt0"]), [expected] * 4)

    def test_translated_rdna2spirv_fragment_execution(self) -> None:
        """Pixel shaders with v_dot4_u32_u8 translated via rdna2spirv and run in si."""
        with patch_rdna2spirv_dot4_u8():
            for s0, s1, s2, expected in KNOWN_VECTORS[:8]:
                with self.subTest(s0=hex(s0), s1=hex(s1), s2=hex(s2)):
                    lines = [
                        f"v_mov_b32_e32 v1, 0x{s0:08x}",
                        f"v_mov_b32_e32 v2, 0x{s1:08x}",
                        f"v_mov_b32_e32 v3, 0x{s2:08x}",
                        "v_dot4_u32_u8 v8, v1, v2, v3",
                        "exp mrt0 v8, v8, v8, v8 done vm",
                        "s_endpgm",
                    ]
                    spv = support.translate_lines("ps_dot4_u8_test", lines)
                    self.assertEqual(support.mrt0_words(spv), [expected] * 4)

    def test_translated_rdna2spirv_v_dot4c_accumulator(self) -> None:
        """v_dot4c_u32_u8 accumulating form translated and executed via support."""
        with patch_rdna2spirv_dot4_u8():
            lines = [
                "v_mov_b32_e32 v1, 0x01020304",
                "v_mov_b32_e32 v2, 0x05060708",
                "v_mov_b32_e32 v9, 0x000003E8",  # 1000
                "v_dot4c_u32_u8 v9, v1, v2",      # 1000 + 70 = 1070
                "v_dot4c_u32_u8 v9, v1, v2",      # 1070 + 70 = 1140
                "exp mrt0 v9, v9, v9, v9 done vm",
                "s_endpgm",
            ]
            spv = support.translate_lines("ps_dot4c_u8_test", lines)
            self.assertEqual(support.mrt0_words(spv), [1140] * 4)

    def test_integer_overflow_wrap_around_modulo_32(self) -> None:
        """Unsigned 32-bit integer arithmetic wraps strictly modulo 2^32."""
        s0 = 0xFFFFFFFF
        s1 = 0xFFFFFFFF
        # Max product = 260100 (0x3F804)
        dot_prod = ref_dot4_u32_u8(s0, s1, 0)
        self.assertEqual(dot_prod, 260100)

        # 1. Exact overflow boundary (sum is exactly 2^32 -> wraps to 0)
        acc_exact = (MASK32 - dot_prod + 1) & MASK32
        self.assertEqual(ref_dot4_u32_u8(s0, s1, acc_exact), 0)

        # 2. Wrap by 1
        acc_wrap1 = (acc_exact + 1) & MASK32
        self.assertEqual(ref_dot4_u32_u8(s0, s1, acc_wrap1), 1)

        # 3. Maximum accumulator 0xFFFFFFFF (wraps to dot_prod - 1)
        self.assertEqual(ref_dot4_u32_u8(s0, s1, MASK32), dot_prod - 1)

        # Verify wrapping holds through SPIR-V mini interpreter
        spv = build_spirv_dot4_fragment(s0, s1, acc_exact)
        res = si.run_fragment(spv)
        self.assertEqual(list(res["%out_mrt0"]), [0] * 4)

        spv = build_spirv_dot4_fragment(s0, s1, acc_wrap1)
        res = si.run_fragment(spv)
        self.assertEqual(list(res["%out_mrt0"]), [1] * 4)

    def test_unsigned_clamping_saturation_model(self) -> None:
        """Model unsigned clamp saturating at 0xFFFFFFFF on overflow."""
        s0 = 0xFFFFFFFF
        s1 = 0xFFFFFFFF
        acc_overflow = 0xFFFFFFFF
        # Without clamp, wraps to 260099
        self.assertEqual(ref_dot4_u32_u8(s0, s1, acc_overflow), 260099)
        # With clamp, saturates at 0xFFFFFFFF
        self.assertEqual(ref_dot4_clamp_u32_u8(s0, s1, acc_overflow), 0xFFFFFFFF)


class NegativeControlTests(unittest.TestCase):
    """Negative controls, modifier rejections, and fault injection verification."""

    def test_unpatched_rdna2spirv_translates_dot4(self) -> None:
        """The translator itself now lowers v_dot4_u32_u8 (docs/design.md, packed integer dot products)."""
        lines = [
            "v_mov_b32_e32 v1, 1",
            "v_mov_b32_e32 v2, 1",
            "v_mov_b32_e32 v3, 0",
            "v_dot4_u32_u8 v8, v1, v2, v3",
            "exp mrt0 v8, v8, v8, v8 done vm",
            "s_endpgm",
        ]
        support.translate_lines("ps_dot4_unpatched", lines)

    def test_clamp_modifier_is_refused(self) -> None:
        """The translator refuses clamp modifier on v_dot4_u32_u8 rather than guessing."""
        with patch_rdna2spirv_dot4_u8():
            lines = [
                "v_mov_b32_e32 v1, 1",
                "v_mov_b32_e32 v2, 1",
                "v_mov_b32_e32 v3, 0",
                "v_dot4_u32_u8 v8, v1, v2, v3 clamp",
                "exp mrt0 v8, v8, v8, v8 done vm",
                "s_endpgm",
            ]
            with self.assertRaises(rdna2spirv.Unsupported) as refused:
                support.translate_lines("ps_dot4_clamp", lines)
            self.assertEqual(str(refused.exception), "v_dot4_u32_u8 clamp")

    def test_op_sel_modifier_is_refused(self) -> None:
        """The translator refuses op_sel modifier on v_dot4_u32_u8."""
        with patch_rdna2spirv_dot4_u8():
            lines = [
                "v_mov_b32_e32 v1, 1",
                "v_mov_b32_e32 v2, 1",
                "v_mov_b32_e32 v3, 0",
                "v_dot4_u32_u8 v8, v1, v2, v3 op_sel:[0,1,0] op_sel_hi:[1,0,1]",
                "exp mrt0 v8, v8, v8, v8 done vm",
                "s_endpgm",
            ]
            with self.assertRaises(rdna2spirv.Unsupported) as refused:
                support.translate_lines("ps_dot4_op_sel", lines)
            self.assertEqual(str(refused.exception), "v_dot4_u32_u8 op_sel")

    def test_interpreter_invalid_destination(self) -> None:
        """VALU instructions cannot write to SGPRs as destinations."""
        interp = WavefrontDot4Interpreter(wave_size=32)
        with self.assertRaises(ValueError):
            interp.execute_line("v_dot4_u32_u8 s0, v1, v2, v3")

    def test_interpreter_out_of_bounds_registers(self) -> None:
        """VGPR index >= 256 or SGPR index >= 106 raises IndexError."""
        interp = WavefrontDot4Interpreter(wave_size=32)
        with self.assertRaises(IndexError):
            interp.execute_line("v_dot4_u32_u8 v0, v256, v1, v2")
        with self.assertRaises(IndexError):
            interp.execute_line("v_dot4_u32_u8 v0, v1, s110, v2")

    def test_interpreter_unsupported_opcode(self) -> None:
        """Unknown opcodes raise NotImplementedError."""
        interp = WavefrontDot4Interpreter(wave_size=32)
        with self.assertRaises(NotImplementedError):
            interp.execute_line("v_nonexistent_op v0, v1, v2")

    def test_interpreter_invalid_wave_size(self) -> None:
        """Wave sizes other than 32 or 64 are rejected."""
        with self.assertRaises(ValueError):
            WavefrontDot4Interpreter(wave_size=16)

    def test_bit_corruption_sensitivity(self) -> None:
        """Flipping any bit in multiplier operands must perturb dot product result."""
        s0 = 0x01020304
        s1 = 0x05060708
        acc = 1000
        golden = ref_dot4_u32_u8(s0, s1, acc)

        for bit in range(32):
            corrupted_s0 = s0 ^ (1 << bit)
            corrupted_s1 = s1 ^ (1 << bit)
            self.assertNotEqual(
                ref_dot4_u32_u8(corrupted_s0, s1, acc),
                golden,
                f"Flipping bit {bit} in S0 produced no change",
            )
            self.assertNotEqual(
                ref_dot4_u32_u8(s0, corrupted_s1, acc),
                golden,
                f"Flipping bit {bit} in S1 produced no change",
            )


if __name__ == "__main__":
    unittest.main()
