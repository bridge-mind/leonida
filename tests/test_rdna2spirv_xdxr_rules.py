"""Tests for XDXR BVH acceleration-structure builder compute shader translation rules:
- buffer_load_ushort / buffer_load_sshort
- tbuffer_load_format_x GFX10_FORMAT_8_UINT
- global_atomic_fmax
- global_atomic_fmin
"""
import os
import struct
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import rdna2spirv  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
DEFAULT_CS_FSSM = bytes(0x138) + struct.pack("<4I", 0x8088, 64, 1, 1)


def translate_cs(name: str, body: str) -> str:
    """Create a temporary test compute shader and translate it to SPIR-V assembly text."""
    with tempfile.TemporaryDirectory() as tmp:
        disasm_dir = os.path.join(tmp, "disasm")
        fssm_dir = os.path.join(tmp, "fssm")
        os.makedirs(disasm_dir, exist_ok=True)
        os.makedirs(fssm_dir, exist_ok=True)

        with open(os.path.join(tmp, f"{name}.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(fssm_dir, f"{name}.fssm"), "wb") as f:
            f.write(DEFAULT_CS_FSSM)

        src = os.path.join(disasm_dir, f"{name}.s")
        full_body = f"; {name} synthetic test\n{body}\n\ts_endpgm\n"
        with open(src, "w", encoding="utf-8") as f:
            f.write(full_body)

        return rdna2spirv.translate(src)


class TestRdna2SpirvXdxrRules(unittest.TestCase):
    def test_buffer_load_ushort(self):
        """buffer_load_ushort loads 16 bits zero-extended into destination register."""
        asm = translate_cs("cs_load_ushort", "\tbuffer_load_ushort v0, v1, s[0:3], 0 offen")
        self.assertIn("OpConvertUToPtr %_ptr_PSB_arr", asm)
        self.assertIn("OpLoad %uint", asm)
        self.assertTrue("65535" in asm or "0xffff" in asm.lower())

    def test_buffer_load_sshort_signed(self):
        """buffer_load_sshort loads 16 bits sign-extended into destination register."""
        asm = translate_cs("cs_load_short", "\tbuffer_load_sshort v0, v1, s[0:3], 0 offen")
        self.assertIn("OpConvertUToPtr %_ptr_PSB_arr", asm)
        self.assertIn("OpLoad %uint", asm)
        self.assertIn("OpBitFieldSExtract", asm)

    def test_tbuffer_load_format_8_uint(self):
        """tbuffer_load_format_x with BUF_FMT_8_UINT decodes format 5 into uint."""
        asm = translate_cs("cs_tb_8_uint", "\ttbuffer_load_format_x v0, v1, s[0:3], 0 format:[BUF_FMT_8_UINT] offen")
        self.assertIn("OpConvertUToPtr %_ptr_PSB_arr", asm)
        self.assertIn("OpLoad %uint", asm)
        self.assertIn("OpBitFieldUExtract %uint", asm)

    def test_global_atomic_fmax(self):
        """global_atomic_fmax emits CAS loop with GLSL NMax."""
        asm = translate_cs("cs_atomic_fmax", "\tglobal_atomic_fmax v0, v1, s[0:1]")
        self.assertIn("OpAtomicCompareExchange %uint", asm)
        self.assertIn("NMax", asm)

    def test_global_atomic_fmin(self):
        """global_atomic_fmin emits CAS loop with GLSL NMin."""
        asm = translate_cs("cs_atomic_fmin", "\tglobal_atomic_fmin v0, v1, s[0:1]")
        self.assertIn("OpAtomicCompareExchange %uint", asm)
        self.assertIn("NMin", asm)

    def test_global_atomic_fmax_with_glc(self):
        """global_atomic_fmax with glc stores the pre-op value into vdst."""
        asm = translate_cs("cs_atomic_fmax_glc", "\tglobal_atomic_fmax v0, v1, v2, s[0:1] glc")
        self.assertIn("OpAtomicCompareExchange %uint", asm)
        self.assertIn("NMax", asm)
        self.assertIn("OpStore", asm)


if __name__ == "__main__":
    unittest.main()
