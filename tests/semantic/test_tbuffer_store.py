"""tbuffer_store_format_* (MTBUF stores with the instruction's own format) through the translator and the interpreter.

cs_tbuffer_store.asm stores into 13 sentinel-filled regions through V#s whose FORMAT differs from the instruction's,
then reads every region back into out[0..51]. check.py's TBUFFER_STORE_WORDS holds the words a Radeon 680M (gfx1035)
gives natively; the rules are in the .asm header. Here the translated fixture runs on the CPU and must give the same
words, so the GPU test, the hardware and the translation are kept in step.
"""
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "runtime", "test", "shaders"))
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))

import check       # noqa: E402  (runtime/test/shaders/check.py; importing it runs nothing)
import rdna2spirv  # noqa: E402
import support     # noqa: E402


def fssm(rsrc2, group):
    """The compute-shader header words the translator reads: COMPUTE_PGM_RSRC2 and the thread-group size."""
    header = bytearray(0x148)
    struct.pack_into("<4I", header, 0x138, rsrc2, *group)
    return bytes(header)


# 4 user SGPRs (the output V#), workgroup id in s4, 64x1x1 threads: the header build.py gives the GPU tests.
CS_FSSM = fssm(0x8088, (64, 1, 1))
# The output V# vk_compute_test passes in s[0:3]: base 0, num_records 0xffffffff, 32_FLOAT, OOB_SELECT 3.
OUT_VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]


def run(spv):
    memory = {}
    outputs = support.run_fragment(spv, inputs={"%wg_id": (0, 0, 0), "%local_id": (0, 0, 0)},
                                   push_constants=OUT_VSHARP + [0] * 28, memory=memory)
    return outputs, memory


class TbufferStoreFixture(unittest.TestCase):
    def test_words_match_the_native_measurement(self):
        outputs, memory = run(support.translate_fixture("cs_tbuffer_store", fssm=CS_FSSM))
        self.assertEqual(outputs, {})
        want = check.TBUFFER_STORE_WORDS
        got = [memory.get(4 * i, 0) for i in range(len(want))]
        for i, (g, w) in enumerate(zip(got, want)):
            with self.subTest(out=i, region=hex(0x200 + 16 * (i // 4))):
                self.assertEqual(f"{g:#010x}", f"{w:#010x}")
        # The regions themselves hold the same words, and nothing lands between out[] and the regions.
        self.assertEqual([memory.get(0x200 + 4 * i, 0) for i in range(len(want))], want)
        self.assertEqual(sorted(a for a in memory if len(want) * 4 <= a < 0x200), [])


class TbufferStoreTranslation(unittest.TestCase):
    def store(self, fmt_mod):
        return support.translate_lines("cs_tbuf_one", [
            "v_mov_b32_e32 v20, 1.0", "v_mov_b32_e32 v5, 0x200",
            f"tbuffer_store_format_x v20, v5, s[0:3], 0{fmt_mod} offen", "s_endpgm"], fssm=CS_FSSM)

    def test_format_outside_formats_is_unsupported(self):
        # BUF_FMT_INVALID (code 0) has no encoder: reported, never guessed. Every code 1-77 has one now (BUF_FMT_8_UINT
        # was the example until #1871 added the 8/16-bit UINT/SINT formats, BUF_FMT_8_USCALED and BUF_FMT_10_11_11_UNORM
        # until the SCALED and 10_11_11 formats).
        with self.assertRaises(rdna2spirv.Unsupported):
            self.store(" format:[BUF_FMT_INVALID]")

    def test_default_format_is_one_unorm_byte(self):
        # No modifier is GFX10's default, BUF_FMT_8_UNORM: 1.0 is 0xff in the low byte, the other three kept.
        spv = self.store("")
        memory = {0x200: 0x11223344}
        support.run_fragment(spv, inputs={"%wg_id": (0, 0, 0), "%local_id": (0, 0, 0)},
                             push_constants=OUT_VSHARP + [0] * 28, memory=memory)
        self.assertEqual(f"{memory[0x200]:#010x}", "0x112233ff")

    def test_data_vgprs_are_read(self):
        # read_before_write_vgprs: a store's vdata is a source, as for buffer_store.
        import shader_ir
        ins = shader_ir.parse_line("\ttbuffer_store_format_xy v[7:8], v5, s[4:7], 0 format:[BUF_FMT_32_32_FLOAT] offen")
        self.assertEqual(sorted(rdna2spirv.read_before_write_vgprs([ins])), [5, 7, 8])


if __name__ == "__main__":
    unittest.main()
