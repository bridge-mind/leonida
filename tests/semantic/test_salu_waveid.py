"""s_get_waveid_in_workgroup: the index of the wave within its workgroup.

Standard library only; no GPU, no LLVM, no game files.

RDNA2 ISA, SOPP S_GET_WAVEID_IN_WORKGROUP: D = the wave's ID within the workgroup. Thread i of a workgroup runs in
wave i / W and lane i % W (W = 32 or 64), the same rule the translator already uses for the hardware lane
(Gen.wave_lane), so D = LocalInvocationIndex / W. Only compute and mesh shaders have a workgroup; elsewhere the
instruction stays Unsupported.

Each case translates a 1-D compute listing (128 threads), runs every invocation with run_workgroup on the CPU and
has each one store its D at buffer word LocalInvocationIndex; the words are compared with i / W.
"""
import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
VSHARP = [0, 0, 0x1000, 0x31016FAC]
THREADS = 128


def fssm(x: int, y: int = 1, z: int = 1) -> bytes:
    """Synthetic compute header: 4 user SGPRs (the output V#), workgroup id in s4, thread ids x, y, z in v0, v1, v2
    (RSRC2 TIDIG_COMP_CNT = 2), group size x * y * z."""
    blob = bytearray(0x148)
    struct.pack_into("<4I", blob, 0x138, 0x88 | 2 << 11, x, y, z)
    return bytes(blob)


def run(body: list, wave64: bool, size=(THREADS, 1, 1), name: str = "cs_waveid", subgroup: int = 32) -> list:
    """Translate `body` (each thread then stores s5 at buffer word = its flat index) and run every invocation."""
    sx, sy, sz = size
    lines = (["s_mov_b64 exec, -1"] if wave64 else []) + list(body) + [
        # flat index = x + y * sx + z * sx * sy from the local id in v0 / v1 / v2, times 4 bytes
        f"v_mad_u32_u24 v3, v1, {sx}, v0", f"v_mad_u32_u24 v3, v2, {sx * sy}, v3", "v_lshlrev_b32 v3, 2, v3",
        "v_mov_b32 v4, s5", "buffer_store_dword v4, v3, s[0:3], 0 offen"]
    with tempfile.TemporaryDirectory() as tmp:
        for d in ("disasm", "fssm"):
            os.mkdir(os.path.join(tmp, d))
        with open(os.path.join(tmp, "disasm", name + ".s"), "w", encoding="utf-8") as f:
            f.write("; " + name + "\n" + "".join(f"\t{ln}\n" for ln in lines) + "\ts_endpgm\n")
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
            f.write(fssm(*size))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(os.path.join(tmp, "disasm", name + ".s"))
    n = sx * sy * sz

    def inputs(i):
        return {"%wg_id": (0, 0, 0), "%local_id": (i % sx, (i // sx) % sy, i // (sx * sy)), "%sg_lane": i % subgroup}
    mem = si.run_workgroup(spv, n, subgroup, inputs_for=inputs, push_constants=VSHARP + [0] * 28, memory={})
    return [mem.get(4 * i) for i in range(n)]


class WaveIdTests(unittest.TestCase):
    def test_wave32(self):
        self.assertEqual(run(["s_get_waveid_in_workgroup s5"], wave64=False), [i // 32 for i in range(THREADS)])

    def test_wave64(self):
        self.assertEqual(run(["s_get_waveid_in_workgroup s5"], wave64=True), [i // 64 for i in range(THREADS)])

    def test_three_dimensional_group(self):
        # 8 x 4 x 4 = 128 threads: the flat index is x + 8y + 32z, so z alone picks the wave in wave32.
        got = run(["s_get_waveid_in_workgroup s5"], wave64=False, size=(8, 4, 4))
        self.assertEqual(got, [i // 32 for i in range(128)])

    def test_other_stages_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(os.path.join(tmp, "disasm"))
            path = os.path.join(tmp, "disasm", "ps_waveid.s")
            with open(path, "w", encoding="utf-8") as f:
                f.write("; ps_waveid\n\ts_get_waveid_in_workgroup s0\n\tv_mov_b32 v0, s0\n"
                        "\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n")
            with open(os.path.join(tmp, "ps_waveid.dxbc"), "wb") as f:
                f.write(STUB_DXBC)
            with self.assertRaises(rdna2spirv.Unsupported):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ResourceWarning)
                    rdna2spirv.translate(path)


if __name__ == "__main__":
    unittest.main()
