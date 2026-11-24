"""A pixel shader's colour exports are packed: exp mrtK writes the K-th declared SV_Target, not render target K.

A pixel shader that declares SV_Target0 and SV_Target2 exports mrt0 and mrt1 (SHADER-210: all 221 such pixel shaders
of the 1.0.2.1 corpus, docs/design.md "Pixel colour exports"). The translator used to decorate mrt1 with Location 1,
so the second value went to render target 1. It now maps export K to the K-th SV_Target of the shader's OSG1.
Hand-built OSG1 containers and listings, no game data.

    python -I tests/semantic/test_sparse_sv_target.py -v
"""
import os
import re
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402
from support import rdna2spirv  # noqa: E402

D3D_NAME_TARGET = 64


def osg1_dxbc(targets):
    """Synthetic DXBC container with one OSG1 chunk declaring SV_Target<n> (float4, register n) for each n."""
    payload = bytearray(struct.pack("<II", len(targets), 8))
    for n in targets:
        elem = bytearray(32)
        struct.pack_into("<6I", elem, 0, 0, 0, n, D3D_NAME_TARGET, 3, n)   # stream, name, index, sysv, float, reg
        elem[24], elem[25] = 0xF, 0xF
        payload += elem
    part = b"OSG1" + struct.pack("<I", len(payload)) + bytes(payload)
    part += bytes((-len(part)) % 4)
    header = b"DXBC" + bytes(16) + struct.pack("<III", 1, 0, 1)
    return header + struct.pack("<I", len(header) + 8) + bytes(4) + part


# Two colour exports: 1.0 then 2.0 in every channel.
TWO_EXPORTS = [
    "v_mov_b32_e32 v0, 1.0",
    "v_mov_b32_e32 v1, 2.0",
    "exp mrt0 v0, v0, v0, v0",
    "exp mrt1 v1, v1, v1, v1 done vm",
    "s_endpgm",
]


def locations(spvasm):
    """{output variable: Location} of the module's mrt outputs."""
    return {m.group(1): int(m.group(2)) for m in re.finditer(r"OpDecorate (%out_mrt\d+) Location (\d+)", spvasm)}


class SparseSvTargetTest(unittest.TestCase):
    def translate(self, targets, lines=TWO_EXPORTS):
        dxbc = osg1_dxbc(targets) if targets is not None else support.STUB_DXBC
        return support.translate_lines("ps_sparse_sv_target", lines, dxbc=dxbc)

    def test_second_export_goes_to_the_second_declared_target(self):
        spvasm = self.translate([0, 2])
        self.assertEqual(locations(spvasm), {"%out_mrt0": 0, "%out_mrt2": 2})
        out = support.run_fragment(spvasm)
        self.assertEqual(sorted(out), ["%out_mrt0", "%out_mrt2"])
        self.assertEqual([struct.unpack("<f", struct.pack("<I", w))[0] for w in out["%out_mrt2"]], [2.0] * 4)

    def test_other_gaps(self):
        self.assertEqual(locations(self.translate([1, 3])), {"%out_mrt1": 1, "%out_mrt3": 3})
        self.assertEqual(locations(self.translate([0, 7])), {"%out_mrt0": 0, "%out_mrt7": 7})

    def test_consecutive_targets_are_unchanged(self):
        self.assertEqual(locations(self.translate([0, 1])), {"%out_mrt0": 0, "%out_mrt1": 1})

    def test_without_an_output_signature_the_index_is_the_target(self):
        # Hand-written listings and run_gpu_tests.py fixtures have no OSG1: mrtK stays render target K.
        self.assertEqual(locations(self.translate(None)), {"%out_mrt0": 0, "%out_mrt1": 1})

    def test_an_export_past_the_declared_targets_is_refused(self):
        with self.assertRaises(rdna2spirv.Unsupported) as cm:
            self.translate([2])
        self.assertIn("mrt1 past the 1 declared SV_Target", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
