"""Unit tests for rdna2spirv.vertex_input_locations (standard library only, no game data).

    python -I tests/test_vertex_input_locations.py

SHADER-644: a packed vertex ISG1 signature declares xy and zw of one register as two elements. Each register must
become one Vulkan vertex input, or the SPIR-V declares %vin{reg} twice and fails spirv-val.
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import rdna2spirv  # noqa: E402

FLOAT, UINT = 3, 1


class VertexInputLocationsTest(unittest.TestCase):
    def test_distinct_registers_keep_order(self):
        sig = [(2, 0xF, 0, FLOAT), (0, 0x7, 0, FLOAT), (1, 0x3, 0, UINT)]
        self.assertEqual(rdna2spirv.vertex_input_locations(sig), [(2, FLOAT), (0, FLOAT), (1, UINT)])

    def test_packed_register_is_declared_once(self):
        sig = [(0, 0x3, 0, FLOAT), (0, 0xC, 0, FLOAT), (1, 0xF, 0, FLOAT)]
        self.assertEqual(rdna2spirv.vertex_input_locations(sig), [(0, FLOAT), (1, FLOAT)])

    def test_three_elements_in_one_register(self):
        sig = [(4, 0x1, 0, UINT), (4, 0x2, 0, UINT), (4, 0xC, 0, UINT)]
        self.assertEqual(rdna2spirv.vertex_input_locations(sig), [(4, UINT)])

    def test_mixed_component_types_in_one_register_are_rejected(self):
        sig = [(0, 0x3, 0, FLOAT), (0, 0xC, 0, UINT)]
        with self.assertRaises(rdna2spirv.Unsupported):
            rdna2spirv.vertex_input_locations(sig)

    def test_empty_signature(self):
        self.assertEqual(rdna2spirv.vertex_input_locations([]), [])


if __name__ == "__main__":
    unittest.main()
