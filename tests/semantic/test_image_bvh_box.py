"""image_bvh64_intersect_ray on box nodes: an empty box (min > max on any axis) is a miss.

One pixel shader per case runs the instruction on a hand-built box node and exports the four child words. The ray
starts at (0, 0, -1) and runs along +z. A box contributes its entry distance only when it is well formed and the
slab interval overlaps [0, extent); empty boxes (min > max) and boxes the ray misses give 0xFFFFFFFF, in distance
order, as tools/xdxr_bvh.py `_box_test` does.

box32 (node type 5) stores a child as six f32 at byte 16 + 24 i; box16 (type 4) stores the same six values as
f16 at byte 16 + 12 i. Both go through the same slab test in tools/rdna2spirv.py.

    python -m unittest discover -s tests/semantic -p "test_image_bvh_box.py" -v
"""
import struct
import unittest

from support import mrt0_words, translate_lines

INF_WORD = 0x7F800000
MISS = 0xFFFFFFFF
NODE = 0x1040                     # descriptor base 0x10 (256-byte units) = 0x1000; node id 0x08 | type -> +0x40


def f2u(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def half(x):
    return struct.unpack("<H", struct.pack("<e", x))[0]


def run(node_type, children, boxes):
    """Run the instruction on one box node; return its four u32 child words."""
    lines = []
    mov = lambda r, v: lines.append(f"v_mov_b32_e32 v{r}, 0x{v:x}")   # noqa: E731
    mov(1, 0x08 | node_type)
    mov(2, 0)
    mov(3, f2u(100.0))                                                 # ray extent
    for i, v in enumerate([f2u(0.0), f2u(0.0), f2u(-1.0), 0, 0, f2u(1.0), INF_WORD, INF_WORD, f2u(1.0)]):
        mov(4 + i, v)                                                  # origin, direction, inverse direction
    lines += ["s_mov_b32 s4, 0x10", "s_mov_b32 s5, 0", "s_mov_b32 s6, -1", "s_mov_b32 s7, 0",
              "image_bvh64_intersect_ray v[20:23], [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12], s[4:7]",
              "exp mrt0 v20, v21, v22, v23 done vm", "s_endpgm"]
    spv = translate_lines(f"ps_bvh_box{node_type}", lines)
    mem = {}
    for i, child in enumerate(children):
        mem[NODE + 4 * i] = child
    for i, box in enumerate(boxes):
        if node_type == 4:
            words = [half(v) for v in box]
            for c in range(3):
                mem[NODE + 16 + 12 * i + 4 * c] = words[2 * c] | words[2 * c + 1] << 16
        else:
            for c, v in enumerate(box):
                mem[NODE + 16 + 24 * i + 4 * c] = f2u(v)
    return list(mrt0_words(spv, memory=mem))


CHILDREN = [0x20, 0x28, 0x30, 0x38]


class EmptyBox(unittest.TestCase):
    def test_inverted_box_is_culled_not_traversed(self):
        # child 1 is empty on x and y but its z slab is 0.5..2: without the check it is the second-nearest hit.
        boxes = [(-1, -1, -1, 1, 1, 1), (1, 1, 0.5, -1, -1, 2), (2, 2, 2, 3, 3, 3), (2, 2, 0, 3, 3, 1)]
        for node_type in (4, 5):
            with self.subTest(node_type=node_type):
                self.assertEqual(run(node_type, CHILDREN, boxes), [0x20, MISS, MISS, MISS])

    def test_inverted_on_one_axis_is_culled(self):
        # child 0 hits on x and z; only y is inverted (min 1 > max -1).
        boxes = [(-1, 1, 0.5, 1, -1, 2), (2, 2, 2, 3, 3, 3), (2, 2, 2, 3, 3, 3), (2, 2, 2, 3, 3, 3)]
        for node_type in (4, 5):
            with self.subTest(node_type=node_type):
                self.assertEqual(run(node_type, CHILDREN, boxes), [MISS, MISS, MISS, MISS])

    def test_zero_thickness_box_is_not_empty(self):
        # min == max is a degenerate but well-formed box; the CPU reference keeps it (lo > hi only).
        boxes = [(-1, -1, 0, 1, 1, 0), (2, 2, 2, 3, 3, 3), (2, 2, 2, 3, 3, 3), (2, 2, 2, 3, 3, 3)]
        for node_type in (4, 5):
            with self.subTest(node_type=node_type):
                self.assertEqual(run(node_type, CHILDREN, boxes), [0x20, MISS, MISS, MISS])

    def test_empty_box_does_not_hide_a_later_hit(self):
        # child 0's box is empty; child 1's overlaps, child 2's starts right after the extent, child 3's is empty.
        boxes = [(1, 1, 4, -1, -1, 5), (-1, -1, 2, 1, 1, 3), (2, 2, 2, 3, 3, 3), (0, 1, 0, 1, 0, 1)]
        # The four results stay in child-slot order; a miss at slot 0 is not compacted away.
        for node_type in (4, 5):
            with self.subTest(node_type=node_type):
                self.assertEqual(run(node_type, CHILDREN, boxes), [MISS, 0x28, MISS, MISS])


if __name__ == "__main__":
    unittest.main()
