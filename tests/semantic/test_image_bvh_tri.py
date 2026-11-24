"""image_bvh64_intersect_ray on triangle nodes: the vertex slots each node type reads and the barycentric remap byte.

One pixel shader per case runs the instruction on a hand-built 64-byte triangle node and exports the four result
words. The node holds a unit quad in slots 0..3, v0 (0,0,0) v1 (1,0,0) v2 (0,1,0) v3 (1,1,0), and a fifth vertex in
slot 4 (words 12..14). The ray starts at (x, y, -1) and runs along +z, so t = 1 and k = |d.z|^3 = 1. The result is
the hardware's (t*det, det, b_i*det, b_j*det), so dividing by .y gives (t, -, b_i, b_j), and on a miss
(copysign(inf, det), det, b_i*det, b_j*det) with the weights of the triangle's plane (docs/design.md, Ray tracing row;
measured on RDNA2, tests/semantic/test_image_bvh_tri_hw.py). det is twice the projected area with the sign of the
winding: -1 for TRI0..TRI2, wound like v0 v1 v2, and +0.5 for TRI3, wound the other way. Until the measurement the
expected words here were the normalised (t/|det|, sign(det), b_i/|det|, b_j/|det|) and a miss (inf, 1, 0, 0); the
slots, the remap selectors and the quotients checked are the same.

Expected rules (docs/design.md, Ray tracing row; docs/bvh.md 7a): TRI0 reads slots (0,1,2), TRI1 (1,3,2), TRI2 (2,3,4)
and TRI3 (2,4,0). Byte `type` of the word at +60 holds two selectors: bits [1:0] for the first weight and [3:2] for
the second, where 0/1/2 name the triangle's vertex 0/1/2. The weights below are written by hand and checked against
a plain-Python solve of the 2x2 system.

The descriptor dword0 is 0x10: the base in 256-byte units (0x1000). The GPU measured this nonzero base against the
CPU walker on the real-BVH first call, so this case also pins the `<< 8`, not a masked byte address.

    python -m unittest tests/semantic/test_image_bvh_tri.py -v
"""
import struct
import unittest

from support import mrt0_words, translate_lines

INF = float("inf")
SLOTS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0.5, 2, 0)]
NODE = 0x1040                     # descriptor base 0x10 (256-byte units) = 0x1000; node id 0x08 | type -> +0x40


def f2u(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def u2f(w):
    return struct.unpack("<f", struct.pack("<I", w))[0]


def run(node_type, tri_id, x, y, slots=SLOTS):
    lines = []
    mov = lambda r, v: lines.append(f"v_mov_b32_e32 v{r}, 0x{v:x}")   # noqa: E731
    mov(1, 0x08 | node_type)
    mov(2, 0)
    mov(3, f2u(100.0))                                                 # ray extent
    for i, v in enumerate([f2u(x), f2u(y), f2u(-1.0), 0, 0, f2u(1.0), 0x7F800000, 0x7F800000, f2u(1.0)]):
        mov(4 + i, v)                                                  # origin, direction, inverse direction
    lines += ["s_mov_b32 s4, 0x10", "s_mov_b32 s5, 0", "s_mov_b32 s6, -1", "s_mov_b32 s7, 0",
              "image_bvh64_intersect_ray v[20:23], [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12], s[4:7]",
              "exp mrt0 v20, v21, v22, v23 done vm", "s_endpgm"]
    spv = translate_lines(f"ps_bvh_tri{node_type}", lines)
    mem = {}
    for s, v in enumerate(slots):
        for c in range(3):
            mem[NODE + 12 * s + 4 * c] = f2u(v[c])
    mem[NODE + 60] = tri_id
    return [u2f(w) for w in mrt0_words(spv, memory=mem)]


def weights(tri, x, y):
    """Barycentric weights of (x, y) in a triangle in the z = 0 plane (plain Python, Cramer's rule)."""
    (ax, ay, _), (bx, by, _), (cx, cy, _) = tri
    det = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
    b1 = ((x - ax) * (cy - ay) - (cx - ax) * (y - ay)) / det
    b2 = ((bx - ax) * (y - ay) - (x - ax) * (by - ay)) / det
    return (1 - b1 - b2, b1, b2)


class TriangleSlots(unittest.TestCase):
    # (type, point inside that type's triangle and outside TRI0's, hand-written weights of vertices 0, 1, 2, det)
    CASES = [
        (0, (0.2, 0.3), (0.5, 0.2, 0.3), -1.0),          # (v0, v1, v2)
        (1, (0.75, 0.6), (0.4, 0.35, 0.25), -1.0),       # (v1, v3, v2)
        (2, (0.6, 1.3), (0.25, 0.45, 0.3), -1.0),        # (v2, v3, v4)
        (3, (0.2, 1.0), (0.2, 0.4, 0.4), 0.5),           # (v2, v4, v0): wound the other way, half the area
    ]
    ORDER = {0: (0, 1, 2), 1: (1, 3, 2), 2: (2, 3, 4), 3: (2, 4, 0)}

    def test_reference_weights(self):
        for t, (x, y), want, _ in self.CASES:
            got = weights([SLOTS[s] for s in self.ORDER[t]], x, y)
            for g, w in zip(got, want):
                self.assertAlmostEqual(g, w, places=12, msg=(t, got))

    def test_each_type_reads_its_slots(self):
        for t, (x, y), want, det in self.CASES:
            with self.subTest(type=t):
                r = run(t, 0x09 << (8 * t), x, y)
                self.assertEqual(r[1], det)                           # k*det, k = 1
                self.assertAlmostEqual(r[0] / r[1], 1.0, places=6)    # t, as the game's code divides by .y
                self.assertAlmostEqual(r[2] / r[1], want[1], places=6)
                self.assertAlmostEqual(r[3] / r[1], want[2], places=6)

    def test_tri1_misses_where_only_tri0_is(self):
        # a miss keeps det and the weights' numerators, here of a point outside the triangle (one weight < 0)
        for t, (x, y) in ((1, (0.2, 0.3)), (0, (0.75, 0.6))):
            with self.subTest(type=t):
                r = run(t, 0x09 << (8 * t), x, y)
                w = weights([SLOTS[s] for s in self.ORDER[t]], x, y)
                self.assertTrue(min(w) < 0)
                self.assertEqual(r[:2], [-INF, -1.0])                 # copysign(inf, det), det
                self.assertAlmostEqual(-r[2], w[1], places=6)
                self.assertAlmostEqual(-r[3], w[2], places=6)


class RemapByte(unittest.TestCase):
    W = (0.5, 0.2, 0.3)                          # TRI0 at (0.2, 0.3)

    def test_every_selector_pair(self):
        for i in range(3):
            for j in range(3):
                with self.subTest(i=i, j=j):
                    r = run(0, i | j << 2, 0.2, 0.3)
                    self.assertAlmostEqual(-r[2], self.W[i], places=6)
                    self.assertAlmostEqual(-r[3], self.W[j], places=6)

    def test_byte_of_the_node_type(self):
        # TRI1 reads byte 1; byte 0 (here 0x06, swapped) belongs to TRI0 and must not affect it
        r = run(1, 0x06 | 0x09 << 8, 0.75, 0.6)
        self.assertAlmostEqual(-r[2], 0.35, places=6)
        self.assertAlmostEqual(-r[3], 0.25, places=6)
        r = run(1, 0x09 | 0x06 << 8, 0.75, 0.6)
        self.assertAlmostEqual(-r[2], 0.25, places=6)
        self.assertAlmostEqual(-r[3], 0.35, places=6)

    def test_selector_3_reads_vertex_0(self):
        r = run(0, 0x3 | 0x3 << 2, 0.2, 0.3)
        self.assertAlmostEqual(-r[2], 0.5, places=6)
        self.assertAlmostEqual(-r[3], 0.5, places=6)

    def test_zero_word_is_not_the_identity(self):
        # a node whose builder left +60 at zero returns vertex 0's weight twice, as the hardware would
        r = run(0, 0, 0.2, 0.3)
        self.assertAlmostEqual(-r[2], 0.5, places=6)
        self.assertAlmostEqual(-r[3], 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
