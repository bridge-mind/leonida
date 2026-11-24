"""Synthetic-data tests for tools/anim_glb.py (skinned part + skeleton + clips -> animated GLB).

    python -I tests/test_anim_glb.py

Every byte is built here: a synthetic skeleton drawable, a synthetic skinned part drawable (system and graphics
pages) and a synthetic anim.rpf from test_anim_clips' encoders. No game file, value or name is used.
"""
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "anim_glb.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import anim_clips  # noqa: E402
import anim_glb  # noqa: E402
from anim_index import joaat32  # noqa: E402
import test_anim_clips as clips_fixture  # noqa: E402

SYSTEM, GRAPHICS = 0x50000000, 0x60000000
# Three joints: the clip fixture animates skel_a and skel_b (rotation) and skel_c (translation).
NAMES = ("SKEL_A", "SKEL_B", "SKEL_C")
PARENTS = (None, 0, 1)
ROTATIONS = ((0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)), (0.0, 0.0, 0.0, 1.0), (0.5, 0.5, 0.5, 0.5))
TRANSLATIONS = ((0.0, 0.0, 1.0), (0.5, 0.0, 0.0), (0.0, 0.25, 0.0))


class Page:
    """System or graphics page builder; `base` is 0x50000000 or 0x60000000."""

    def __init__(self, base=SYSTEM, head=0x100):
        self.b = bytearray(head)
        self.base = base

    def alloc(self, data, align=16):
        self.b += bytes((-len(self.b)) % align)
        at = len(self.b)
        self.b += data
        return at

    def ptr(self, at, target, base=None):
        struct.pack_into("<Q", self.b, at, (self.base if base is None else base) + target)


def drawable_header(p, name):
    """Fields resource_kind reads for a drawable, and the page directory (1 system page, `graphics` pages)."""
    p.ptr(16, p.alloc(bytes(64)))
    struct.pack_into("<f", p.b, 44, 1.0)
    struct.pack_into("<3f", p.b, 48, -2.0, -2.0, -2.0)
    struct.pack_into("<3f", p.b, 64, 2.0, 2.0, 2.0)
    p.ptr(168, p.alloc(name.encode() + b"\0"))
    directory = p.alloc(bytes(16))
    p.ptr(8, directory)
    return directory


def world_transforms(rotations=ROTATIONS, translations=TRANSLATIONS):
    out = []
    for i, parent in enumerate(PARENTS):
        q, t = rotations[i], translations[i]
        if parent is None:
            out.append((q, t))
        else:
            pq, pt = out[parent]
            out.append((anim_glb.quat_mul(pq, q), tuple(a + b for a, b in zip(pt, anim_glb.quat_rotate(pq, t)))))
    return out


def rotation_rows(q):
    """Column-vector rotation matrix of q, row by row."""
    x, y, z, w = q
    return ((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)))


def inverse_bind(q, t):
    """Row-vector (D3D) inverse of [R | t], stored row by row: rows of R, then -t R."""
    r = rotation_rows(q)
    last = [-sum(t[j] * r[j][k] for j in range(3)) for k in range(3)]
    return [r[0][0], r[0][1], r[0][2], 0.0, r[1][0], r[1][1], r[1][2], 0.0,
            r[2][0], r[2][1], r[2][2], 0.0, last[0], last[1], last[2], 1.0]


# Both bind matrices have rows 0 and 1 equal (zero determinant) and the same recovered origin (0, 0, 1),
# so only the determinant distinguishes them from bone 0's valid bind. The first has a zero middle
# cofactor; the second keeps it non-zero so a sign flip in that term is caught.
SINGULAR_ZERO_COFACTOR = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 1.0]
SINGULAR_LIVE_COFACTOR = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 1.0]


def skeleton_resource(shift_bind=0.0, singular_bind=None):
    p = Page()
    directory = drawable_header(p, "synthetic_skeleton")
    p.b[directory + 8], p.b[directory + 9] = 1, 0
    p.ptr(80, p.alloc(bytes(16)))          # an empty LOD keeps the drawable signature
    n = len(NAMES)
    header = p.alloc(bytes(0x70))
    records = p.alloc(bytes(0x50 * n))
    world = world_transforms()
    binds = [inverse_bind(q, (t[0] + shift_bind, t[1], t[2])) for q, t in world]
    if singular_bind is not None:
        binds[0] = singular_bind
    matrices = p.alloc(b"".join(struct.pack("<16f", *m) for m in binds))
    parents = p.alloc(b"".join(struct.pack("<H", 0xFFFF if x is None else x) for x in PARENTS))
    for i, name in enumerate(NAMES):
        r = records + 0x50 * i
        struct.pack_into("<4f", p.b, r, *ROTATIONS[i])
        struct.pack_into("<3f", p.b, r + 0x10, *TRANSLATIONS[i])
        struct.pack_into("<I", p.b, r + 0x1C, joaat32(name.lower()))
        struct.pack_into("<3f", p.b, r + 0x20, 1.0, 1.0, 1.0)
        p.ptr(r + 0x30, p.alloc(name.encode() + b"\0"))
        struct.pack_into("<HH", p.b, r + 0x46, 0xFFFF if PARENTS[i] is None else PARENTS[i], i)
    p.ptr(0x18, header)
    p.ptr(header + 0x28, records)
    p.ptr(header + 0x30, matrices)
    p.ptr(header + 0x40, parents)
    struct.pack_into("<HH", p.b, header + 0x6C, n, n)
    return bytes(p.b)


POSITIONS = [(0.0, 0.0, 1.0), (0.5, 0.0, 1.0), (0.5, 0.25, 1.0), (0.0, 0.25, 1.0)]
TRIANGLES = [(0, 1, 2), (0, 2, 3)]


def part_resource(palette, layout="u8x4", skin=None):
    """One model, one geometry, float positions, one cluster. `skin` per vertex: [(model bone, weight byte)]."""
    p, g = Page(), Page(GRAPHICS, head=0)
    directory = drawable_header(p, "synthetic_part.#dr")
    p.b[directory + 8], p.b[directory + 9] = 1, 1
    n = len(POSITIONS)
    lod, model = p.alloc(bytes(16)), p.alloc(bytes(0x40))
    geometry, vb, fvf = p.alloc(bytes(160)), p.alloc(bytes(80)), p.alloc(bytes(0x140))
    desc, ib = p.alloc(bytes(16)), p.alloc(bytes(64))
    p.ptr(80, lod)
    p.ptr(lod, p.alloc(struct.pack("<Q", SYSTEM + model)))
    struct.pack_into("<HH", p.b, lod + 8, 1, 1)
    p.ptr(model + 8, p.alloc(struct.pack("<Q", SYSTEM + geometry)))
    struct.pack_into("<HHH", p.b, model + 16, 1, 1, len(palette))
    struct.pack_into("<H", p.b, model + 54, 1)
    p.ptr(model + 0x28, p.alloc(b"".join(struct.pack("<I", h) for h in palette)))
    p.ptr(geometry + 8, vb)
    p.ptr(geometry + 64, desc)
    p.ptr(desc, ib)
    p.ptr(vb + 64, fvf)
    # vertex buffer: descriptor words, then the position stream at 128 and the skin streams after it
    vbuf = bytearray(128)
    struct.pack_into("<I", vbuf, 0, 0x63)
    slots = {0: (12, 6, b"".join(struct.pack("<3f", *v) for v in POSITIONS))}
    skin = skin or [[(0, 255)] for _ in POSITIONS]
    if layout in ("u8x4", "packed"):
        sets = 2 if max(len(pairs) for pairs in skin) > 4 else 1      # 8 influences use slots 16/17 and 20/21
        for k in range(sets):
            weights, bones = bytearray(), bytearray()
            for pairs in skin:
                pairs = (pairs + [(0, 0)] * 8)[4 * k:4 * k + 4]
                weights += bytes(w for _b, w in pairs)
                bones += bytes(b for b, _w in pairs) if layout == "u8x4" else \
                    b"".join(struct.pack("<I", b | 0xABC000) for b, _w in pairs)   # bits 12.. are not the index
            slots[16 + k] = (4, 28, bytes(weights))
            slots[20 + k] = (4, 30, bytes(bones)) if layout == "u8x4" else (16, 3, bytes(bones))
    elif layout == "rigid-u8":
        slots[20] = (1, 62, bytes(pairs[0][0] for pairs in skin))
    for semantic, (size, fmt, data) in slots.items():
        at = len(vbuf)
        vbuf += data + bytes((-len(data)) % 16)
        struct.pack_into("<I", p.b, fvf + 4 * semantic, at)
        p.b[fvf + 0xD0 + semantic], p.b[fvf + 0x104 + semantic] = size, fmt
    struct.pack_into("<I", p.b, fvf + 0x138, (n << 10) | 1)
    vat = g.alloc(bytes(vbuf))
    struct.pack_into("<IH", p.b, vb + 8, len(vbuf), 1)
    p.ptr(vb + 24, vat, GRAPHICS)
    # index buffer: one cluster with n vertex references and two triangles, 16-bit references
    ibuf = struct.pack("<III", 0x80000000 | 1, len(TRIANGLES), n)
    ibuf += struct.pack("<IIII", (n - 1) | ((len(TRIANGLES) - 1) << 7), 0, 0, 0)
    ibuf += b"".join(struct.pack("<I", a | b << 7 | c << 14) for a, b, c in TRIANGLES)
    ibuf += b"".join(struct.pack("<H", i) for i in range(n))
    ibuf += bytes((-len(ibuf)) % 16)
    iat = g.alloc(ibuf)
    struct.pack_into("<IH", p.b, ib + 8, len(ibuf) // 4, 4)
    p.ptr(ib + 24, iat, GRAPHICS)
    return bytes(p.b), bytes(g.b)


def write_archive(path, resources):
    """Resources are lists of pages; each page is a zlib stream at a 4096-aligned offset + 16. -> offsets."""
    out, offsets = bytearray(0x1000), []
    for pages in resources:
        offsets.append(len(out) + 16)
        for k, page in enumerate(pages):
            if k == 0:
                out += bytes(16)
            out += zlib.compress(page)
            out += bytes((-len(out)) % 16)
        out += bytes((-len(out)) % 0x1000)
    with open(path, "wb") as f:
        f.write(out)
    return offsets


def read_glb(path):
    with open(path, "rb") as f:
        data = f.read()
    magic, version, total = struct.unpack_from("<4sII", data, 0)
    assert magic == b"glTF" and version == 2 and total == len(data)
    jlen, _ = struct.unpack_from("<II", data, 12)
    gltf = json.loads(data[20:20 + jlen])
    blen, _ = struct.unpack_from("<II", data, 20 + jlen)
    return gltf, data[28 + jlen:28 + jlen + blen]


def accessor_values(gltf, binary, index):
    acc = gltf["accessors"][index]
    view = gltf["bufferViews"][acc["bufferView"]]
    width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[acc["type"]]
    code = {5126: "f", 5123: "H", 5125: "I"}[acc["componentType"]]
    flat = struct.unpack_from(f"<{acc['count'] * width}{code}", binary, view["byteOffset"])
    return [flat[i:i + width] for i in range(0, len(flat), width)]


class SkeletonAndPartTest(unittest.TestCase):
    def test_skeleton_reads_default_transforms_and_matches_inverse_binds(self):
        sys_page = skeleton_resource()
        joints, worst = anim_glb.read_skeleton(sys_page, anim_glb.drawable_at(sys_page, None))
        self.assertEqual([j["name"] for j in joints], list(NAMES))
        self.assertEqual([j["parent"] for j in joints], list(PARENTS))
        self.assertEqual([j["hash"] for j in joints], [joaat32(n.lower()) for n in NAMES])
        self.assertLess(worst, 1e-6)
        self.assertAlmostEqual(joints[2]["translation"][1], 0.25)

    def test_a_skeleton_whose_binds_disagree_reports_the_error(self):
        sys_page = skeleton_resource(shift_bind=0.3)
        _joints, worst = anim_glb.read_skeleton(sys_page, anim_glb.drawable_at(sys_page, None))
        self.assertGreater(worst, 0.29)

    def test_a_singular_inverse_bind_is_refused_even_when_its_origin_matches(self):
        for matrix in (SINGULAR_ZERO_COFACTOR, SINGULAR_LIVE_COFACTOR):
            with self.subTest(matrix=matrix):
                sys_page = skeleton_resource(singular_bind=matrix)
                # Before the fix the origin check passed and the degenerate matrix was exported as a joint.
                with self.assertRaisesRegex(anim_glb.ExportError, "inverse bind matrix 0"):
                    anim_glb.read_skeleton(sys_page, anim_glb.drawable_at(sys_page, None))

    def test_part_binds_model_bones_to_joints_by_hash(self):
        joints, _ = anim_glb.read_skeleton(skeleton_resource(), 0)
        # model bone 0 is SKEL_C and model bone 1 is SKEL_A: the model's list is in its own order
        palette = [joaat32("skel_c"), joaat32("skel_a")]
        skin = [[(0, 200), (1, 55)], [(1, 255)], [(0, 128), (1, 127)], [(1, 255), (0, 0)]]
        sys_page, gfx = part_resource(palette, "u8x4", skin)
        meshes = anim_glb.bind_part(anim_glb.read_part(sys_page, gfx, anim_glb.drawable_at(sys_page, None)), joints)
        mesh = meshes[0]
        self.assertEqual(mesh["positions"], [tuple(map(float, v)) for v in POSITIONS])
        self.assertEqual(mesh["indices"], [i for t in TRIANGLES for i in t])
        self.assertEqual(mesh["joints"][0][:2], [2, 0])
        self.assertEqual(mesh["joints"][1][:1], [0])
        self.assertAlmostEqual(mesh["weights"][0][0], 200 / 255)
        for row in mesh["weights"]:
            self.assertAlmostEqual(sum(row), 1.0, places=6)
        self.assertEqual(mesh["weights"][3], [1.0, 0.0, 0.0, 0.0])   # the zero-weight component is dropped

    def test_eight_influences_make_two_sets_or_the_largest_four(self):
        joints, _ = anim_glb.read_skeleton(skeleton_resource(), 0)
        palette = [joaat32("skel_a"), joaat32("skel_b"), joaat32("skel_c")]
        skin = [[(0, 60), (1, 50), (2, 40), (0, 35), (1, 30), (2, 25), (0, 10), (1, 5)]] * 4
        sys_page, gfx = part_resource(palette, "u8x4", skin)
        mesh = anim_glb.bind_part(anim_glb.read_part(sys_page, gfx, 0), joints)[0]
        self.assertEqual((mesh["sets"], len(mesh["weights"][0])), (2, 8))
        self.assertAlmostEqual(sum(mesh["weights"][0]), 1.0, places=6)
        mesh = anim_glb.bind_part(anim_glb.read_part(sys_page, gfx, 0), joints, max_influences=4)[0]
        self.assertEqual((mesh["sets"], len(mesh["weights"][0])), (1, 4))
        self.assertAlmostEqual(mesh["weights"][0][0], 60 / 185)
        self.assertAlmostEqual(sum(mesh["weights"][0]), 1.0, places=6)

    def test_packed_indices_use_bits_0_to_11(self):
        joints, _ = anim_glb.read_skeleton(skeleton_resource(), 0)
        palette = [joaat32("skel_b"), joaat32("skel_c")]
        sys_page, gfx = part_resource(palette, "packed", [[(1, 255)]] * 4)
        mesh = anim_glb.bind_part(anim_glb.read_part(sys_page, gfx, 0), joints)[0]
        self.assertEqual(mesh["layout"], "packed")
        self.assertEqual({row[0] for row in mesh["joints"]}, {2})

    def test_rigid_indices(self):
        joints, _ = anim_glb.read_skeleton(skeleton_resource(), 0)
        sys_page, gfx = part_resource([joaat32("skel_b")], "rigid-u8", [[(0, 255)]] * 4)
        mesh = anim_glb.bind_part(anim_glb.read_part(sys_page, gfx, 0), joints)[0]
        self.assertEqual((mesh["layout"], mesh["joints"][0][0], mesh["weights"][0][0]), ("rigid-u8", 1, 1.0))

    def test_a_weighted_bone_missing_from_the_skeleton_is_refused(self):
        joints, _ = anim_glb.read_skeleton(skeleton_resource(), 0)
        sys_page, gfx = part_resource([joaat32("skel_a"), 0x12345678], "u8x4", [[(1, 255)]] * 4)
        with self.assertRaisesRegex(anim_glb.ExportError, "0x12345678"):
            anim_glb.bind_part(anim_glb.read_part(sys_page, gfx, 0), joints)

    def test_an_unweighted_bone_missing_from_the_skeleton_is_fine(self):
        joints, _ = anim_glb.read_skeleton(skeleton_resource(), 0)
        sys_page, gfx = part_resource([joaat32("skel_a"), 0x12345678], "u8x4", [[(0, 255), (1, 0)]] * 4)
        mesh = anim_glb.bind_part(anim_glb.read_part(sys_page, gfx, 0), joints)[0]
        self.assertEqual({row[0] for row in mesh["joints"]}, {0})


class ClipChannelTest(unittest.TestCase):
    def setUp(self):
        self.real = anim_clips.decode_animation
        frames = [(0.0, 0.0, 0.0, 1.0)] * 5
        anim_clips.decode_animation = lambda _s, anim: ({0: frames, 1: [(1.0, 2.0, 3.0)] * 5, 2: [(9.0,)] * 5,
                                                         3: [(0.0, 0.0, 0.0, 1.0)] * 5}, 5)
        self.anim = {"offset": 0x100, "duration": 4 / 30, "tracks": [
            (joaat32("skel_a"), joaat32("bonerotation"), 0), (0, joaat32("bonetranslation"), 2),
            (0x77, joaat32("genericcontrol"), 4), (0x999, joaat32("bonerotation"), 0)]}
        self.dictionary = {"packs": [{"animations": [self.anim]}]}

    def tearDown(self):
        anim_clips.decode_animation = self.real

    def channels(self, start, end, rate, root_motion=True):
        clip = {"animations": [{"animation": 0x100, "start": start, "end": end, "rate": rate}]}
        tally = __import__("collections").Counter()
        out = anim_glb.clip_channels(None, self.dictionary, clip, {joaat32("skel_a"): 0}, root_motion, tally)
        return out, tally

    def test_window_and_rate_map_animation_time_to_clip_time(self):
        out, tally = self.channels(1 / 30, 3 / 30, 2.0)
        times = [t for t, _v in out[(0, "rotation")]]
        self.assertEqual(len(times), 3)
        for got, want in zip(times, (0.0, 1 / 60, 2 / 60)):
            self.assertAlmostEqual(got, want, places=6)
        self.assertEqual((tally["tracks_exported"], tally["tracks_other_type"], tally["tracks_bone_not_in_skeleton"]),
                         (2, 1, 1))

    def test_a_frame_just_before_the_window_starts_at_zero(self):
        out, _ = self.channels(1 / 30 + 5e-5, 3 / 30, 1.0)
        times = [t for t, _v in out[(0, "rotation")]]
        self.assertEqual(times[0], 0.0)
        self.assertTrue(all(b > a for a, b in zip(times, times[1:])))

    def test_bone_hash_zero_drives_the_mover(self):
        out, _ = self.channels(0.0, 4 / 30, 1.0)
        self.assertIn(("mover", "translation"), out)
        out, tally = self.channels(0.0, 4 / 30, 1.0, root_motion=False)
        self.assertNotIn(("mover", "translation"), out)
        self.assertEqual(tally["tracks_mover_skipped"], 1)

    def test_single_frame_pose_window_keeps_only_that_frame(self):
        anim_clips.decode_animation = lambda _s, _anim: (
            {0: [(float(k), 0.0, 0.0, 1.0) for k in range(5)],
             1: [(float(k * 10), 0.0, 0.0) for k in range(5)]},
            5,
        )
        out, _ = self.channels(2 / 30, 2 / 30, 1.0)
        self.assertEqual(out[("mover", "translation")], [(0.0, (20.0, 0.0, 0.0))])
        # Off-grid start == end windows pick the nearest frame instead of falling back to frame 0.
        out_mid, _ = self.channels(0.075, 0.075, 1.0)
        self.assertEqual(out_mid[("mover", "translation")], [(0.0, (20.0, 0.0, 0.0))])
        out_hi, _ = self.channels(0.095, 0.095, 1.0)
        self.assertEqual(out_hi[("mover", "translation")], [(0.0, (30.0, 0.0, 0.0))])
        # Even when start == end sits slightly before a frame within 1e-4, timestamp stays 0.0.
        self.anim["duration"] = 4 / 32
        t = 2 / 32 - 2**-14
        out_tol, _ = self.channels(t, t, 1 / 1024)
        self.assertEqual(out_tol[("mover", "translation")], [(0.0, (20.0, 0.0, 0.0))])

    def test_select_clips_handles_non_hex_0x_names_and_bad_selectors(self):
        d = {
            "clips": [
                {"kind": "clip", "name": "other.clip", "hash": 0x11111111},
                {"kind": "clip", "name": "0x_intro.clip", "hash": 0x22222222},
            ]
        }
        self.assertEqual([c["hash"] for c in anim_glb.select_clips(d, ["0x_intro"])], [0x22222222])
        with self.assertRaisesRegex(anim_glb.ExportError, "0xbad"):
            anim_glb.select_clips(d, ["0xbad"])


class SamplerTimeTest(unittest.TestCase):
    def test_fractional_time_bounds_match_the_binary_accessor(self) -> None:
        skeleton = skeleton_resource()
        joints, _ = anim_glb.read_skeleton(skeleton, anim_glb.drawable_at(skeleton, None))
        keys = [(1 / 30, (0.0, 0.0, 0.0)), (2 / 30, (1.0, 0.0, 0.0))]
        glb, _ = anim_glb.build(joints, [], [("sample", {(0, "translation"): keys})])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sampler.glb")
            with open(path, "wb") as out:
                out.write(glb)
            gltf, binary = read_glb(path)

        sampler = gltf["animations"][0]["samplers"][0]
        index = sampler["input"]
        min_time, max_time = accessor_values(gltf, binary, index)
        self.assertEqual(gltf["accessors"][index]["min"], list(min_time))
        self.assertEqual(gltf["accessors"][index]["max"], list(max_time))

    def test_keys_that_collapse_to_one_stored_time_are_refused(self) -> None:
        skeleton = skeleton_resource()
        joints, _ = anim_glb.read_skeleton(skeleton, anim_glb.drawable_at(skeleton, None))
        keys = [(1.0, (0.0, 0.0, 0.0)), (1.0 + 2 ** -25, (1.0, 0.0, 0.0))]
        with self.assertRaisesRegex(ValueError, "key times are not increasing"):
            anim_glb.build(joints, [], [("sample", {(0, "translation"): keys})])


class ExportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.packs = os.path.join(self.tmp.name, "packs.rpf")
        self.anim = os.path.join(self.tmp.name, "anim.rpf")
        palette = [joaat32("skel_c"), joaat32("skel_a")]
        skin = [[(0, 200), (1, 55)], [(1, 255)], [(0, 128), (1, 127)], [(1, 255)]]
        self.offsets = write_archive(self.packs, [[skeleton_resource()], list(part_resource(palette, "u8x4", skin))])
        self.dict_offset = clips_fixture.write_rpf(self.anim, [clips_fixture.two_sequence_dictionary("first@dict")])[0]

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, *args):
        return subprocess.run([sys.executable, "-I", TOOL, *args], capture_output=True, text=True)

    def export(self, *extra):
        out = os.path.join(self.tmp.name, "out.glb")
        report = os.path.join(self.tmp.name, "report.json")
        r = self.run_tool(self.packs, out, "--skeleton", hex(self.offsets[0]), "--part", hex(self.offsets[1]),
                          "--anim", self.anim, "--dictionary", hex(self.dict_offset), "--report", report, *extra)
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(report, encoding="utf-8") as f:
            return read_glb(out), json.load(f)

    def test_export_writes_skin_joints_and_clip(self):
        (gltf, binary), report = self.export()
        self.assertEqual(gltf["nodes"][0]["rotation"], anim_glb.Z_UP_TO_Y_UP)
        skin = gltf["skins"][0]
        self.assertEqual([gltf["nodes"][i]["name"] for i in skin["joints"]], list(NAMES))
        ibm = accessor_values(gltf, binary, skin["inverseBindMatrices"])
        world = world_transforms()
        for got, (q, t) in zip(ibm, world):
            for a, b in zip(got, inverse_bind(q, t)):
                self.assertAlmostEqual(a, b, places=5)
        prim = gltf["meshes"][0]["primitives"][0]
        joints = accessor_values(gltf, binary, prim["attributes"]["JOINTS_0"])
        weights = accessor_values(gltf, binary, prim["attributes"]["WEIGHTS_0"])
        self.assertEqual(joints[0][:2], (2, 0))
        for row in weights:
            self.assertAlmostEqual(sum(row), 1.0, places=5)
        # skinned mesh nodes sit at the scene root, outside the turned skeleton
        mesh_node = next(n for n in gltf["nodes"] if "mesh" in n)
        self.assertNotIn("rotation", mesh_node)
        self.assertEqual(mesh_node["skin"], 0)
        # the fixture clip: rotation on skel_a and skel_b, translation on skel_c, two control tracks left out
        (animation,) = gltf["animations"]
        targets = sorted((gltf["nodes"][c["target"]["node"]]["name"], c["target"]["path"])
                         for c in animation["channels"])
        self.assertEqual(targets, [("SKEL_A", "rotation"), ("SKEL_B", "rotation"), ("SKEL_C", "translation")])
        self.assertEqual(report["clips"][0]["tracks_other_type"], 2)
        # the sampled values are the decoded ones
        s, d = anim_glb.load_dictionary(self.anim, self.dict_offset)
        tracks, frames = anim_clips.decode_animation(s, d["packs"][0]["animations"][0])
        channel = next(c for c in animation["channels"] if gltf["nodes"][c["target"]["node"]]["name"] == "SKEL_A")
        sampler = animation["samplers"][channel["sampler"]]
        times = accessor_values(gltf, binary, sampler["input"])
        values = accessor_values(gltf, binary, sampler["output"])
        self.assertEqual(len(times), frames)
        self.assertAlmostEqual(times[-1][0], (frames - 1) / 30.0, places=5)
        for got, want in zip(values, tracks[0]):
            for a, b in zip(got, want):
                self.assertAlmostEqual(a, b, places=5)

    def test_dictionary_lookup_stops_before_adjacent_resource(self):
        data = clips_fixture.two_sequence_dictionary()
        _, selected, _ = clips_fixture.write_rpf(self.anim, [data] * 3)
        with patch.object(anim_glb.anim_index, "classify", wraps=anim_glb.anim_index.classify) as classify:
            system, _dictionary = anim_glb.load_dictionary(self.anim, selected)
        self.assertEqual(system.d, data)
        self.assertEqual([call.args[1] for call in classify.call_args_list], [selected])

    def test_export_reads_page_map_on_sixth_page(self):
        data = bytearray(clips_fixture.two_sequence_dictionary())
        page_map = len(data)
        data.extend(bytes(96))
        struct.pack_into("<Q", data, 8, SYSTEM + page_map)
        data[page_map + 8] = 6
        cuts = [0, 256, 512, 768, 1024, page_map, len(data)]
        pages = [data[a:b] for a, b in zip(cuts, cuts[1:])]
        self.dict_offset = clips_fixture.write_rpf(self.anim, pages)[0]

        (gltf, binary), _report = self.export()
        (animation,) = gltf["animations"]
        channel = next(c for c in animation["channels"]
                       if gltf["nodes"][c["target"]["node"]]["name"] == "SKEL_A")
        self.assertEqual(channel["target"]["path"], "rotation")
        sampler = animation["samplers"][channel["sampler"]]
        values = accessor_values(gltf, binary, sampler["output"])
        self.assertEqual(len(values), clips_fixture.NF)
        for rotation, x in zip(values, clips_fixture.X_INTS):
            self.assertAlmostEqual(rotation[0], 0.002 * x - 0.05, places=5)

    def test_missing_page_leaves_adjacent_dictionary_selectable(self):
        data = clips_fixture.two_sequence_dictionary("incomplete")
        first, _ = clips_fixture.split(data, len(data), move_map=False)
        other = clips_fixture.two_sequence_dictionary("neighbor")
        selected, neighbor = clips_fixture.write_rpf(self.anim, [first, other])
        with self.assertRaisesRegex(anim_glb.ExportError, "no clip dictionary"):
            anim_glb.load_dictionary(self.anim, selected)
        system, _ = anim_glb.load_dictionary(self.anim, neighbor)
        self.assertEqual(system.d, other)
        _, summary = anim_glb.anim_clips.scan(self.anim)
        self.assertEqual(summary["unresolved"], {"system page not the next stream": 1})
        self.assertEqual(summary["counts"]["dictionaries"], 1)

    def test_root_like_continuation_with_incidental_name(self):
        data = clips_fixture.two_sequence_dictionary()
        first, _ = clips_fixture.split(data, len(data), move_map=False)
        tail = bytearray(first)
        tail[0x68] ^= 1  # Root prefix and valid name, but no matching dictionary hash.
        selected, _ = clips_fixture.write_rpf(self.anim, [first, tail])
        system, _ = anim_glb.load_dictionary(self.anim, selected)
        self.assertEqual(system.d, first + tail)

    def test_missing_pack_table_cannot_claim_a_named_neighbor(self):
        data = clips_fixture.two_sequence_dictionary("incomplete")
        refs = anim_clips.System(data).ptr(0x70)
        first, _ = clips_fixture.split(data, refs, move_map=False)
        other = clips_fixture.two_sequence_dictionary("neighbor")
        selected, neighbor = clips_fixture.write_rpf(self.anim, [first, other])
        with self.assertRaisesRegex(anim_glb.ExportError, "no clip dictionary"):
            anim_glb.load_dictionary(self.anim, selected)
        self.assertEqual(anim_glb.load_dictionary(self.anim, neighbor)[0].d, other)
        _, summary = anim_clips.scan(self.anim)
        self.assertEqual(summary["unresolved"], {"system page not the next stream": 1})
        self.assertEqual(summary["counts"]["dictionaries"], 1)

    def test_named_pack_at_continuation_boundary(self):
        data = bytearray(clips_fixture.two_sequence_dictionary("abc"))
        data.extend(bytes(96))
        s = anim_clips.System(data)
        refs = s.ptr(0x70)
        pack = s.ptr(refs)
        data[pack + 8:pack + 16] = data[8:16]
        struct.pack_into("<15sB", data, pack + 0x38, b"abc-0", 0x85)
        # These layouts parse when the resource extent is known independently.
        for cuts in ([pack], [pack, refs]):
            data[s.ptr(8) + 8] = len(cuts) + 1
            pages = [data[a:b] for a, b in zip([0, *cuts], [*cuts, len(data)])]
            with self.subTest(ambiguous_cuts=cuts):
                self.assertEqual(anim_clips.parse_dictionary(s)["packs"][0]["name"], "abc-0")
                root, *_ = clips_fixture.write_rpf(self.anim, pages)
                with self.assertRaisesRegex(anim_glb.ExportError, "no clip dictionary"):
                    anim_glb.load_dictionary(self.anim, root)
                rows, _ = anim_clips.scan(self.anim)
                self.assertNotIn(root, [row["offset"] for row in rows])

        # Move the pack header behind its reference table. The table can then
        # be on the first page or on an already accepted continuation.
        copied_pack = len(data)
        data.extend(data[pack:pack + 0x70])
        struct.pack_into("<Q", data, refs, SYSTEM + copied_pack)
        for cuts in ([copied_pack], [refs, copied_pack]):
            with self.subTest(cuts=cuts):
                data[s.ptr(8) + 8] = len(cuts) + 1
                pages = [data[a:b] for a, b in zip([0, *cuts], [*cuts, len(data)])]
                root, *_ = clips_fixture.write_rpf(self.anim, pages)
                system, dictionary = anim_glb.load_dictionary(self.anim, root)
                self.assertEqual(system.d, data)
                self.assertEqual(dictionary["packs"][0]["name"], "abc-0")
                _, summary = anim_clips.scan(self.anim)
                self.assertEqual(summary["unresolved"], {})
                self.assertEqual(summary["counts"]["dictionaries"], 1)

    def test_neighbor_cannot_supply_its_own_pack_reference(self):
        for later in (False, True):
            with self.subTest(table_on_later_page=later):
                owner = bytearray(clips_fixture.two_sequence_dictionary("owner"))
                neighbor = bytearray(clips_fixture.two_sequence_dictionary("neighbor"))
                owner[anim_clips.System(owner).ptr(8) + 8] = 3 if later else 2
                table = len(neighbor) if later else 0x60
                struct.pack_into("<Q", owner, 0x70, SYSTEM + len(owner) + table)
                if later:
                    neighbor[anim_clips.System(neighbor).ptr(8) + 8] = 2
                    tail = bytearray(96)
                    struct.pack_into("<QI", tail, 0, SYSTEM + len(owner), joaat32("neighbor"))
                    pages = [neighbor, tail]
                else:
                    # The neighbor's own hash at +0x68 completes the false reference.
                    struct.pack_into("<Q", neighbor, table, SYSTEM + len(owner))
                    pages = [neighbor]
                root, following, *_ = clips_fixture.write_rpf(self.anim, [owner, *pages])
                with self.assertRaisesRegex(anim_glb.ExportError, "no clip dictionary"):
                    anim_glb.load_dictionary(self.anim, root)
                self.assertEqual(anim_glb.load_dictionary(self.anim, following)[0].d, b"".join(pages))
                rows, summary = anim_clips.scan(self.anim)
                self.assertEqual([row["offset"] for row in rows], [following])
                self.assertEqual(summary["unresolved"], {"system page not the next stream": 1})

    def test_middle_dictionary_page_count_straddles_fifth_and_sixth_pages(self):
        neighbor = clips_fixture.two_sequence_dictionary()
        data = bytearray(clips_fixture.two_sequence_dictionary("middle"))
        page_map = len(data)
        data.extend(bytes(96))
        struct.pack_into("<Q", data, 8, SYSTEM + page_map)
        data[page_map + 8] = 6
        cuts = [0, 256, 512, 768, 1024, page_map + 9, len(data)]
        pages = [data[a:b] for a, b in zip(cuts, cuts[1:])]
        offsets = clips_fixture.write_rpf(self.anim, [neighbor, *pages, neighbor])
        with patch.object(anim_glb.anim_index, "classify", wraps=anim_glb.anim_index.classify) as classify:
            system, _ = anim_glb.load_dictionary(self.anim, offsets[1])
        self.assertEqual(system.d, data)
        self.assertEqual([call.args[1] for call in classify.call_args_list], offsets[1:-1])

    def test_no_root_motion_leaves_the_mover_node_out(self):
        (gltf, _binary), _report = self.export("--no-root-motion")
        self.assertNotIn("mover", [n.get("name") for n in gltf["nodes"]])

    def test_clip_selection_must_have_an_animation_export(self):
        out = os.path.join(self.tmp.name, "out.glb")
        report = os.path.join(self.tmp.name, "report.json")
        common = ["--skeleton", hex(self.offsets[0]), "--report", report]
        for selector in (["--clip", "walk"], ["--dictionary", "0"]):
            for mode, args in (("--anim", [out, "--part", hex(self.offsets[1])]),
                               ("--coverage", ["--coverage", "--anim", self.anim])):
                with self.subTest(selector=selector, mode=mode):
                    result = self.run_tool(self.packs, *args, *common, *selector)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn(selector[0], result.stderr.splitlines()[-1])
                    self.assertIn(mode, result.stderr.splitlines()[-1])
                    self.assertFalse(os.path.exists(out))
                    self.assertFalse(os.path.exists(report))

        # The same authored part is still a valid static export without selectors.
        result = self.run_tool(self.packs, out, "--part", hex(self.offsets[1]), *common)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("animations", read_glb(out)[0])

    def test_coverage_counts_bone_tracks(self):
        report = os.path.join(self.tmp.name, "coverage.json")
        r = self.run_tool(self.packs, "--coverage", "--skeleton", hex(self.offsets[0]), "--anim", self.anim,
                          "--report", report)
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(report, encoding="utf-8") as f:
            counts = json.load(f)["counts"]
        self.assertEqual((counts["dictionaries"], counts["bone_tracks"], counts["bone_tracks_in_skeleton"],
                          counts["dictionaries_all_bones_in_skeleton"]), (1, 3, 3, 1))

    def test_a_jobs_count_below_one_is_refused(self):
        # coverage() used max(1, jobs), so 0 and negatives ran one serial worker and still exited 0.
        report = os.path.join(self.tmp.name, "coverage.json")
        for jobs in ("0", "-4"):
            r = self.run_tool(self.packs, "--coverage", "--skeleton", hex(self.offsets[0]), "--anim", self.anim,
                              "--report", report, "--jobs", jobs)
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("--jobs", r.stderr)
            self.assertNotIn("Traceback", r.stderr)
            self.assertFalse(os.path.exists(report), "the run still wrote a report")

    def test_a_skeleton_whose_binds_disagree_is_refused(self):
        write_archive(self.packs, [[skeleton_resource(shift_bind=0.3)]])
        r = self.run_tool(self.packs, os.path.join(self.tmp.name, "x.glb"), "--skeleton", hex(self.offsets[0]),
                          "--part", hex(self.offsets[0]))
        self.assertEqual(r.returncode, 2)
        self.assertIn("inverse bind", r.stderr)

    def test_outputs_over_an_input_are_refused(self):
        for out in (self.packs, self.anim):
            r = self.run_tool(self.packs, out, "--skeleton", hex(self.offsets[0]), "--part", hex(self.offsets[1]),
                              "--anim", self.anim, "--dictionary", hex(self.dict_offset))
            self.assertEqual(r.returncode, 2)
            self.assertIn("refusing to overwrite", r.stderr)
        r = self.run_tool(self.packs, os.path.join(self.tmp.name, "x.glb"), "--skeleton", hex(self.offsets[0]),
                          "--part", hex(self.offsets[1]), "--report", self.packs)
        self.assertEqual(r.returncode, 2)

    def test_report_directory_is_refused_before_writing_a_glb(self):
        out = os.path.join(self.tmp.name, "out.glb")
        report = os.path.join(self.tmp.name, "report")
        os.mkdir(report)
        r = self.run_tool(self.packs, out, "--skeleton", hex(self.offsets[0]),
                          "--part", hex(self.offsets[1]), "--report", report)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("--report is a directory", r.stderr)
        self.assertNotIn("Traceback", r.stderr)
        self.assertFalse(os.path.exists(out))

    def test_report_write_failure_is_one_error_line(self):
        out = os.path.join(self.tmp.name, "out.glb")
        report = os.path.join(self.tmp.name, "missing", "report.json")
        r = self.run_tool(self.packs, out, "--skeleton", hex(self.offsets[0]),
                          "--part", hex(self.offsets[1]), "--report", report)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertEqual(len(r.stderr.strip().splitlines()), 1, r.stderr)
        self.assertIn("cannot write report", r.stderr)
        self.assertIn(report, r.stderr)
        self.assertEqual(r.stdout, "")

    def test_a_bad_offset_is_one_error_line(self):
        r = self.run_tool(self.packs, os.path.join(self.tmp.name, "x.glb"), "--skeleton", "0x20", "--part", "0x20")
        self.assertEqual(r.returncode, 2)
        self.assertEqual(len(r.stderr.strip().splitlines()), 1, r.stderr)
        self.assertTrue(r.stderr.startswith("ERROR:"), r.stderr)

    def test_a_missing_archive_prints_its_path_without_repr_escaping(self):
        missing = os.path.join(self.tmp.name, "missing", "packs.rpf")
        r = self.run_tool(missing, os.path.join(self.tmp.name, "x.glb"),
                          "--skeleton", "0x1000", "--part", "0x1000")
        self.assertEqual(r.returncode, 2)
        self.assertEqual(r.stderr.strip(), f"ERROR: {missing}: {os.strerror(2)}")
        self.assertEqual(r.stdout, "")


if __name__ == "__main__":
    unittest.main()
