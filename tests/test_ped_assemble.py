"""Pure synthetic unit tests for tools/ped_assemble.py (zero game files).

    python -I tests/test_ped_assemble.py

Constructs synthetic overlapping modular body-part meshes (upper torso + lower
pelvis sharing a coplanar waist overlap band, plus female nipple and genital
patches, plus nearby non-coplanar geometry) to verify:
- exact coplanar overlap triangle removal counts,
- watertight seam preservation (zero unintended boundary-edge holes, zero
  non-manifold overlap edges after stitching),
- preservation of nearby non-coplanar triangles,
- standing-space +1.0 m SKEL_ROOT alignment and RAGE Z-up -> glTF Y-up math,
- valid binary glTF 2.0 (.glb) export (static textured and 8-influence skinned,
  cross-checked with tools/glb_writer.py and tools/glb_skinning.py),
- CLI execution (`--help`, `assemble`, `cull-overlap`, `inspect-seams`).
"""
from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import glb_skinning  # noqa: E402
import glb_writer  # noqa: E402
import ped_assemble as pa  # noqa: E402


def _make_1x1_png(r: int, g: int, b: int, a: int = 255) -> bytes:
    """Build a minimal valid 1x1 RGBA PNG using only stdlib zlib/struct."""
    def _chunk(tag: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    raw_scanline = bytes([0, r, g, b, a])
    idat = zlib.compress(raw_scanline)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", idat)
        + _chunk(b"IEND", b"")
    )


def _cylinder_band(
    z_levels: list[float],
    n_seg: int = 16,
    radius: float = 0.15,
    *,
    split_islands: bool = False,
    radial_offset: float = 0.0,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]], list[tuple[float, float]]]:
    """Generate a synthetic tubular mesh in RAGE Z-up spanning ``z_levels``.

    When ``split_islands=True``, each quad band is emitted with duplicated
    vertex indices (mirroring RAGE 9's physically split ``_nude_cull`` islands
    whose positions still weld seamlessly).
    """
    positions: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    r = radius + radial_offset

    if not split_islands:
        for zi, z in enumerate(z_levels):
            v_coord = zi / max(len(z_levels) - 1, 1)
            for si in range(n_seg):
                ang = 2.0 * math.pi * si / n_seg
                positions.append((r * math.cos(ang), r * math.sin(ang), z))
                uvs.append((si / n_seg, v_coord))
        for zi in range(len(z_levels) - 1):
            row0 = zi * n_seg
            row1 = (zi + 1) * n_seg
            for si in range(n_seg):
                sn = (si + 1) % n_seg
                a, b = row0 + si, row0 + sn
                c, d = row1 + si, row1 + sn
                triangles.append((a, b, d))
                triangles.append((a, d, c))
        return positions, triangles, uvs

    for zi in range(len(z_levels) - 1):
        z0, z1 = z_levels[zi], z_levels[zi + 1]
        v0 = zi / max(len(z_levels) - 1, 1)
        v1 = (zi + 1) / max(len(z_levels) - 1, 1)
        for si in range(n_seg):
            sn = (si + 1) % n_seg
            a0 = 2.0 * math.pi * si / n_seg
            a1 = 2.0 * math.pi * sn / n_seg
            base = len(positions)
            positions.extend([
                (r * math.cos(a0), r * math.sin(a0), z0),
                (r * math.cos(a1), r * math.sin(a1), z0),
                (r * math.cos(a0), r * math.sin(a0), z1),
                (r * math.cos(a1), r * math.sin(a1), z1),
            ])
            uvs.extend([
                (si / n_seg, v0),
                ((si + 1) / n_seg, v0),
                (si / n_seg, v1),
                ((si + 1) / n_seg, v1),
            ])
            triangles.append((base + 0, base + 1, base + 3))
            triangles.append((base + 0, base + 3, base + 2))
    return positions, triangles, uvs


def _extract_sub_patch(
    positions: list[tuple[float, float, float]],
    triangles: list[tuple[int, int, int]],
    tri_indices: list[int],
    *,
    normal_shift: float = 0.0,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    """Extract a subset of triangles as a standalone patch, optionally nudged slightly along +radius."""
    remap: dict[int, int] = {}
    out_pos: list[tuple[float, float, float]] = []
    out_tris: list[tuple[int, int, int]] = []
    for ti in tri_indices:
        tri = triangles[ti]
        nt = []
        for vi in tri:
            if vi not in remap:
                x, y, z = positions[vi]
                if normal_shift != 0.0:
                    rad = math.hypot(x, y) or 1.0
                    x += normal_shift * (x / rad)
                    y += normal_shift * (y / rad)
                remap[vi] = len(out_pos)
                out_pos.append((x, y, z))
            nt.append(remap[vi])
        out_tris.append((nt[0], nt[1], nt[2]))
    return out_pos, out_tris


class StandingSpaceTests(unittest.TestCase):
    def test_standing_offset_and_zup_to_yup(self):
        # Bind-space biped: feet at Z = -1.0 m, head top at Z = +0.72 m (1.72 m tall)
        pos_bind = [(0.0, 0.0, -1.0), (0.1, 0.2, 0.0), (0.0, 0.0, 0.72)]
        self.assertTrue(pa.is_bind_space_biped(pos_bind, coord_space="z-up"))

        shifted_z = pa.apply_standing_offset(pos_bind, offset_m=1.0, coord_space="z-up")
        self.assertFalse(pa.is_bind_space_biped(shifted_z, coord_space="z-up"))
        self.assertAlmostEqual(shifted_z[0][2], 0.0, places=7)
        self.assertAlmostEqual(shifted_z[1][2], 1.0, places=7)
        self.assertAlmostEqual(shifted_z[2][2], 1.72, places=7)

        part = pa.MeshPart(
            name="hand_fr1_000_nude_cull",
            positions=pos_bind,
            triangles=[(0, 1, 2)],
            normals=[(0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)],
            coord_space="z-up",
        )
        aligned = pa.align_part_to_standing_space(part, standing_offset=1.0, target_space="y-up")
        self.assertEqual(aligned.coord_space, "y-up")
        # RAGE (0, 0, 0.0) -> glTF (0, 0, 0); RAGE (0.1, 0.2, 1.0) -> glTF (0.1, 1.0, -0.2)
        self.assertAlmostEqual(aligned.positions[0][1], 0.0, places=7)
        self.assertAlmostEqual(aligned.positions[1][0], 0.1, places=7)
        self.assertAlmostEqual(aligned.positions[1][1], 1.0, places=7)
        self.assertAlmostEqual(aligned.positions[1][2], -0.2, places=7)
        self.assertAlmostEqual(aligned.positions[2][1], 1.72, places=7)
        # Normals rotated identically: (0,1,0)->(0,0,-1), (0,0,1)->(0,1,0)
        self.assertEqual(aligned.normals[0], (0.0, 0.0, -1.0))
        self.assertEqual(aligned.normals[1], (0.0, 1.0, 0.0))
        # Roundtrip back to RAGE Z-up
        for p in aligned.positions:
            back = pa.rage_to_yup_vec3(pa.yup_to_rage_vec3(p))
            for a, b in zip(p, back):
                self.assertAlmostEqual(a, b, places=7)

    def test_role_classification(self):
        self.assertEqual(pa.classify_part_role("hand_fr1_000_nude_cull"), pa.ROLE_UPPER_BODY)
        self.assertEqual(pa.classify_part_role("hand_mr1_000_nude_cull.glb"), pa.ROLE_UPPER_BODY)
        self.assertEqual(pa.classify_part_role("feet_fr1_000_nude_cull"), pa.ROLE_LOWER_BODY)
        self.assertEqual(pa.classify_part_role("bodynipplesgen_fr1_000_regular_ws00_full"), pa.ROLE_NIPPLES)
        self.assertEqual(pa.classify_part_role("bodygenitalsgen_fr1_008_regular_ws00_full"), pa.ROLE_GENITALS_FEMALE)
        self.assertEqual(pa.classify_part_role("bodygenitalsgen_mr1_000_regular_ws00_full"), pa.ROLE_GENITALS_MALE)
        self.assertEqual(pa.classify_part_role("uppr_000"), pa.ROLE_UPPER_BODY)
        self.assertEqual(pa.classify_part_role("lowr_000"), pa.ROLE_LOWER_BODY)


class CoplanarCullingAndSeamTests(unittest.TestCase):
    def test_weld_collapsed_faces_do_not_contribute_edge_incidence(self):
        part = pa.MeshPart(name="panel", positions=[(0, 0, 0), (1, 0, 0), (0, 1, 0),
                                                    (0.00001, 0.00001, 0), (0, 0.00001, 0)],
                           triangles=[(0, 1, 2), (0, 3, 1), (0, 3, 4)])
        report = pa.inspect_seams([part], weld_tol=1e-4)
        self.assertEqual(report["parts"][0]["triangles"], 3)
        self.assertEqual(report["combined_topology"], {
            "unique_edges": 3, "boundary_edges": 3, "manifold_edges": 0,
            "non_manifold_edges": 0, "collapsed_triangles": 2,
        })
        # At a finer weld resolution all three faces retain their own corners.
        fine = pa.analyze_edge_topology([part], weld_tol=1e-6)
        self.assertEqual(fine["collapsed_triangles"], 0)
        self.assertEqual(fine["manifold_edges"], 2)

    def _build_synthetic_body_parts(
        self,
        *,
        waist_radial_jitter: float = 0.0,
        patch_normal_shift: float = 0.0,
    ):
        # Upper torso spans Z = [0.90, 1.40] with 5 bands (6 z-levels), 16 segments -> 5 * 32 = 160 tris.
        # Lower pelvis spans Z = [0.50, 1.04] with 4 bands (5 z-levels), 16 segments -> 4 * 32 = 128 tris.
        # Exact overlap band is Z = [0.90, 0.97, 1.04] -> 2 bands * 32 tris = 64 coplanar waist overlap tris!
        n_seg = 16
        upper_z = [0.90, 0.97, 1.04, 1.16, 1.28, 1.40]
        lower_z = [0.50, 0.70, 0.90, 0.97, 1.04]

        u_pos, u_tris, u_uvs = _cylinder_band(upper_z, n_seg=n_seg, split_islands=True)
        l_pos, l_tris, l_uvs = _cylinder_band(lower_z, n_seg=n_seg, split_islands=True, radial_offset=waist_radial_jitter)

        # Nipple patch overlaps 4 triangles on upper chest band (band index 3: Z = [1.16, 1.28], segments 0 and 8)
        nip_tri_ids = [3 * (2 * n_seg) + 0, 3 * (2 * n_seg) + 1, 3 * (2 * n_seg) + 16, 3 * (2 * n_seg) + 17]
        n_pos, n_tris = _extract_sub_patch(u_pos, u_tris, nip_tri_ids, normal_shift=patch_normal_shift)

        # Female genital patch overlaps 6 triangles on lower pelvis band (band index 1: Z = [0.70, 0.90])
        gen_tri_ids = [1 * (2 * n_seg) + k for k in range(6)]
        g_pos, g_tris = _extract_sub_patch(l_pos, l_tris, gen_tri_ids, normal_shift=patch_normal_shift)

        upper = pa.MeshPart(
            name="hand_fr1_000_nude_cull",
            positions=u_pos,
            triangles=u_tris,
            uvs=u_uvs,
            coord_space="z-up",
            material=pa.MaterialSpec(name="upper_skin", base_color_factor=(0.82, 0.65, 0.55, 1.0)),
        )
        lower = pa.MeshPart(
            name="feet_fr1_000_nude_cull",
            positions=l_pos,
            triangles=l_tris,
            uvs=l_uvs,
            coord_space="z-up",
            material=pa.MaterialSpec(name="lower_skin", base_color_factor=(0.82, 0.65, 0.55, 1.0)),
        )
        nipples = pa.MeshPart(
            name="bodynipplesgen_fr1_000_regular_ws00_full",
            positions=n_pos,
            triangles=n_tris,
            coord_space="z-up",
        )
        genitals = pa.MeshPart(
            name="bodygenitalsgen_fr1_008_regular_ws00_full",
            positions=g_pos,
            triangles=g_tris,
            coord_space="z-up",
        )
        return upper, lower, nipples, genitals

    def test_exact_waist_nipple_genital_culling_and_watertight_stitch(self):
        upper, lower, nipples, genitals = self._build_synthetic_body_parts(
            waist_radial_jitter=0.0,
            patch_normal_shift=0.0,
        )

        # Physical islands on split_islands upper mesh = 5 bands * 16 quads = 80 islands, welded = 1 island
        phys = pa.count_mesh_islands(upper.positions, upper.triangles, by_position=False)
        welded = pa.count_mesh_islands(upper.positions, upper.triangles, by_position=True)
        self.assertEqual(phys["island_count"], 80)
        self.assertEqual(welded["island_count"], 1)

        # Before culling: waist overlap produces non-manifold double-drawn edges
        before_body = pa.analyze_edge_topology([upper, lower], weld_tol=1e-4)
        self.assertGreater(before_body["non_manifold_edges"], 0)

        # Run standard 3-pass culling
        stitched, reports = pa.apply_standard_ped_culling([upper, lower, nipples, genitals])
        by_pass = {r["pass"]: r for r in reports}
        self.assertEqual(by_pass["upper_vs_lower_waist"]["culled_total"], 64)
        self.assertEqual(by_pass["upper_vs_nipples"]["culled_total"], 4)
        self.assertEqual(by_pass["lower_vs_female_genitals"]["culled_total"], 6)

        stitched_upper, stitched_lower, stitched_nip, stitched_gen = stitched
        self.assertEqual(len(stitched_upper.triangles), 160 - 64 - 4)
        self.assertEqual(len(stitched_lower.triangles), 128 - 6)
        self.assertEqual(len(stitched_nip.triangles), 4)
        self.assertEqual(len(stitched_gen.triangles), 6)

        # Watertight check across all 4 stitched parts at 0.1 mm weld tolerance:
        # zero non-manifold edges, and boundary edges equal ONLY the 16 top-cap +
        # 16 bottom-cap rim edges = 32 (zero unintended holes along any seam!).
        after_topo = pa.analyze_edge_topology(stitched, weld_tol=1e-4)
        self.assertEqual(after_topo["non_manifold_edges"], 0)
        self.assertEqual(after_topo["boundary_edges"], 32)

        # Re-inspecting seams must show 0 remaining pairwise coplanar overlaps
        after_seams = pa.inspect_seams(stitched, weld_tol=1e-4, overlap_tol=0.004)
        self.assertEqual(after_seams["pairwise_overlaps"], [])

    def test_submillimeter_perturbed_waist_and_patch_overlap_culling(self):
        # 0.8 mm radial offset on waist (< 1.5 mm measured RAGE 9 tolerance) and 0.5 mm shift on
        # nipple/genital patches fail exact vertex-key matching and are caught by spatial-hashed
        # coplanar proximity culling.
        upper, lower, nipples, genitals = self._build_synthetic_body_parts(
            waist_radial_jitter=0.0008,
            patch_normal_shift=0.0005,
        )
        stitched, reports = pa.apply_standard_ped_culling([upper, lower, nipples, genitals])
        by_pass = {r["pass"]: r for r in reports}
        self.assertEqual(by_pass["upper_vs_lower_waist"]["culled_exact_key"], 0)
        self.assertEqual(by_pass["upper_vs_lower_waist"]["culled_coplanar_proximity"], 64)
        self.assertEqual(by_pass["upper_vs_nipples"]["culled_total"], 4)
        self.assertEqual(by_pass["lower_vs_female_genitals"]["culled_total"], 6)
        self.assertEqual(len(stitched[0].triangles), 160 - 64 - 4)
        self.assertEqual(len(stitched[1].triangles), 128 - 6)

    def test_non_coplanar_nearby_triangles_are_preserved(self):
        # Reference triangle lies in XY plane (Z = 0, normal = +Z), centroid at (0.01, 0.01, 0.0)
        ref = pa.MeshPart(
            name="feet_fr1_000_nude_cull",
            positions=[(0.0, 0.0, 0.0), (0.03, 0.0, 0.0), (0.0, 0.03, 0.0)],
            triangles=[(0, 1, 2)],
            coord_space="z-up",
        )
        # Target contains:
        #   tri 0: coplanar duplicate (Z = 0.0005, normal = +Z) -> MUST BE CULLED
        #   tri 1: perpendicular fin sharing exact same centroid (normal = +Y) -> MUST BE PRESERVED
        #   tri 2: parallel offset layer 6 mm above (Z = 0.006, normal = +Z) -> MUST BE PRESERVED
        #   tri 3: tilted 45-deg wedge sharing exact same centroid -> MUST BE PRESERVED
        target = pa.MeshPart(
            name="hand_fr1_000_nude_cull",
            positions=[
                # tri 0 (coplanar)
                (0.0, 0.0, 0.0005), (0.03, 0.0, 0.0005), (0.0, 0.03, 0.0005),
                # tri 1 (perpendicular XZ plane passing through centroid (0.01, 0.01, 0.0))
                (0.0, 0.01, -0.01), (0.03, 0.01, -0.01), (0.0, 0.01, 0.02),
                # tri 2 (parallel cloth hovering 6 mm outside skin)
                (0.0, 0.0, 0.006), (0.03, 0.0, 0.006), (0.0, 0.03, 0.006),
                # tri 3 (tilted ~45 deg passing through centroid)
                (0.0, 0.0, -0.01), (0.03, 0.0, 0.01), (0.0, 0.03, 0.0),
            ],
            triangles=[(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)],
            coord_space="z-up",
        )
        culled, stats = pa.cull_part_overlap(
            target,
            [ref],
            centroid_tol=0.004,
            plane_tol=0.002,
            normal_cos_min=0.90,
        )
        self.assertEqual(stats["culled_total"], 1)
        self.assertEqual(culled.triangles, [(3, 4, 5), (6, 7, 8), (9, 10, 11)])

    def test_compact_vertices_remaps_attributes(self):
        part = pa.MeshPart(
            name="test_part",
            positions=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (5, 5, 5), (6, 5, 5), (5, 6, 5)],
            triangles=[(3, 4, 5)],
            uvs=[(0, 0), (1, 0), (0, 1), (0.5, 0.5), (0.6, 0.5), (0.5, 0.6)],
            coord_space="z-up",
        )
        compacted = pa.compact_part_vertices(part)
        self.assertEqual(len(compacted.positions), 3)
        self.assertEqual(compacted.triangles, [(0, 1, 2)])
        self.assertEqual(compacted.uvs, [(0.5, 0.5), (0.6, 0.5), (0.5, 0.6)])


class GlbExportAndCliTests(unittest.TestCase):
    def test_static_textured_glb_roundtrip(self):
        png = _make_1x1_png(220, 180, 150)
        part_a = pa.MeshPart(
            name="hand_fr1_000_nude_cull",
            positions=[(0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.2, 0.1)],
            triangles=[(0, 1, 2)],
            uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
            coord_space="z-up",
            material=pa.MaterialSpec(name="skin_upper", base_color_png=png, normal_png=png),
        )
        glb_bytes, stats = pa.build_assembled_glb([part_a], standing_offset=1.0, apply_standing=True)
        self.assertEqual(stats["parts"], 1)
        self.assertEqual(stats["triangles"], 1)
        self.assertEqual(stats["textures"], 1)  # deduplicated identical PNG bytes

        summary = glb_writer.read_glb_summary(glb_bytes)
        self.assertEqual(summary["version"], 2)
        self.assertEqual(len(summary["json"]["meshes"]), 1)
        self.assertEqual(len(summary["json"]["materials"]), 1)
        self.assertEqual(len(summary["json"]["images"]), 1)

        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "assembled.glb")
            pa.write_assembled_glb(out_path, [part_a], standing_offset=1.0, apply_standing=True)
            reloaded = pa.read_glb_parts(out_path)
            self.assertEqual(len(reloaded), 1)
            self.assertEqual(reloaded[0].coord_space, "y-up")
            # Original Z=0.0 + standing offset 1.0 -> Y=1.0 in glTF Y-up
            self.assertAlmostEqual(reloaded[0].positions[0][1], 1.0, places=5)
            self.assertAlmostEqual(reloaded[0].positions[2][1], 1.1, places=5)

    def test_skinned_glb_validates_with_glb_skinning(self):
        # 2-joint skeleton in RAGE Z-up with identity IBMs and 8-influence vertex weights
        skel = pa.SkeletonSpec(
            name="ambient_femalebody",
            coord_space="z-up",
            joints=[
                pa.SkeletonJoint(name="SKEL_ROOT", parent=-1, translation=(0.0, 0.0, 0.0)),
                pa.SkeletonJoint(name="SKEL_Pelvis", parent=0, translation=(0.0, 0.0, 0.02)),
            ],
        )
        part = pa.MeshPart(
            name="hand_fr1_000_nude_cull",
            positions=[(0.0, 0.0, -0.1), (0.1, 0.0, -0.1), (0.0, 0.0, 0.2)],
            triangles=[(0, 1, 2)],
            joints_0=[(0, 1, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0)],
            weights_0=[(0.5, 0.5, 0.0, 0.0), (0.25, 0.75, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)],
            joints_1=[(0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)],
            weights_1=[(0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)],
            coord_space="z-up",
        )
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "skinned.glb")
            stats = pa.write_assembled_glb(out_path, [part], skeleton=skel, standing_offset=1.0)
            self.assertEqual(stats["skins"], 1)
            self.assertEqual(stats["joints"], 2)

            # Cross-validate the produced GLB with Leonida's strict glb_skinning validator!
            proc = subprocess.run(
                [sys.executable, "-I", os.path.join(TOOLS, "glb_skinning.py"), out_path],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            # Also confirm standing-space root wrapper translation [0, 1.0, 0]
            with open(out_path, "rb") as f:
                summary = glb_writer.read_glb_summary(f.read())
            root_idx = summary["json"]["skins"][0]["skeleton"]
            self.assertEqual(summary["json"]["nodes"][root_idx]["translation"], [0.0, 1.0, 0.0])

    def test_cli_subcommands_end_to_end(self):
        script = os.path.join(TOOLS, "ped_assemble.py")
        # 1. --help
        r_help = subprocess.run(
            [sys.executable, "-I", script, "--help"],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(r_help.returncode, 0, r_help.stderr)
        self.assertIn("assemble", r_help.stdout)
        self.assertIn("cull-overlap", r_help.stdout)
        self.assertIn("inspect-seams", r_help.stdout)

        with tempfile.TemporaryDirectory() as tmp:
            # Build overlapping synthetic upper + lower JSON parts
            u_pos, u_tris, u_uvs = _cylinder_band([0.90, 0.97, 1.04, 1.20], n_seg=8, split_islands=True)
            l_pos, l_tris, l_uvs = _cylinder_band([0.60, 0.90, 0.97, 1.04], n_seg=8, split_islands=True)
            upper_json = os.path.join(tmp, "hand_mr1_000_nude_cull.json")
            lower_json = os.path.join(tmp, "feet_mr1_000_nude_cull.json")
            with open(upper_json, "w", encoding="utf-8") as f:
                json.dump({"name": "hand_mr1_000_nude_cull", "positions": u_pos, "triangles": u_tris, "uvs": u_uvs}, f)
            with open(lower_json, "w", encoding="utf-8") as f:
                json.dump({"name": "feet_mr1_000_nude_cull", "positions": l_pos, "triangles": l_tris, "uvs": l_uvs}, f)

            # 2. inspect-seams
            seam_report = os.path.join(tmp, "seams.json")
            r_seam = subprocess.run(
                [sys.executable, "-I", script, "inspect-seams", upper_json, lower_json, "--json", seam_report],
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(r_seam.returncode, 0, r_seam.stderr)
            with open(seam_report, encoding="utf-8") as f:
                sdata = json.load(f)
            self.assertEqual(sdata["part_count"], 2)
            self.assertEqual(len(sdata["pairwise_overlaps"]), 1)
            self.assertEqual(sdata["pairwise_overlaps"][0]["overlapping_triangles_in_a"], 32)

            # 3. cull-overlap
            culled_json = os.path.join(tmp, "culled_upper.json")
            cull_stats = os.path.join(tmp, "cull_stats.json")
            r_cull = subprocess.run(
                [
                    sys.executable, "-I", script, "cull-overlap",
                    "--target", upper_json, "--ref", lower_json,
                    "-o", culled_json, "--json", cull_stats, "--compact-vertices",
                ],
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(r_cull.returncode, 0, r_cull.stderr)
            with open(cull_stats, encoding="utf-8") as f:
                cdata = json.load(f)
            self.assertEqual(cdata["culled_total"], 32)
            self.assertEqual(cdata["target_triangles_after"], 16)

            # 4. assemble -> GLB + report (also test reading back OBJ + GLB inputs)
            out_glb = os.path.join(tmp, "ped.glb")
            asm_report = os.path.join(tmp, "asm.json")
            r_asm = subprocess.run(
                [
                    sys.executable, "-I", script, "assemble",
                    upper_json, lower_json, "-o", out_glb, "--json", asm_report,
                ],
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(r_asm.returncode, 0, r_asm.stderr)
            with open(asm_report, encoding="utf-8") as f:
                adata = json.load(f)
            self.assertEqual(adata["after_seams"]["combined_topology"]["non_manifold_edges"], 0)
            self.assertEqual(adata["after_seams"]["pairwise_overlaps"], [])
            self.assertEqual(adata["glb"]["triangles"], 16 + 48)
            self.assertTrue(os.path.isfile(out_glb))

            # 5. Re-inspect the assembled GLB directly via inspect-seams
            r_glb_seam = subprocess.run(
                [sys.executable, "-I", script, "inspect-seams", out_glb, "--source-space", "y-up"],
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(r_glb_seam.returncode, 0, r_glb_seam.stderr)
            self.assertIn("non_manifold=0", r_glb_seam.stdout)
            self.assertIn("overlapping_pairs=0", r_glb_seam.stdout)

    def test_obj_manifest_and_error_paths(self):
        script = os.path.join(TOOLS, "ped_assemble.py")
        with tempfile.TemporaryDirectory() as tmp:
            obj_path = os.path.join(tmp, "feet_fr1_000_nude_cull.obj")
            with open(obj_path, "w", encoding="utf-8") as f:
                f.write("o feet_fr1_000_nude_cull\n")
                f.write("v 0.0 0.0 -1.0\n")
                f.write("v 0.2 0.0 -1.0\n")
                f.write("v 0.0 0.2 -0.8\n")
                f.write("f 1 2 3\n")

            manifest_path = os.path.join(tmp, "ped_manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump({
                    "coord_space": "z-up",
                    "parts": [{"path": "feet_fr1_000_nude_cull.obj", "name": "feet_fr1_000_nude_cull"}],
                }, f)

            out_glb = os.path.join(tmp, "from_manifest.glb")
            r_asm = subprocess.run(
                [sys.executable, "-I", script, "assemble", "--manifest", manifest_path, "-o", out_glb],
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(r_asm.returncode, 0, r_asm.stderr)
            reloaded = pa.read_glb_parts(out_glb)
            self.assertEqual(len(reloaded), 1)
            # Z = -1.0 + 1.0 standing offset -> Y = 0.0 on ground plane
            self.assertAlmostEqual(reloaded[0].positions[0][1], 0.0, places=5)

            # Unsupported extension exits cleanly with status 2
            bad_path = os.path.join(tmp, "bad.xyz")
            with open(bad_path, "w", encoding="utf-8") as f:
                f.write("x")
            r_bad = subprocess.run(
                [sys.executable, "-I", script, "inspect-seams", bad_path],
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(r_bad.returncode, 2)
            self.assertIn("ERROR:", r_bad.stderr)


def _trs_mat4_col(
    t: tuple[float, float, float] = (0.0, 0.0, 0.0),
    q: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    s: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[float, ...]:
    """Column-major 4x4 TRS matrix (glTF convention)."""
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    r00 = (1.0 - 2.0 * (yy + zz)) * s[0]
    r10 = (2.0 * (xy + wz)) * s[0]
    r20 = (2.0 * (xz - wy)) * s[0]
    r01 = (2.0 * (xy - wz)) * s[1]
    r11 = (1.0 - 2.0 * (xx + zz)) * s[1]
    r21 = (2.0 * (yz + wx)) * s[1]
    r02 = (2.0 * (xz + wy)) * s[2]
    r12 = (2.0 * (yz - wx)) * s[2]
    r22 = (1.0 - 2.0 * (xx + yy)) * s[2]
    return (
        r00, r10, r20, 0.0,
        r01, r11, r21, 0.0,
        r02, r12, r22, 0.0,
        float(t[0]), float(t[1]), float(t[2]), 1.0,
    )


def _mul_mat4_col(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    out = [0.0] * 16
    for col in range(4):
        for row in range(4):
            out[col * 4 + row] = sum(a[k * 4 + row] * b[col * 4 + k] for k in range(4))
    return tuple(out)


def _transform_point_col(m: tuple[float, ...], p: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
        m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
        m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
    )


def _evaluate_glb_skinned_vertices(
    glb_path: str,
    rotations_by_joint_name: dict[str, tuple[float, float, float, float]],
) -> dict[str, list[tuple[float, float, float]]]:
    """Evaluate glTF linear-blend skinning directly from a GLB container."""
    with open(glb_path, "rb") as f:
        raw = f.read()
    doc, bin_chunk, _ = glb_skinning.parse_glb(raw)
    nodes = doc["nodes"]
    skins = doc["skins"]
    meshes = doc["meshes"]

    parent_of: dict[int, int] = {}
    for ni, node in enumerate(nodes):
        for child in node.get("children", []):
            parent_of[child] = ni

    world_cache: dict[int, tuple[float, ...]] = {}

    def _world_matrix(ni: int) -> tuple[float, ...]:
        if ni in world_cache:
            return world_cache[ni]
        node = nodes[ni]
        t = tuple(float(v) for v in node.get("translation", (0.0, 0.0, 0.0)))
        name = node.get("name", "")
        q = rotations_by_joint_name.get(
            name,
            tuple(float(v) for v in node.get("rotation", (0.0, 0.0, 0.0, 1.0))),
        )
        s = tuple(float(v) for v in node.get("scale", (1.0, 1.0, 1.0)))
        local = _trs_mat4_col(t, q, s)
        mat = _mul_mat4_col(_world_matrix(parent_of[ni]), local) if ni in parent_of else local
        world_cache[ni] = mat
        return mat

    results: dict[str, list[tuple[float, float, float]]] = {}
    for node in nodes:
        if "mesh" not in node or "skin" not in node:
            continue
        skin = skins[node["skin"]]
        ibms = pa._read_glb_accessor(doc, bin_chunk, skin["inverseBindMatrices"])
        joint_mats = [
            _mul_mat4_col(_world_matrix(j_node_idx), tuple(float(x) for x in ibms[slot]))
            for slot, j_node_idx in enumerate(skin["joints"])
        ]
        prim = meshes[node["mesh"]]["primitives"][0]
        attrs = prim["attributes"]
        positions = pa._read_glb_accessor(doc, bin_chunk, attrs["POSITION"])
        j0 = pa._read_glb_accessor(doc, bin_chunk, attrs["JOINTS_0"])
        w0 = pa._read_glb_accessor(doc, bin_chunk, attrs["WEIGHTS_0"])
        j1 = pa._read_glb_accessor(doc, bin_chunk, attrs["JOINTS_1"]) if "JOINTS_1" in attrs else None
        w1 = pa._read_glb_accessor(doc, bin_chunk, attrs["WEIGHTS_1"]) if "WEIGHTS_1" in attrs else None

        skinned: list[tuple[float, float, float]] = []
        for vi, pos in enumerate(positions):
            acc = [0.0, 0.0, 0.0]
            pairs = list(zip(j0[vi], w0[vi]))
            if j1 is not None and w1 is not None:
                pairs.extend(zip(j1[vi], w1[vi]))
            for j_idx, weight in pairs:
                if weight == 0.0:
                    continue
                tp = _transform_point_col(joint_mats[int(j_idx)], pos)
                acc[0] += weight * tp[0]
                acc[1] += weight * tp[1]
                acc[2] += weight * tp[2]
            skinned.append((acc[0], acc[1], acc[2]))
        results[node["name"]] = skinned
    return results


def _repack_glb_with_mutated_doc(glb_path: str, out_path: str, mutate_fn) -> None:
    """Parse a GLB, mutate its JSON dict via ``mutate_fn(doc)``, and write a valid GLB."""
    with open(glb_path, "rb") as f:
        raw = f.read()
    doc, bin_chunk, _ = glb_skinning.parse_glb(raw)
    mutate_fn(doc)
    json_bytes = bytearray(json.dumps(doc, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))
    pa._pad4(json_bytes, 0x20)
    bin_bytes = bytearray(bin_chunk or b"")
    pa._pad4(bin_bytes, 0)
    total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    out = (
        struct.pack("<4sII", pa.GLB_MAGIC, pa.GLB_VERSION, total_len)
        + struct.pack("<II", len(json_bytes), pa.CHUNK_TYPE_JSON)
        + bytes(json_bytes)
        + struct.pack("<II", len(bin_bytes), pa.CHUNK_TYPE_BIN)
        + bytes(bin_bytes)
    )
    with open(out_path, "wb") as f:
        f.write(out)


class SkinnedJointRemappingTests(unittest.TestCase):
    def _joint_dict(self, name: str, parent: int, translation: tuple[float, float, float]) -> dict:
        tx, ty, tz = translation
        ibm = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            -tx, -ty, -tz, 1.0,
        ]
        return {
            "name": name,
            "parent": parent,
            "translation": [tx, ty, tz],
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "scale": [1.0, 1.0, 1.0],
            "inverse_bind_matrix": ibm,
        }

    def _hinge_skeleton_dict(self, order: tuple[str, str, str]) -> dict:
        transforms = {
            "Root": (0.0, 0.0, 0.0),
            "Hinge": (0.25, 0.5, 0.0),
            "Guide": (-0.5, 0.25, 0.0),
        }
        joints = []
        for idx, name in enumerate(order):
            parent = -1 if name == "Root" else order.index("Root")
            joints.append(self._joint_dict(name, parent, transforms[name]))
        return {"name": "rig", "coord_space": "y-up", "joints": joints}

    def _build_hinge_source_glb(
        self,
        tmp: str,
        part_name: str,
        order: tuple[str, str, str],
    ) -> str:
        script = os.path.join(TOOLS, "ped_assemble.py")
        hinge_i = order.index("Hinge")
        guide_i = order.index("Guide")
        root_i = order.index("Root")
        manifest = {
            "coord_space": "y-up",
            "skeleton": self._hinge_skeleton_dict(order),
            "parts": [{
                "name": part_name,
                "coord_space": "y-up",
                "positions": [
                    [0.75, 0.5, 0.0],
                    [0.25, 1.0, 0.0],
                    [0.25, 0.5, 0.25],
                ],
                "triangles": [[0, 1, 2]],
                "joints_0": [
                    [hinge_i, 0, 0, 0],
                    [hinge_i, guide_i, 0, 0],
                    [root_i, 0, 0, 0],
                ],
                "weights_0": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.75, 0.25, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ],
            }],
        }
        mpath = os.path.join(tmp, f"{part_name}_src.json")
        out_glb = os.path.join(tmp, f"{part_name}.glb")
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(manifest, f)
        proc = subprocess.run(
            [
                sys.executable, "-I", "-S", script, "assemble",
                "--source-space", "y-up", "--no-cull", "--no-standing-offset",
                "--manifest", mpath, "-o", out_glb,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
        return out_glb

    def test_swapped_sibling_joint_orders_remap_both_master_orders_and_skin_motion(self):
        script = os.path.join(TOOLS, "ped_assemble.py")
        q_hinge_90z = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))

        with tempfile.TemporaryDirectory() as tmp:
            glb_a = self._build_hinge_source_glb(tmp, "part_a", ("Root", "Hinge", "Guide"))
            glb_b = self._build_hinge_source_glb(tmp, "part_b", ("Root", "Guide", "Hinge"))

            # Verify read_glb_parts retains each input GLB's own source skin joint names.
            parts_a = pa.read_glb_parts(glb_a)
            parts_b = pa.read_glb_parts(glb_b)
            self.assertEqual(parts_a[0].joint_names, ["Root", "Hinge", "Guide"])
            self.assertEqual(parts_b[0].joint_names, ["Root", "Guide", "Hinge"])

            for master_order in (("Root", "Hinge", "Guide"), ("Root", "Guide", "Hinge")):
                for standing_enabled in (False, True):
                    tag = f"{'_'.join(master_order)}_{'standing' if standing_enabled else 'bind'}"
                    master_manifest = {
                        "coord_space": "y-up",
                        "skeleton": self._hinge_skeleton_dict(master_order),
                        "parts": [glb_a, glb_b],
                    }
                    mpath = os.path.join(tmp, f"master_{tag}.json")
                    out_glb = os.path.join(tmp, f"assembled_{tag}.glb")
                    with open(mpath, "w", encoding="utf-8") as f:
                        json.dump(master_manifest, f)

                    cmd = [
                        sys.executable, "-I", "-S", script, "assemble",
                        "--source-space", "y-up", "--no-cull",
                        "--manifest", mpath, "-o", out_glb,
                    ]
                    if not standing_enabled:
                        cmd.append("--no-standing-offset")
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)

                    # Strict glb_skinning validation must pass.
                    val = subprocess.run(
                        [sys.executable, "-I", os.path.join(TOOLS, "glb_skinning.py"), out_glb],
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    self.assertEqual(val.returncode, 0, val.stderr + val.stdout)

                    # Parse back and verify EVERY vertex's joint slots in BOTH parts!
                    assembled = {p.name: p for p in pa.read_glb_parts(out_glb)}
                    self.assertEqual(set(assembled.keys()), {"part_a", "part_b"})
                    dst_root = master_order.index("Root")
                    dst_hinge = master_order.index("Hinge")
                    dst_guide = master_order.index("Guide")

                    for pname in ("part_a", "part_b"):
                        p = assembled[pname]
                        self.assertEqual(p.joint_names, list(master_order))
                        self.assertEqual(p.joints_0[0][0], dst_hinge)
                        self.assertEqual(p.joints_0[1][0], dst_hinge)
                        self.assertEqual(p.joints_0[1][1], dst_guide)
                        self.assertEqual(p.joints_0[2][0], dst_root)

                    # Evaluate 90-degree Hinge rotation in Y-up and verify world positions within 1e-5 m!
                    y_off = 1.0 if standing_enabled else 0.0
                    posed = _evaluate_glb_skinned_vertices(out_glb, {"Hinge": q_hinge_90z})
                    expected = [
                        (0.25, 1.0 + y_off, 0.0),
                        (-0.125, 0.625 + y_off, 0.0),
                        (0.25, 0.5 + y_off, 0.25),
                    ]
                    for pname in ("part_a", "part_b"):
                        for vi, exp in enumerate(expected):
                            got = posed[pname][vi]
                            err = math.dist(got, exp)
                            self.assertLess(
                                err, 1e-5,
                                f"{tag} {pname} v{vi}: got {got}, expected {exp} (err={err})",
                            )

    def test_eight_influence_remapping_survives_overlap_culling_and_compaction(self):
        # Upper and lower skinned parts share one coplanar triangle (culled from upper),
        # with swapped joint orders and active weights in both WEIGHTS_0 and WEIGHTS_1,
        # plus a zero-weight slot pointing at a joint absent from the master skeleton.
        master_skel = pa.SkeletonSpec(
            name="master",
            coord_space="y-up",
            joints=[
                pa.SkeletonJoint(name="Root", parent=-1),
                pa.SkeletonJoint(name="Spine", parent=0),
                pa.SkeletonJoint(name="Pelvis", parent=0),
                pa.SkeletonJoint(name="Chest", parent=1),
            ],
        )
        upper = pa.MeshPart(
            name="hand_fr1_000_nude_cull",
            coord_space="y-up",
            positions=[
                # tri 0 (coplanar overlap with lower -> will be culled)
                (0.0, 0.9, 0.0), (0.2, 0.9, 0.0), (0.0, 1.0, 0.0),
                # tri 1 (kept upper torso triangle)
                (0.0, 1.1, 0.0), (0.2, 1.1, 0.0), (0.0, 1.3, 0.0),
            ],
            triangles=[(0, 1, 2), (3, 4, 5)],
            # Swapped order + extra unweighted source joint "HelperUnused" at slot 4
            joint_names=["Root", "Chest", "Pelvis", "Spine", "HelperUnused"],
            joints_0=[(1, 3, 2, 0)] * 6,
            weights_0=[(0.4, 0.3, 0.1, 0.0)] * 6,
            joints_1=[(2, 4, 0, 0)] * 6,
            weights_1=[(0.2, 0.0, 0.0, 0.0)] * 6,
        )
        lower = pa.MeshPart(
            name="feet_fr1_000_nude_cull",
            coord_space="y-up",
            positions=[(0.0, 0.9, 0.0), (0.2, 0.9, 0.0), (0.0, 1.0, 0.0)],
            triangles=[(0, 1, 2)],
            joint_names=["Root", "Pelvis", "Spine"],
            joints_0=[(1, 2, 0, 0)] * 3,
            weights_0=[(0.8, 0.2, 0.0, 0.0)] * 3,
        )

        culled, reports = pa.apply_standard_ped_culling([upper, lower], compact_vertices=True)
        self.assertEqual(reports[0]["culled_total"], 1)
        self.assertEqual(len(culled[0].positions), 3)
        self.assertEqual(culled[0].joint_names, upper.joint_names)

        with tempfile.TemporaryDirectory() as tmp:
            out_glb = os.path.join(tmp, "culled_skinned.glb")
            pa.write_assembled_glb(out_glb, culled, skeleton=master_skel, apply_standing=False)
            val = subprocess.run(
                [sys.executable, "-I", os.path.join(TOOLS, "glb_skinning.py"), out_glb],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(val.returncode, 0, val.stderr + val.stdout)

            reloaded = {p.name: p for p in pa.read_glb_parts(out_glb)}
            u = reloaded["hand_fr1_000_nude_cull"]
            l = reloaded["feet_fr1_000_nude_cull"]
            # Master order: Root=0, Spine=1, Pelvis=2, Chest=3
            # Upper src (Chest=1, Spine=3, Pelvis=2, Root=0) -> (3, 1, 2, 0)
            self.assertEqual(u.joints_0[0], (3, 1, 2, 0))
            # Upper joints_1 src (Pelvis=2 [w=0.2], HelperUnused=4 [w=0.0], Root=0, Root=0) -> (2, 0, 0, 0)
            self.assertEqual(u.joints_1[0], (2, 0, 0, 0))
            # Lower src (Pelvis=1, Spine=2, Root=0) -> (2, 1, 0, 0)
            self.assertEqual(l.joints_0[0], (2, 1, 0, 0))

    def test_multi_skin_glb_and_ambiguous_or_skinless_glb_errors(self):
        master_skel = pa.SkeletonSpec(
            name="master",
            coord_space="y-up",
            joints=[
                pa.SkeletonJoint(name="Root", parent=-1),
                pa.SkeletonJoint(name="Hinge", parent=0),
                pa.SkeletonJoint(name="Guide", parent=0),
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            base_glb = self._build_hinge_source_glb(tmp, "part_a", ("Root", "Hinge", "Guide"))

            # 1. Single GLB with two nodes referencing the SAME mesh under TWO different skins
            multi_skin_glb = os.path.join(tmp, "multi_skin.glb")

            def _add_second_skin(doc: dict) -> None:
                skin0 = doc["skins"][0]
                j0 = skin0["joints"]  # [Root, Hinge, Guide]
                doc["skins"].append({
                    "name": "swapped_skin",
                    "joints": [j0[0], j0[2], j0[1]],  # [Root, Guide, Hinge]
                    "inverseBindMatrices": skin0["inverseBindMatrices"],
                })
                mesh_node = next(n for n in doc["nodes"] if "mesh" in n)
                new_node_idx = len(doc["nodes"])
                doc["nodes"].append({
                    "name": "part_swapped",
                    "mesh": mesh_node["mesh"],
                    "skin": 1,
                })
                doc["scenes"][0]["nodes"].append(new_node_idx)

            _repack_glb_with_mutated_doc(base_glb, multi_skin_glb, _add_second_skin)
            multi_parts = {p.name: p for p in pa.read_glb_parts(multi_skin_glb)}
            self.assertEqual(multi_parts["part_a"].joint_names, ["Root", "Hinge", "Guide"])
            self.assertEqual(multi_parts["part_swapped"].joint_names, ["Root", "Guide", "Hinge"])

            # 2. Ambiguous multi-skin GLB: skinned mesh node omits "skin" while 2 skins exist
            ambig_glb = os.path.join(tmp, "ambiguous_skin.glb")

            def _make_ambiguous(doc: dict) -> None:
                _add_second_skin(doc)
                mesh_node = next(n for n in doc["nodes"] if n.get("name") == "part_a")
                del mesh_node["skin"]

            _repack_glb_with_mutated_doc(base_glb, ambig_glb, _make_ambiguous)
            with self.assertRaises(ValueError):
                pa.read_glb_parts(ambig_glb)

            # 3. Skinless GLB with JOINTS_0 cannot silently bind to a master skeleton
            skinless_glb = os.path.join(tmp, "skinless.glb")

            def _strip_skins(doc: dict) -> None:
                doc.pop("skins", None)
                for n in doc["nodes"]:
                    n.pop("skin", None)

            _repack_glb_with_mutated_doc(base_glb, skinless_glb, _strip_skins)
            skinless_parts = pa.read_glb_parts(skinless_glb)
            self.assertEqual(skinless_parts[0].joint_names, [])
            with self.assertRaises(ValueError):
                pa.build_assembled_glb(skinless_parts, skeleton=master_skel, apply_standing=False)

    def test_joint_remapping_error_cases(self):
        master_skel = pa.SkeletonSpec(
            name="master",
            coord_space="y-up",
            joints=[
                pa.SkeletonJoint(name="Root", parent=-1),
                pa.SkeletonJoint(name="Hinge", parent=0),
                pa.SkeletonJoint(name="Guide", parent=0),
            ],
        )
        base_positions = [(0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.2, 0.0)]
        base_tris = [(0, 1, 2)]

        # 1. Weighted source joint missing from master skeleton
        missing_part = pa.MeshPart(
            name="missing_joint_part",
            coord_space="y-up",
            positions=base_positions,
            triangles=base_tris,
            joint_names=["Root", "GhostJoint", "Guide"],
            joints_0=[(1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)],
            weights_0=[(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)],
        )
        with self.assertRaisesRegex(ValueError, "GhostJoint"):
            pa.build_assembled_glb([missing_part], skeleton=master_skel, apply_standing=False)

        # 2. Duplicate/ambiguous joint name in source skin (weighted)
        dup_src_part = pa.MeshPart(
            name="dup_src_part",
            coord_space="y-up",
            positions=base_positions,
            triangles=base_tris,
            joint_names=["Root", "Hinge", "Hinge"],
            joints_0=[(1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)],
            weights_0=[(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)],
        )
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            pa.build_assembled_glb([dup_src_part], skeleton=master_skel, apply_standing=False)

        # 3. Duplicate/ambiguous joint name in master skeleton (weighted)
        dup_master = pa.SkeletonSpec(
            name="dup_master",
            coord_space="y-up",
            joints=[
                pa.SkeletonJoint(name="Root", parent=-1),
                pa.SkeletonJoint(name="Hinge", parent=0),
                pa.SkeletonJoint(name="Hinge", parent=0),
            ],
        )
        good_src_part = pa.MeshPart(
            name="good_src",
            coord_space="y-up",
            positions=base_positions,
            triangles=base_tris,
            joint_names=["Root", "Hinge"],
            joints_0=[(1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)],
            weights_0=[(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)],
        )
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            pa.build_assembled_glb([good_src_part], skeleton=dup_master, apply_standing=False)

        # 4. Unnamed weighted source joint (both via API and via mutated GLB without node name)
        unnamed_part = pa.MeshPart(
            name="unnamed_src",
            coord_space="y-up",
            positions=base_positions,
            triangles=base_tris,
            joint_names=["Root", "", "Guide"],
            joints_0=[(1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)],
            weights_0=[(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)],
        )
        with self.assertRaisesRegex(ValueError, "unnamed"):
            pa.build_assembled_glb([unnamed_part], skeleton=master_skel, apply_standing=False)

        with tempfile.TemporaryDirectory() as tmp:
            src_glb = self._build_hinge_source_glb(tmp, "part_u", ("Root", "Hinge", "Guide"))
            unnamed_glb = os.path.join(tmp, "unnamed_joint.glb")

            def _drop_hinge_name(doc: dict) -> None:
                hinge_node_idx = doc["skins"][0]["joints"][1]
                doc["nodes"][hinge_node_idx].pop("name", None)

            _repack_glb_with_mutated_doc(src_glb, unnamed_glb, _drop_hinge_name)
            loaded = pa.read_glb_parts(unnamed_glb)
            self.assertEqual(loaded[0].joint_names, ["Root", "", "Guide"])
            with self.assertRaisesRegex(ValueError, "unnamed"):
                pa.build_assembled_glb(loaded, skeleton=master_skel, apply_standing=False)

            # Also verify CLI returns clean exit code 2 with ERROR: on stderr
            master_manifest = {
                "coord_space": "y-up",
                "skeleton": self._hinge_skeleton_dict(("Root", "Hinge", "Guide")),
                "parts": [unnamed_glb],
            }
            mpath = os.path.join(tmp, "bad_master.json")
            out_glb = os.path.join(tmp, "bad_out.glb")
            with open(mpath, "w", encoding="utf-8") as f:
                json.dump(master_manifest, f)
            proc = subprocess.run(
                [
                    sys.executable, "-I", "-S", os.path.join(TOOLS, "ped_assemble.py"),
                    "assemble", "--source-space", "y-up", "--no-cull",
                    "--manifest", mpath, "-o", out_glb,
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(proc.returncode, 2)
            self.assertIn("ERROR:", proc.stderr)
            self.assertIn("unnamed", proc.stderr)

        # 5. Out-of-range weighted joint index (raw part without joint_names)
        oob_raw_part = pa.MeshPart(
            name="oob_raw",
            coord_space="y-up",
            positions=base_positions,
            triangles=base_tris,
            joints_0=[(9, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)],
            weights_0=[(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)],
        )
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            pa.build_assembled_glb([oob_raw_part], skeleton=master_skel, apply_standing=False)


if __name__ == "__main__":
    unittest.main()
