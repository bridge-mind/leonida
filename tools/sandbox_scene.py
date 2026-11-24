#!/usr/bin/env python3
"""Pack the placed props around one spot into a scene file for runtime/test/vk_sandbox.

Reads the same inputs as ``world_build.py --placements`` (``world_index.py --entities-out`` rows, the
``rpf9dump models`` manifests and the ``rpf9dump ttpf`` texture manifests), keeps the entities within
--radius metres of --center (RAGE world x y, Z-up), and writes one little-endian file:

    "LSB1", u32 version 1, u32 textures, u32 meshes, u32 draws, f32 center[3]
    textures: u32 w, u32 h, w*h RGBA8 (the material's DiffuseTex, mip 0 box-scaled to --tex-size)
    meshes:   u32 vertex count, u32 index count, vertices (f32 pos[3], f32 normal[3], f32 uv[2]), u32 indices
    draws:    u32 mesh, i32 texture (-1: none), f32 rows[4][3]

Positions stay in model space; a draw's rows are the entity's world transform as three basis rows plus the
translation, world = x*row0 + y*row1 + z*row2 + row3 (RAGE Z-up). The file holds decoded game data, so it
stays in the user's work folder. Standard library only.
Usage: `python -I tools/sandbox_scene.py --help` lists the options.
"""
import argparse
import math
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dds_png  # noqa: E402
import glb_writer  # noqa: E402
import world_build  # noqa: E402

DIFFUSE_TEX = "C8E8C282"   # joaat("DiffuseTex"), the albedo Param of a material texture reference
# Two-layer blend materials have no DiffuseTex; their first layer's albedo sits in this slot (name not known yet;
# beside it are the second layer's albedo, two normal maps, packed material maps and the blend mask).
LAYER0_ALBEDO = "346394EA"


def select_rows(rows, center, radius, limit):
    """Rows within radius of center (x, y), nearest first, at most limit."""
    cx, cy = center
    near = []
    for row in rows:
        dx, dy = row["_pos"][0] - cx, row["_pos"][1] - cy
        d = math.hypot(dx, dy)
        if d <= radius:
            near.append(row)
    near.sort(key=lambda row: math.hypot(row["_pos"][0] - cx, row["_pos"][1] - cy))
    return near if limit is None or limit >= len(near) else near[:limit]


def draw_rows(rotation, scale_xy, scale_z, position):
    """The VS's world rows (basis vectors then translation) for one entity."""
    m, _, _ = world_build.entity_transform(rotation, scale_xy, scale_z)
    rows = [[m[i][j] for i in range(3)] for j in range(3)]   # column j of M is the image of model axis j
    rows.append([float(v) for v in position])
    return rows


def mesh_bytes(prim):
    pos, nrm, uvs, idx = prim["positions"], prim.get("normals") or [], prim.get("uvs") or [], prim["indices"]
    out = bytearray(struct.pack("<II", len(pos), len(idx)))
    for i, p in enumerate(pos):
        n = nrm[i] if i < len(nrm) else (0.0, 0.0, 1.0)
        uv = uvs[i] if i < len(uvs) else (0.0, 0.0)
        out += struct.pack("<8f", p[0], p[1], p[2], n[0], n[1], n[2], uv[0], uv[1])
    out += struct.pack(f"<{len(idx)}I", *idx)
    return bytes(out)


def texture_bytes(dds_path, max_size):
    try:
        with open(dds_path, "rb") as f:
            w, h, rgba = dds_png.decode_dds(f.read())
        w, h, rgba = dds_png.downscale_rgba(w, h, rgba, max_size)
    except (dds_png.DdsError, OSError, ValueError):
        return None
    return struct.pack("<II", w, h) + bytes(rgba)


def build(args):
    rows, counts = world_build.load_placements(args.entities)
    entries = []
    for path in world_build.discover_model_manifests(args.models):
        data = world_build._load_json_dict(path) or {}
        entries += [(os.path.dirname(path), it) for it in data.get("Items") or [] if isinstance(it, dict)]
    index = world_build.model_name_index(entries, world_build._get_world_names())
    ttpf = {}
    for root in args.textures:
        for k, v in world_build.index_ttpf_textures(root).items():
            ttpf.setdefault(k, v)
    if args.center is None:   # the 50 m cell holding the most entities that resolve to a model
        cells = {}
        for row in rows:
            model, _ = world_build.resolve_placement(index, row["_arch"], row.get("key"))
            if model is not None:
                key = (math.floor(row["_pos"][0] / 50), math.floor(row["_pos"][1] / 50))
                cells[key] = cells.get(key, 0) + 1
        if not cells:
            raise SystemExit("no entity resolves to an exported model")
        best = max(cells, key=cells.get)
        args.center = [(best[0] + 0.5) * 50, (best[1] + 0.5) * 50]
    # Missing models or empty geometry must not exhaust a positive placement budget.
    candidate_limit = len(rows) if args.max_props > 0 else 0
    picked = select_rows(rows, args.center, args.radius, candidate_limit)

    textures, tex_ids = [], {}
    meshes, mesh_ids = [], {}
    draws = []
    stats = {"entities": len(picked), "placed": 0, "unresolved": 0, "textured_draws": 0, "z": []}
    for row in picked:
        model, _ = world_build.resolve_placement(index, row["_arch"], row.get("key"))
        if model is None:
            stats["unresolved"] += 1
            continue
        models_dir, item = model
        obj = os.path.join(models_dir, item.get("File") or "")
        key = (os.path.normcase(obj), id(item))   # archetypes sharing an OBJ can differ in materials
        if key not in mesh_ids:
            try:
                prims = glb_writer.parse_obj(obj, is_path=True, up_axis="y")
            except (OSError, glb_writer.GlbError, ValueError):
                prims = []
            tex_map = world_build.resolve_model_textures(models_dir, item, ttpf, LAYER0_ALBEDO)
            tex_map.update(world_build.resolve_model_textures(models_dir, item, ttpf, DIFFUSE_TEX))
            shaders = world_build._list_field(item, "MeshShaders")
            ids = []
            for p in prims:
                if not p.get("positions") or not p.get("indices"):
                    continue
                g = p.get("group_index", 0)
                sidx = shaders[g] if 0 <= g < len(shaders) else g
                tex = -1
                dds = tex_map.get(sidx)
                if dds:
                    dkey = os.path.normcase(dds)
                    if dkey not in tex_ids:
                        blob = texture_bytes(dds, args.tex_size)
                        tex_ids[dkey] = len(textures) if blob else -1
                        if blob:
                            textures.append(blob)
                    tex = tex_ids[dkey]
                ids.append((len(meshes), tex))
                meshes.append(mesh_bytes(p))
            mesh_ids[key] = ids
        if not mesh_ids[key]:
            stats["unresolved"] += 1
            continue
        rws = draw_rows(row["_rot"], row["_sxy"], row["_sz"], row["_pos"])
        for mesh, tex in mesh_ids[key]:
            draws.append(struct.pack("<Ii12f", mesh, tex, *[v for r in rws for v in r]))
            stats["textured_draws"] += tex >= 0
        stats["placed"] += 1
        stats["z"].append(row["_pos"][2])
        if stats["placed"] >= args.max_props:
            break
    z = sorted(stats.pop("z")) or [0.0]
    center = [args.center[0], args.center[1], z[len(z) // 2]]
    with open(args.out, "wb") as f:
        f.write(b"LSB1" + struct.pack("<4I3f", 1, len(textures), len(meshes), len(draws), *center))
        for blob in textures + meshes + draws:
            f.write(blob)
    stats.update(textures=len(textures), meshes=len(meshes), draws=len(draws), center=center,
                 selected_rows=counts["selected"])
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--entities", required=True, help="world_index.py --entities-out file (JSON lines)")
    ap.add_argument("--models", required=True, help="folder holding rpf9dump models manifests")
    ap.add_argument("--textures", action="append", default=[], help="folder holding rpf9dump ttpf manifests (repeatable)")
    ap.add_argument("--center", type=float, nargs=2, metavar=("X", "Y"),
                    help="RAGE world x y (default: densest 50 m cell of resolved model placements)")
    ap.add_argument("--radius", type=float, default=150.0)
    ap.add_argument("--max-props", type=int, default=400, help="entities to place, nearest first (each can be several draws)")
    ap.add_argument("--tex-size", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if not math.isfinite(args.radius) or args.radius < 0:
        ap.error("--radius must be finite and non-negative")
    if args.max_props < 0:
        ap.error("--max-props must be non-negative")
    if os.path.isdir(args.out):
        print(f"ERROR: --out is a folder, not a scene file: {args.out}", file=sys.stderr)
        return 2
    try:
        stats = build(args)
    except OSError as e:
        # An unreadable --entities/--models or an unwritable --out used to end in a traceback.
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print(" ".join(f"{k}={v}" for k, v in stats.items()))
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
