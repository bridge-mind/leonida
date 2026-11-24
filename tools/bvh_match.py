"""Find the drawable geometry a dehydrated BLAS was built from, over `rpf9dump`'s model export.

A dehydrated BLAS (docs/bvh.md §3.1, §4a) keeps, for every leaf, the index of the geometry it came from and the
primitive it uses. That index counts the geometries of one model in the drawable's primary LOD, leaving out the
geometries whose shader the BVH does not keep. So a BLAS is a triangle count per geometry index, and the drawable it
came from has exactly those counts among the geometries of one of its models:

1. read that signature from the BLAS leaves (a leaf holding two primitives counts twice);
2. keep every (drawable, model) of a `rpf9dump models` manifest whose per-geometry triangle counts contain the
   signature as an ordered subsequence - the geometries a model uses are `MeshTris`, grouped by `MeshModels`;
3. for each survivor, read the geometries it points at from the model's OBJ (positions are the decoder's exact
   float32, faces are in primitive order) and check the rehydration against what the BLAS stores with
   tools/bvh_geometry_check.py (bounds, overrides, paired-leaf edges, internal area).

Which geometries the BVH leaves out follows from their material's render bucket (docs/bvh.md §4a, ASSET-349): the
byte at +57 of the shader record, which `rpf9dump models` writes as `RenderBucket`. A geometry is kept when the low 7
bits of that byte are 0 or 3 and left out otherwise (KEPT_BUCKETS; bit 7 is ignored). When a geometry of a
model has no bucket in the manifest (its shader index is not in the material list, or an older export without the
field) the rule cannot be applied to that model and every geometry is a candidate; the checks then decide.
--no-bucket-rule turns it off. --skip-shaders (or BVH_SKIP_SHADERS) still leaves out geometries by shader name hash
on top of it; the hashes are game data, for local use only.

The instance transform is the 3x4 at +0x50 of the BLAS's 208-byte instance descriptor (docs/bvh.md §3.2). Without
it, every leaf of a rotated structure falls outside the header bounds and a correct geometry is rejected.

The manifest is one window of an archive: `rpf9dump models <slice> <out>` exports every drawable it recovers from
that slice, and a slice holds one region's. A BLAS whose drawable sits elsewhere in the archive therefore finds no
candidate in it, which is not the same as a geometry that fails the checks; the tool says which case it is.

Standard library only. Reads the user's own package and writes nothing by default. Exit status is 0 when a
candidate is consistent, 1 when none is, and 2 on an input error.

Usage:
    python -I tools/bvh_match.py <dehydrated.bin> <manifest.json> --descriptor <instance208.bin>
                                [--models DIR] [--header-offset N] [--descriptor-offset N]
                                [--transform 12 floats] [--skip-shaders HEX,HEX] [--no-bucket-rule] [--unordered]
                                [--embeddings N] [--top N] [--names] [--write DIR] [--json FILE]

--write DIR writes the matched geometries as float3/uint32 buffers plus an export.json, the shape
tools/bvh_rehydrate.py takes: the transform is recorded in it rather than applied to the vertices.
"""
import argparse
import json
import os
import struct
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bvh_geometry_check as C  # noqa: E402
import bvh_rehydrate as R  # noqa: E402
from report_paths import same_report_path  # noqa: E402

INSTANCE_TRANSFORM_OFFSET = 0x50
SKIP_SHADERS_ENV = "BVH_SKIP_SHADERS"
# Render buckets (low 7 bits of the byte at shader record +57, `RenderBucket` in the manifest) whose geometries the
# BVH keeps; every other bucket (1, 2, 4, 5, 6) is left out. Bit 7 does not change the choice (docs/bvh.md §4a,
# ASSET-349).
KEPT_BUCKETS = frozenset((0, 3))
BUCKET_MASK = 0x7F


def bucket_kept(bucket):
    """True when the BVH keeps a geometry whose material has this render bucket, None when the bucket is unknown."""
    return None if bucket is None else (bucket & BUCKET_MASK) in KEPT_BUCKETS


def leaf_signature(src, header_offset=0):
    """Triangles per leaf geometry index; a leaf that holds two primitives counts as two triangles."""
    hdr = R.Header(src, header_offset)
    body = bytes(src[header_offset:header_offset + hdr.total_bytes])   # the leaf arrays count from the header
    per = {}
    for _, geometry, _, _, prim1, _ in R.leaf_records(body, hdr):
        per[geometry] = per.get(geometry, 0) + (1 if prim1 is None else 2)
    return [per.get(i, 0) for i in range(max(per) + 1)] if per else []


def parse_shader_hashes(text):
    """'0x1234abcd,89abcdef' -> frozenset of ints; empty or None -> empty set (the rule is not applied)."""
    return frozenset(int(x, 16) for x in (text or "").replace(" ", "").split(",") if x)


def read_transform(path, at=0):
    """The 3x4 at +0x50 of a 208-byte instance descriptor, as three rows of four floats."""
    with open(path, "rb") as f:
        f.seek(at + INSTANCE_TRANSFORM_OFFSET)
        raw = f.read(48)
    if len(raw) != 48:
        raise ValueError("%s is shorter than a 208-byte instance descriptor at %d" % (path, at))
    values = struct.unpack("<12f", raw)
    return tuple(values[4 * row:4 * row + 4] for row in range(3))


def embeddings(signature, counts, cap, ordered=True):
    """Every way to pick distinct geometries of `counts` matching `signature` (index lists), at most `cap` of them.

    ordered: the picked indices must increase (a prefix is the common case); otherwise any order is allowed.
    """
    if cap <= 0 or len(signature) > len(counts):
        return []

    latest = [len(counts) - 1] * len(signature)
    if ordered:
        # Match the signature backwards once. An index later than latest[i] leaves
        # too few matching geometries for the rest of the signature.
        k = len(counts) - 1
        for i in reversed(range(len(signature))):
            while k >= 0 and counts[k] != signature[i]:
                k -= 1
            if k < 0:
                return []
            latest[i] = k
            k -= 1
    elif Counter(signature) - Counter(counts):
        # In any order, each triangle count just needs enough distinct geometries.
        return []

    out = []

    def walk(i, j, acc):
        if i == len(signature):
            out.append(list(acc))
            return
        for k in range(j, latest[i] + 1):
            if counts[k] == signature[i] and (ordered or k not in acc):
                acc.append(k)
                walk(i + 1, k + 1 if ordered else 0, acc)
                acc.pop()
                if len(out) == cap:
                    return

    walk(0, 0, [])
    return out


def model_geometries(item):
    """[(mesh index, model, geometry, triangles, shader hash or None, render bucket or None)] for one manifest item,
    in mesh order.

    The mesh index is the geometry's `o` group in the item's OBJ: the manifest arrays and the OBJ groups are
    written in the same order.
    """
    tris = item.get("MeshTris")
    if tris is None:
        raise ValueError("the manifest has no MeshTris: re-export with an rpf9dump that writes per-geometry counts")
    models = item.get("MeshModels") or [0] * len(tris)
    geometries = item.get("MeshGeoms") or list(range(len(tris)))
    shaders = item.get("MeshShaders") or [None] * len(tris)
    materials = item.get("Materials") or []
    hashes, buckets = [], []
    for index in shaders:
        material = materials[index] if index is not None and 0 <= index < len(materials) else {}
        hashes.append(int(material["Shader"], 16) if "Shader" in material else None)
        bucket = material.get("RenderBucket")
        buckets.append(bucket if isinstance(bucket, int) and not isinstance(bucket, bool) else None)
    return list(zip(range(len(tris)), models, geometries, tris, hashes, buckets))


def match_candidates(items, signature, skip, cap, ordered=True, buckets=True):
    """Every (item, model, used geometry indices, kept geometries) whose triangle counts contain the signature.

    buckets: leave out the geometries whose render bucket the BVH does not keep (bucket_kept), in every model whose
    geometries all have a bucket. skip: shader hashes left out as well, in every model whose geometries all have one.
    """
    out = []
    for item in items:
        meshes = model_geometries(item)
        for model in sorted({mesh[1] for mesh in meshes}):
            group = [mesh for mesh in meshes if mesh[1] == model]
            # A geometry the BVH leaves out is not in the BLAS, so it takes no index in the signature. Without the
            # bucket (or, for --skip-shaders, the hash) of every geometry of the group a rule cannot be applied to
            # it: keep all of them and let the checks decide.
            by_bucket = buckets and all(mesh[5] is not None for mesh in group)
            rule = bool(skip) and all(mesh[4] is not None for mesh in group)
            kept = [mesh for mesh in group
                    if (not by_bucket or bucket_kept(mesh[5])) and (not rule or mesh[4] not in skip)]
            for used in embeddings(signature, [mesh[3] for mesh in kept], cap, ordered):
                out.append({"item": item, "model": model, "used": used, "kept": kept,
                            "rule": rule, "skipped": bool(skip), "by_bucket": by_bucket, "buckets": buckets})
    return out


def read_groups(path):
    """[(positions, faces)] per `o` group of an OBJ, in file order; faces are 0-based indices into that group.

    tools/rpf9dump reads `RageModelDecoder.CreateObj` output: one group per geometry, its vertices in buffer order
    and its faces in primitive order (docs/bvh.md §4a: the leaves address triangles in primitive order).
    """
    groups, positions, faces, base, started = [], [], [], 0, False

    def flush():
        nonlocal positions, faces, base, started
        if started or positions or faces:
            groups.append((positions, faces))
            base += len(positions)
        positions, faces, started = [], [], False

    with open(path, "rb") as f:
        for raw in f:
            line = raw.decode("ascii", "replace").strip()
            if len(line) < 2:
                continue
            tag, _, rest = line.partition(" ")
            if tag == "o":
                flush()
                started = True
            elif tag == "v":
                x, y, z = rest.split()[:3]
                positions.append((float(x), float(y), float(z)))
            elif tag == "f":
                corners = rest.split()
                if len(corners) != 3:
                    raise ValueError("%s: a face with %d corners; only triangles are expected" % (path, len(corners)))
                faces.append(tuple(int(corner.split("/")[0]) - 1 - base for corner in corners))
    flush()
    return groups


def build_geometry(group, transform=None):
    """A bvh_rehydrate.Geometry from one OBJ group: float3 vertices, uint32 triangles, the instance transform."""
    positions, faces = group
    if not faces:
        raise ValueError("a geometry without triangles cannot be an index of a BLAS leaf")
    for face in faces:
        if min(face) < 0 or max(face) >= len(positions):
            raise ValueError("an OBJ face refers to a vertex outside its group")
    vb = b"".join(struct.pack("<3f", *p) for p in positions)
    ib = b"".join(struct.pack("<3I", *face) for face in faces)
    return R.Geometry(vb, 12, ib, False, transform)


def score(src, candidate, groups, transform, header_offset=0):
    """Rehydrate the candidate's geometries and run bvh_geometry_check over them."""
    used = [candidate["kept"][k] for k in candidate["used"]]
    for mesh in used:
        if mesh[0] >= len(groups):
            raise ValueError("the manifest lists %d geometries but its OBJ holds %d: re-export the window"
                             % (max(m[0] for m in used) + 1, len(groups)))
    geometries = [build_geometry(groups[mesh[0]], transform) for mesh in used]
    return C.check(src, geometries, header_offset)


def export_buffer_names(count):
    """The (vertex, index) buffer names write_export gives each of `count` matched geometries."""
    return [("g%d_f3.bin" % i, "g%d_tri.bin" % i) for i in range(count)]


def write_export(directory, candidate, groups, transform):
    """Write the matched geometries as float3/uint32 buffers plus the export.json tools/bvh_rehydrate.py is given.

    The vertices are the OBJ's float32 as they are, so the instance transform is recorded rather than applied: pass
    it to bvh_rehydrate.py with --transform, exactly as the descriptor gave it here.
    """
    os.makedirs(directory, exist_ok=True)
    rows = []
    names = export_buffer_names(len(candidate["used"]))
    for k in candidate["used"]:
        mesh = candidate["kept"][k]
        geometry = build_geometry(groups[mesh[0]])
        vb, ib = names[len(rows)]
        with open(os.path.join(directory, vb), "wb") as f:
            f.write(geometry.vertices)
        with open(os.path.join(directory, ib), "wb") as f:
            f.write(geometry.indices)
        rows.append({"index": len(rows), "model": candidate["model"], "drawable_geometry": mesh[2],
                     "vertices": len(geometry.vertices) // 12, "triangles": len(geometry.indices) // 12,
                     "vertex_format": 6, "index_format": 42, "stride": 12, "vb": vb, "ib": ib,
                     "instance_transform": False,
                     "transform": None if transform is None else [v for row in transform for v in row]})
    with open(os.path.join(directory, "export.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(rows, f, indent=1)
    return rows


def describe(candidate, names):
    item = candidate["item"]
    used = [candidate["kept"][k][2] for k in candidate["used"]]
    label = ("%-40s " % item["Name"]) if names else ""
    note = " (shader rule not applied)" if candidate["skipped"] and not candidate["rule"] else ""
    if candidate.get("buckets") and not candidate.get("by_bucket"):
        note += " (bucket rule not applied: a geometry has no render bucket)"
    return "%sstream 0x%-10x model %d geometry %s%s" % (label, item["Offset"], candidate["model"], used, note)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Find the drawable geometry a dehydrated BLAS was built from.")
    ap.add_argument("dehydrated")
    ap.add_argument("manifest")
    ap.add_argument("--models", help="directory of the OBJs (default: the manifest's directory)")
    ap.add_argument("--descriptor", help="208-byte instance descriptor; its 3x4 at +0x50 is the geometry transform")
    ap.add_argument("--descriptor-offset", type=int, default=0)
    ap.add_argument("--transform", help="the same 3x4 as 12 comma-separated floats, instead of --descriptor")
    ap.add_argument("--header-offset", type=int, default=0, help="the 64-byte header's offset in the file")
    ap.add_argument("--skip-shaders", default=os.environ.get(SKIP_SHADERS_ENV, ""),
                    help="shader name hashes of the geometries the BVH leaves out (game data; local use only)")
    ap.add_argument("--no-bucket-rule", action="store_true",
                    help="do not leave out geometries by render bucket (by default only buckets 0 and 3 are kept)")
    ap.add_argument("--unordered", action="store_true", help="also try the geometries in any order")
    ap.add_argument("--embeddings", type=int, default=64, help="at most this many index choices per model")
    ap.add_argument("--top", type=int, default=8, help="how many scored candidates to print")
    ap.add_argument("--names", action="store_true", help="print drawable names (game data: local use only)")
    ap.add_argument("--write", metavar="DIR", help="write the matched geometries here as float3/uint32 buffers "
                                                  "and an export.json for tools/bvh_rehydrate.py")
    ap.add_argument("--json", help="write the report here")
    a = ap.parse_args(argv)
    if a.json:
        if os.path.isdir(a.json):
            print("error: --json is a directory: %s" % a.json, file=sys.stderr)
            return 2
        # The report is written last, after the inputs have been read, so an aliased input would silently
        # be replaced by it. Refuse the alias before any input is opened, as the other report tools do.
        for what, other in (("the dehydrated input", a.dehydrated), ("the manifest", a.manifest),
                            ("the --descriptor input", a.descriptor)):
            if other and same_report_path(a.json, other):
                print("error: --json would overwrite %s: %s" % (what, other), file=sys.stderr)
                return 2
        if a.write and same_report_path(a.json, os.path.join(a.write, "export.json")):
            print("error: --json would overwrite --write's export.json: %s" % a.json, file=sys.stderr)
            return 2
    try:
        with open(a.dehydrated, "rb") as f:
            src = f.read()
        with open(a.manifest, encoding="utf-8") as f:
            manifest = json.load(f)
        if not isinstance(manifest, dict):
            raise ValueError("%s is not a JSON object (got %s)" % (a.manifest, type(manifest).__name__))
        items = manifest.get("Items") or []
        if not items:
            raise ValueError("%s has no Items: the export found no model in its window, so there is nothing to match "
                             "the BLAS against" % a.manifest)
        models_dir = a.models or os.path.dirname(os.path.abspath(a.manifest))
        transform = None
        if a.transform:
            transform = R.parse_transform(a.transform)
        elif a.descriptor:
            transform = read_transform(a.descriptor, a.descriptor_offset)
        signature = leaf_signature(src, a.header_offset)
        if not signature:
            raise ValueError("the BLAS has no leaves")
        candidates = match_candidates(items, signature, parse_shader_hashes(a.skip_shaders), a.embeddings,
                                      not a.unordered, not a.no_bucket_rule)
    except (OSError, ValueError, KeyError, IndexError, TypeError, struct.error) as e:
        print("error: %s" % e, file=sys.stderr)
        return 2
    if transform is None:
        print("no --descriptor or --transform: the geometry is checked without the instance transform", file=sys.stderr)
    print("triangles per geometry index: %s" % signature)
    print("%d candidates from %d models in the manifest" % (len(candidates), len(items)))
    if not candidates:
        # Distinguishing "the geometry is not in this export" from "the rule does not fit" saves the confusion of an
        # empty run: a window holds one region's drawables, and a BLAS whose drawable lives elsewhere cannot match.
        print("nothing to score: no model in this manifest has those triangle counts per geometry. Either the drawable "
              "is outside the exported window (the manifest's Scan.Start/Scan.Length, when present, says which window "
              "it is), or its geometry list does not follow the rule (docs/bvh.md §4a)")
    results, groups, errors = [], {}, 0
    for candidate in candidates:
        item = candidate["item"]
        path = os.path.join(models_dir, item["File"])
        try:
            if path not in groups:
                groups[path] = read_groups(path)
            report = score(src, candidate, groups[path], transform, a.header_offset)
        except (OSError, ValueError, KeyError, IndexError, TypeError, struct.error) as e:
            errors += 1
            print("   %s: %s" % (describe(candidate, a.names), e))
            continue
        results.append((not report["consistent"], report["internal_area"], report, candidate))
    results.sort(key=lambda r: (r[0], r[1]))
    best = next((r[1] for r in results if not r[0]), None)
    for bad, area, report, candidate in results[:max(1, a.top)]:
        ratio = "" if best is None or not best else " (x%.3f)" % (area / best)
        print("   %-10s %s | outside %d of %d, gap %.1e, overrides %d/%d, area %.6g%s" % (
            "consistent" if not bad else "rejected", describe(candidate, a.names), report["leaves_outside"], report["leaves"],
            report["bounds_gap"], report["overrides_consistent"], report["overrides_inconsistent"], area, ratio))
    consistent = [r for r in results if not r[0]]
    print("%d of %d scored candidates are consistent%s" % (len(consistent), len(results), "" if not errors else " (%d could not be read)" % errors))
    if len(consistent) > 1:
        print("more than one candidate is consistent: the signature and the checks do not decide between them; "
              "the internal box area (the smallest one) is the tie-break")
    if a.write:
        if not consistent:
            print("nothing written: no candidate is consistent")
        else:
            match = consistent[0][3]
            if a.json:
                # The report is written after the export, so it must not land on one of the buffers the
                # export is about to write either. Their names depend on the match, hence the check here.
                for vb, ib in export_buffer_names(len(match["used"])):
                    for name in (vb, ib):
                        if same_report_path(a.json, os.path.join(a.write, name)):
                            print("error: --json would overwrite --write's %s: %s" % (name, a.json), file=sys.stderr)
                            return 2
            rows = write_export(a.write, match, groups[os.path.join(models_dir, match["item"]["File"])], transform)
            print("wrote %d geometries to %s%s" % (len(rows), a.write,
                  "; pass the transform in export.json to bvh_rehydrate.py with --transform" if transform else ""))
    if a.json:
        match = consistent[0][3] if consistent else None
        try:
            with open(a.json, "w", encoding="utf-8", newline="\n") as f:
                json.dump({"signature": signature, "consistent": bool(consistent), "scored": len(results),
                           "matched": None if match is None else {
                               "offset": match["item"]["Offset"], "model": match["model"],
                               "geometry": [match["kept"][k][2] for k in match["used"]],
                               "triangles": [match["kept"][k][3] for k in match["used"]],
                               "name": match["item"]["Name"] if a.names else None}},
                          f, indent=1)
        except OSError as e:
            print("error: cannot write %s: %s" % (a.json, e.strerror or e), file=sys.stderr)
            return 2
    return 0 if consistent else 1


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
