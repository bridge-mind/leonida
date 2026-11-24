"""Compare multi-slot BVH transforms against drawable groups selected by topology.

Inputs are the core BVH archive, its generic archive index (files[].off/stored/size/head),
and one or more primary-LOD levels model manifests with their exported OBJs.
Use --save-evidence and --prior to combine private window scans without reopening earlier OBJs.
Only BLAS with at least 30 paired primitives enter the topology screen.
The candidate universe is the supplied exports. Stdout contains aggregate counts only.
Usage: `python -I tools/bvh_slot_check.py --help` lists the options.
"""
import argparse
import hashlib
import json
import math
import os
import struct
import sys
import tempfile
import zlib
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bvh_geometry_check as C
import bvh_index as I
import bvh_match as M
import bvh_rehydrate as R


# Bump when signature ordering, source matching or matrix verdict semantics change.
EVIDENCE_FORMAT = "bvh-slot-evidence-v1"
PAIR = struct.Struct("<IIB")  # first primitive, second primitive, fourth vertex index


def slots(descriptor, buffer):
    count, entries = struct.unpack_from("<QQ", descriptor, 16)
    entries -= 0x50000000
    if struct.unpack_from("<Q", descriptor, 56)[0] != 0x50000050:
        raise ValueError("invalid matrix pointer")
    if count < 2 or entries < 80 + 48 * count or entries + 8 * count > len(descriptor):
        raise ValueError("invalid instance slot table")
    matrices = []
    bodies = []
    pos = 0
    for i in range(count):
        matrix = tuple(struct.unpack_from("<4f", descriptor, 80 + 48 * i + 16 * r) for r in range(3))
        if not all(math.isfinite(value) for row in matrix for value in row):
            raise ValueError("non-finite matrix")
        matrices.append(matrix)
        size, = struct.unpack_from("<I", descriptor, entries + 8 * i)
        if not size:
            bodies.append(None)
            continue
        block = buffer[pos:pos + size]
        header = I.classify(block[:I.HEAD], len(block))
        if header["kind"] != "bvh4" or len(block) != size or (header["declared_bytes"] + 255) // 256 * 256 != size:
            raise ValueError("slot size disagrees with BLAS wrapper")
        bodies.append(block[256:header["declared_bytes"]])
        pos += size
    if any(buffer[pos:]):
        raise ValueError("unexpected data after last slot")
    return matrices, bodies


def compare(body, geometries, matrices, slot):
    passed = []
    for matrix in dict.fromkeys(matrices):
        transformed = [R.Geometry(g.vertices, g.stride, g.indices, g.index16, matrix) for g in geometries]
        if C.check(body, transformed)["consistent"]:
            passed.append(matrix)
    if not passed:
        return "unknown_no_matrix_passed"
    if matrices[slot] not in passed:
        return "alternative_only"
    if len(passed) > 1 or matrices.count(passed[0]) > 1:
        return "non_discriminating"
    return "own_only"


def topology_matches(faces, pairs):
    """Require both the common edge and the encoded fourth vertex of every pair."""
    for first, second, fourth in pairs:
        a, b = faces[first], faces[second]
        if b[fourth] in a or len(set(a).intersection(b)) != 2:
            return False
    return True


def read_stream(f, row):
    f.seek(row["off"])
    data = zlib.decompress(f.read(row["stored"]))
    if len(data) != row["size"] or not data.startswith(bytes.fromhex(row["head"])):
        raise ValueError("index disagrees with archive")
    return data


def signature(records):
    counts = Counter()
    pairs = defaultdict(bytearray)
    largest = {}
    for _, gi, _, a, b, order in records:
        counts[gi] += 1
        largest[gi] = max(largest.get(gi, 0), a)
        if b is not None:
            counts[gi] += 1
            fourth = 0 if order & 0x20 else 1 if order & 0x80 else 2
            largest[gi] = max(largest[gi], b)
            pairs[gi].extend(PAIR.pack(a, b, fourth))
    if not counts or max(counts) >= len(counts):
        return None
    if any(largest[k] >= counts[k] for k in counts) or sum(map(len, pairs.values())) < 30 * PAIR.size:
        return None
    return tuple((counts[k], bytes(pairs[k])) for k in range(len(counts)))


def source_candidates(sig, meshes, groups):
    """Ordered embeddings in one drawable model, retaining only topology matches."""
    choices = []
    for n, pairs in sig:
        choices.append([mesh for mesh in meshes if len(groups[mesh][1]) == n
                        and topology_matches(groups[mesh][1], PAIR.iter_unpack(pairs))])

    # Remove choices that cannot be followed by a complete ordered suffix.
    for i in reversed(range(len(choices))):
        if not choices[i]:
            return
        if i:
            end = bisect_left(choices[i - 1], choices[i][-1])
            choices[i - 1] = choices[i - 1][:end]

    def walk(prefix):
        if len(prefix) == len(sig):
            yield prefix
            return
        for mesh in choices[len(prefix)]:
            if not prefix or mesh > prefix[-1]:
                yield from walk(prefix + (mesh,))

    yield from walk(())


def match_exports(queries, manifests, stats):
    # Only inspect signatures for which the model has the largest required geometry.
    by_count = defaultdict(list)
    for sig in queries:
        by_count[max(n for n, _ in sig)].append(sig)
    matches = {}
    for manifest in manifests:
        items = json.loads(manifest.read_bytes())["Items"]
        stats["manifest_items"] += len(items)
        for item in items:
            models = defaultdict(list)
            for mesh, model, _, _, _ in M.model_geometries(item):
                models[model].append(mesh)
            groups = None
            buffers = {}
            for meshes in models.values():
                counts = Counter(item["MeshTris"][mesh] for mesh in meshes)
                for count in counts:
                    for sig in by_count[count]:
                        if sig in matches and matches[sig] is None:
                            continue
                        if Counter(n for n, _ in sig) - counts:
                            continue
                        if groups is None:
                            groups = M.read_groups(manifest.parent / item["File"])
                            if [len(faces) for _, faces in groups] != item["MeshTris"]:
                                raise ValueError("OBJ and manifest disagree on mesh triangle counts")
                        for selected in source_candidates(sig, meshes, groups):
                            for mesh in selected:
                                if mesh not in buffers:
                                    if not all(math.isfinite(x) for vertex in groups[mesh][0] for x in vertex):
                                        raise ValueError("non-finite source vertex")
                                    buffers[mesh] = M.build_geometry(groups[mesh])
                            geometry = tuple(buffers[mesh] for mesh in selected)
                            previous = matches.get(sig)
                            if previous and any(a.indices != b.indices or a.vertices != b.vertices
                                                for a, b in zip(previous, geometry)):
                                matches[sig] = None
                                break
                            matches[sig] = geometry
    return matches


def buffer_digest(geometries):
    digest = hashlib.sha256()
    for geometry in geometries:
        for buffer in (geometry.vertices, geometry.indices):
            digest.update(struct.pack("<Q", len(buffer)))
            digest.update(buffer)
    return digest.hexdigest()


def merge_evidence(matches, key, candidate):
    # Missing means no source. None means distinct buffers were seen and is permanent.
    if key not in matches:
        matches[key] = candidate
    elif matches[key] is None or candidate is None or matches[key][0] != candidate[0]:
        matches[key] = None
    elif matches[key][1] != candidate[1]:
        raise ValueError("same source and archive have conflicting saved verdicts")


def load_evidence(path, scope, locations_by_key, evidence):
    verdicts = {"own_only", "alternative_only", "non_discriminating", "unknown_no_matrix_passed"}
    state = json.loads(path.read_bytes())
    if state["format"] != EVIDENCE_FORMAT or state["scope"] != scope:
        raise ValueError("evidence version or archive/index differs")
    count = state["manifest_items"]
    if type(count) is not int or count < 0 or not isinstance(state["matches"], dict):
        raise ValueError("invalid evidence header")
    for key, candidate in state["matches"].items():
        k = int(key)
        if not 0 <= k < len(locations_by_key) or str(k) != key:
            raise ValueError("invalid evidence signature index")
        if candidate is not None:
            fingerprint, results = candidate
            if (not isinstance(results, list) or len(fingerprint) != 64
                    or any(c not in "0123456789abcdef" for c in fingerprint) or len(results) != len(locations_by_key[k])
                    or any(result not in verdicts for result in results)):
                raise ValueError("invalid evidence candidate")
        merge_evidence(evidence, k, candidate)
    return count


def save_evidence(path, record):
    f = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False)
    temporary = Path(f.name)
    try:
        with f:
            json.dump(record, f)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def census(archive, index, manifests, prior=(), save=None):
    if save and save.resolve() in {path.resolve() for path in (archive, index, *manifests)}:
        raise ValueError("evidence output would overwrite a census input")
    raw = index.read_bytes()
    digest = hashlib.sha256(raw) if prior or save else None
    catalog = json.loads(raw)
    del raw
    rows = catalog["files"]
    with archive.open("rb") as f:
        f.seek(4)
        start, = struct.unpack("<I", f.read(4))
    if catalog["summary"]["span"] != archive.stat().st_size - start or len(rows) != catalog["summary"]["files"]:
        raise ValueError("a full archive carve index is required")
    # Group BLAS with the same topology so the OBJ corpus is read only once.
    alternative_counts = {}
    queries = defaultdict(list)
    stats = Counter()
    with archive.open("rb") as f:
        for i, row in enumerate(rows):
            head = bytes.fromhex(row["head"])
            if row["size"] <= 208 or head[:8] != bytes(8):
                continue
            ptr, = struct.unpack_from("<Q", head, 8)
            if not 0x50000000 <= ptr < 0x51000000:
                continue
            desc = read_stream(f, row)
            if struct.unpack_from("<Q", desc, 32)[0] != 0x60000000:
                continue
            buffer = read_stream(f, rows[i + 1])
            if digest is not None:
                digest.update(desc)
                digest.update(buffer)
            matrices, bodies = slots(desc, buffer)
            del buffer
            alternative_counts[i] = len(set(matrices)) - 1
            stats["descriptors"] += 1
            for k, body in enumerate(bodies):
                if body is None:
                    stats["empty_slots"] += 1
                    continue
                stats["occupied_slots"] += 1
                if alternative_counts[i] == 0:
                    stats["non_discriminating_identical_matrices"] += 1
                    continue
                sig = signature(R.leaf_records(body, R.Header(body, 0)))
                if sig is None:
                    stats["unknown_insufficient_topology"] += 1
                    continue
                queries[sig].append((i, k))

    evidence = {}
    # Signature indices and location order follow the full carve index in both scans.
    locations_by_key = list(queries.values())
    scope = digest.hexdigest() if digest is not None else None
    for path in prior:
        stats["manifest_items"] += load_evidence(path, scope, locations_by_key, evidence)

    matches = match_exports(queries, manifests, stats)
    supporting_descriptors = set()
    with archive.open("rb") as f:
        for key, (sig, locations) in enumerate(queries.items()):
            if sig in matches:
                candidate = None
                geometry = matches[sig]
                if geometry is not None:
                    results = []
                    for i, k in locations:
                        matrices, bodies = slots(read_stream(f, rows[i]), read_stream(f, rows[i + 1]))
                        results.append(compare(bodies[k], geometry, matrices, k))
                    candidate = [buffer_digest(geometry) if digest is not None else "", results]
                merge_evidence(evidence, key, candidate)
            if evidence.get(key) is None:
                reason = "unknown_multiple_sources" if key in evidence else "unknown_no_source"
                stats[reason] += len(locations)
                continue
            for (i, k), outcome in zip(locations, evidence[key][1]):
                stats[outcome] += 1
                if outcome == "own_only":
                    supporting_descriptors.add(i)
                    stats["alternative_matrices_rejected_for_own_only"] += alternative_counts[i]
    if save:
        save_evidence(save, {"format": EVIDENCE_FORMAT, "scope": scope,
                             "manifest_items": stats["manifest_items"], "matches": evidence})
    stats["descriptors_with_own_only"] = len(supporting_descriptors)
    return dict(sorted(stats.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("archive", type=Path)
    ap.add_argument("index", type=Path)
    ap.add_argument("manifests", type=Path, nargs="*")
    ap.add_argument("--prior", action="append", default=[], type=Path, help="combine private evidence (repeatable)")
    ap.add_argument("--save-evidence", type=Path, help="write combined private evidence for a later run")
    args = ap.parse_args()
    if not args.manifests and not args.prior:
        ap.error("supply a manifest or --prior evidence")
    try:
        print(json.dumps(census(args.archive, args.index, args.manifests, args.prior, args.save_evidence), indent=2))
    except (ValueError, TypeError, OverflowError, IndexError, KeyError, OSError, struct.error, zlib.error) as exc:
        ap.exit(2, "error: %s\n" % exc)


if __name__ == "__main__":
    main()
