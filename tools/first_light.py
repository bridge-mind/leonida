"""First light: draw an exported game model with a selected pair of translated game shaders.

This assembles the smallest end-to-end path from the user's package to a Vulkan image: carve the game's
`shaders.rpf`, translate one real VS/PS pair with the repository's translator, convert one exported model to the
mesh the GPU harness reads, and run it on the device. Nothing is written outside `--work`, and no game data is
committed: the tool is the recipe, the user's copy is the input.

Without a DDS texture, the vertex shader forwards clip-space POSITION and TEXCOORD. The default pixel shader
writes the interpolated TEXCOORD, so the first image is the model's own UV map. Both are ordinary game shaders;
`--vs` and `--ps` take other names from the carve index.

Usage:
    python -I first_light.py --game "<package>" --obj <model.obj> --work <dir> [--out <dir>] [--run]
    python -I first_light.py --game "<package>" --obj <model.obj> --work <dir> --camera [--yaw 30 --pitch 25] --run
    python -I first_light.py --game "<package>" --work <dir> --list          # show VS/PS candidates

The mesh file the harness reads is: u32 vertex_count, u32 index_count, vertex_count x (x,y,z,u,v) f32, then
index_count u32 indices. By default the draw is orthographic: the model is centred and scaled to NDC on X/Y and
flattened to z = 0.5, because the chosen VS does not apply a matrix (see docs/first_light.md).

`--camera` draws in perspective instead, with a game VS that reads a world matrix and a view-projection matrix
from two constant buffers (CAMERA_VS). The mesh keeps the model's own coordinates; the tool builds a camera, writes
the matrices, the two V#s and the descriptor table into the harness heap (VKH_HEAP, VKH_SGPRS) and predicts the
image with the same matrices, interpolating UV perspective-correctly as the rasterizer does. Its output goes to
<work>/first_light/camera, so the orthographic folder that d3d12_first_light_test reads is left alone.

The model is an OBJ from `rpf9dump models` (tools/rpf9dump). Building that tool and exporting an archive is the
only step outside this script:

    git submodule update --init
    dotnet build tools/rpf9dump/rpf9dump.csproj -c Release
    DOTNET_ROLL_FORWARD=Major tools/rpf9dump/bin/Release/net8.0/rpf9dump models "<package>/data.rpf" <work>/assets

Standard library only; the project's tools run with `python -I`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)
sys.path.insert(0, os.path.join(ROOT, "runtime", "test", "shaders"))
from compute_fixture import find_test_program  # noqa: E402

DEFAULT_VS = "VS_main_170e32e6_Wrapped"
DEFAULT_PS = "$PS_Rage_Untextured_14f8529b_Wrapped"
DEFAULT_TEXTURED_PS = "PS_Copy_3ea4b38a_Wrapped"
DEFAULT_TSHARP_VA = 0x100
DEFAULT_PTR_SGPR = 9
DEFAULT_SSHARP = [0x92, 0xC00000, 0x14500000, 0]
MESH_HEADER = struct.Struct("<II")
TEX_HEADER = struct.Struct("<II")

# --camera (TEST-523). CAMERA_VS forwards TEXCOORD like DEFAULT_VS but builds clip space from two constant buffers;
# its scalar loads give the layout. The descriptor table's 32-bit address is in s9 (high half 0). The V# at table +
# 0x20 holds the world rows (X, Y, Z and translation, one float4 each at 0x00..0x30; the VS reads xyz of each), and
# the V# at table + 0x60 the four view-projection rows at 0x160..0x19f. Both multiply a row vector:
# world = [x y z 1] W, clip = [world 1] VP. On 1.0.2.1, all 9 translated VS modules with exactly POSITION and
# TEXCOORD inputs and a computed clip w read 16 dwords at 0x160 of a constant buffer, as do 1,266 of all 2,236 VS
# modules (docs/first_light.md section 7).
CAMERA_VS = "VS_TransformVT_e6ebb353_Wrapped"
CAMERA_TABLE_SGPR = 9
CAMERA_WORLD_VSHARP = 0x20
CAMERA_VIEW_VSHARP = 0x60
CAMERA_VIEWPROJ = 0x160
TABLE_VA, WORLD_VA, VIEW_VA = 0x100, 0x1000, 0x2000  # Xbox VAs inside vk_harness's 64 KiB VKH_HEAP


def run(argv, **kwargs):
    """Run a repository tool with this interpreter; fail loud with its command in the message."""
    cmd = [sys.executable, "-I"] + argv
    print("+ " + " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True, **kwargs)


def reset_run_folders(work: str) -> str:
    """Remove <work>/first_light/dxbc and spirv as an earlier run left them; return a new, empty dxbc.

    The unpacker, the disassembler and the translator each take every file in the folder they are given, and the
    translator merges into an index.json it finds, so a container staged by an earlier run would be translated
    again beside this run's pair. Both folders hold only copies from <work>/shaders and files derived from them,
    so they are removed whole; first_light/out, another --out and <work>/shaders are not touched. A folder that
    is a symlink or a junction, or is reached through one at first_light, is refused: nothing outside the work
    folder is removed. shutil.rmtree unlinks a link inside a folder without entering it.
    """
    base = os.path.join(work, "first_light")
    real_base = os.path.join(os.path.realpath(work), "first_light")
    # first_light itself is checked first: when it is a link to a folder with no dxbc or spirv in it, there is
    # nothing stale to judge below, and the new dxbc would be created on the far side of the link.
    if os.path.lexists(base) and (
        not os.path.isdir(base) or os.path.normcase(os.path.realpath(base)) != os.path.normcase(real_base)
    ):
        raise SystemExit(
            f"{base} is a symlink, a junction or a file: each run removes and creates folders under it and does "
            "not follow a link to do it; remove it and run again"
        )
    stale = [os.path.join(base, name) for name in ("dxbc", "spirv") if os.path.lexists(os.path.join(base, name))]
    for folder in stale:  # both are checked before either is removed
        real = os.path.join(real_base, os.path.basename(folder))
        if not os.path.isdir(folder) or os.path.normcase(os.path.realpath(folder)) != os.path.normcase(real):
            raise SystemExit(
                f"{folder} is a symlink, a junction or a file, or first_light above it is a link: each run "
                "removes this folder and does not follow a link to do it; remove it and run again"
            )
    for folder in stale:
        try:
            shutil.rmtree(folder)
        except OSError as e:
            raise SystemExit(
                f"cannot remove {e.filename or folder} left by an earlier run ({e.strerror or e}); "
                f"close what holds it or delete {folder}, then run again"
            ) from e
    dxbc = os.path.join(base, "dxbc")
    try:
        os.makedirs(dxbc)
    except OSError as e:
        raise SystemExit(f"cannot create {dxbc}: {e.strerror or e}") from e
    return dxbc


def carve_index(work: str) -> str:
    return os.path.join(work, "shaders", "dxbc", "index.json")


def ensure_carve(game: str, work: str) -> str:
    index = carve_index(work)
    if not os.path.isfile(index):
        run(
            [
                os.path.join(TOOLS, "shader_carve.py"),
                os.path.join(game, "shaders.rpf"),
                os.path.join(work, "shaders", "dxbc"),
            ]
        )
    return index


def choose(index_path: str, vs: str, ps: str) -> list[dict]:
    with open(index_path, encoding="utf-8") as f:
        entries = json.load(f)
    by_stage_and_name = {(entry["stage"], entry["name"]): entry for entry in entries}
    requested = (("vs", vs), ("ps", ps))
    missing = [f"{stage} {name}" for stage, name in requested if (stage, name) not in by_stage_and_name]
    if missing:
        raise SystemExit(
            f"shader not in the carve index: {', '.join(missing)} (use --list)"
        )
    return [by_stage_and_name[key] for key in requested]


def translate(work: str, picked: list[dict]) -> dict:
    """Translate just the two chosen containers, in the layout rdna2spirv.py reads."""
    base = os.path.join(work, "first_light")
    dxbc = reset_run_folders(work)
    index = []
    for entry in picked:
        shutil.copy(
            os.path.join(work, "shaders", "dxbc", entry["file"]),
            os.path.join(dxbc, entry["file"]),
        )
        index.append(
            {"name": entry["name"], "file": entry["file"], "stage": entry["stage"]}
        )
    with open(os.path.join(dxbc, "index.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(index, f, indent=0)
    run([os.path.join(TOOLS, "fssm_unpack.py"), dxbc])
    run(
        [
            os.path.join(TOOLS, "shader_disasm.py"),
            os.path.join(dxbc, "fssm"),
            "--out",
            os.path.join(dxbc, "disasm"),
            "--mcpu",
            "gfx1013",
        ]
    )
    spirv = os.path.join(base, "spirv")
    run(
        [
            os.path.join(TOOLS, "rdna2spirv.py"),
            os.path.join(dxbc, "disasm"),
            "--out",
            spirv,
        ]
    )
    with open(os.path.join(spirv, "index.json"), encoding="utf-8") as f:
        meta = json.load(f)
    result = {}
    for entry in picked:
        stem = os.path.splitext(entry["file"])[0]
        item = meta.get(stem)
        if item is None or item.get("stage") != entry["stage"]:
            raise SystemExit(f"translator has no {entry['stage']} result for {stem}")
        result[entry["stage"]] = item
    return result


def _obj_lines(handle):
    """Yield logical OBJ lines with trailing-backslash continuations joined and # comments stripped."""
    pending = ""
    for raw_line in handle:
        stripped = raw_line.rstrip("\r\n")
        if stripped.endswith("\\"):
            pending += stripped[:-1]
            continue
        yield (pending + stripped).split("#", 1)[0]
        pending = ""
    if pending:
        yield pending.split("#", 1)[0]


def _parse_obj_int(token: str, path: str, kind: str) -> int:
    try:
        return int(token)
    except ValueError:
        raise SystemExit(f"{path}: invalid {kind} index {token!r}") from None


def read_obj(
    path: str,
) -> tuple[list[tuple[float, float, float, float, float]], list[int]]:
    """Positions, UVs and triangles as (x, y, z, u, v) vertices plus a u32 index list."""
    positions, uvs, faces = [], [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in _obj_lines(f):
            p = line.split()
            if not p:
                continue
            if p[0] == "v":
                if len(p) < 4:
                    raise SystemExit(f"{path}: short vertex line: {line.strip()!r}")
                try:
                    position = tuple(float(x) for x in p[1:4])
                except ValueError:
                    raise SystemExit(f"{path}: invalid vertex coordinates: {line.strip()!r}") from None
                if not all(math.isfinite(value) for value in position):
                    raise SystemExit(f"{path}: non-finite vertex coordinates: {line.strip()!r}")
                positions.append(position)
            elif p[0] == "vt":
                if len(p) < 3:
                    raise SystemExit(f"{path}: short texture-coordinate line: {line.strip()!r}")
                try:
                    u, v = map(float, p[1:3])
                except ValueError:
                    raise SystemExit(f"{path}: invalid texture coordinates: {line.strip()!r}") from None
                if not (math.isfinite(u) and math.isfinite(v)):
                    raise SystemExit(f"{path}: non-finite texture coordinates: {line.strip()!r}")
                try:
                    struct.pack("<2f", u, v)  # Validate the mesh encoding before opening the output.
                except OverflowError:
                    raise SystemExit(f"{path}: texture coordinates exceed the binary32 range: {line.strip()!r}") from None
                uvs.append((u, v))
            elif p[0] == "f":
                face = []
                for word in p[1:]:
                    parts = word.split("/")
                    raw_vi = _parse_obj_int(parts[0], path, "vertex")
                    if raw_vi == 0:
                        raise SystemExit(f"{path}: vertex index 0 is out of range (OBJ indices are 1-based)")
                    vi = len(positions) + raw_vi if raw_vi < 0 else raw_vi - 1
                    if raw_vi < 0 and vi < 0:
                        raise SystemExit(f"{path}: vertex index {raw_vi} is out of range for {len(positions)} vertex(es)")
                    if len(parts) > 1 and parts[1]:
                        raw_ti = _parse_obj_int(parts[1], path, "texture-coordinate")
                        if raw_ti == 0:
                            ti = -1
                        else:
                            ti = len(uvs) + raw_ti if raw_ti < 0 else raw_ti - 1
                            if raw_ti < 0 and ti < 0:
                                ti = -1
                    else:
                        ti = -1
                    face.append((vi, ti))
                for k in range(1, len(face) - 1):
                    faces.extend((face[0], face[k], face[k + 1]))
    if not positions or not faces:
        raise SystemExit(f"no vertices or faces in {path}")
    unique, vertices, indices = {}, [], []
    for vi, ti in faces:
        if not 0 <= vi < len(positions):
            raise SystemExit(f"{path}: vertex index {vi + 1} is out of range for {len(positions)} vertex(es)")
        key = (vi, ti)
        if key not in unique:
            unique[key] = len(vertices)
            x, y, z = positions[vi]
            u, v = uvs[ti] if 0 <= ti < len(uvs) else (0.0, 0.0)
            vertices.append((x, y, z, u, v))
        indices.append(unique[key])
    return vertices, indices


def fit(vertices, width: int, height: int, fit_xy: bool, flatten_z: bool):
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    span = max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
    scale = 1.8 / span
    cz = (min(zs) + max(zs)) / 2 if not flatten_z else 0.0
    aspect = width / height if height else 1.0
    out = []
    for x, y, z, u, v in vertices:
        nx = (x - cx) * scale * (1.0 / aspect if aspect > 1 else 1.0)
        ny = (y - cy) * scale * (aspect if aspect < 1 else 1.0)
        nz = 0.5 if flatten_z else (z - cz) * scale + 0.5
        out.append((nx, ny, nz, u, v))
    return out


def f32(x: float) -> float:
    """x rounded to float32, so the predicted image uses the matrices the GPU reads."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _unit(a):
    n = math.sqrt(_dot(a, a))
    return tuple(x / n for x in a)


def row_times(v, m):
    """The row vector v times the 4x4 matrix m (a list of rows), as CAMERA_VS multiplies."""
    return tuple(sum(v[i] * m[i][j] for i in range(4)) for j in range(4))


def camera_matrices(vertices, width: int, height: int, yaw: float, pitch: float, fov: float, distance: float = 3.0):
    """The world rows and view-projection rows (row-vector 4x4, float32) for --camera.

    World centres the model's bounding box and scales its bounding sphere to radius 1. The camera looks at the
    centre from `distance` radii, `yaw` degrees around the up axis (Z, the game's) from -Y and `pitch` degrees above
    the XY plane, with a vertical field of view of `fov` degrees. The projection is Vulkan's: x right, y down the
    image like the harness's viewport, depth 0..1 between a near and far plane just outside the sphere, so depth
    clips no vertex.
    """
    cx, cy, cz = ((min(v[i] for v in vertices) + max(v[i] for v in vertices)) / 2 for i in range(3))
    radius = max(math.dist(v[:3], (cx, cy, cz)) for v in vertices) or 1.0
    s = 1.0 / radius
    world = [[s, 0.0, 0.0, 0.0], [0.0, s, 0.0, 0.0], [0.0, 0.0, s, 0.0], [-cx * s, -cy * s, -cz * s, 1.0]]
    ya, pa = math.radians(yaw), math.radians(pitch)
    eye = (distance * math.cos(pa) * math.sin(ya), -distance * math.cos(pa) * math.cos(ya), distance * math.sin(pa))
    forward = _unit(_sub((0.0, 0.0, 0.0), eye))
    right = _unit(_cross(forward, (0.0, 0.0, 1.0)))
    up = _cross(right, forward)
    view = [[right[i], up[i], forward[i], 0.0] for i in range(3)]
    view.append([-_dot(eye, right), -_dot(eye, up), -_dot(eye, forward), 1.0])
    near, far = distance - 1.25, distance + 1.25
    sy = 1.0 / math.tan(math.radians(fov) / 2)
    sx = sy * height / width
    a, b = far / (far - near), -near * far / (far - near)
    proj = [[sx, 0.0, 0.0, 0.0], [0.0, -sy, 0.0, 0.0], [0.0, 0.0, a, 1.0], [0.0, 0.0, b, 0.0]]
    viewproj = [row_times(row, proj) for row in view]
    return [[f32(x) for x in row] for row in world], [[f32(x) for x in row] for row in viewproj]


def project(vertices, world, viewproj):
    """(x/w, y/w, z/w, u, v, w) per vertex: what CAMERA_VS computes, divided into the harness's NDC.

    The trailing w makes uv_at interpolate perspective-correctly. A vertex the draw would clip (w <= 0, or depth
    outside 0..1) is an error: the prediction does not model clipping.
    """
    out = []
    for x, y, z, u, v in vertices:
        wx, wy, wz, _ = row_times((x, y, z, 1.0), world)
        cx, cy, cz, cw = row_times((wx, wy, wz, 1.0), viewproj)
        if cw <= 0 or not 0.0 <= cz / cw <= 1.0:
            raise SystemExit(f"vertex {(x, y, z)} is clipped by the camera (clip {cx, cy, cz, cw})")
        out.append((cx / cw, cy / cw, cz / cw, u, v, cw))
    return out


def vsharp(base: int, size: int) -> list[int]:
    """A raw buffer V#: 32-bit base address (the heap is below 4 GiB), stride 0, size in bytes, word 3 zero."""
    return [base, 0, size, 0]


def camera_constants(world, viewproj) -> tuple[str, str]:
    """VKH_SGPRS and VKH_HEAP for CAMERA_VS: the table address in s9, the two V#s in the table, the two matrices."""
    sgprs = [0] * (CAMERA_TABLE_SGPR + 1)
    sgprs[CAMERA_TABLE_SGPR] = TABLE_VA
    heap = {}

    def put(addr, words):
        for i, word in enumerate(words):
            heap[addr + 4 * i] = word

    def f32_words(m):
        return [struct.unpack("<I", struct.pack("<f", x))[0] for row in m for x in row]

    put(TABLE_VA + CAMERA_WORLD_VSHARP, vsharp(WORLD_VA, 0x40))
    put(TABLE_VA + CAMERA_VIEW_VSHARP, vsharp(VIEW_VA, CAMERA_VIEWPROJ + 0x40))
    put(WORLD_VA, f32_words(world))
    put(VIEW_VA + CAMERA_VIEWPROJ, f32_words(viewproj))
    return (
        ",".join(str(w) for w in sgprs),
        ",".join(f"{addr:#x}={word:#x}" for addr, word in sorted(heap.items())),
    )


def write_mesh(path: str, vertices, indices) -> None:
    with open(path, "wb") as f:
        f.write(MESH_HEADER.pack(len(vertices), len(indices)))
        f.writelines(struct.pack("<5f", *v) for v in vertices)
        f.write(struct.pack(f"<{len(indices)}I", *indices))


def decode_dds_to_rgba(path: str) -> tuple[int, int, bytes, str]:
    """Decode a DDS texture (BC1/BC3/BC7/RGBA8/etc.) via tools/dds_png.py."""
    if TOOLS not in sys.path:
        sys.path.insert(0, TOOLS)
    import dds_png

    with open(path, "rb") as f:
        raw = f.read()
    width, height, rgba = dds_png.decode_dds(raw)
    return width, height, rgba, hashlib.sha256(raw).hexdigest()


def write_texture_rgba(path: str, width: int, height: int, rgba: bytes) -> None:
    if len(rgba) != width * height * 4:
        raise ValueError(f"expected {width * height * 4} RGBA bytes, got {len(rgba)}")
    with open(path, "wb") as f:
        f.write(TEX_HEADER.pack(width, height))
        f.write(rgba)


def sample_rgba_bilinear(
    width: int, height: int, rgba: bytes, u: float, v: float
) -> list[float]:
    """Bilinear sample of an R8G8B8A8_UNORM texture with clamp-to-edge addressing."""
    u = max(0.0, min(1.0, u))
    v = max(0.0, min(1.0, v))
    fx = u * width - 0.5
    fy = v * height - 0.5
    x0 = max(0, min(width - 1, int(fx) if fx >= 0 else 0))
    y0 = max(0, min(height - 1, int(fy) if fy >= 0 else 0))
    x1 = min(width - 1, x0 + 1)
    y1 = min(height - 1, y0 + 1)
    tx = max(0.0, min(1.0, fx - x0))
    ty = max(0.0, min(1.0, fy - y0))

    def texel(x: int, y: int) -> tuple[float, float, float, float]:
        off = (y * width + x) * 4
        return (
            rgba[off] / 255.0,
            rgba[off + 1] / 255.0,
            rgba[off + 2] / 255.0,
            rgba[off + 3] / 255.0,
        )

    c00, c10 = texel(x0, y0), texel(x1, y0)
    c01, c11 = texel(x0, y1), texel(x1, y1)
    out = []
    for c in range(4):
        top = c00[c] * (1.0 - tx) + c10[c] * tx
        bot = c01[c] * (1.0 - tx) + c11[c] * tx
        out.append(round(top * (1.0 - ty) + bot * ty, 6))
    return out


def harness_env(manifest: dict, out_dir: str, base_env: dict | None = None) -> dict:
    """The environment vk_harness draws a first_light manifest with: the mesh and its layout, plus the constants of a
    --camera draw (VKH_SGPRS and VKH_HEAP) or of a --dds draw (the texture, T#, S# and its pointer SGPR).

    tests/test_first_light.py uses it too. VKH_SGPRS and VKH_HEAP from the caller are dropped first, so a stale value
    cannot change an orthographic draw.
    """
    env = dict(os.environ if base_env is None else base_env)
    env["VKH_DRAW_MESH"] = os.path.join(out_dir, manifest["files"]["mesh"])
    env["VKH_VERTEX_LAYOUT"] = "0:3f,1:2f"
    env.pop("VKH_SGPRS", None)
    env.pop("VKH_HEAP", None)
    camera = manifest.get("camera")
    if camera:
        env["VKH_SGPRS"] = camera["sgprs"]
        env["VKH_HEAP"] = camera["heap"]
    tex = manifest.get("texture")
    if tex:
        tex_file = manifest["files"]["texture"]
        ptr_sgpr = int(tex["ptr_sgpr"])
        tsharp_va = int(tex["tsharp_va"])
        ssharp = tex["ssharp"]
        sgprs = ["0"] * (ptr_sgpr + 1)
        sgprs[ptr_sgpr] = hex(tsharp_va)
        env["VKH_TEX_RGBA"] = os.path.join(out_dir, tex_file)
        env["VKH_TSHARP_VA"] = hex(tsharp_va)
        env["VKH_SSHARP"] = ",".join(hex(int(w)) for w in ssharp)
        env["VKH_SGPRS"] = ",".join(sgprs)
        ps_sg = manifest.get("ps", {}).get("subgroup_size")
        if ps_sg:
            env["VKH_SUBGROUP_SIZE"] = str(int(ps_sg))
    return env


def pixel_ndc(px: int, py: int, width: int, height: int) -> tuple[float, float]:
    """The NDC of a pixel centre under the harness's viewport (y grows down, like Vulkan)."""
    return (px + 0.5) / width * 2.0 - 1.0, (py + 0.5) / height * 2.0 - 1.0


def barycentric(px, py, ax, ay, bx, by, cx, cy):
    """The three barycentric weights of (px, py) in the triangle, or None outside it (either winding)."""
    det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if abs(det) < 1e-12:
        return None
    w0 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / det
    w1 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / det
    w2 = 1.0 - w0 - w1
    if min(w0, w1, w2) < -1e-6 or max(w0, w1, w2) > 1.0 + 1e-6:
        return None
    return w0, w1, w2


def uv_at(vertices, indices, px: int, py: int, width: int, height: int):
    """The UV the last triangle covering this pixel centre writes (no depth test: the last primitive wins).

    Vertices from project() carry clip w as a sixth item; their UV is interpolated perspective-correctly (each
    screen-space weight divided by its vertex's w, renormalized), as the rasterizer does for a perspective varying.
    """
    x, y = pixel_ndc(px, py, width, height)
    found = None
    for i in range(0, len(indices), 3):
        a, b, c = (vertices[indices[i + k]] for k in range(3))
        weights = barycentric(x, y, a[0], a[1], b[0], b[1], c[0], c[1])
        if weights and len(a) > 5:
            q = (weights[0] / a[5], weights[1] / b[5], weights[2] / c[5])
            weights = tuple(w / sum(q) for w in q)
        if weights:
            if not (0 <= a[2] <= 1 and 0 <= b[2] <= 1 and 0 <= c[2] <= 1):
                # The passthrough VS has w=1. Use the float32 depths written to mesh.bin.
                za, zb, zc = struct.unpack("<3f", struct.pack("<3f", a[2], b[2], c[2]))
                z = za + weights[1] * (zb - za) + weights[2] * (zc - za)
                if not 0 <= z <= 1:
                    continue
            found = (
                weights[0] * a[3] + weights[1] * b[3] + weights[2] * c[3],
                weights[0] * a[4] + weights[1] * b[4] + weights[2] * c[4],
            )
    return found


def clear_pixel(vertices, indices, width: int, height: int):
    """A corner the mesh does not cover, which must keep the harness clear value."""
    for px, py in ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)):
        if uv_at(vertices, indices, px, py, width, height) is None:
            return [px, py]
    return None


def uv_probes(vertices, indices, width: int, height: int, limit: int = 4) -> list[dict]:
    """Pixels whose predicted UV has U and V far apart, so a swapped-U/V image cannot pass.

    The centre alone is not enough: a symmetric model can have U ~= V there. The scan is coarse (16 x 16 at
    128 px) and uses the same last-triangle rule as the centre.
    """
    step = max(1, min(width, height) // 16)
    found = []
    for py in range(step // 2, height, step):
        for px in range(step // 2, width, step):
            uv = uv_at(vertices, indices, px, py, width, height)
            if uv and abs(uv[0] - uv[1]) > 0.15:
                found.append(
                    {"pixel": [px, py], "uv": [round(uv[0], 6), round(uv[1], 6)]}
                )
    if len(found) > limit:
        found = [found[int(i * len(found) / limit)] for i in range(limit)]
    return found


def find_harness(explicit: str | None) -> str:
    executable = "vk_harness.exe" if os.name == "nt" else "vk_harness"
    build = os.path.join(ROOT, "runtime", "build")
    build_dirs = (os.path.join(build, "linux"), os.path.join(build, "windows"), build)
    for candidate in (
        explicit,
        os.environ.get("LEONIDA_VK_HARNESS"),
        *(os.path.join(folder, executable) for folder in build_dirs),
    ):
        if candidate and os.path.isfile(candidate):
            return candidate
    for folder in build_dirs:
        configured = find_test_program(folder, "vk_harness")
        if configured:
            return os.path.join(configured, executable)
    raise SystemExit(
        "vk_harness not found: build runtime/build/linux or runtime/build/windows (target vk_harness), "
        "or set --harness/LEONIDA_VK_HARNESS"
    )


CLEAR = -1.0
FALLBACK_RGBA = (1.0, 0.0, 1.0, 1.0)


def _close(got: float, want: float, tol: float) -> bool:
    return math.isfinite(got) and abs(got - want) <= tol


def _distinguishable_from_fallback(rgba: list[float] | tuple[float, ...], tol: float) -> bool:
    return any(abs(c - m) > tol for c, m in zip(rgba, FALLBACK_RGBA))


def rgba_probes_for_mesh(
    vertices,
    indices,
    width: int,
    height: int,
    tw: int,
    th: int,
    rgba: bytes,
    uv_probe_list: list[dict] | None = None,
    tol: float = 0.02,
    limit: int = 4,
) -> list[dict]:
    """Select covered RGBA probes for textured verification.

    Starts from ``uv_probe_list`` when available; if that is empty (e.g. an off-centre mesh with
    default (0, 0) UVs), scans covered pixels so textured draws always have at least one covered
    RGBA probe. Prefers samples whose expected colour differs from the harness's magenta fallback.
    """
    candidates = list(uv_probe_list) if uv_probe_list else []
    if not candidates and vertices:
        step = max(1, min(width, height) // 16)
        for py in range(step // 2, height, step):
            for px in range(step // 2, width, step):
                uv = uv_at(vertices, indices, px, py, width, height)
                if uv is not None:
                    candidates.append(
                        {"pixel": [px, py], "uv": [round(uv[0], 6), round(uv[1], 6)]}
                    )
        if not candidates:
            for py in range(height):
                for px in range(width):
                    uv = uv_at(vertices, indices, px, py, width, height)
                    if uv is not None:
                        candidates.append(
                            {"pixel": [px, py], "uv": [round(uv[0], 6), round(uv[1], 6)]}
                        )
                        if len(candidates) >= limit * 4:
                            break
                if len(candidates) >= limit * 4:
                    break
    sampled = [
        {
            "pixel": p["pixel"],
            "uv": p["uv"],
            "rgba": sample_rgba_bilinear(tw, th, rgba, p["uv"][0], p["uv"][1]),
        }
        for p in candidates
    ]
    distinct = [p for p in sampled if _distinguishable_from_fallback(p["rgba"], tol)]
    pool = distinct if distinct else sampled
    if len(pool) > limit:
        pool = [pool[int(i * len(pool) / limit)] for i in range(limit)]
    return pool


def check_image(data: bytes, manifest: dict) -> dict:
    """The readback oracle: exact shape, coverage window, a clear corner and the centre UV.

    tests/test_first_light.py imports this function, so the test and ``--check`` agree.
    """
    def check_rgba_shape(values: object, field: str) -> None:
        if not isinstance(values, (list, tuple)) or len(values) != 4:
            raise AssertionError(f"{field} must contain exactly four RGBA values")

    width, height = manifest["width"], manifest["height"]

    def pixel_offset(point, field: str) -> int:
        if (
            not isinstance(point, (list, tuple)) or len(point) != 2
            or any(type(coord) is not int for coord in point)
            or not (0 <= point[0] < width and 0 <= point[1] < height)
        ):
            raise AssertionError(f"invalid {field} coordinates {point!r} for {width}x{height} image")
        x, y = point
        return (y * width + x) * 4

    if len(data) != width * height * 16:
        raise AssertionError(
            f"raw is {len(data)} bytes, expected {width * height * 16}"
        )
    pixels = struct.unpack(f"<{len(data) // 4}f", data)
    clear_val = manifest.get("clear", CLEAR)
    textured = (
        manifest.get("texture") is not None
        or manifest.get("centre_rgba") is not None
        or manifest.get("rgba_probes") is not None
    )
    if textured:
        # UNORM texels can have any alpha in [0, 1] (including 0.0), so distinguish covered fragments
        # from the harness clear sentinel (-1.0) instead of testing alpha > 0.5.
        covered = sum(
            1
            for i in range(0, len(pixels), 4)
            if not all(_close(pixels[i + c], clear_val, 1e-5) for c in range(4))
        )
    else:
        covered = sum(1 for i in range(3, len(pixels), 4) if pixels[i] > 0.5)
    ratio = covered / (width * height)
    if (
        not manifest.get("min_coverage", 0.0)
        <= ratio
        <= manifest.get("max_coverage", 1.0)
    ):
        raise AssertionError(
            f"coverage {ratio:.4f} outside "
            f"[{manifest.get('min_coverage', 0.0)}, {manifest.get('max_coverage', 1.0)}]"
        )
    if (clear := manifest.get("clear_pixel")) is not None:
        i = pixel_offset(clear, "clear_pixel")
        if any(not _close(pixels[i + c], clear_val, 1e-5) for c in range(4)):
            raise AssertionError(
                f"clear pixel {clear} is {pixels[i : i + 4]}, expected {clear_val}"
            )
    if textured:
        tol = manifest.get("rgba_tolerance", 0.02)
        expected_rgbas = []
        if (rgba := manifest.get("centre_rgba")) is not None:
            expected_rgbas.append(rgba)
        for probe in manifest.get("rgba_probes") or []:
            expected_rgbas.append(probe["rgba"])
        if not expected_rgbas:
            raise AssertionError(
                "textured manifest has no covered RGBA samples (centre_rgba or rgba_probes)"
            )
        if not any(_distinguishable_from_fallback(r, tol) for r in expected_rgbas):
            raise AssertionError(
                f"textured manifest has no expected colour distinguishable from fallback {FALLBACK_RGBA} (tol {tol})"
            )
    centre = manifest.get("centre_pixel")
    if centre is None or centre == []:
        centre = [width // 2, height // 2]
    if (rgba := manifest.get("centre_rgba")) is not None:
        i = pixel_offset(centre, "centre_pixel")
        tol = manifest.get("rgba_tolerance", 0.02)
        check_rgba_shape(rgba, "centre_rgba")
        for c, want in enumerate(rgba):
            if not _close(pixels[i + c], want, tol):
                raise AssertionError(
                    f"centre {centre} is {pixels[i : i + 4]}, expected {rgba} (tol {tol})"
                )
    elif (uv := manifest.get("centre_uv")) is not None:
        i = pixel_offset(centre, "centre_pixel")
        for c, want in enumerate((uv[0], uv[1], 0.0, 1.0)):
            if not _close(pixels[i + c], want, 2e-3):
                raise AssertionError(
                    f"centre {centre} is {pixels[i : i + 4]}, expected {want} at {c}"
                )
    if (rgba_probes := manifest.get("rgba_probes")) is not None:
        tol = manifest.get("rgba_tolerance", 0.02)
        for probe in rgba_probes:
            i = pixel_offset(probe.get("pixel"), "rgba_probes pixel")
            want_rgba = probe["rgba"]
            check_rgba_shape(want_rgba, "rgba_probes[].rgba")
            for c, want in enumerate(want_rgba):
                if not _close(pixels[i + c], want, tol):
                    raise AssertionError(
                        f"RGBA probe {probe['pixel']} is {pixels[i : i + 4]}, expected {want_rgba} (tol {tol})"
                    )
    uv_probes = manifest.get("uv_probes")
    if uv_probes is None:
        uv_probes = []
    for probe in uv_probes:
        i = pixel_offset(probe.get("pixel"), "uv_probes pixel")
        if rgba_probes is not None:
            continue
        u, v = probe["uv"]
        for c, want in enumerate((u, v, 0.0, 1.0)):
            if not _close(pixels[i + c], want, 2e-3):
                raise AssertionError(
                    f"UV probe {probe['pixel']} is {pixels[i : i + 4]}, expected {want} at {c}"
                )
    return {"covered": covered, "ratio": ratio, "pixels": pixels}


def write_ppm(path: str, pixels, width: int, height: int) -> None:
    """Write RGB bytes, saturating infinities and rendering NaN as zero."""
    def channel_byte(value: float) -> int:
        return round(min(1.0, value) * 255) if value > 0.0 else 0

    with open(path, "wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode())
        f.writelines(
            bytes(channel_byte(pixels[i + c]) for c in range(3))
            for i in range(0, len(pixels), 4)
        )


def run_draw(args, manifest: dict, out_dir: str) -> dict:
    if manifest.get("vertex_count") == 0 or manifest.get("index_count") == 0:
        raise SystemExit("first-light draw requires an OBJ with faces. Prepare this output with --obj before drawing.")
    harness = find_harness(args.harness)
    raw = os.path.join(out_dir, "image.raw")
    ppm = os.path.join(out_dir, "image.ppm")
    manifest_path = os.path.join(out_dir, "manifest.json")
    env = harness_env(manifest, out_dir)
    cmd = [
        harness,
        os.path.join(out_dir, "vs.spv"),
        os.path.join(out_dir, "ps.spv"),
        raw,
        str(manifest["width"]),
        str(manifest["height"]),
    ]
    # A new draw must not reuse an old readback or leave an old success report on failure.
    for path in (raw, ppm):
        if os.path.exists(path):
            os.unlink(path)
    if "result" in manifest:
        del manifest["result"]
        with open(manifest_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(manifest, f, indent=1)
    subprocess.run(cmd, check=True, env=env)
    if not os.path.isfile(raw):
        raise SystemExit(f"the harness exited successfully but did not write {raw}")
    with open(raw, "rb") as f:
        data = f.read()
    try:
        result = check_image(data, manifest)
    except AssertionError as e:
        raise SystemExit(str(e)) from e
    write_ppm(ppm, result["pixels"], manifest["width"], manifest["height"])
    print(
        f"covered {result['covered']}/{manifest['width'] * manifest['height']} "
        f"({result['ratio']:.3f}); {ppm}"
    )
    manifest["result"] = {
        "covered": result["covered"],
        "total": manifest["width"] * manifest["height"],
        "ratio": result["ratio"],
        "ppm": ppm,
    }
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1)
    return result


def load_json(
    path: str,
    kind: str,
    expect: type,
    keys: tuple[str, ...] = (),
    entry_keys: tuple[str, ...] = (),
):
    """Read a JSON file from the user's folders; a bad read exits naming it, never a traceback.

    Invalid or truncated JSON, non-UTF-8 bytes, the wrong top-level shape, a missing required
    key, and an array entry that is not an object or lacks one are the user's broken files, so
    they fail like the tool's other errors: one SystemExit line with the path and the reason.
    """
    wanted = "an object" if expect is dict else "an array"
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except UnicodeDecodeError as e:
        raise SystemExit(f"the {kind} {path} is not valid UTF-8 ({e.reason})") from e
    except json.JSONDecodeError as e:
        raise SystemExit(
            f"the {kind} {path} is not valid JSON ({e.msg} at line {e.lineno} column {e.colno})"
        ) from e
    if not isinstance(data, expect):
        raise SystemExit(f"the {kind} {path} is a {type(data).__name__}, expected {wanted}")
    if expect is dict:
        if gaps := [k for k in keys if k not in data]:
            raise SystemExit(f"the {kind} {path} has no key {', '.join(repr(k) for k in gaps)}")
    else:
        for i, entry in enumerate(data):
            if not isinstance(entry, dict):
                raise SystemExit(
                    f"the {kind} {path} entry {i} is a {type(entry).__name__}, expected an object"
                )
            if gaps := [k for k in entry_keys if k not in entry]:
                raise SystemExit(
                    f"the {kind} {path} entry {i} has no key {', '.join(repr(k) for k in gaps)}"
                )
    return data


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Draw an exported model with translated game shaders. Display its UVs or supply a texture with --dds."
    )
    ap.add_argument("--game", help="the package folder")
    ap.add_argument("--obj", help="an OBJ exported by rpf9dump models, with finite vertex positions and UVs")
    ap.add_argument(
        "--work",
        default=os.path.expanduser("~/leonida-work"),
        help="the work folder make_game.py uses (default ~/leonida-work)",
    )
    ap.add_argument("--out", help="output folder (default <work>/first_light/out)")
    ap.add_argument(
        "--vs",
        help=f"vertex shader name in the carve index (default {DEFAULT_VS}, or {CAMERA_VS} with --camera)",
    )
    ap.add_argument(
        "--ps", default=None, help="pixel shader name in the carve index (default untextured, or PS_Copy with --dds)"
    )
    ap.add_argument(
        "--dds",
        help="an exported DDS texture (e.g. BCPack-decoded TTPF albedo) to bind via set-0 T# and sampler S#",
    )
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument(
        "--min-coverage",
        type=float,
        help="the oracle's lowest covered fraction (default 0.5, or 0.2 with --camera: a model in perspective "
        "fills less of the image)",
    )
    ap.add_argument(
        "--no-fit",
        action="store_true",
        help="keep the model's coordinates (must already be in NDC)",
    )
    ap.add_argument("--keep-z", action="store_true", help="do not flatten z to 0.5")
    ap.add_argument(
        "--camera",
        action="store_true",
        help="draw in perspective with CAMERA_VS and a camera's world/view-projection constants",
    )
    ap.add_argument("--yaw", type=float, default=30.0, help="--camera: degrees around Z from -Y (default 30)")
    ap.add_argument(
        "--pitch", type=float, default=25.0, help="--camera: degrees above the XY plane, -80..80 (default 25)"
    )
    ap.add_argument("--fov", type=float, default=50.0, help="--camera: vertical field of view, degrees (default 50)")
    ap.add_argument(
        "--harness",
        help="vk_harness binary (default LEONIDA_VK_HARNESS or runtime/build/linux)",
    )
    ap.add_argument(
        "--run", action="store_true", help="run the draw and write image.raw/image.ppm"
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="run the draw and fail if the coverage is too low",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="list VS/PS names in the carve index and exit",
    )
    a = ap.parse_args(argv)
    if a.min_coverage is not None and not 0.0 <= a.min_coverage <= 1.0:
        ap.error("--min-coverage must be between 0 and 1")
    if a.camera:
        if not math.isfinite(a.yaw):
            ap.error("--yaw must be finite")
        if a.vs not in (None, CAMERA_VS):
            ap.error(f"--camera writes the constant layout of {CAMERA_VS} only")
        if a.no_fit or a.keep_z:
            ap.error("--camera keeps the model's coordinates; --no-fit and --keep-z do not apply")
        if not -80.0 <= a.pitch <= 80.0 or not 1.0 <= a.fov <= 150.0:
            ap.error("--pitch must be within -80..80 and --fov within 1..150 degrees")
        if a.dds:
            ap.error("--camera and --dds both put a pointer in s9 and data at heap VA 0x100; draw the camera untextured")
    a.vs = a.vs or (CAMERA_VS if a.camera else DEFAULT_VS)
    if a.min_coverage is None:
        a.min_coverage = 0.2 if a.camera else 0.5
    work = os.path.abspath(os.path.expanduser(a.work))
    out_dir = (
        os.path.abspath(os.path.expanduser(a.out))
        if a.out
        else os.path.join(work, "first_light", "camera" if a.camera else "out")
    )

    if a.list:
        if not a.game:
            raise SystemExit("--list needs --game")
        index = ensure_carve(os.path.abspath(a.game), work)
        for e in load_json(index, "carve index", list, entry_keys=("stage", "name")):
            if e["stage"] in ("vs", "ps"):
                print(f"{e['stage']} {e['name']}")
        return 0
    existing = os.path.join(out_dir, "manifest.json")
    if a.check and not a.game and os.path.isfile(existing):
        manifest = load_json(existing, "first-light manifest", dict, keys=("width", "height"))
        run_draw(a, manifest, out_dir)
        return 0
    if not a.game:
        raise SystemExit("--game is required")
    game = os.path.abspath(os.path.expanduser(a.game))
    if not a.no_fit and not a.obj:
        raise SystemExit("--obj is required unless --no-fit")
    if a.dds and not a.obj:
        raise SystemExit("--dds requires --obj so textured probes can be checked")

    index = ensure_carve(game, work)
    ps_name = a.ps if a.ps is not None else (DEFAULT_TEXTURED_PS if a.dds else DEFAULT_PS)
    picked = choose(index, a.vs, ps_name)
    meta = translate(work, picked)
    os.makedirs(out_dir, exist_ok=True)
    for entry in picked:
        stem = os.path.splitext(entry["file"])[0]
        stage = entry["stage"]
        shutil.copy(
            os.path.join(work, "first_light", "spirv", f"{stem}.spv"),
            os.path.join(out_dir, f"{stage}.spv"),
        )
    # For d3d12_first_light_test: the game's containers (the PSO finds each stage by the FSSM hash inside) and
    # the translator's index entries, keyed vs/ps so the runtime's shader library reads vs.spv and ps.spv here.
    for entry in picked:
        shutil.copy(
            os.path.join(work, "shaders", "dxbc", entry["file"]),
            os.path.join(out_dir, f"{entry['stage']}.dxbc"),
        )
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"vs": meta["vs"], "ps": meta["ps"]}, f, indent=1)
    vertices, indices = read_obj(a.obj) if a.obj else ([], [])
    obj_sha256 = None
    if a.obj:
        with open(a.obj, "rb") as f:
            obj_sha256 = hashlib.sha256(f.read()).hexdigest()
    camera = None
    if vertices and a.camera:
        # The mesh keeps the model's coordinates (CAMERA_VS applies the matrices); the prediction below works on
        # the same vertices projected on the CPU.
        write_mesh(os.path.join(out_dir, "mesh.bin"), vertices, indices)
        world, viewproj = camera_matrices(vertices, a.width, a.height, a.yaw, a.pitch, a.fov)
        sgprs, heap = camera_constants(world, viewproj)
        camera = {
            "yaw": a.yaw,
            "pitch": a.pitch,
            "fov": a.fov,
            "world": world,
            "viewproj": viewproj,
            "sgprs": sgprs,
            "heap": heap,
        }
        vertices = project(vertices, world, viewproj)
    elif vertices:
        if not a.no_fit:
            vertices = fit(vertices, a.width, a.height, True, not a.keep_z)
        write_mesh(os.path.join(out_dir, "mesh.bin"), vertices, indices)
    else:  # a run without a model must not draw the mesh an earlier run left in the folder
        try:
            os.unlink(os.path.join(out_dir, "mesh.bin"))
        except FileNotFoundError:
            pass
    centre_pixel = [a.width // 2, a.height // 2]
    centre_uv = (
        uv_at(vertices, indices, centre_pixel[0], centre_pixel[1], a.width, a.height)
        if vertices
        else None
    )
    clear = clear_pixel(vertices, indices, a.width, a.height) if vertices else None
    probes = uv_probes(vertices, indices, a.width, a.height) if vertices else []
    manifest = {
        "vs": {
            "name": picked[0]["name"],
            "fssm_sha256": meta["vs"].get("fssm_sha256"),
            "verified": True,
        },
        "ps": {
            "name": picked[1]["name"],
            "fssm_sha256": meta["ps"].get("fssm_sha256"),
            "subgroup_size": meta["ps"].get("subgroup_size"),
            "verified": True,
        },
        "vertex_count": len(vertices),
        "index_count": len(indices),
        "triangle_count": len(indices) // 3,
        "obj_sha256": obj_sha256,
        "width": a.width,
        "height": a.height,
        "min_coverage": a.min_coverage,
        "max_coverage": 0.99,
        "centre_pixel": centre_pixel,
        "centre_uv": list(centre_uv) if centre_uv else None,
        "clear_pixel": clear,
        "uv_probes": probes,
        "clear": -1.0,
        "camera": camera,
        "files": {
            "vs": "vs.spv",
            "ps": "ps.spv",
            "mesh": "mesh.bin",
            "vs_dxbc": "vs.dxbc",
            "ps_dxbc": "ps.dxbc",
            "index": "index.json",
        },
    }
    if a.dds:
        tw, th, rgba, dds_sha = decode_dds_to_rgba(a.dds)
        write_texture_rgba(os.path.join(out_dir, "texture.rgba"), tw, th, rgba)
        ps_meta = meta["ps"]
        ptr_sgpr = next(
            (
                int(r["sgpr"])
                for r in (ps_meta.get("user_sgpr_reads") or [])
                if r.get("kind") in ("ptr32_lo", "ptr64_lo")
            ),
            DEFAULT_PTR_SGPR,
        )
        lit_samplers = ps_meta.get("literal_samplers") or []
        ssharp = list(lit_samplers[0]) if lit_samplers else list(DEFAULT_SSHARP)
        manifest["texture"] = {
            "width": tw,
            "height": th,
            "dds_sha256": dds_sha,
            "ptr_sgpr": ptr_sgpr,
            "tsharp_va": DEFAULT_TSHARP_VA,
            "ssharp": ssharp,
        }
        manifest["files"]["texture"] = "texture.rgba"
        manifest["rgba_tolerance"] = 0.02
        if centre_uv is not None:
            manifest["centre_rgba"] = sample_rgba_bilinear(
                tw, th, rgba, centre_uv[0], centre_uv[1]
            )
        tex_probes = rgba_probes_for_mesh(
            vertices,
            indices,
            a.width,
            a.height,
            tw,
            th,
            rgba,
            uv_probe_list=probes,
            tol=manifest["rgba_tolerance"],
        )
        if tex_probes:
            manifest["rgba_probes"] = tex_probes
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1)
    print(f"{len(vertices)} vertices, {len(indices) // 3} triangles -> {out_dir}")
    if not (a.run or a.check):
        env_preview = harness_env(manifest, out_dir, base_env={})
        env_str = " ".join(f"{k}={v}" for k, v in env_preview.items())
        print(
            f"run: {env_str} "
            f"<vk_harness> {os.path.join(out_dir, 'vs.spv')} {os.path.join(out_dir, 'ps.spv')} "
            f"{os.path.join(out_dir, 'image.raw')} {a.width} {a.height}"
            + (" (with VKH_SGPRS and VKH_HEAP from manifest.json's camera)" if camera else "")
        )
        return 0
    run_draw(a, manifest, out_dir)
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
