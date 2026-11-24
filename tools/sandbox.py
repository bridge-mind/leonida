#!/usr/bin/env python3
"""Real-shader sandbox: walk, fly or drive through placed props drawn by the game's own translated shaders.

    python -I tools/sandbox.py --work <work> --entities <entities.jsonl> --models <assets> --textures <ttpf out> \
        [--center X Y] [--radius M] [--max-props N] (--window | --check)

It packs the props around --center with tools/sandbox_scene.py, picks a translated VS/PS pair from
<work>/shaders/spirv/index.json and runs runtime/build/vk_sandbox. The pair is chosen by what the shaders read,
not by name (the material -> program table is in the executable, #2519): a vertex shader with POSITION, NORMAL
and TEXCOORD inputs, user SGPR 8 as its only user data and no bindings, whose listing loads three descriptors from
s[8:9] at 0x0/0x40/0x60, a world translation at 0x30 and a 4x4 at 0x160; and a pixel shader with TEXCOORD and
COLOR inputs that loads one T# from s[8:9] + 0x20 and samples it once. vk_sandbox fills that table (see its
header). --check renders one frame headless and prints only counts: exit 0 when the frame is covered and, with
textures, shows texture detail; 77 when the package, the pair or a GPU is missing. Standard library only.
Usage: `python -I tools/sandbox.py --help` lists the options.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sandbox_scene  # noqa: E402

VS_LAYOUT = (r"s_load_dwordx4 s\[\d+:\d+\], s\[8:9\], (?:null|0)\b", r"s_load_dwordx4 s\[\d+:\d+\], s\[8:9\], 0x40\b",
             r"s_load_dwordx4 s\[\d+:\d+\], s\[8:9\], 0x60\b", r"s_buffer_load_dwordx16 s\[\d+:\d+\], s\[\d+:\d+\], 0x160\b",
             r"s_buffer_load_dwordx4 s\[\d+:\d+\], s\[\d+:\d+\], 0x30\b")
SKY = bytes((140, 173, 217))   # vk_sandbox's clear colour (0.55, 0.68, 0.85) in 8 bits
PS_LAYOUT = (r"s_load_dwordx8 s\[\d+:\d+\], s\[8:9\], 0x20\b", r"\bimage_sample\b", r"\bexp mrt0\b")


def _signature(entry):
    return [(e.get("semantic"), e.get("semantic_index"), e.get("mask")) for e in entry.get("input_signature") or []]


def _listing_matches(disasm_dir, name, patterns):
    try:
        with open(os.path.join(disasm_dir, name + ".s"), encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return False
    return text.count("image_sample") <= 1 and all(re.search(p, text) for p in patterns)


def pick_pair(shaders_dir):
    """(vs.spv, ps.spv) paths of the first matching pair in name order, or None."""
    spirv = os.path.join(shaders_dir, "spirv")
    disasm = os.path.join(shaders_dir, "dxbc", "disasm")
    try:
        with open(os.path.join(spirv, "index.json"), encoding="utf-8-sig") as f:
            index = json.load(f)
    except (OSError, ValueError):
        return None
    ptr8 = {"kind": "ptr32_lo", "sgpr": 8}
    vs = ps = None
    for name in sorted(index):
        e = index[name]
        if not os.path.isfile(os.path.join(spirv, name + ".spv")):
            continue
        if (vs is None and e.get("stage") == "vs" and not e.get("bindings")
                and e.get("user_sgpr_reads") == [ptr8]
                and _signature(e) == [("POSITION", 0, 7), ("NORMAL", 0, 7), ("TEXCOORD", 0, 3)]
                and _listing_matches(disasm, name, VS_LAYOUT)):
            vs = name
        elif (ps is None and e.get("stage") == "ps" and e.get("bindings") == [0, 1, 2]
              and ptr8 in (e.get("user_sgpr_reads") or [])
              and [s[0] for s in _signature(e)] == ["SV_Position", "TEXCOORD", "COLOR"]
              and _listing_matches(disasm, name, PS_LAYOUT)):
            ps = name
        if vs and ps:
            return os.path.join(spirv, vs + ".spv"), os.path.join(spirv, ps + ".spv")
    return None


def find_tool(build):
    for name in ("vk_sandbox.exe", "vk_sandbox"):
        path = os.path.join(build, name)
        if os.path.isfile(path):
            return path
    return None


def read_ppm(path):
    with open(path, "rb") as f:
        data = f.read()
    head = data.split(b"\n", 3)
    w, h = (int(v) for v in head[1].split())
    return w, h, head[3]


def check_frame(path, textured):
    """(ok, message) from the rendered frame: coverage and, with textures, colour variety."""
    w, h, px = read_ppm(path)
    sky = SKY
    covered, colours = 0, set()
    for i in range(0, len(px), 3):
        p = px[i:i + 3]
        if abs(p[0] - sky[0]) > 1 or abs(p[1] - sky[1]) > 1 or abs(p[2] - sky[2]) > 1:
            covered += 1
            colours.add(p)
    ok = covered >= w * h // 100 and (not textured or len(colours) >= 256)
    return ok, f"{covered} of {w * h} pixels covered, {len(colours)} distinct colours"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", required=True, help="make_game.py work folder (holds shaders/spirv and shaders/dxbc)")
    ap.add_argument("--entities", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--textures", action="append", default=[])
    ap.add_argument("--center", type=float, nargs=2, metavar=("X", "Y"))
    ap.add_argument("--radius", type=float, default=150.0)
    ap.add_argument("--max-props", type=int, default=400)
    ap.add_argument("--build", default=os.path.join(HERE, "..", "runtime", "build"))
    ap.add_argument("--scene", help="scene file to write (default: a temporary file)")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--window", action="store_true")
    mode.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)

    tool = find_tool(a.build)
    pair = pick_pair(os.path.join(a.work, "shaders"))
    if not tool or not pair or not os.path.isfile(a.entities):
        print("unsupported: " + ("vk_sandbox is not built" if not tool else
                                 "no matching translated shader pair" if not pair else "no entities file"))
        return 77
    tmp = frame = None
    scene = a.scene
    if not scene:
        fd, scene = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        tmp = scene
    try:
        try:
            stats = sandbox_scene.build(argparse.Namespace(entities=a.entities, models=a.models, textures=a.textures,
                                                           center=a.center, radius=a.radius, max_props=a.max_props,
                                                           tex_size=256, out=scene))
        except SystemExit as e:   # build() exits when no entity resolves to an exported model
            print(f"unsupported: {e}")
            return 77
        print(f"scene: {stats['placed']} props, {stats['draws']} draws, {stats['textures']} textures, "
              f"{stats['textured_draws']} textured draws")
        if stats["draws"] == 0:
            print("unsupported: no prop near the centre resolves to an exported model")
            return 77
        x, y, z = stats["center"]
        cam = [str(x - 40), str(y - 40), str(z + 25), "45", "-20"]
        if a.window:
            return subprocess.call([tool, scene, *pair, "--window", "--cam", *cam])
        frame = scene + ".ppm"
        r = subprocess.run([tool, scene, *pair, "--out", frame, "--size", "640", "360", "--cam", *cam],
                           capture_output=True, text=True)
        if r.returncode:
            print((r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else "vk_sandbox failed")
            return r.returncode
        ok, msg = check_frame(frame, stats["textured_draws"] > 0)
        print(("PASS: " if ok else "FAIL: ") + msg)
        return 0 if ok else 1
    finally:
        for path in (tmp, frame):
            if path and os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
