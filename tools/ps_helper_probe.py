r"""SHADER-681: measure pixel-shader helper lanes at a triangle edge on an AMD GPU (raw RDNA2 ISA, ps_oracle.py).

docs/design.md's whole-quad-mode row left "helper-invocation behaviour at triangle edges" untested, and the
translator's model of a pixel wave's entry EXEC (helpers off until s_wqm turns their quad on) came from the ISA guide
and ACO. This draws one triangle whose edge cuts 2x2 quads of a 16 x 16 target (vertices (-1, -1), (1.3, -1),
(-1, 0.9); the varying at location 0 is the vertex position, so every lane's interpolated value is its own pixel
centre in NDC) and runs seven wave64 programs as the pixel shader:

  masks, self       each covered lane exports its lane id, the wave's EXEC at entry and after s_wqm_b64
  nb12, nb3         after s_wqm, each covered lane reads its quad neighbours' interpolated x/y through DPP quad_perm,
                    so a helper's values reach the target through a covered pixel
  nowqm_ij          no s_wqm: a helper's raw I/J (v0/v1) and an unwritten VGPR, read with fetch-inactive DPP, and a
                    DPP read of an inactive lane without fi into a preloaded destination
  nowqm_masks       no s_wqm: EXEC at entry and later
  helper_export     after s_wqm (helpers on in EXEC) an `exp mrt0 ... done vm`: are pixels outside the triangle written?

The template pixel shader takes a derivative (OpDPdx), so the driver builds it as a shader that needs quads. Every
check and its result is printed; --json writes the raw words. Standard library only, no game files.

    python -I tools/ps_helper_probe.py --build <runtime build> [--json out.json]

Select the GPU with LEONIDA_VK_DEVICE, as for ps_oracle. Exit codes: 0 every check holds (the gfx1036
measurement), 1 a check differs, 2 a harness error, 77 no pixel-shader oracle on this device (see ps_oracle) or a
wave32 pixel stage (the programs are wave64).
Usage: `python -I tools/ps_helper_probe.py --help` lists the options.
"""
import argparse
import json
import os
import struct
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ps_oracle  # noqa: E402

SIZE = (16, 16)
TRI = ((-1.0, -1.0), (1.3, -1.0), (-1.0, 0.9))
CLEAR = (0xBF800000,) * 4            # vk_harness clears to -1.0
SENTINEL = 0x11111111

VS_EDGE = """\
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %vid %pos %out0
               OpDecorate %vid BuiltIn VertexIndex
               OpDecorate %pos BuiltIn Position
               OpDecorate %out0 Location 0
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %uint = OpTypeInt 32 0
        %int = OpTypeInt 32 1
      %float = OpTypeFloat 32
         %v4 = OpTypeVector %float 4
       %p_in = OpTypePointer Input %int
      %p_out = OpTypePointer Output %v4
         %u3 = OpConstant %uint 3
        %arr = OpTypeArray %v4 %u3
      %p_arr = OpTypePointer Function %arr
       %p_v4 = OpTypePointer Function %v4
        %fm1 = OpConstant %float -1
         %f0 = OpConstant %float 0
         %f1 = OpConstant %float 1
        %fx1 = OpConstant %float {bx}
        %fy2 = OpConstant %float {cy}
         %p0 = OpConstantComposite %v4 %fm1 %fm1 %f0 %f1
         %p1 = OpConstantComposite %v4 %fx1 %fm1 %f0 %f1
         %p2 = OpConstantComposite %v4 %fm1 %fy2 %f0 %f1
         %pa = OpConstantComposite %arr %p0 %p1 %p2
        %vid = OpVariable %p_in Input
        %pos = OpVariable %p_out Output
       %out0 = OpVariable %p_out Output
       %main = OpFunction %void None %fn
          %e = OpLabel
         %vp = OpVariable %p_arr Function
               OpStore %vp %pa
          %i = OpLoad %int %vid
         %pp = OpAccessChain %p_v4 %vp %i
          %p = OpLoad %v4 %pp
               OpStore %pos %p
               OpStore %out0 %p
               OpReturn
               OpFunctionEnd
""".format(bx=TRI[1][0], cy=TRI[2][1])

PS_DERIV = """\
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %in0 %out
               OpExecutionMode %main OriginUpperLeft
               OpDecorate %out Location 0
               OpDecorate %in0 Location 0
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
         %v4 = OpTypeVector %float 4
       %p_in = OpTypePointer Input %v4
      %p_out = OpTypePointer Output %v4
        %out = OpVariable %p_out Output
        %in0 = OpVariable %p_in Input
         %f1 = OpConstant %float 1
       %main = OpFunction %void None %fn
          %e = OpLabel
          %a = OpLoad %v4 %in0
          %x = OpCompositeExtract %float %a 0
          %y = OpCompositeExtract %float %a 1
         %dx = OpDPdx %float %x
          %r = OpCompositeConstruct %v4 %x %y %dx %f1
               OpStore %out %r
               OpReturn
               OpFunctionEnd
"""

DPP = " row_mask:0xf bank_mask:0xf"
# v0/v1 = I/J and M0 from s2, as --show-template prints for this template on gfx1036 (RADV)
HEAD = ["s_mov_b64 s[20:21], exec", "{wqm}", "s_mov_b64 s[22:23], exec", "s_mov_b32 m0, s2",
        "v_interp_p1_f32_e32 v2, v0, attr0.x", "v_interp_p2_f32_e32 v2, v1, attr0.x",
        "v_interp_p1_f32_e32 v3, v0, attr0.y", "v_interp_p2_f32_e32 v3, v1, attr0.y",
        "v_mbcnt_lo_u32_b32 v4, -1, 0", "v_mbcnt_hi_u32_b32 v4, -1, v4"]
RESTORE = ["s_mov_b64 exec, s[20:21]"]
X1, X2, X3 = "quad_perm:[1,0,3,2]", "quad_perm:[2,3,0,1]", "quad_perm:[3,2,1,0]"   # lane ^ 1, ^ 2, ^ 3
PROGRAMS = {
    "masks": (True, RESTORE + ["v_mov_b32_e32 v5, s20", "v_mov_b32_e32 v6, s21", "v_mov_b32_e32 v7, s22",
                               "exp mrt0, v4, v5, v6, v7 done vm", "s_endpgm"]),
    "self": (True, RESTORE + ["v_mov_b32_e32 v5, s23", "exp mrt0, v4, v5, v2, v3 done vm", "s_endpgm"]),
    "nb12": (True, [f"v_mov_b32_dpp v5, v2 {X1}{DPP}", f"v_mov_b32_dpp v6, v3 {X1}{DPP}",
                    f"v_mov_b32_dpp v7, v2 {X2}{DPP}", f"v_mov_b32_dpp v8, v3 {X2}{DPP}"] + RESTORE +
             ["exp mrt0, v5, v6, v7, v8 done vm", "s_endpgm"]),
    "nb3": (True, [f"v_mov_b32_dpp v5, v2 {X3}{DPP}", f"v_mov_b32_dpp v6, v3 {X3}{DPP}"] + RESTORE +
            ["exp mrt0, v5, v6, v5, v6 done vm", "s_endpgm"]),
    "nowqm_ij": (False, ["v_mov_b32_e32 v9, 0x7fc0dead",
                         f"v_mov_b32_dpp v5, v0 {X1}{DPP} fi:1", f"v_mov_b32_dpp v6, v1 {X1}{DPP} fi:1",
                         f"v_mov_b32_dpp v7, v2 {X1}{DPP} fi:1",
                         f"v_mov_b32_e32 v8, 0x{SENTINEL:08x}", f"v_mov_b32_dpp v8, v9 {X1}{DPP}",
                         "exp mrt0, v5, v6, v7, v8 done vm", "s_endpgm"]),
    "nowqm_masks": (False, ["v_mov_b32_e32 v5, s20", "v_mov_b32_e32 v6, s21", "v_mov_b32_e32 v7, s22",
                            "exp mrt0, v4, v5, v6, v7 done vm", "s_endpgm"]),
    "helper_export": (True, ["v_mov_b32_e32 v5, 0x3f800000", "exp mrt0, v5, v5, v5, v5 done vm", "s_endpgm"]),
}


def program(name):
    wqm, tail = PROGRAMS[name]
    return [ln.format(wqm="s_wqm_b64 exec, exec" if wqm else "s_nop 0") for ln in HEAD] + tail


def f32(w):
    return struct.unpack("<f", struct.pack("<I", w))[0]


def ndc(p, size=SIZE):
    """The NDC position of pixel p's centre (Vulkan: y down)."""
    return 2 * (p[0] + 0.5) / size[0] - 1, 2 * (p[1] + 0.5) / size[1] - 1


def bary(p, size=SIZE, tri=TRI):
    """(I, J) of pixel p's centre: P = A + I (B - A) + J (C - A), AMD's weights of vertices 1 and 2."""
    (ax, ay), (bx, by), (cx, cy) = tri
    px, py = ndc(p, size)
    px, py = px - ax, py - ay
    d = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
    return (px * (cy - ay) - (cx - ax) * py) / d, ((bx - ax) * py - px * (by - ay)) / d


def neighbour(p, k):
    """The pixel of lane ^ k in p's 2x2 quad."""
    return (p[0] ^ (k & 1), p[1] ^ (k >> 1))


def analyze(res, size=SIZE, tri=TRI):
    """[(check, ok, detail)] from res: {program name: {(x, y): 4 words}}."""
    cov = {p for p, w in res["masks"].items() if tuple(w) != CLEAR}
    lane = {p: res["masks"][p][0] for p in cov}
    entry = {p: res["masks"][p][1] | res["masks"][p][2] << 32 for p in cov}
    wqm = {p: res["masks"][p][3] | res["self"][p][1] << 32 for p in cov}
    helpers = {(p, k) for p in cov for k in (1, 2, 3) if neighbour(p, k) not in cov}
    out = [("partial quads at the edge", bool(helpers), f"{len(cov)} covered pixels, {len(helpers)} (covered lane, "
            "helper neighbour) pairs")]
    waves = {}
    for p in cov:
        waves.setdefault(entry[p], []).append(p)
    entry_ok, wqm_ok, packed, detail = True, True, True, []
    for e, pix in sorted(waves.items()):
        lanes = [lane[p] for p in pix]
        quads = {ln // 4 for ln in lanes}
        w = {wqm[p] for p in pix}
        bits = lambda m: {i for i in range(64) if m >> i & 1}   # noqa: E731
        entry_ok &= len(set(lanes)) == len(lanes) and bits(e) == set(lanes)
        wqm_ok &= len(w) == 1 and bits(w.pop() if len(w) == 1 else 0) == {4 * q + i for q in quads for i in range(4)}
        packed &= quads == set(range(len(quads)))
        detail.append(f"0x{e:016x}: {len(pix)} covered lanes in {len(quads)} quads")
    out.append(("entry EXEC = the covered pixels (helpers off)", entry_ok, f"{len(waves)} waves: " + "; ".join(detail)))
    out.append(("s_wqm turns on exactly the quads with a covered pixel", wqm_ok, ""))
    out.append(("quads packed from lane 0; none without a covered pixel", packed, ""))
    order = {(p[0] & 1) + 2 * (p[1] & 1) == lane[p] % 4 for p in cov}
    out.append(("lane % 4 = (x & 1) + 2 (y & 1): 0 top-left, 1 top-right, 2 bottom-left, 3 bottom-right",
                order == {True}, ""))
    err_c = max(max(abs(f32(res["self"][p][2]) - ndc(p, size)[0]), abs(f32(res["self"][p][3]) - ndc(p, size)[1]))
                for p in cov)
    err_h = 0.0
    for p, k in helpers:
        got = {1: res["nb12"][p][0:2], 2: res["nb12"][p][2:4], 3: res["nb3"][p][0:2]}[k]
        ex = ndc(neighbour(p, k), size)
        err_h = max(err_h, abs(f32(got[0]) - ex[0]), abs(f32(got[1]) - ex[1]))
    out.append(("after s_wqm a helper interpolates its own pixel centre, as precisely as a covered pixel",
                err_h <= 1.5 * err_c + 1e-6, f"helper |error| max {err_h:.2e}, covered {err_c:.2e}"))
    nm = res["nowqm_masks"]
    out.append(("without s_wqm EXEC stays the entry mask", all(
        (nm[p][1] | nm[p][2] << 32) == entry[p] and nm[p][3] == entry[p] & 0xFFFFFFFF for p in cov), ""))
    h1 = [p for p in cov if neighbour(p, 1) not in cov]
    ij = {p: (f32(res["nowqm_ij"][p][0]), f32(res["nowqm_ij"][p][1])) for p in h1}
    err_ij = max([max(abs(g[0] - bary(neighbour(p, 1), size, tri)[0]), abs(g[1] - bary(neighbour(p, 1), size, tri)[1]))
                  for p, g in ij.items()] or [0.0])
    out.append(("without s_wqm a helper's v0/v1 already hold its own pixel's I/J (outside: I + J > 1)",
                bool(ij) and err_ij < 1e-4 and all(i + j > 1 for i, j in ij.values()),
                f"{len(ij)} helpers, |error| max {err_ij:.2e}"))
    out.append(("DPP without fi from an inactive lane leaves the destination unchanged",
                bool(h1) and {res["nowqm_ij"][p][3] for p in h1} == {SENTINEL}, ""))
    written = [p for p, w in res["helper_export"].items() if p not in cov and tuple(w) != CLEAR]
    out.append(("an exp ... vm with helpers on in EXEC writes no pixel outside the triangle", not written,
                f"{len(written)} written"))
    return out


def coverage_map(res, size=SIZE):
    cov = {p for p, w in res["masks"].items() if tuple(w) != CLEAR}
    return ["".join("#" if (x, y) in cov else "." for x in range(size[0])) for y in range(size[1])]


def measure(build):
    """(device, {program: {(x, y): words}}) on the selected GPU."""
    with tempfile.TemporaryDirectory(prefix="ps_helper_probe_") as d:
        vs, ps = os.path.join(d, "vs_edge.spvasm"), os.path.join(d, "ps_deriv.spvasm")
        for path, text in ((vs, VS_EDGE), (ps, PS_DERIV)):
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
        return _run(ps_oracle.Oracle(build, vs=vs, ps=ps, size=SIZE, work=os.path.join(d, "work")))


def _run(o):
    try:
        if not o.wave64:
            raise ps_oracle.Unavailable("the pixel stage is wave32; these programs are wave64")
        res = {}
        for name in PROGRAMS:
            px = o.run(program(name))
            res[name] = {(i % SIZE[0], i // SIZE[0]): tuple(w) for i, w in enumerate(px)}
        return o.device, res
    finally:
        o.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", required=True, help="runtime build folder with vk_harness")
    ap.add_argument("--json", help="write {device, checks, words} here")
    a = ap.parse_args(argv)
    try:
        device, res = measure(a.build)
    except ps_oracle.Unavailable as e:
        print(f"ps_helper_probe unavailable: {e}")
        return 77
    except (OSError, ValueError, RuntimeError) as e:
        print(f"ps_helper_probe: {e}")
        return 2
    print(f"device: {device}")
    print("coverage (# covered):")
    for row in coverage_map(res):
        print("  " + row)
    checks = analyze(res)
    for name, ok, detail in checks:
        print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f" ({detail})" if detail else ""))
    good = sum(ok for _n, ok, _d in checks)
    print(f"{good}/{len(checks)} checks hold")
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"device": device, "checks": [[n, ok, d] for n, ok, d in checks],
                       "words": {n: {f"{x},{y}": list(w) for (x, y), w in px.items()} for n, px in res.items()}},
                      fh, indent=1)
    return 0 if good == len(checks) else 1


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
