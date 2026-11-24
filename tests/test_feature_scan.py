"""feature_scan: stage census, SFI0 flags, output streams, MMA/GDS opcodes and code-base counts on a synthetic work folder.

    python -I tests/test_feature_scan.py

Every input is built here: tiny DXBC containers with only the parts the scanner reads, three hand-written .s
listings and a two-entry spirv/index.json. No game data.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile

TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
sys.path.insert(0, TOOLS)
import feature_scan as F  # noqa: E402

failures = 0


def check(cond, what):
    global failures
    print(("PASS " if cond else "FAIL ") + what)
    failures += not cond


def signature(streams):
    """ISG1/OSG1 payload: count, first-element offset, 32-byte elements with the stream index first."""
    body = struct.pack("<II", len(streams), 8)
    for i, s in enumerate(streams):
        body += struct.pack("<6I", s, 0, 0, 0, 3, i) + bytes([0xF, 0xF, 0, 0]) + struct.pack("<I", 0)
    return body


def short_signature():
    """An OSG1 part whose declared five elements do not fit in the eight payload bytes present."""
    return struct.pack("<II", 5, 8) + bytes(8)


def dxbc(parts):
    """A DXBC container: 32-byte header (part count last), part offsets, then fourcc + size + payload per part."""
    head = 32 + 4 * len(parts)
    offsets, blob = [], b""
    for fourcc, payload in parts:
        offsets.append(head + len(blob))
        blob += fourcc.encode() + struct.pack("<I", len(payload)) + payload
    total = head + len(blob)
    header = b"DXBC" + bytes(16) + struct.pack("<II", 1, total) + struct.pack("<I", len(parts))
    return header + b"".join(struct.pack("<I", o) for o in offsets) + blob


def build(root):
    sh = os.path.join(root, "shaders")
    dx = os.path.join(sh, "dxbc")
    os.makedirs(os.path.join(dx, "disasm"))
    os.makedirs(os.path.join(sh, "spirv"))
    shaders = [
        # name, stage, SFI0 mask, output streams, listing
        ("a_ps", "ps", 0x1000 | 0x80000, [0], "\tv_mov_b32 v0, v1\n\ts_endpgm\n"),
        ("b_gs", "gs", 0, [0, 1], "\tds_ordered_count v0, v1 offset:772 gds\n\ts_endpgm\n"),
        ("c_cs", "cs", 0x8000000, [], "\tv_mfma_f32_32x32x8f16 a[0:15], v[0:1], v[2:3], a[0:15]\n\ts_endpgm\n"),
        ("d_vs", "vs", 0, [0], None),
        # s_getpc for data (the address is kept and offset), and s_getpc for a far branch (setpc follows)
        ("e_vs", "vs", 0, [0], "\ts_getpc_b64 s[4:5]\n\ts_add_u32 s4, s4, 0x40\n\ts_load_dword s6, s[4:5], 0x0\n\ts_endpgm\n"),
        ("f_ps", "ps", 0, [0], "\ts_getpc_b64 s[4:5]\n\ts_add_u32 s4, s4, 0x40\n\ts_addc_u32 s5, s5, 0\n\ts_setpc_b64 s[4:5]\n"),
    ]
    index = []
    for name, stage, mask, streams, listing in shaders:
        parts = [("SFI0", struct.pack("<Q", mask)), ("OSG1", signature(streams))]
        with open(os.path.join(dx, name + ".dxbc"), "wb") as f:
            f.write(dxbc(parts))
        index.append({"name": name, "file": name + ".dxbc", "stage": stage})
        if listing:
            with open(os.path.join(dx, "disasm", name + ".s"), "w", encoding="utf-8") as f:
                f.write(f"; {name}\n" + listing)
    with open(os.path.join(dx, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f)
    with open(os.path.join(sh, "spirv", "index.json"), "w", encoding="utf-8") as f:
        json.dump({"a_ps": {"stage": "ps", "uses_code_base": True}, "d_vs": {"stage": "vs"}}, f)
    return sh, index


def test_feature_scan():
    """Runs under pytest (CI's translator-tests job) and as a plain script."""
    with tempfile.TemporaryDirectory() as tmp:
        sh, index = build(tmp)
        d = F.scan_dxbc(os.path.join(sh, "dxbc"), index)
        check(d["stages"]["ps"] == 2 and d["stages"]["gs"] == 1 and d["stages"]["hs"] == 0 and d["stages"]["as"] == 0,
              "stage census counts each stage and lists the absent ones as 0")
        check(d["sfi0"].get("ROVS") == {"ps": 1} and d["sfi0"].get("SHADING_RATE") == {"ps": 1},
              "SFI0 bits 0x1000 and 0x80000 decode to ROVS and SHADING_RATE")
        check(d["sfi0"].get("WAVE_MMA") == {"cs": 1}, "SFI0 bit 0x8000000 decodes to WAVE_MMA")
        check(d["output_on_stream_other_than_0"] == {"gs": 1} and d["gs_output_stream_count"] == {2: 1},
              "an OSG1 element on stream 1 is counted, stream 0 alone is not")

        stage_of = {e["name"]: e["stage"] for e in index}
        s = F.scan_disasm(os.path.join(sh, "dxbc", "disasm"), stage_of)
        check(s["files"] == 5, "five listings scanned")
        check(s["getpc_for_data_shaders"] == {"vs": 1},
              "s_getpc kept for data counts as a code base, s_getpc into s_setpc (far branch) does not")
        check(s["wave_mma_shaders"] == {"cs": 1} and "v_mfma_f32_32x32x8f16" in s["wave_mma_ops"], "v_mfma is found")
        check(s["ordered_or_gws_shaders"] == {"gs": 1} and s["gds_shaders"] == {"gs": 1},
              "ds_ordered_count with gds is found")
        check("ps" not in s["wave_mma_shaders"] and "ps" not in s["gds_shaders"], "a plain listing has neither")

        sp = F.scan_spirv(os.path.join(sh, "spirv", "index.json"))
        check(sp["uses_code_base"] == {"ps": 1} and sp["translated"] == {"ps": 1, "vs": 1}, "uses_code_base per stage")

        r = subprocess.run([sys.executable, "-I", os.path.join(TOOLS, "feature_scan.py"), sh, "--out",
                            os.path.join(tmp, "out.json")], capture_output=True, text=True)
        check(r.returncode == 0 and "SFI0 WAVE_MMA: 1" in r.stdout and "hs 0" in r.stdout, "the CLI prints the summary")
        check(os.path.isfile(os.path.join(tmp, "out.json")), "--out writes the JSON")
        # --out in a folder that does not exist creates it, instead of losing the scan to a traceback (#934's class).
        nested = os.path.join(tmp, "nested", "dir", "out.json")
        r = subprocess.run([sys.executable, "-I", os.path.join(TOOLS, "feature_scan.py"), sh, "--out", nested],
                           capture_output=True, text=True)
        check(r.returncode == 0 and "Traceback" not in r.stderr, "--out creates a missing folder")
        check(os.path.isfile(nested), "the JSON lands in the created folder")
        check("a_ps" not in r.stdout and "b_gs" not in r.stdout, "the summary prints counts, not shader names")
    assert failures == 0, f"{failures} feature_scan check(s) failed"


def test_short_signature():
    """A malformed signature is ignored instead of aborting the count-only scan with struct.error."""
    check(F.streams(short_signature()) == set(),
          "a declared element table longer than the part yields no streams")
    check(F.streams(struct.pack("<II", 1, 0xFFFF)) == set(),
          "an element offset past the part yields no streams")
    check(F.streams(struct.pack("<II", 1, 0) + bytes(32)) == set(),
          "an element offset inside the count/offset header is ignored")
    check(F.streams(signature([0])) == {0} and F.streams(signature([0, 1])) == {0, 1},
          "well-formed signatures still report every element's stream")

    with tempfile.TemporaryDirectory() as tmp:
        sh, _ = build(tmp)
        dx = os.path.join(sh, "dxbc")
        with open(os.path.join(dx, "bad.dxbc"), "wb") as f:
            f.write(dxbc([("SFI0", struct.pack("<Q", 0)), ("OSG1", short_signature())]))
        with open(os.path.join(dx, "index.json"), encoding="utf-8") as f:
            index = json.load(f)
        index.append({"name": "bad", "file": "bad.dxbc", "stage": "ps"})
        with open(os.path.join(dx, "index.json"), "w", encoding="utf-8") as f:
            json.dump(index, f)

        d = F.scan_dxbc(dx, index)
        check(d["stages"]["ps"] == 3 and d["unreadable"] == 0,
              "scan_dxbc still counts a shader whose OSG1 is truncated")
        check(d["output_on_stream_other_than_0"] == {"gs": 1},
              "the truncated signature adds no bogus stream and keeps the real one")
        r = subprocess.run([sys.executable, "-I", os.path.join(TOOLS, "feature_scan.py"), sh],
                           capture_output=True, text=True)
        check(r.returncode == 0 and "SFI0 WAVE_MMA: 1" in r.stdout,
              "the CLI still prints the summary with a truncated OSG1 present")
    assert failures == 0, f"{failures} short-signature check(s) failed"


def test_part_payload_outside_container_is_unreadable():
    # Allow the flags to fit physically, while the declared part extends past EOF.
    data = bytearray(dxbc([("SFI0", struct.pack("<Q", 0x8000000))]))
    offset, = struct.unpack_from("<I", data, 32)
    struct.pack_into("<I", data, offset + 4, 16)
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "bad.dxbc"), "wb") as f:
            f.write(data)
        with open(os.path.join(tmp, "valid.dxbc"), "wb") as f:
            f.write(dxbc([("SFI0", struct.pack("<Q", 0x8000000))]))
        rows = [{"file": "bad.dxbc", "stage": "ps"}, {"file": "valid.dxbc", "stage": "gs"}]
        result = F.scan_dxbc(tmp, rows)
        check(result["unreadable"] == 1, "a DXBC part extending past EOF counts as unreadable")
        check(result["sfi0"] == {"WAVE_MMA": {"gs": 1}}, "only the valid file contributes feature flags")
    assert failures == 0, f"{failures} DXBC bounds check(s) failed"


def test_bad_shaders_folder_is_one_line():
    tool = os.path.join(TOOLS, "feature_scan.py")
    with tempfile.TemporaryDirectory() as tmp:
        as_file = os.path.join(tmp, "shaders.txt")
        with open(as_file, "w", encoding="utf-8") as f:
            f.write("not a folder")
        no_index = os.path.join(tmp, "no_index")
        os.makedirs(os.path.join(no_index, "dxbc"))
        broken = os.path.join(tmp, "broken")
        os.makedirs(os.path.join(broken, "dxbc"))
        with open(os.path.join(broken, "dxbc", "index.json"), "w", encoding="utf-8") as f:
            f.write("[{")
        not_list = os.path.join(tmp, "not_list")
        os.makedirs(os.path.join(not_list, "dxbc"))
        with open(os.path.join(not_list, "dxbc", "index.json"), "w", encoding="utf-8") as f:
            json.dump({"files": []}, f)
        out = os.path.join(tmp, "out.json")
        for src, expect in ((os.path.join(tmp, "missing"), "does not exist"), (as_file, "is not a folder"),
                            (no_index, "Run the shaders stage"), (broken, "cannot read"),
                            (not_list, "is not a shader list")):
            r = subprocess.run([sys.executable, "-I", tool, src, "--out", out], capture_output=True, text=True)
            check(r.returncode == 2 and expect in r.stderr and "Traceback" not in r.stderr and not r.stdout,
                  f"{os.path.basename(src)}: exit 2 with one line ({expect!r}), got {r.returncode}: "
                  f"{r.stderr.strip().splitlines()[-1:] if r.stderr.strip() else ''}")
        check(not os.path.exists(out), "no --out is written for a bad shaders folder")
    assert failures == 0, f"{failures} bad-input check(s) failed"



def test_out_inside_corpus_is_a_usage_error():
    """--out used to replace the index (or a .dxbc) it had just read, and the next run could not scan."""
    tool = os.path.join(TOOLS, "feature_scan.py")
    with tempfile.TemporaryDirectory() as tmp:
        sh, _ = build(tmp)
        index_path = os.path.join(sh, "dxbc", "index.json")
        shader_path = os.path.join(sh, "dxbc", "a_ps.dxbc")
        spirv_path = os.path.join(sh, "spirv", "index.json")
        before = {p: open(p, "rb").read() for p in (index_path, shader_path, spirv_path)}
        out = os.path.join(tmp, "report.json")
        for bad, what in ((index_path, "the corpus index"), (shader_path, "a scanned .dxbc"),
                          (os.path.join(sh, "dxbc", "disasm", "new.s"), "a file in dxbc/disasm"),
                          (spirv_path, "spirv/index.json")):
            r = subprocess.run([sys.executable, "-I", tool, sh, "--out", bad], capture_output=True, text=True)
            check(r.returncode == 2 and "--out" in r.stderr and "is not a shader list" not in r.stderr
                  and "Traceback" not in r.stderr and not r.stdout,
                  f"--out naming {what}: exit 2 naming the path, got {r.returncode}: {r.stderr.strip()!r}")
        check(all(open(p, "rb").read() == data for p, data in before.items()),
              "no input is modified when --out names one")
        check(not os.path.exists(out), "nothing is written when --out is refused")
        for good in (out, os.path.join(sh, "report.json"), os.path.join(tmp, "deep", "er", "report.json")):
            r = subprocess.run([sys.executable, "-I", tool, sh, "--out", good], capture_output=True, text=True)
            check(r.returncode == 0 and os.path.isfile(good), f"--out {os.path.basename(good)} still writes (rc={r.returncode})")
        r = subprocess.run([sys.executable, "-I", tool, sh], capture_output=True, text=True)
        check(r.returncode == 0 and "shaders" in r.stdout, "a run without --out still scans")
    assert failures == 0, f"{failures} --out guard check(s) failed"


def test_out_via_directory_alias_into_corpus_is_a_usage_error():
    tool = os.path.join(TOOLS, "feature_scan.py")
    with tempfile.TemporaryDirectory() as tmp:
        sh, _ = build(tmp)
        corpus = os.path.join(sh, "dxbc")
        alias = os.path.join(tmp, "corpus-alias")
        if os.name == "nt":
            # A junction works even when the Windows runner lacks symlink privileges.
            subprocess.run(["cmd.exe", "/d", "/c", "mklink", "/J", os.path.basename(alias),
                            os.path.relpath(corpus, tmp)], cwd=tmp, check=True)
        else:
            os.symlink(corpus, alias, target_is_directory=True)
        index_path = os.path.join(corpus, "index.json")
        with open(index_path, "rb") as f:
            before = f.read()
        for name in ("report.json", "index.json"):
            output = os.path.join(alias, name)
            r = subprocess.run([sys.executable, "-I", tool, sh, "--out", output],
                               capture_output=True, text=True)
            assert r.returncode == 2, (output, r.returncode, r.stdout, r.stderr)
            assert "inside the scanned corpus" in r.stderr, r.stderr
        assert not os.path.exists(os.path.join(corpus, "report.json"))
        with open(index_path, "rb") as f:
            assert f.read() == before


def test_out_hard_link_to_carved_index_is_a_usage_error():
    tool = os.path.join(TOOLS, "feature_scan.py")
    with tempfile.TemporaryDirectory() as tmp:
        sh, _ = build(tmp)
        index_path = os.path.join(sh, "dxbc", "index.json")
        report = os.path.join(tmp, "report.json")
        os.link(index_path, report)
        with open(index_path, "rb") as f:
            before = f.read()
        r = subprocess.run([sys.executable, "-I", tool, sh, "--out", report],
                           capture_output=True, text=True)
        assert r.returncode == 2, (r.returncode, r.stderr)
        assert index_path in r.stderr, r.stderr
        with open(index_path, "rb") as f:
            assert f.read() == before


def test_duplicate_shader_names_keep_file_stages():
    with tempfile.TemporaryDirectory() as tmp:
        dx = os.path.join(tmp, "dxbc")
        dis = os.path.join(dx, "disasm")
        os.makedirs(dis)
        index = []
        for stem, stage in (("same", "gs"), ("same~1", "ms")):
            with open(os.path.join(dx, stem + ".dxbc"), "wb") as f:
                f.write(dxbc([]))
            with open(os.path.join(dis, stem + ".s"), "w", encoding="utf-8") as f:
                f.write("ds_ordered_count v0 gds\n")
            index.append({"name": "same", "file": stem + ".dxbc", "stage": stage})
        with open(os.path.join(dx, "index.json"), "w", encoding="utf-8") as f:
            json.dump(index, f)

        out = os.path.join(tmp, "out.json")
        subprocess.run([sys.executable, "-I", os.path.join(TOOLS, "feature_scan.py"), tmp, "--out", out],
                       capture_output=True, text=True, check=True)
        with open(out, encoding="utf-8") as f:
            ordered = json.load(f)["disasm"]["ordered_or_gws_shaders"]
        check(ordered == {"gs": 1, "ms": 1}, f"ordered-count shaders keep their file stages: {ordered}")


def test_v_perm_b32_selectors():
    """v_perm_b32 selector bytes: immediate, s_mov/v_mov const-reg, clobber invalidation, and no name leakage."""
    with tempfile.TemporaryDirectory() as tmp:
        sh = os.path.join(tmp, "shaders")
        dx = os.path.join(sh, "dxbc")
        dis = os.path.join(dx, "disasm")
        os.makedirs(dis)
        synthetic = [
            # Immediate selector 0x0c050600 -> bytes 0, 6, 5 (three 0..7) and 12 (one 12)
            ("perm_imm_ps", "ps",
             "\tv_perm_b32 v0, v1, v2, 0x0c050600\n"
             "\ts_endpgm\n"),
            # Constant SGPR selector 0x0b0a0908 -> bytes 8, 9, 10, 11 (four 8..11 sign-extend)
            ("perm_sreg_cs", "cs",
             "\ts_mov_b32 s4, 0x0b0a0908\n"
             "\tv_perm_b32 v3, v1, v2, s4\n"
             "\ts_endpgm\n"),
            # Constant VGPR selector 0x0d0e0f08 -> bytes 8 (one 8..11) and 15, 14, 13 (three >= 13)
            ("perm_vreg_vs", "vs",
             "\tv_mov_b32_e32 v6, 0x0d0e0f08\n"
             "\tv_perm_b32 v7, v1, v2, v6\n"
             "\ts_endpgm\n"),
            # Clobbered SGPR and VGPR before v_perm_b32 -> both unresolved
            ("perm_clobber_ps", "ps",
             "\ts_mov_b32 s5, 0x0c050600\n"
             "\ts_add_u32 s5, s5, 1\n"
             "\tv_perm_b32 v4, v1, v2, s5\n"
             "\tv_mov_b32 v8, 0x0c050600\n"
             "\tv_add_nc_u32 v8, v8, v1\n"
             "\tv_perm_b32 v9, v1, v2, v8\n"
             "\ts_endpgm\n"),
            # The scalar carry is the second destination; wave64 writes both halves of its pair.
            ("perm_carry_w32", "cs",
             "\ts_mov_b32 s4, 0x08080808\n"
             "\tv_add_co_u32 v8, s4, v1, v2\n"
             "\tv_perm_b32 v9, v1, v2, s4\n"
             "\ts_endpgm\n"),
            ("perm_carry_w64", "vs",
             "\ts_mov_b32 s5, 0x08080808\n"
             "\tv_add_co_u32 v8, s[4:5], v1, v2\n"
             "\tv_perm_b32 v9, v1, v2, s5\n"
             "\ts_endpgm\n"),
        ]
        index = []
        for name, stage, listing in synthetic:
            with open(os.path.join(dx, name + ".dxbc"), "wb") as f:
                f.write(dxbc([("SFI0", struct.pack("<Q", 0)), ("OSG1", signature([0]))]))
            index.append({"name": name, "file": name + ".dxbc", "stage": stage})
            with open(os.path.join(dis, name + ".s"), "w", encoding="utf-8") as f:
                f.write(f"; {name}\n" + listing)
        with open(os.path.join(dx, "index.json"), "w", encoding="utf-8") as f:
            json.dump(index, f)

        stage_of = {e["name"]: e["stage"] for e in index}
        s = F.scan_disasm(dis, stage_of)
        vp = s["v_perm_b32"]
        check(vp["shaders"] == {"ps": 2, "cs": 2, "vs": 2}, f"v_perm_b32 shader counts by stage: {vp['shaders']}")
        check(vp["insns"] == 7, f"v_perm_b32 total instructions == 7 (got {vp['insns']})")
        check((vp["immediate"], vp["const_reg"], vp["unresolved"]) == (1, 2, 4),
              f"v_perm_b32 resolution split (1 imm, 2 const_reg, 4 unresolved), got {vp}")
        check(vp["sign_extend_8_11_shaders"] == 2 and vp["sign_extend_8_11_insns"] == 2,
              f"v_perm_b32 8..11 sign-extend shader/insn counts == (2, 2), got {vp}")
        check(vp["selector_classes"] == {"0_7": 3, "8_11": 5, "12": 1, "13_plus": 3},
              f"v_perm_b32 selector byte class counts: {vp['selector_classes']}")

        out_json = os.path.join(tmp, "out.json")
        r = subprocess.run([sys.executable, "-I", os.path.join(TOOLS, "feature_scan.py"), sh, "--out", out_json],
                           capture_output=True, text=True)
        check(r.returncode == 0 and "v_perm_b32: 6 shaders" in r.stdout and "selectors 8-11: 2 shaders / 2 insns" in r.stdout,
              f"CLI prints v_perm_b32 summary counts (rc={r.returncode})")
        raw_json = open(out_json, encoding="utf-8").read()
        for name, _, _ in synthetic:
            check(name not in r.stdout and name not in raw_json,
                  f"synthetic shader name {name!r} does not leak into stdout or JSON")
    assert failures == 0, f"{failures} v_perm_b32 check(s) failed"


def psv0(records, size=24, truncate=False):
    """PSV0 payload: a 24-byte runtime info, then the resource count, record size and (type, kind) records."""
    body = struct.pack("<I", 24) + bytes(24) + struct.pack("<II", len(records), size)
    for t, k in records:
        rec = struct.pack("<4I", t, 0, 0, 0) + (struct.pack("<II", k, 0) if size >= 24 else b"")
        body += rec + bytes(size - len(rec))
    return body[:-4] if truncate else body


def listing(lines):
    """A .s listing with `; encoding:` comments of the right length, so pcs and branch targets are real."""
    return "".join(f"\t{code} ; encoding: [{','.join(['0x00'] * n)}]\n" for code, n in lines)


def test_typed_buffers_and_psv0():
    """§8.2 #16: PSV0 counter/typed-buffer bindings, MTBUF formats and where each MUBUF V# word 3 comes from."""
    with tempfile.TemporaryDirectory() as tmp:
        sh = os.path.join(tmp, "shaders")
        dx = os.path.join(sh, "dxbc")
        dis = os.path.join(dx, "disasm")
        os.makedirs(dis)
        cs = listing([
            ("s_load_dwordx4 s[8:11], s[0:1], 0x0", 8),                         # pc 0
            ("s_mov_b32 s15, 0x38000", 8),                                      # 8: format 56, 8_8_8_8_UNORM
            ("s_mov_b32 s19, 0x24000", 8),                                      # 16: format 36, 10_11_11_FLOAT
            ("s_or_b32 s23, s23, 0x1000", 8),                                   # 24: computed
            ("s_cmp_eq_u32 s11, 0", 4),                                         # 32: reads s11, writes nothing
            ("buffer_load_format_xyzw v[0:3], v4, s[8:11], 0 idxen", 8),        # 36: memory
            ("buffer_load_format_x v5, v4, s[12:15], 0 idxen", 8),              # 44: literal
            ("buffer_store_format_x v5, v4, s[16:19], 0 idxen", 8),             # 52: literal, not in the table
            ("buffer_load_format_x v6, v4, s[20:23], 0 idxen", 8),              # 60: computed
            ("buffer_load_format_x v7, v4, s[24:27], 0 idxen", 8),              # 68: never written: shader input
            ("s_cbranch_scc0 1", 4),                                            # 76: to 84; 80 and 84 lead blocks
            ("s_nop 0", 4),                                                     # 80
            ("buffer_load_format_x v8, v4, s[8:11], 0 idxen", 8),               # 84: loaded in an earlier block
            ("buffer_load_format_x v9, v4, s[12:15], 0 idxen", 8),              # 92: literal from an earlier block
            ("tbuffer_load_format_xy v[0:1], v4, s[8:11], 0 format:[BUF_FMT_16_16_SNORM] idxen", 8),
            ("tbuffer_load_format_x v2, v4, s[8:11], 0 idxen", 8),             # no format: GFX10 default 8_UNORM
            ("tbuffer_store_format_x v2, v4, s[8:11], 0 format:[BUF_FMT_10_11_11_FLOAT] idxen", 8),
            ("s_endpgm", 4)])
        ps = listing([("s_mov_b32 s11, 0", 4),                                    # word 3 0: FORMAT INVALID
                      ("buffer_load_format_x v0, v1, s[0:3], 0 idxen", 8),
                      ("buffer_load_format_x v2, v1, s[8:11], 0 idxen", 8), ("s_endpgm", 4)])
        vs = listing([("v_mov_b32_e32 v0, v1", 4), ("s_endpgm", 4)])
        synthetic = [
            ("tb_cs", "cs", cs, psv0([(9, 12), (3, 10), (6, 2)])),     # counter UAV, typed buffer SRV, typed 2D UAV
            ("tb_ps", "ps", ps, psv0([(6, 10), (6, 10)])),             # two typed buffer UAVs
            ("tb_vs", "vs", vs, psv0([(3, 10)], size=16)),             # v0 records carry no kind
            ("tb_gs", "gs", None, psv0([(9, 12)], truncate=True)),     # records past the end of the part
        ]
        index = []
        for name, stage, text, psv in synthetic:
            with open(os.path.join(dx, name + ".dxbc"), "wb") as f:
                f.write(dxbc([("SFI0", struct.pack("<Q", 0)), ("OSG1", signature([0])), ("PSV0", psv)]))
            index.append({"name": name, "file": name + ".dxbc", "stage": stage})
            if text:
                with open(os.path.join(dis, name + ".s"), "w", encoding="utf-8") as f:
                    f.write(f"; {name}\n" + text)
        with open(os.path.join(dx, "index.json"), "w", encoding="utf-8") as f:
            json.dump(index, f)

        pv = F.scan_dxbc(dx, index)["psv0"]
        check(pv["records"] == 6 and pv["unreadable"] == 1, f"PSV0: 6 records read, 1 truncated table ({pv})")
        check(pv["shaders"] == {"uav_with_counter": 1, "typed_buffer_srv": 1, "typed_buffer_uav": 1},
              f"PSV0 shaders with a counter UAV / typed buffer SRV / typed buffer UAV ({pv['shaders']})")
        check(pv["records_of"] == {"uav_with_counter": 1, "typed_buffer_srv": 1, "typed_buffer_uav": 2},
              "a typed UAV of a texture kind and a v0 record without a kind are not typed buffers")
        check(pv["by_type"] == {"UAVStructuredWithCounter": 1, "SRVTyped": 2, "UAVTyped": 3}, f"by type {pv['by_type']}")

        tb = F.scan_typed_buffers(dis, {e["name"]: e["stage"] for e in index})
        check(tb["mtbuf"] == {"load 16_16_SNORM": 1, "load 8_UNORM": 1, "store 10_11_11_FLOAT": 1},
              f"MTBUF formats from the instruction, 8_UNORM without format: ({tb['mtbuf']})")
        check(tb["mubuf_word3"] == {"load computed": 1, "load earlier_literal": 1, "load earlier_memory": 1,
                                    "load literal": 2, "load memory": 1, "load shader_input": 2,
                                    "store literal": 1},
              f"MUBUF word 3 by source, s_cmp is no write, a branch target starts a block ({tb['mubuf_word3']})")
        check(tb["mubuf_literal_formats"] == {"load 8_8_8_8_UNORM": 2, "load INVALID": 1,
                                              "store 10_11_11_FLOAT": 1}, f"literal word 3 formats ({tb['mubuf_literal_formats']})")
        # 10_11_11_FLOAT stores are in rdna2spirv.Gen.FORMATS since #2022/#2432, so nothing here is outside it
        check(tb["not_in_translator"] == {},
              f"formats outside rdna2spirv.Gen.FORMATS are listed, INVALID (reads 0) is not ({tb['not_in_translator']})")
        check(tb["shaders"] == {"mtbuf": {"cs": 1}, "mubuf": {"cs": 1, "ps": 1}, "mubuf_literal": {"cs": 1, "ps": 1},
                                "mubuf_runtime_format": {"cs": 1, "ps": 1}}, f"shaders per stage ({tb['shaders']})")

        out_json = os.path.join(tmp, "out.json")
        r = subprocess.run([sys.executable, "-I", os.path.join(TOOLS, "feature_scan.py"), sh, "--out", out_json],
                           capture_output=True, text=True)
        check(r.returncode == 0 and "PSV0 bindings: 6 records" in r.stdout and "uav_with_counter 1" in r.stdout
              and "10_11_11_FLOAT" in r.stdout, f"CLI prints the #16 summary (rc={r.returncode})")
        raw_json = open(out_json, encoding="utf-8").read()
        check(all(n not in r.stdout and n not in raw_json for n, *_ in synthetic), "no shader name in stdout or JSON")
    assert failures == 0, f"{failures} typed-buffer check(s) failed"


def test_a_test_that_raises_fails_the_run_and_the_rest_still_run():
    """The plain-script run used to hold every test in one try block that swallowed AssertionError: a failing bare
    assert (test_duplicate_shader_names_keep_file_stages has one) skipped the tests after it and printed OK, exit 0."""
    ran, reported = [], []

    def raises():
        ran.append("raises")
        raise AssertionError("a bare assert failed")

    def passes():
        ran.append("passes")

    raised = run_tests([raises, passes], report=reported.append)
    check(raised == ["raises"], f"a test that raises is reported as failed ({raised})")
    check(ran == ["raises", "passes"], f"the tests after it still run ({ran})")
    check(reported == ["FAIL raises: a bare assert failed"], f"its message is printed ({reported})")
    assert failures == 0, f"{failures} script-runner check(s) failed"


def test_a_failed_check_fails_only_its_own_test():
    """Each test ends in `assert failures == 0` over the module-wide count, so one failed check used to fail every
    test after it as well. run_tests gives each test its own count and adds it back to the run's total."""
    global failures
    reported = []

    def fails_a_check():
        global failures
        failures += 1   # what check(False, ...) counts, without printing a FAIL line into this run's output
        assert failures == 0, f"{failures} check(s) failed"

    def passes():
        assert failures == 0, f"{failures} check(s) failed"

    before = failures
    raised = run_tests([fails_a_check, passes], report=reported.append)
    counted, failures = failures - before, before   # the failure above was staged: keep it out of this run's total
    check(raised == ["fails_a_check"], f"only the test with the failed check fails ({raised})")
    check(counted == 1, f"its failed check still counts toward the run's total ({counted})")
    # first line only: under pytest the assert's message gets pytest's "assert 1 == 0" explanation appended
    check([r.splitlines()[0] for r in reported] == ["FAIL fails_a_check: 1 check(s) failed"],
          f"with its own count in the message ({reported})")
    assert failures == 0, f"{failures} script-runner check(s) failed"


def run_tests(tests, report=print):
    """Run each test on its own, for the plain-script run (CI's standard-library job): one that raises
    AssertionError is reported and the rest still run. Each test starts from a zero check count, so its closing
    `assert failures == 0` sees only its own failed checks; the counts are added back to the run's total afterwards.
    Returns the names of the tests that raised."""
    global failures
    raised = []
    for test in tests:
        before, failures = failures, 0
        try:
            test()
        except AssertionError as error:
            raised.append(test.__name__)
            report(f"FAIL {test.__name__}: {error}")
        finally:
            failures += before
    return raised


if __name__ == "__main__":
    # Every test_* function in the file, in file order, so a new test can't be left out of the script run.
    raised = run_tests([fn for name, fn in list(globals().items()) if name.startswith("test_") and callable(fn)])
    ok = not failures and not raised
    print(f"test_feature_scan: {'OK' if ok else 'FAILED'} ({failures} failed check{'s' if failures != 1 else ''}, "
          f"{len(raised)} failed test{'s' if len(raised) != 1 else ''})")
    sys.exit(0 if ok else 1)
