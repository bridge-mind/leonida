"""tools/ps_oracle.py: the msgpack subset, the PAL ELF rewrite that puts new pixel-shader code in a pipeline binary,
the binary file format, and (with LEONIDA_PS_ORACLE_BUILD set to a runtime build and an AMD GPU selected) a real
draw with patched code. Standard library only."""
import os
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import ps_oracle as P  # noqa: E402

S_ENDPGM = struct.pack("<I", 0xBF810000)
S_VERSION_W64 = struct.pack("<I", 0xB0802004)


def _meta(vgprs=4, sgprs=6):
    return {"amdpal.version": [3, 6], "amdpal.pipelines": [{
        ".api": "Vulkan",
        ".hardware_stages": {".gs": {".vgpr_count": 46, ".sgpr_count": 20},
                             ".ps": {".vgpr_count": vgprs, ".sgpr_count": sgprs, ".entry_point_symbol": P.PS_SYMBOL,
                                     ".checksum_value": 780697862}},
        ".graphics_registers": {".spi_ps_input_cntl": [{".offset": 0, ".fp16_interp_mode": False}],
                                ".spi_ps_in_control": {".num_interps": 1}},
        ".scale": P.F32(0.5),
    }]}


def _elf(gs_code, ps_code, ps_at, meta):
    """A small ELF laid out like the AMD driver's pipeline binary: .shstrtab, .text (gs then ps), .comment, .strtab,
    .symtab and the AMDGPU metadata note, no program headers."""
    names = [b"", b".shstrtab", b".text", b".comment", b".strtab", b".symtab", b".note"]
    shstr = b"\0".join(names) + b"\0"
    off = {n: shstr.index(n + b"\0") if n else 0 for n in names}
    text = gs_code + b"\0" * (ps_at - len(gs_code)) + ps_code
    strtab = b"\0_amdgpu_gs_main\0" + P.PS_SYMBOL.encode() + b"\0"
    sym = b"\0" * 24
    sym += struct.pack("<IBBHQQ", 1, 0x12, 0, 2, 0, len(gs_code))
    sym += struct.pack("<IBBHQQ", strtab.index(P.PS_SYMBOL.encode()), 0x12, 0, 2, ps_at, len(ps_code))
    desc = P.mp_dump(meta)
    note = struct.pack("<III", 7, len(desc), 32) + b"AMDGPU\0\0" + desc
    note += b"\0" * (-len(note) % 4)
    secs = [(".shstrtab", 3, shstr, 0, 0), (".text", 1, text, 0, 0), (".comment", 1, b"test\0", 0, 0),
            (".strtab", 3, strtab, 0, 0), (".symtab", 2, sym, 4, 24), (".note", 7, note, 0, 0)]
    body, hdrs, at = bytearray(), [(0,) * 10], 64
    for name, typ, data, link, ent in secs:
        hdrs.append((off[name.encode()], typ, 6 if name == ".text" else 0, 0, at, len(data), link, 0,
                     256 if name == ".text" else 0, ent))
        body += data
        at += len(data)
    while (64 + len(body)) % 8:
        body += b"\0"
    head = bytearray(64)
    head[:16] = b"\x7fELF\x02\x01\x01\x41" + bytes(8)
    struct.pack_into("<HHIQQQIHHHHHH", head, 16, 1, 0xE0, 1, 0, 0, 64 + len(body), 0x945, 64, 0, 0, 64,
                     len(hdrs), 1)
    return bytes(head) + bytes(body) + b"".join(struct.pack(P._SH, *h) for h in hdrs)


class Msgpack(unittest.TestCase):
    def test_round_trip(self):
        values = [0, 127, 128, 255, 256, 65536, 1 << 40, -1, -32, -33, -200, -40000, -(1 << 40), True, False, None,
                  "", "x" * 31, "y" * 32, "z" * 300, b"\x01\x02", [], list(range(20)), {"a": 1, "b": [1, {"c": None}]},
                  {str(i): i for i in range(20)}, 1.5, P.F32(0.25)]
        for v in values:
            with self.subTest(v=v if not isinstance(v, str) else v[:5]):
                got, end = P.mp_load(P.mp_dump(v))
                self.assertEqual(got, v)
                self.assertEqual(end, len(P.mp_dump(v)))
                self.assertEqual(type(got) is P.F32, type(v) is P.F32)

    def test_minimal_encodings(self):
        self.assertEqual(P.mp_dump(5), b"\x05")
        self.assertEqual(P.mp_dump(200), b"\xcc\xc8")
        self.assertEqual(P.mp_dump(-5), b"\xfb")
        self.assertEqual(P.mp_dump(P.F32(1.0)), b"\xca\x3f\x80\x00\x00")
        self.assertEqual(P.mp_dump({"a": 1}), b"\x81\xa1a\x01")
        with self.assertRaises(ValueError):
            P.mp_load(b"\xc1")


class PatchPixelShader(unittest.TestCase):
    def setUp(self):
        self.gs = S_VERSION_W64 + S_ENDPGM
        self.ps = S_VERSION_W64 + struct.pack("<II", 0x7E0002FF, 0x3F800000) + S_ENDPGM
        self.elf = _elf(self.gs, self.ps, 0x100, _meta())

    def test_the_test_elf_reads_back(self):
        e = P.Elf(self.elf)
        self.assertEqual([s[0] for s in e.symbols()], ["", "_amdgpu_gs_main", P.PS_SYMBOL])
        self.assertEqual(P.ps_stage(e.metadata())[".vgpr_count"], 4)
        self.assertEqual(P.ps_code(self.elf), (0x100, self.ps))
        self.assertTrue(P.is_wave64(self.elf))

    def test_longer_code_grows_text_and_keeps_everything_else(self):
        code = S_VERSION_W64 + struct.pack("<II", 0x7E0002FF, 0x12345678) * 50 + S_ENDPGM
        out = P.patch_ps(self.elf, code, ["v_mov_b32 v40, 0", "s_mov_b32 s[20:21], 0"],
                         [(".graphics_registers/.spi_ps_input_cntl/0/.fp16_interp_mode", True)])
        e = P.Elf(out)
        text = bytes(e.data[e.index(".text")])
        self.assertEqual(text[:len(self.gs)], self.gs)                    # the vertex stage is untouched
        self.assertEqual(text[0x100:0x100 + len(code)], code)
        pad = text[0x100 + len(code):]
        self.assertGreaterEqual(len(pad), 256)                            # s_code_end past the end, as the driver
        self.assertEqual(pad, P.S_CODE_END * (len(pad) // 4))
        self.assertEqual((0x100 + len(code) + len(pad)) % 64, 0)
        sym = {s[0]: s for s in e.symbols()}
        self.assertEqual(sym[P.PS_SYMBOL][1:3], (0x100, len(code) + (len(pad) - 256)))
        meta = e.metadata()
        ps = P.ps_stage(meta)
        self.assertEqual((ps[".vgpr_count"], ps[".sgpr_count"]), (41, 22))   # raised to the registers named
        self.assertEqual(ps[".checksum_value"], 780697862)
        regs = meta["amdpal.pipelines"][0][".graphics_registers"]
        self.assertIs(regs[".spi_ps_input_cntl"][0][".fp16_interp_mode"], True)
        self.assertIs(type(meta["amdpal.pipelines"][0][".scale"]), P.F32)
        self.assertEqual(bytes(e.data[e.index(".comment")]), b"test\0")
        self.assertEqual(P.Elf(e.to_bytes()).to_bytes(), out)            # stable when written again

    def test_counts_are_never_lowered_and_an_untouched_note_is_kept(self):
        out = P.patch_ps(_elf(self.gs, self.ps, 0x100, _meta(vgprs=64, sgprs=40)), self.ps, ["v_mov_b32 v1, s2"])
        e = P.Elf(out)
        self.assertEqual(P.ps_stage(e.metadata())[".vgpr_count"], 64)
        self.assertEqual(bytes(e.data[e.index(".note")]),
                         bytes(P.Elf(_elf(self.gs, self.ps, 0x100, _meta(64, 40))).data[e.index(".note")]))

    def test_refusals(self):
        e = P.Elf(self.elf)                            # the vertex stage moved after the pixel stage
        struct.pack_into("<Q", e.data[e.index(".symtab")], 24 + 8, 0x200)
        with self.assertRaises(ValueError):
            P.patch_ps(e.to_bytes(), self.ps)
        with self.assertRaises(ValueError):
            P.patch_ps(self.elf, self.ps, [], [(".graphics_registers/.no_such_register", 1)])
        with self.assertRaises(P.Unavailable):
            P.Elf(b"MZ" + bytes(100))

    def test_registers_named(self):
        self.assertEqual(P.regs_used(["v_interp_p1_f32_e32 v4, v0, attr0.z", "s_mov_b32 m0, s2",
                                      "v_mad_u64_u32 v[8:9], s[10:11], v1, v1, v[2:3] ; v200"]), (10, 12))


class BinaryFile(unittest.TestCase):
    def test_round_trip(self):
        bins = [(16, bytes(range(32)), b"\x7fELF" + bytes(40)), (32, b"\xff" * 32, b"")]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "b.bin")
            P.write_binaries(p, bins)
            self.assertEqual(P.read_binaries(p), bins)


def _radv(code, exec_size=None, info_size=600, wave=64, stage=4, next_stage=0xFF, vgprs=8, sgprs=108,
          stats=b"", ir=b"", tail=b"", btype=0):
    """One RADV stage binary laid out as radv_shader_serialize writes it (Mesa 26.3 radv_pipeline_cache.c): type,
    ac_shader_config, radv_shader_info (its size changes between Mesa versions; wave_size, ngg_lds_scratch_size,
    stage, next_stage at the end here), total_size, 4 bytes of padding, the six sizes, then stats, code, ir."""
    granule = 8 if wave == 32 else 4
    rsrc1 = (vgprs - 1) // granule | 0xC0 << 12 | 1 << 21                 # VGPRS, FLOAT_MODE, DX10_CLAMP
    config = struct.pack("<10I4x3I", sgprs, vgprs, 0, 0, 0, 0, 2, 2, 0xC0, 0, rsrc1, 4, 0)
    info = bytearray(info_size)
    info[-8:-4] = bytes([wave, 0, stage, next_stage])
    base = struct.pack("<I", btype) + config + bytes(info)
    exec_size = len(code) if exec_size is None else exec_size
    total = len(base) + 4 + (-(len(base) + 4) % 8) + 24 + len(stats) + len(code) + len(ir)
    head = base + struct.pack("<I", total) + b"\xab\x55\0\0" * (-(len(base) + 4) % 8 // 4)
    head += struct.pack("<6I", len(code), exec_size, len(ir), 0, len(stats), 0)
    return head + stats + code + ir + tail


class RadvBinary(unittest.TestCase):
    CODE = S_ENDPGM + P.S_CODE_END * 4

    def test_fields_are_found_whatever_the_info_size(self):
        for info_size in (600, 404, 1000):
            with self.subTest(info_size=info_size):
                data = _radv(self.CODE, exec_size=4, info_size=info_size, stats=b"S" * 12, ir=b"ir\0", tail=b"T" * 9)
                lay = P.radv_layout(data)
                self.assertEqual(data[lay.code:lay.code + lay.code_size], self.CODE)
                self.assertEqual((lay.exec_size, lay.vgprs, lay.sgprs, lay.wave, lay.is_ps), (4, 8, 108, 64, True))

    def test_wave32_and_the_vertex_stage(self):
        self.assertEqual(P.radv_layout(_radv(self.CODE, wave=32, vgprs=16)).wave, 32)
        self.assertFalse(P.radv_layout(_radv(self.CODE, stage=0, next_stage=4)).is_ps)

    def test_patch_replaces_the_code_and_raises_the_counts(self):
        data = _radv(self.CODE, exec_size=4, stats=b"S" * 12, ir=b"ir\0", tail=b"T" * 9)
        new = S_ENDPGM * 7
        out = P.patch_radv_ps(data, new, ["v_mov_b32 v40, s90", "s_endpgm"])
        lay = P.radv_layout(out)
        self.assertEqual(out[lay.code:lay.code + lay.exec_size], new)
        self.assertEqual(set(struct.unpack_from(f"<{(lay.code_size - len(new)) // 4}I", out, lay.code + len(new))),
                         {0xBF9F0000})
        self.assertEqual((lay.vgprs, lay.sgprs), (44, 108))         # v40 -> 41 VGPRs, rounded to the wave64 granule 4
        rsrc1 = struct.unpack_from("<I", out, lay.rsrc1_at)[0]
        self.assertEqual(rsrc1, 43 // 4 | 0xC0 << 12 | 1 << 21)
        self.assertEqual(out[lay.code - 12:lay.code], b"S" * 12)       # stats stay before the code
        self.assertTrue(out.endswith(b"ir\0" + b"T" * 9))                # ir and the debug tail stay after it
        self.assertEqual(len(out), len(data) + lay.code_size - len(self.CODE))

    def test_counts_are_never_lowered(self):
        out = P.patch_radv_ps(_radv(self.CODE, wave=32, vgprs=24), S_ENDPGM, ["v_mov_b32 v1, v0"])
        lay = P.radv_layout(out)
        self.assertEqual((lay.vgprs, lay.wave), (24, 32))
        self.assertEqual(struct.unpack_from("<I", out, lay.rsrc1_at)[0] & 0x3F, 23 // 8)

    def test_unknown_layouts_are_refused(self):
        for bad in (_radv(self.CODE, btype=1), _radv(self.CODE)[:-4], _radv(self.CODE)[:60] + bytes(8)):
            with self.subTest(n=len(bad)), self.assertRaises(P.Unavailable):
                P.radv_layout(bad)

    def test_the_pixel_stage_is_picked(self):
        bins = [(32, b"k" * 32, _radv(self.CODE, stage=0, next_stage=4)), (32, b"q" * 32, _radv(self.CODE))]
        self.assertEqual(P.radv_ps_index(bins), 1)
        with self.assertRaises(P.Unavailable):
            P.radv_ps_index(bins[:1])

    def test_architecture(self):
        self.assertEqual(P.radv_arch("AMD Radeon Graphics (RADV RAPHAEL_MENDOCINO) (subgroup 32..64)"), "gfx1036")
        self.assertEqual(P.radv_arch("AMD Custom GPU 0405 (RADV VANGOGH)"), "gfx1033")
        for name, chip in (("AMD Radeon RX 9070 XT (RADV GFX1201) (subgroup 32..64)", "GFX1201"),
                           ("AMD Radeon RX 7900 XTX (RADV NAVI31)", "NAVI31")):
            with self.subTest(chip=chip), self.assertRaisesRegex(P.Unavailable, f"unsupported architecture.*{chip}"):
                P.radv_arch(name)
        with self.assertRaisesRegex(P.Unavailable, "not RADV"):
            P.radv_arch("NVIDIA GeForce RTX 3090")


class HarnessEnvironment(unittest.TestCase):
    def test_child_receives_selected_format_even_when_shell_sets_one(self):
        # Use Python as the child to observe the actual harness environment, without a GPU.
        args = (sys.executable, "-c", "import os; print(os.environ['VKH_COLOR_FORMAT'])",
                "unused.raw", (1, 1), {})
        with patch.dict(os.environ, VKH_COLOR_FORMAT="107:16"):
            default = P.run_harness(*args)
            narrow = P.run_harness(*args, fmt="r32u")
            self.assertEqual(os.environ["VKH_COLOR_FORMAT"], "107:16")
        for result, expected in ((default, "109:16"), (narrow, "98:4")):
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), expected)


class RadvOracleWithoutGpu(unittest.TestCase):
    """The Oracle on a stand-in vk_harness that hands back RADV binaries: the RDNA4 refusal and --set."""

    def _oracle(self, device):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        for name in ("vk_harness", "vs.spv", "ps.spv"):
            open(os.path.join(d.name, name), "wb").close()
        bins = [(32, b"k" * 32, _radv(RadvBinary.CODE, stage=0, next_stage=4)), (32, b"q" * 32, _radv(RadvBinary.CODE))]

        def fake(exe, vs, ps, out_raw, size, env, fmt="rgba32f"):
            P.write_binaries(env["VKH_BINARY_OUT"], bins)
            with open(out_raw, "wb") as f:
                f.write(bytes(16 * size[0] * size[1]))
            return type("R", (), {"returncode": 0, "stderr": f"device: {device}\n"})()
        old, P.run_harness = P.run_harness, fake
        self.addCleanup(setattr, P, "run_harness", old)
        return P.Oracle(d.name, os.path.join(d.name, "vs.spv"), os.path.join(d.name, "ps.spv"), size=(2, 1),
                        work=os.path.join(d.name, "w"))

    def test_rdna2_is_taken(self):
        o = self._oracle("AMD Radeon Graphics (RADV RAPHAEL_MENDOCINO) (subgroup 32..64)")
        self.assertEqual((o.backend, o.wave64, o.ps_index), ("radv", True, 1))
        with self.assertRaisesRegex(ValueError, "--set"):
            o.run(["s_endpgm"], [(".graphics_registers/.spi_ps_in_control/.num_interps", 2)])

    def test_rdna4_is_refused(self):
        with self.assertRaisesRegex(P.Unavailable, "unsupported architecture.*GFX1201"):
            self._oracle("AMD Radeon RX 9070 XT (RADV GFX1201) (subgroup 32..64)")


class Cli(unittest.TestCase):
    def test_nothing_to_do_is_a_usage_error(self):
        with self.assertRaises(SystemExit) as cm:
            P.main(["--build", "."])
        self.assertEqual(cm.exception.code, 2)

    def test_no_harness_is_unavailable(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(P.main(["--build", d, "--show-template"]), 77)

    def test_json_overwriting_an_input_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            asm = os.path.join(d, "prog.txt")
            with open(asm, "w", encoding="utf-8") as f:
                f.write("s_endpgm\n")
            with self.assertRaises(SystemExit) as cm:    # the report is written after the run, so the guard comes first
                P.main(["--build", d, "--asm", asm, "--json", asm])
            self.assertEqual(cm.exception.code, 2)
            with open(asm, encoding="utf-8") as f:
                self.assertEqual(f.read(), "s_endpgm\n")
            with self.subTest(flag="--ps"), self.assertRaises(SystemExit) as cm:
                P.main(["--build", d, "--asm", asm, "--ps", asm, "--json", asm])
            self.assertEqual(cm.exception.code, 2)


@unittest.skipUnless(os.environ.get("LEONIDA_PS_ORACLE_BUILD"), "set LEONIDA_PS_ORACLE_BUILD to a runtime build "
                     "and select an AMD GPU (LEONIDA_VK_DEVICE) to run the pixel shader on hardware")
class OnHardware(unittest.TestCase):
    def test_patched_code_runs_as_the_pixel_shader(self):
        # interp2: location 0 is (u, v, 0, 1), u = 0 at vertex 0 and 2 at vertex 1, so on a 4 x 1 target pixel x has
        # u = (x + 0.5) / 4. The parameter words in LDS are P0 = 0 (vertex 0) and P10 = 2.0 (vertex 1 - vertex 0).
        try:
            o = P.Oracle(os.environ["LEONIDA_PS_ORACLE_BUILD"], template="interp2", size=(4, 1))
        except P.Unavailable as e:
            self.skipTest(str(e))
        body = ["s_mov_b32 m0, s2", "v_interp_p1_f32_e32 v2, v0, attr0.x", "v_interp_p2_f32_e32 v2, v1, attr0.x",
                "v_interp_mov_f32_e32 v3, p0, attr0.x", "v_interp_mov_f32_e32 v4, p10, attr0.x",
                "v_mov_b32_e32 v40, {}", "v_mov_b32_e32 v5, v40", "exp mrt0, v2, v3, v4, v5 done vm", "s_endpgm"]
        try:
            px = o.run([ln.format("0x12345678") for ln in body])
            # a second patch must change the result: nothing of the first one is cached by the driver
            again = o.run([ln.format("0x9abcdef0") for ln in body])
        finally:
            o.close()
        want_u = [0x3E000000, 0x3EC00000, 0x3F200000, 0x3F600000]    # 0.125, 0.375, 0.625, 0.875
        p10 = 0x40000000                                              # 2.0
        if o.backend == "radv":
            # Mesa moves the template's in0 + in1 into the vertex shader: one parameter, attr0 = (2u, 2v, 0, 1)
            want_u = [0x3E800000, 0x3F400000, 0x3FA00000, 0x3FE00000]  # 0.25, 0.75, 1.25, 1.75
            p10 = 0x40800000                                          # 4.0
        self.assertEqual(px, [[u, 0, p10, 0x12345678] for u in want_u])
        self.assertEqual(again, [[u, 0, p10, 0x9ABCDEF0] for u in want_u])
        self.assertNotEqual(o.template_pixels, px)                     # the template's own shader gives other words

    def test_other_target_formats(self):
        # The driver picks the export format for the target: a compressed export reaches an RGBA16_UNORM target
        # bit for bit (the halves are unorm16 values, not f16), and an RG32F target (32_GR) takes R from the last
        # enabled of x/z and G from the last enabled of y/w (measured on gfx1036).
        cases = [("rgba16unorm", ["v_mov_b32_e32 v2, 0x80017fff", "v_mov_b32_e32 v3, 0xc0003c01",
                                  "exp mrt0, v2, v2, v3, v3 done compr vm"], [0x80017FFF, 0xC0003C01]),
                 ("rg32f", ["v_mov_b32_e32 v2, 1", "v_mov_b32_e32 v3, 2", "v_mov_b32_e32 v4, 3", "v_mov_b32_e32 v5, 4",
                            "exp mrt0, v2, v3, v4, v5 done vm"], [3, 4]),
                 ("rg32f", ["v_mov_b32_e32 v2, 1", "v_mov_b32_e32 v5, 4", "exp mrt0, v2, off, off, v5 done vm"], [1, 4])]
        for fmt, body, want in cases:
            with self.subTest(fmt=fmt, body=body[-1]):
                try:
                    o = P.Oracle(os.environ["LEONIDA_PS_ORACLE_BUILD"], template="const", size=(1, 1), fmt=fmt)
                except P.Unavailable as e:
                    self.skipTest(str(e))
                try:
                    self.assertEqual(o.run(body + ["s_endpgm"]), [want])
                finally:
                    o.close()


if __name__ == "__main__":
    unittest.main()
