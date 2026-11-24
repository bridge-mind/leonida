"""Unit tests for tools/pe_inspect.py: standard-library PE parsing, directory walks and the report it writes.

The fixtures are built here, byte by byte, so the test needs no game files and no third-party packages. The 32-bit
image exists because a delay-import descriptor without dlattrRva stores virtual addresses in 32-bit fields: that form
can only be built for an image whose base fits in 32 bits, and it is what the parser has to convert.

    python -I tests/test_pe_inspect.py
"""
import contextlib
import io
import json
import os
import struct
import sys
import shutil
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
TOOLS_DIR = os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import pe_inspect  # noqa: E402

IMAGE_BASE = 0x180000000
BASE32 = 0x10000000
TEXT_RVA = 0x1000
TEXT_RAW = 0x200
TEXT_SIZE = 0x1000


def place(buf: bytearray, rva: int, blob: bytes) -> None:
    at = TEXT_RAW + (rva - TEXT_RVA)
    assert at + len(blob) <= len(buf), f"fixture does not fit at {rva:#x}"
    buf[at:at + len(blob)] = blob


def build_image(is64: bool = True, with_directories: bool = True) -> bytes:
    """A one-section image with imports (by name and by ordinal), a delay import, exports (two named, one
    ordinal-only), .pdata entries, TLS callbacks and a relocation directory. The 32-bit build also carries a
    delay-import descriptor whose fields are virtual addresses rather than RVAs."""
    base = IMAGE_BASE if is64 else BASE32
    opt_size = 240 if is64 else 224
    assert 0x58 + opt_size + 40 <= TEXT_RAW
    buf = bytearray(TEXT_RAW + TEXT_SIZE)

    buf[0:2] = b"MZ"
    struct.pack_into("<I", buf, 0x3C, 0x40)
    buf[0x40:0x44] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", buf, 0x44,
                     0x8664 if is64 else 0x14C,     # machine
                     1,                            # sections
                     0x5F000000,                   # timestamp
                     0, 0,                         # symbol table
                     opt_size,
                     0x2022 if is64 else 0x0102)   # DLL+executable+large-address, or executable+32-bit

    opt = 0x58
    if is64:
        struct.pack_into("<H", buf, opt, 0x20B)
        struct.pack_into("<I", buf, opt + 16, TEXT_RVA)               # AddressOfEntryPoint
        struct.pack_into("<Q", buf, opt + 24, base)                   # ImageBase
        struct.pack_into("<I", buf, opt + 108, 16)                    # NumberOfRvaAndSizes
    else:
        struct.pack_into("<H", buf, opt, 0x10B)
        struct.pack_into("<I", buf, opt + 16, TEXT_RVA)
        struct.pack_into("<I", buf, opt + 28, base)
        struct.pack_into("<I", buf, opt + 92, 16)
    struct.pack_into("<I", buf, opt + 32, 0x1000)                     # SectionAlignment
    struct.pack_into("<I", buf, opt + 36, 0x200)                      # FileAlignment
    struct.pack_into("<I", buf, opt + 56, 0x2000)                     # SizeOfImage
    struct.pack_into("<I", buf, opt + 60, TEXT_RAW)                   # SizeOfHeaders
    struct.pack_into("<H", buf, opt + 68, 2)                          # Subsystem: GUI
    struct.pack_into("<H", buf, opt + 70, 0x40)                       # DllCharacteristics: dynamic base

    directories = {}
    if with_directories:
        directories = {
            pe_inspect.DIR_EXPORT: (0x1700, 40),
            pe_inspect.DIR_IMPORT: (0x1100, 40),
            pe_inspect.DIR_EXCEPTION: (0x1800, 36),
            pe_inspect.DIR_BASERELOC: (0x1A00, 12),
            pe_inspect.DIR_TLS: (0x1900, 40 if is64 else 24),
            pe_inspect.DIR_DELAY_IMPORT: (0x1500, 32 if is64 else 64),
        }
    dirs_at = opt + (112 if is64 else 96)
    for index, (rva, size) in directories.items():
        struct.pack_into("<II", buf, dirs_at + index * 8, rva, size)

    section = opt + opt_size
    buf[section:section + 8] = b".text\0\0\0"
    struct.pack_into("<IIII", buf, section + 8, TEXT_SIZE, TEXT_RVA, TEXT_SIZE, TEXT_RAW)
    struct.pack_into("<I", buf, section + 36, 0x60000020)              # code, execute, read

    if not with_directories:
        return bytes(buf)

    thunk = "Q" if is64 else "I"
    ordinal = (1 << (63 if is64 else 31)) | 7

    # Import table: one import by name, one by ordinal.
    place(buf, 0x1100, struct.pack("<IIIII", 0x1200, 0, 0, 0x1300, 0x1240))
    place(buf, 0x1114, struct.pack("<IIIII", 0x1280, 0, 0, 0x1320, 0x12C0))
    place(buf, 0x1128, bytes(20))                                      # terminator
    place(buf, 0x1200, struct.pack(f"<{thunk}{thunk}", 0x1400, 0))
    place(buf, 0x1240, bytes(16))
    place(buf, 0x1280, struct.pack(f"<{thunk}{thunk}", ordinal, 0))
    place(buf, 0x12C0, bytes(16))
    place(buf, 0x1300, b"kernel32.dll\0")
    place(buf, 0x1320, b"xmem.dll\0")

    # Delay-import table. An x64 image can only use dlattrRva: without it the fields are 32-bit virtual addresses,
    # which cannot hold a base of 0x180000000.
    place(buf, 0x1500, struct.pack("<IIIIIIII", 1, 0x1340, 0, 0x1600, 0x1580, 0, 0, 0))
    if not is64:
        place(buf, 0x1520, struct.pack("<IIIIIIII", 0, base + 0x1360, 0, base + 0x1640,
                                       base + 0x1620, 0, 0, 0))
    place(buf, 0x1540, bytes(32))                                      # terminator
    place(buf, 0x1580, struct.pack(f"<{thunk}{thunk}", 0x1420, 0))
    place(buf, 0x1600, bytes(16))
    place(buf, 0x1620, struct.pack(f"<{thunk}{thunk}", 0x1440, 0))
    place(buf, 0x1640, bytes(16))
    place(buf, 0x1340, b"xmem.dll\0")
    place(buf, 0x1360, b"d3d12_xs.dll\0")
    place(buf, 0x1400, struct.pack("<H", 0) + b"GetCurrentProcessId\0")
    place(buf, 0x1420, struct.pack("<H", 0) + b"XMemAlloc\0")
    place(buf, 0x1440, struct.pack("<H", 0) + b"D3D12SerializeVersionedRootSignature\0")

    # Export directory: functions 5 and 7 are named, function 6 is ordinal-only.
    place(buf, 0x1700, struct.pack("<IIHHIIIIIII", 0, 0, 0, 0, 0x1780, 5, 3, 2, 0x1730, 0x1740, 0x1750))
    place(buf, 0x1730, struct.pack("<III", 0x1000, 0x1008, 0x1010))
    place(buf, 0x1740, struct.pack("<II", 0x1760, 0x1770))
    place(buf, 0x1750, struct.pack("<HH", 0, 2))
    place(buf, 0x1760, b"ExportedA\0")
    place(buf, 0x1770, b"ExportedB\0")
    place(buf, 0x1780, b"sample.dll\0")

    # .pdata: three RUNTIME_FUNCTION entries. TLS: two callbacks then the zero terminator.
    place(buf, 0x1800, bytes(36))
    tls = bytearray(40 if is64 else 24)
    struct.pack_into("<Q" if is64 else "<I", tls, 24 if is64 else 12, base + 0x1940)
    place(buf, 0x1900, bytes(tls))
    cb = "QQQ" if is64 else "III"
    place(buf, 0x1940, struct.pack(f"<{cb}", base + 0x1000, base + 0x1008, 0))
    place(buf, 0x1A00, bytes(12))
    return bytes(buf)


EXPORT_FUNCS = 0x1F00  # AddressOfFunctions for the rebuilt export tables below: last in the section, so a cut can reach it
FORWARDER = 0x1790     # inside the export directory once its size is widened: a forwarder string lives here


def with_exports(slots: list[int], names: dict[int, str], base: int = 10, is64: bool = True) -> bytes:
    """build_image() with its export directory replaced: `slots` is AddressOfFunctions, `names` maps an index into
    it to the export name that points at it."""
    buf = bytearray(build_image(is64=is64))
    img = pe_inspect.PeImage(bytes(buf))
    dir_entry = (0x58 + (112 if is64 else 96)) + pe_inspect.DIR_EXPORT * 8
    struct.pack_into("<II", buf, dir_entry, 0x1700, 0x100)
    ordered = sorted(names.items(), key=lambda kv: kv[1])
    place(buf, 0x1700, struct.pack("<IIHHIIIIIII", 0, 0, 0, 0, 0x1780, base, len(slots), len(ordered),
                                   EXPORT_FUNCS, 0x1C00, 0x1C80))
    place(buf, 0x1780, b"sample.dll\0")
    place(buf, FORWARDER, b"other.Target\0")
    place(buf, EXPORT_FUNCS, struct.pack(f"<{len(slots)}I", *slots))
    text = 0x1D00
    for n, (index, name) in enumerate(ordered):
        place(buf, 0x1C00 + 4 * n, struct.pack("<I", text))
        place(buf, 0x1C80 + 2 * n, struct.pack("<H", index))
        place(buf, text, name.encode() + b"\0")
        text += len(name) + 1
    assert img.offset(EXPORT_FUNCS) is not None
    return bytes(buf)


class PeInspectTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = cls.tmp.name
        cls.image = os.path.join(cls.dir, "sample.dll")
        with open(cls.image, "wb") as f:
            f.write(build_image())
        cls.image32 = os.path.join(cls.dir, "sample32.exe")
        with open(cls.image32, "wb") as f:
            f.write(build_image(is64=False))
        cls.report = pe_inspect.inspect(cls.image, cls.dir)
        cls.report32 = pe_inspect.inspect(cls.image32, cls.dir)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_headers(self):
        r = self.report
        self.assertEqual(r["machine"], "x86-64")
        self.assertTrue(r["is_dll"])
        self.assertEqual(r["subsystem"], "GUI")
        self.assertEqual(r["image_base"], "0x180000000")
        self.assertEqual(r["size_of_image"], 0x2000)
        self.assertEqual(r["entry_rva"], "0x1000")
        self.assertTrue(r["has_relocs"])
        self.assertTrue(r["dynamic_base"])

    def test_sections(self):
        sections = self.report["sections"]
        self.assertEqual([s["name"] for s in sections], [".text"])
        self.assertTrue(sections[0]["exec"])
        self.assertFalse(sections[0]["write"])
        self.assertFalse(sections[0]["looks_encrypted"])
        self.assertGreater(sections[0]["entropy"], 0.0)

    def test_imports_by_name_and_ordinal(self):
        imports = {(i["dll"], i["kind"]): i for i in self.report["imports"]}
        self.assertEqual(imports[("kernel32.dll", "normal")]["functions"], ["GetCurrentProcessId"])
        self.assertEqual(imports[("xmem.dll", "normal")]["functions"], ["#7"])

    def test_delay_import_with_rva_descriptor(self):
        delay = [i for i in self.report["imports"] if i["kind"] == "delay"]
        self.assertEqual(len(delay), 1)
        self.assertEqual(delay[0]["dll"], "xmem.dll")
        self.assertEqual(delay[0]["functions"], ["XMemAlloc"])

    def test_delay_import_descriptor_with_virtual_addresses(self):
        imports = {(i["dll"], i["kind"]): i for i in self.report32["imports"]}
        self.assertEqual(imports[("xmem.dll", "delay")]["functions"], ["XMemAlloc"])
        self.assertEqual(imports[("d3d12_xs.dll", "delay")]["functions"],
                         ["D3D12SerializeVersionedRootSignature"])

    def test_exports_include_ordinal_only_entry(self):
        for report in (self.report, self.report32):
            self.assertEqual(report["export_count"], 3)
            self.assertEqual(set(report["exports"]), {"ExportedA", "ExportedB", "#6"})

    def exports_of(self, data: bytes) -> list[str]:
        return pe_inspect._exports(pe_inspect.PeImage(data))

    def test_unused_ordinal_slot_is_not_an_export(self):
        for is64 in (True, False):
            data = with_exports([0x1000, 0, 0x1010], {0: "Alpha"}, is64=is64)
            self.assertEqual(self.exports_of(data), ["Alpha", "#12"])
            path = os.path.join(self.dir, "gap.dll")
            with open(path, "wb") as f:
                f.write(data)
            report = pe_inspect.inspect(path, None)
            self.assertEqual(report["exports"], ["Alpha", "#12"])
            self.assertEqual(report["export_count"], 2)

    def test_named_exports_require_a_populated_address_table_slot(self) -> None:
        for is64 in (True, False):
            with self.subTest(is64=is64):
                data = bytearray(with_exports([0x1000, 0, FORWARDER, 0x1010],
                                              {0: "Alpha", 1: "Empty", 2: "Forwarded", 4: "Outside"},
                                              is64=is64))
                # The bytes after AddressOfFunctions are not part of its declared four slots.
                place(data, EXPORT_FUNCS + 16, struct.pack("<I", 0x1020))
                self.assertEqual(self.exports_of(bytes(data)), ["Alpha", "Forwarded", "#13"])

    def test_unused_slots_in_a_table_without_names(self):
        data = with_exports([0, 0x1000, 0, 0x1010, 0], {})
        self.assertEqual(self.exports_of(data), ["#11", "#13"])

    def test_unnamed_forwarder_is_still_an_export(self):
        data = with_exports([0x1000, FORWARDER, 0], {0: "Alpha"})
        self.assertEqual(self.exports_of(data), ["Alpha", "#11"])

    def test_function_table_cut_by_end_of_file_lists_the_readable_slots(self):
        data = with_exports([0x1000, 0, 0x1010, 0x1020], {})
        at = pe_inspect.PeImage(data).offset(EXPORT_FUNCS)
        for keep, expected in ((8, ["#10"]), (10, ["#10"]), (12, ["#10", "#12"]), (0, [])):
            with self.subTest(bytes_kept=keep):
                self.assertEqual(self.exports_of(data[:at + keep]), expected)

    def test_table_without_gaps_is_listed_as_before(self):
        data = with_exports([0x1000, 0x1008, 0x1010, 0x1018], {1: "Beta", 3: "Delta"}, base=1)
        self.assertEqual(self.exports_of(data), ["Beta", "Delta", "#1", "#3"])

    def test_pdata_and_tls_counts(self):
        for report in (self.report, self.report32):
            self.assertEqual(report["pdata_functions"], 3)
            self.assertEqual(report["tls_callbacks"], 2)

    def test_pe32_headers_and_thunks(self):
        r = self.report32
        self.assertEqual(r["machine"], "x86")
        self.assertEqual(r["image_base"], "0x10000000")
        self.assertFalse(r["is_dll"])
        imports = {(i["dll"], i["kind"]): i for i in r["imports"]}
        self.assertEqual(imports[("kernel32.dll", "normal")]["functions"], ["GetCurrentProcessId"])
        self.assertEqual(imports[("xmem.dll", "normal")]["functions"], ["#7"])

    def test_bundled_module_is_classified_from_the_game_dir(self):
        with tempfile.TemporaryDirectory() as game:
            path = os.path.join(game, "xmem.dll")
            open(path, "wb").close()
            self.assertEqual(pe_inspect.classify_dll("xmem.dll", game), "bundled")
        self.assertEqual(pe_inspect.classify_dll("xmem.dll", None), "shim")

    @unittest.skipUnless(os.name == "nt", "system DLL classification needs Windows")
    def test_system_dll_is_classified_system(self):
        self.assertEqual(pe_inspect.classify_dll("kernel32.dll", None), "system")

    def test_apiset_query_returns_a_bool_and_unknown_contracts_do_not_resolve(self):
        self.assertIsInstance(pe_inspect._apiset_resolves("api-ms-win-core-file-l1-2-0"), bool)
        self.assertFalse(pe_inspect._apiset_resolves("api-ms-win-not-a-real-contract-l9-9-9"))

    def test_markdown_report(self):
        md = pe_inspect.to_markdown(self.report)
        self.assertIn("# sample.dll", md)
        self.assertIn("| dll | kind | resolves | functions |", md)
        self.assertIn("| xmem.dll | normal | shim | 1 |", md)
        self.assertIn("## Needs checking or a shim", md)
        self.assertIn("`XMemAlloc`", md)

    def test_runtime_load_candidates(self):
        """DLL names held as strings but not imported: what a run-time LoadLibrary* reaches
        (docs/runtime_imports.md)."""
        base = {d["name"] for d in self.report["runtime_load_candidates"]}
        self.assertEqual(base, {"d3d12_xs.dll"}, "the 64-bit fixture names d3d12_xs.dll without importing it")
        buf = bytearray(build_image())
        place(buf, 0x1E00, "XGameRuntime.dll".encode("utf-16-le") + b"\0\0")
        place(buf, 0x1E40, b"gameplatformservices.dll\0xmem.dll\0")
        path = os.path.join(self.dir, "loads.dll")
        with open(path, "wb") as f:
            f.write(buf)
        r = pe_inspect.inspect(path, self.dir)
        found = {d["name"]: d for d in r["runtime_load_candidates"]}
        self.assertEqual(set(found) - base, {"gameplatformservices.dll", "xgameruntime.dll"}, "xmem.dll is imported")
        self.assertEqual(found["xgameruntime.dll"]["encoding"], "utf-16")
        self.assertEqual(found["gameplatformservices.dll"]["encoding"], "ascii")
        self.assertIn(found["xgameruntime.dll"]["where"], ("system", "shim"))
        md = pe_inspect.to_markdown(r)
        self.assertIn("## Run-time load candidates", md)
        self.assertIn("| xgameruntime.dll | utf-16 |", md)

    def test_dll_strings_encodings_own_name_and_near_misses(self):
        data = (b"a.dll\0" + "b.dll".encode("utf-16-le") + b"\0\0" + "a.dll".encode("utf-16-le") +
                b" self.DLL x.dllz .dll y.dll")
        got = pe_inspect.dll_strings(data, {"y.dll"}, "Self.dll")
        self.assertEqual(got, [{"name": "a.dll", "encoding": "both"}, {"name": "b.dll", "encoding": "utf-16"}])

    def test_non_pe_input_is_rejected(self):
        bad = os.path.join(self.dir, "not_pe.bin")
        with open(bad, "wb") as f:
            f.write(b"not a PE file")
        with self.assertRaises(pe_inspect.PeFormatError):
            pe_inspect.inspect(bad, None)

    def test_minimal_image_without_directories(self):
        path = os.path.join(self.dir, "bare.bin")
        with open(path, "wb") as f:
            f.write(build_image(with_directories=False))
        r = pe_inspect.inspect(path, None)
        self.assertEqual(r["imports"], [])
        self.assertEqual(r["exports"], [])
        self.assertEqual(r["tls_callbacks"], 0)
        self.assertEqual(r["pdata_functions"], 0)
        self.assertFalse(r["has_relocs"])

    def test_cli_reports_an_unusable_output_directory(self):
        out = os.path.join(self.dir, "report-file")
        with open(out, "w", encoding="utf-8") as f:
            f.write("keep me")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.assertEqual(pe_inspect.main([self.image, "--out", out]), 1)
        self.assertIn(f"cannot create output directory {out}", stderr.getvalue())
        with open(out, encoding="utf-8") as f:
            self.assertEqual(f.read(), "keep me")

    def test_cli_writes_json_and_markdown(self):
        out = os.path.join(self.dir, "reports")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(pe_inspect.main([self.image, "--out", out]), 0)
        with open(os.path.join(out, "sample.dll.json"), encoding="utf-8") as f:
            written = json.load(f)
        self.assertEqual(written["machine"], "x86-64")
        self.assertEqual(written["imports"][0]["dll"], "kernel32.dll")
        with open(os.path.join(out, "sample.dll.md"), encoding="utf-8") as f:
            self.assertIn("# sample.dll", f.read())

    def run_cli(self, images: list[str], out: str) -> str:
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            self.assertEqual(pe_inspect.main([*images, "--out", out]), 0)
        return err.getvalue()

    def read_report(self, out: str, name: str) -> tuple[dict, str]:
        with open(os.path.join(out, name + ".json"), encoding="utf-8") as f:
            data = json.load(f)
        with open(os.path.join(out, name + ".md"), encoding="utf-8") as f:
            return data, f.read()

    def test_cli_exe_and_dll_with_the_same_stem_get_separate_reports(self):
        sub = os.path.join(self.dir, "stem")
        os.makedirs(sub)
        exe, dll = os.path.join(sub, "game.exe"), os.path.join(sub, "game.dll")
        with open(exe, "wb") as f:
            f.write(build_image(is64=False))
        with open(dll, "wb") as f:
            f.write(build_image())
        out = os.path.join(self.dir, "reports_stem")
        self.run_cli([exe, dll], out)
        self.assertEqual(sorted(os.listdir(out)), ["game.dll.json", "game.dll.md", "game.exe.json", "game.exe.md"])
        exe_r, exe_md = self.read_report(out, "game.exe")
        dll_r, dll_md = self.read_report(out, "game.dll")
        self.assertEqual((exe_r["file"], exe_r["machine"]), ("game.exe", self.report32["machine"]))
        self.assertEqual((dll_r["file"], dll_r["machine"]), ("game.dll", "x86-64"))
        self.assertNotEqual(exe_r["machine"], dll_r["machine"])
        self.assertIn("# game.exe", exe_md)
        self.assertIn("# game.dll", dll_md)

    def test_cli_same_file_name_from_two_directories_gets_a_numbered_suffix(self):
        a, b = os.path.join(self.dir, "a"), os.path.join(self.dir, "b")
        os.makedirs(a)
        os.makedirs(b)
        first, second = os.path.join(a, "x.exe"), os.path.join(b, "x.exe")
        with open(first, "wb") as f:
            f.write(build_image())
        with open(second, "wb") as f:
            f.write(build_image(is64=False))
        out = os.path.join(self.dir, "reports_dup")
        err = self.run_cli([first, second, first], out)
        self.assertEqual(sorted(os.listdir(out)), ["x.exe-2.json", "x.exe-2.md", "x.exe.json", "x.exe.md"])
        r1, _ = self.read_report(out, "x.exe")
        r2, _ = self.read_report(out, "x.exe-2")
        self.assertEqual(r1["machine"], "x86-64")
        self.assertEqual(r2["machine"], self.report32["machine"])
        self.assertNotEqual(r1["machine"], r2["machine"])
        self.assertIn("report written as x.exe-2.json", err)
        self.assertIn("given more than once", err)

    def check_same_image_under_another_folder(self, tag: str, link) -> None:
        a, b = os.path.join(self.dir, tag + "_a"), os.path.join(self.dir, tag + "_b")
        os.makedirs(a)
        os.makedirs(b)
        first, second = os.path.join(a, "x.exe"), os.path.join(b, "x.exe")
        with open(first, "wb") as f:
            f.write(build_image())
        link(first, second)
        open(os.path.join(b, "xmem.dll"), "wb").close()
        out = os.path.join(self.dir, "reports_" + tag)
        self.run_cli([first, second], out)
        self.assertEqual(sorted(os.listdir(out)), ["x.exe-2.json", "x.exe-2.md", "x.exe.json", "x.exe.md"])

        def resolves(report: dict) -> dict:
            return {(i["dll"], i["kind"]): i["where"] for i in report["imports"]}

        for name, img in (("x.exe", first), ("x.exe-2", second)):
            single = os.path.join(self.dir, "reports_" + tag + "_single_" + name)
            self.run_cli([img], single)
            combined, _ = self.read_report(out, name)
            alone, _ = self.read_report(single, "x.exe")
            self.assertEqual(resolves(combined), resolves(alone))
        self.assertEqual(resolves(self.read_report(out, "x.exe")[0])[("xmem.dll", "normal")], "shim")
        self.assertEqual(resolves(self.read_report(out, "x.exe-2")[0])[("xmem.dll", "normal")], "bundled")

    def test_cli_symlinked_image_under_another_folder_gets_its_own_report(self):
        def link(src, dst):
            try:
                os.symlink(src, dst)
            except (OSError, NotImplementedError) as e:
                self.skipTest(f"os.symlink unavailable: {e}")
        self.check_same_image_under_another_folder("symlink", link)

    def test_cli_copied_image_under_another_folder_gets_its_own_report(self):
        self.check_same_image_under_another_folder("copy", shutil.copyfile)

    def test_cli_reports_a_non_pe_file(self):
        bad = os.path.join(self.dir, "not_pe_cli.bin")
        with open(bad, "wb") as f:
            f.write(b"MZ" + bytes(10))
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            self.assertEqual(pe_inspect.main([bad, "--out", os.path.join(self.dir, "reports2")]), 1)
        self.assertIn("not a PE image", err.getvalue())

    def test_same_output_path_identity(self):
        a = os.path.join(self.dir, "test.exe")
        self.assertTrue(pe_inspect.same_output_path(a, a))
        b = os.path.join(self.dir, "other.exe")
        self.assertFalse(pe_inspect.same_output_path(a, b))

    def test_cli_refuses_to_overwrite_input_file_with_report(self):
        out = os.path.join(self.dir, "reports_collision")
        os.makedirs(out, exist_ok=True)
        colliding_file = os.path.join(out, "sample.dll.json")
        with open(colliding_file, "w", encoding="utf-8") as f:
            f.write("original preserved text")

        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            rc = pe_inspect.main([self.image, colliding_file, "--out", out])
        self.assertEqual(rc, 1)
        self.assertIn("would overwrite input file", err.getvalue())
        with open(colliding_file, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "original preserved text")

    def test_cli_refuses_existing_file_as_out_dir(self):
        file_out = os.path.join(self.dir, "some_file.txt")
        with open(file_out, "w", encoding="utf-8") as f:
            f.write("file not dir")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(pe_inspect.main([self.image, "--out", file_out]), 1)
        self.assertIn("path is a file, not a directory", err.getvalue())


OPT = 0x58   # where build_image puts the optional header


class DirectoryCountTest(unittest.TestCase):
    """NumberOfRvaAndSizes counts the data directories; a slot past it is not one, whatever it holds."""

    def report(self, count, is64=True):
        image = bytearray(build_image(is64=is64))
        struct.pack_into("<I", image, OPT + (108 if is64 else 92), count)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "counted.dll")
            with open(path, "wb") as f:
                f.write(image)
            return pe_inspect.inspect(path, tmp)

    def test_slots_past_the_count_are_not_directories(self):
        for is64 in (True, False):
            with self.subTest(is64=is64):
                # export (0) and import (1) stay; exception (3), relocations (5), TLS (9) and delay import (13)
                # are stale bytes now
                r = self.report(2, is64)
                self.assertEqual(r["export_count"], 3)
                self.assertEqual([e["kind"] for e in r["imports"]], ["normal", "normal"])
                self.assertEqual((r["pdata_functions"], r["tls_callbacks"], r["has_relocs"]), (0, 0, False))

    def test_the_count_cuts_between_two_used_slots(self):
        r = self.report(pe_inspect.DIR_BASERELOC)          # slots 0-4: the relocation directory is the first one out
        self.assertEqual((r["pdata_functions"], r["has_relocs"]), (3, False))
        r = self.report(pe_inspect.DIR_BASERELOC + 1)
        self.assertEqual((r["pdata_functions"], r["has_relocs"], r["tls_callbacks"]), (3, True, 0))

    def test_a_count_above_16_reads_the_16_there_are(self):
        full, more = self.report(16), self.report(0x1000)
        self.assertEqual(full, more)
        self.assertEqual((full["pdata_functions"], full["tls_callbacks"], full["has_relocs"]), (3, 2, True))

    def test_no_directories(self):
        r = self.report(0)
        self.assertEqual((r["imports"], r["exports"], r["has_relocs"]), ([], [], False))


class ImportDescriptorBoundsTest(unittest.TestCase):
    DIRECTORIES = (
        ("normal", pe_inspect.DIR_IMPORT, 20, ["kernel32.dll", "xmem.dll"]),
        ("delay", pe_inspect.DIR_DELAY_IMPORT, 32, ["xmem.dll", "d3d12_xs.dll"]),
    )

    def test_import_descriptors_fit_the_declared_directory(self):
        for kind, index, step, dlls in self.DIRECTORIES:
            for size, count in ((step - 1, 0), (2 * step - 1, 1), (2 * step, 2)):
                with self.subTest(kind=kind, size=size):
                    data = bytearray(build_image(is64=False))
                    struct.pack_into("<I", data, OPT + 96 + index * 8 + 4, size)
                    imports = pe_inspect._imports(pe_inspect.PeImage(data), kind, index, delay=kind == "delay")
                    self.assertEqual([entry["dll"] for entry in imports], dlls[:count])

    def test_import_descriptors_fit_the_file_backed_section(self):
        for kind, index, step, dlls in self.DIRECTORIES:
            for extra in (0, 1):
                with self.subTest(kind=kind, extra=extra):
                    data = bytearray(build_image(is64=False)) + b"\0"
                    directory = OPT + 96 + index * 8
                    source_rva = struct.unpack_from("<I", data, directory)[0]
                    source = TEXT_RAW + source_rva - TEXT_RVA
                    rva = TEXT_RVA + TEXT_SIZE - step + extra
                    place(data, rva, data[source:source + step])
                    struct.pack_into("<II", data, directory, rva, step)
                    imports = pe_inspect._imports(pe_inspect.PeImage(data), kind, index, delay=kind == "delay")
                    self.assertEqual([entry["dll"] for entry in imports], dlls[:1 - extra])


class SectionSpanTest(unittest.TestCase):
    """An RVA belongs to the section whose virtual span holds it, not to an earlier section's raw tail."""

    @staticmethod
    def two_sections(a_vsize, a_raw):
        """`.a` at RVA 0x1000 (raw 0x200) and `.b` at RVA 0x2000 (raw 0x200 + a_raw), each with a marker."""
        b_raw = 0x200 + a_raw
        image = bytearray(build_image(with_directories=False)[:0x200]) + bytearray(a_raw + 0x200)
        struct.pack_into("<H", image, 0x46, 2)                                   # NumberOfSections
        table = OPT + 240
        struct.pack_into("<IIII", image, table + 8, a_vsize, 0x1000, a_raw, 0x200)
        image[table + 40:table + 48] = b".b\0\0\0\0\0\0"
        struct.pack_into("<IIII", image, table + 48, 0x200, 0x2000, 0x200, b_raw)
        struct.pack_into("<I", image, table + 76, 0x40000040)
        struct.pack_into("<I", image, OPT + 56, 0x3000)                          # SizeOfImage
        image[0x200:0x20A] = b"NAME_IN_A\0"
        image[b_raw:b_raw + 10] = b"NAME_IN_B\0"
        return pe_inspect.PeImage(bytes(image)), b_raw

    def test_raw_data_longer_than_the_virtual_span_does_not_shadow_the_next_section(self):
        img, b_raw = self.two_sections(a_vsize=0x1000, a_raw=0x2000)
        self.assertEqual(img.offset(0x1000), 0x200)
        self.assertEqual(img.offset(0x1FFF), 0x11FF)
        self.assertEqual(img.offset(0x2000), b_raw)
        self.assertEqual((img.string_at(0x1000), img.string_at(0x2000)), ("NAME_IN_A", "NAME_IN_B"))

    def test_virtual_bytes_past_the_raw_data_have_no_file_offset(self):
        img, _ = self.two_sections(a_vsize=0x1000, a_raw=0x400)
        self.assertEqual(img.offset(0x13FF), 0x5FF)
        self.assertIsNone(img.offset(0x1400))

    def test_virtual_size_zero_uses_the_raw_size(self):
        img, b_raw = self.two_sections(a_vsize=0, a_raw=0x400)
        self.assertEqual(img.offset(0x13FF), 0x5FF)
        self.assertIsNone(img.offset(0x1400))
        self.assertEqual(img.offset(0x2000), b_raw)


class LongNameTest(unittest.TestCase):
    def test_a_long_decorated_import_name_is_read_whole(self):
        # the section is 0x1000 bytes, so the longest name that fits after the fixtures is used: 1,200 characters
        name = "?Long@" + "V" * 1190 + "@@QE"
        self.assertEqual(len(name), 1200)
        image = bytearray(build_image())
        place(image, 0x1B00, struct.pack("<H", 0) + name.encode() + b"\0")
        place(image, 0x1200, struct.pack("<QQ", 0x1B00, 0))                       # the first import's lookup entry
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "long.dll")
            with open(path, "wb") as f:
                f.write(image)
            r = pe_inspect.inspect(path, tmp)
        self.assertEqual(r["imports"][0]["functions"], [name])

    def test_an_unterminated_name_is_still_bounded(self):
        data = b"A" * (3 * pe_inspect.NAME_LIMIT)
        self.assertIsNone(pe_inspect._cstring(data, 0))
        self.assertEqual(pe_inspect._cstring(b"A" * 4096 + b"\0", 0), "A" * 4096)


class TruncatedImageTest(unittest.TestCase):
    """A file cut off inside a header or table is a bad image, not a struct.error that stops the run."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name

    def write(self, name: str, data: bytes) -> str:
        path = os.path.join(self.dir, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def cut_inside(self, full: bytes, index: int, into: int) -> bytes:
        """The image cut `into` bytes after the start of data directory `index`."""
        rva, _size = pe_inspect.PeImage(full).directory(index)
        return full[:pe_inspect.PeImage(full).offset(rva) + into]

    def test_each_table_cut_short_is_a_pe_format_error(self):
        for is64 in (True, False):
            full = build_image(is64=is64)
            cases = {
                "optional header": full[:0x58 + 2 + 16],     # magic present, AddressOfEntryPoint cut off
                "export directory": self.cut_inside(full, pe_inspect.DIR_EXPORT, 10),
                "tls directory": self.cut_inside(full, pe_inspect.DIR_TLS, 20 if is64 else 10),
            }
            for what, data in cases.items():
                with self.subTest(is64=is64, cut=what):
                    path = self.write("cut.dll", data)
                    with self.assertRaises(pe_inspect.PeFormatError):
                        pe_inspect.inspect(path, None)

    def test_every_cut_length_gives_a_report_or_a_pe_format_error(self):
        for is64 in (True, False):
            full = build_image(is64=is64)
            for n in range(0, len(full), 16):
                with self.subTest(is64=is64, length=n):
                    path = self.write("sweep.dll", full[:n])
                    try:
                        pe_inspect.inspect(path, None)
                    except pe_inspect.PeFormatError:
                        pass

    def test_cli_reports_a_truncated_file_and_keeps_going(self):
        cut = self.write("cut.dll", build_image()[:0x58 + 2 + 16])
        good = self.write("good.dll", build_image())
        out = os.path.join(self.dir, "reports")
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            self.assertEqual(pe_inspect.main([cut, good, "--out", out]), 1)
        self.assertIn("cut.dll: not a PE image (truncated optional header)", err.getvalue())
        self.assertEqual(sorted(os.listdir(out)), ["good.dll.json", "good.dll.md"])

    def test_export_counts_past_the_table_cap_do_not_run_away(self):
        full = bytearray(build_image())
        img = pe_inspect.PeImage(bytes(full))
        at = img.offset(img.directory(pe_inspect.DIR_EXPORT)[0])
        struct.pack_into("<I", full, at + 20, 0xFFFFFFFF)  # NumberOfFunctions
        path = self.write("huge_exports.dll", bytes(full))
        exports = pe_inspect.inspect(path, None)["exports"]
        self.assertLessEqual(len(exports), pe_inspect.MAX_ENTRIES + 2)


class DeclaredOptionalHeaderSizeTest(unittest.TestCase):
    """Reject optional headers smaller than the fields the parser reads."""

    def test_declared_optional_header_too_small(self):
        for is64 in (False, True):
            for declared_size in (0, 2, 71):
                with self.subTest(is64=is64, declared_size=declared_size):
                    data = bytearray(build_image(is64=is64))
                    pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
                    struct.pack_into("<H", data, pe_offset + 20, declared_size)

                    with self.assertRaisesRegex(
                        pe_inspect.PeFormatError, "truncated optional header"
                    ):
                        pe_inspect.PeImage(bytes(data))

    def test_declared_optional_header_extends_beyond_file(self):
        for is64 in (False, True):
            with self.subTest(is64=is64):
                data = bytearray(build_image(is64=is64))
                pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
                struct.pack_into("<H", data, pe_offset + 20, 0xFFFF)

                with self.assertRaisesRegex(
                    pe_inspect.PeFormatError, "truncated optional header"
                ):
                    pe_inspect.PeImage(bytes(data))


class ShortHeaderTest(unittest.TestCase):
    """A file too short for a header names the header it stops short of, not a signature that is present."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name

    def write(self, name: str, data: bytes) -> str:
        path = os.path.join(self.dir, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def dos_header(self, size: int, nt: int | None = None) -> bytes:
        """`size` bytes starting with MZ, optionally with e_lfanew at 0x3c pointing at `nt`."""
        data = bytearray(size)
        data[0:2] = b"MZ"
        if nt is not None:
            struct.pack_into("<I", data, 0x3C, nt)
        return bytes(data)

    def check_rejected(self, name: str, data: bytes, expected: str, not_expected: str = "") -> None:
        """PeImage raises the message, and the CLI prints it to stderr and exits 1; no traceback on either path."""
        with self.assertRaisesRegex(pe_inspect.PeFormatError, expected):
            pe_inspect.PeImage(data)
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            self.assertEqual(pe_inspect.main([self.write(name, data), "--out", os.path.join(self.dir, "reports")]), 1)
        self.assertIn(expected, err.getvalue())
        if not_expected:
            self.assertNotIn(not_expected, err.getvalue())

    def test_mz_shorter_than_the_msdos_header_reports_the_length(self):
        # a 4-byte truncated copy: the signature is there, the 64-byte MS-DOS header (e_lfanew lives at 0x3c) is not
        self.check_rejected("short4.bin", self.dos_header(4),
                            "file is 4 bytes, shorter than the 64-byte MS-DOS header", not_expected="no MZ header")

    def test_mz_one_byte_short_of_the_msdos_header_reports_the_length(self):
        self.check_rejected("short63.bin", self.dos_header(63),
                            "file is 63 bytes, shorter than the 64-byte MS-DOS header", not_expected="no MZ header")

    def test_file_without_the_mz_signature_still_reports_no_mz_header(self):
        for name, data in (("aaaa.bin", b"AAAA"), ("empty.bin", b""), ("one_byte.bin", b"A"), ("m.bin", b"M")):
            with self.subTest(name=name):
                self.check_rejected(name, data, "no MZ header")

    def test_e_lfanew_past_the_end_reports_a_short_file_not_a_missing_signature(self):
        # 64 bytes: the MS-DOS header fits, the PE header it points at at 0x40 does not
        self.check_rejected("nt_past.bin", self.dos_header(64, 0x40),
                            "file is 64 bytes, too short to hold the PE header at 0x40",
                            not_expected="no PE signature")

    def test_pe_header_cut_after_the_signature_reports_a_short_file(self):
        # PE\0\0 at 0x40 is there, the 20 bytes of file header after it are cut off
        data = bytearray(self.dos_header(68, 0x40))
        data[0x40:0x44] = b"PE\0\0"
        self.check_rejected("coff_cut.bin", bytes(data),
                            "file is 68 bytes, too short to hold the PE header at 0x40",
                            not_expected="no PE signature")

    def test_a_long_enough_file_with_a_wrong_signature_still_reports_no_pe_signature(self):
        data = bytearray(self.dos_header(88, 0x40))
        data[0x40:0x44] = b"XX\0\0"
        self.check_rejected("bad_sig.bin", bytes(data), "no PE signature")


class SectionBoundaryReadTest(unittest.TestCase):
    """Strings and multi-byte directory entries must not cross section/header boundaries or alias 64-bit thunks."""

    @staticmethod
    def two_section_bytes(is64: bool = True, a_vsize: int = 0x200, a_raw: int = 0x200) -> tuple[bytearray, int]:
        """Build raw bytes for a two-section image with `.a` at 0x1000 and `.b` immediately after it."""
        b_raw = 0x200 + a_raw
        image = bytearray(build_image(is64=is64, with_directories=False)[:0x200]) + bytearray(a_raw + 0x200)
        struct.pack_into("<H", image, 0x46, 2)
        table = OPT + (240 if is64 else 224)
        struct.pack_into("<IIII", image, table + 8, a_vsize, 0x1000, a_raw, 0x200)
        image[table + 40:table + 48] = b".b\0\0\0\0\0\0"
        struct.pack_into("<IIII", image, table + 48, 0x200, 0x2000, 0x200, b_raw)
        struct.pack_into("<I", image, table + 76, 0x40000040)
        struct.pack_into("<I", image, OPT + 56, 0x3000)
        return image, b_raw

    def test_string_at_does_not_cross_section_or_header_boundary(self):
        # `.a` (0x200..0x400) immediately followed by `.b` (0x400..0x600)
        image, b_raw = self.two_section_bytes(a_vsize=0x200, a_raw=0x200)
        image[0x3FA:0x400] = b"UNTERM"
        image[b_raw:b_raw + 7] = b"LEAKED\0"
        img = pe_inspect.PeImage(bytes(image))
        self.assertIsNone(img.string_at(0x11FA))

        # `.a` virtual span (0x200) shorter than its raw size (0x400), followed by unmapped raw tail
        tail_image, _ = self.two_section_bytes(a_vsize=0x200, a_raw=0x400)
        tail_image[0x3FA:0x400] = b"UNTERM"
        tail_image[0x400:0x405] = b"TAIL\0"
        self.assertIsNone(pe_inspect.PeImage(bytes(tail_image)).string_at(0x11FA))

        # Unterminated string in headers stops at SizeOfHeaders (0x200) before `.a`
        hdr_image, _ = self.two_section_bytes(a_vsize=0x200, a_raw=0x200)
        hdr_image[0x1FD:0x200] = b"HDR"
        hdr_image[0x200:0x206] = b"SEC_A\0"
        self.assertIsNone(pe_inspect.PeImage(bytes(hdr_image)).string_at(0x1FD))

        # Positive control: a NUL byte on the very last mapped byte of `.a` is read cleanly
        good_image, b_raw = self.two_section_bytes(a_vsize=0x200, a_raw=0x200)
        good_image[0x3FA:0x400] = b"EXACT\0"
        good_image[b_raw:b_raw + 7] = b"LEAKED\0"
        self.assertEqual(pe_inspect.PeImage(bytes(good_image)).string_at(0x11FA), "EXACT")

    def test_thunk_entry_straddling_section_boundary_stops_cleanly(self):
        for is64 in (True, False):
            with self.subTest(is64=is64):
                width = 8 if is64 else 4
                ordinal_flag = 1 << (63 if is64 else 31)
                image, b_raw = self.two_section_bytes(is64=is64, a_vsize=0x200, a_raw=0x200)
                # Place one valid Hint/Name at RVA 0x1010 (raw 0x210)
                image[0x210:0x21A] = b"\x00\x00GoodSym\0"
                # Place thunk table so entry 0 fits in `.a` and entry 1 starts 2 bytes before `.a` ends
                lookup_rva = 0x1200 - width - 2
                entry0_off = 0x200 + (lookup_rva - 0x1000)
                fmt = "<Q" if is64 else "<I"
                struct.pack_into(fmt, image, entry0_off, 0x1010)
                entry1 = struct.pack(fmt, ordinal_flag | 9)
                image[0x3FE:0x400] = entry1[:2]
                image[b_raw:b_raw + width - 2] = entry1[2:]
                img = pe_inspect.PeImage(bytes(image))
                self.assertEqual(img.thunk_names(lookup_rva), ["GoodSym"])

    def test_pe64_non_ordinal_thunk_high_bits_are_not_masked(self):
        image = bytearray(build_image(is64=True))
        place(image, 0x1B00, b"\x00\x00AliasedSym\0")
        place(image, 0x1200, struct.pack("<QQQ", 0x100001B00, 0x80001B00, 0))
        img = pe_inspect.PeImage(bytes(image))
        self.assertEqual(img.thunk_names(0x1200), ["?", "?"])

    def test_export_directory_and_tables_straddling_section_boundary_do_not_cross(self):
        # Straddling export directory itself (starts 20 bytes before `.a` ends)
        image, b_raw = self.two_section_bytes(is64=True, a_vsize=0x200, a_raw=0x200)
        dirs_at = OPT + 112
        struct.pack_into("<II", image, dirs_at + pe_inspect.DIR_EXPORT * 8, 0x11EC, 40)
        image[0x210:0x21A] = b"leaked.dll"
        image[0x21A] = 0
        struct.pack_into("<IIII", image, 0x220, 0x1080, 0, 0, 0)
        image[0x240:0x247] = b"ExpSym\0"
        struct.pack_into("<I", image, 0x250, 0x1040)
        struct.pack_into("<H", image, 0x254, 0)
        struct.pack_into("<IIHHIIIIIII", image, 0x3EC, 0, 0, 0, 0, 0x1010, 1, 1, 1, 0x1020, 0x1050, 0x1054)
        img = pe_inspect.PeImage(bytes(image))
        self.assertEqual(pe_inspect._exports(img), [])
        self.assertEqual(pe_inspect._export_module_name(img), "")

        # Straddling export table entries (functions / names / ordinals)
        image2, b_raw2 = self.two_section_bytes(is64=True, a_vsize=0x200, a_raw=0x200)
        struct.pack_into("<II", image2, dirs_at + pe_inspect.DIR_EXPORT * 8, 0x1020, 40)
        image2[0x250:0x256] = b"First\0"
        image2[b_raw2:b_raw2 + 16] = b"LeakedSym\0\0\0\0\0\0\0"
        # AddressOfFunctions at 0x11FA (slot 0 in `.a`, slot 1 straddles 2 bytes in `.a` + 2 bytes in `.b`)
        struct.pack_into("<I", image2, 0x3FA, 0x1080)
        image2[0x3FE:0x400] = b"\x00\x00"
        image2[b_raw2:b_raw2 + 2] = b"\x10\x00"
        # AddressOfNames at 0x1100 (1 valid entry pointing to 0x1050), AddressOfNameOrdinals at 0x1110 ([0])
        struct.pack_into("<I", image2, 0x300, 0x1050)
        struct.pack_into("<H", image2, 0x310, 0)
        struct.pack_into("<IIHHIIIIIII", image2, 0x220, 0, 0, 0, 0, 0, 1, 2, 1, 0x11FA, 0x1100, 0x1110)
        self.assertEqual(pe_inspect._exports(pe_inspect.PeImage(bytes(image2))), ["First"])

        # Straddling AddressOfNames and AddressOfNameOrdinals
        image3, b_raw3 = self.two_section_bytes(is64=True, a_vsize=0x200, a_raw=0x200)
        struct.pack_into("<II", image3, dirs_at + pe_inspect.DIR_EXPORT * 8, 0x1020, 40)
        image3[0x250:0x256] = b"First\0"
        image3[0x260:0x26A] = b"SecondSym\0"
        struct.pack_into("<II", image3, 0x280, 0x1080, 0x1090)
        struct.pack_into("<I", image3, 0x3FA, 0x1050)
        straddle_name = struct.pack("<I", 0x1060)
        image3[0x3FE:0x400] = straddle_name[:2]
        image3[b_raw3:b_raw3 + 2] = straddle_name[2:]
        struct.pack_into("<HH", image3, 0x300, 0, 1)
        struct.pack_into("<IIHHIIIIIII", image3, 0x220, 0, 0, 0, 0, 0, 1, 2, 2, 0x1080, 0x11FA, 0x1100)
        self.assertEqual(pe_inspect._exports(pe_inspect.PeImage(bytes(image3))), ["First", "#2"])

        image4, b_raw4 = self.two_section_bytes(is64=True, a_vsize=0x200, a_raw=0x200)
        struct.pack_into("<II", image4, dirs_at + pe_inspect.DIR_EXPORT * 8, 0x1020, 40)
        image4[0x250:0x256] = b"First\0"
        image4[0x260:0x26A] = b"SecondSym\0"
        struct.pack_into("<II", image4, 0x280, 0x1080, 0x1090)
        struct.pack_into("<II", image4, 0x300, 0x1050, 0x1060)
        struct.pack_into("<H", image4, 0x3FD, 0)
        straddle_ord = struct.pack("<H", 1)
        image4[0x3FF:0x400] = straddle_ord[:1]
        image4[b_raw4:b_raw4 + 1] = straddle_ord[1:]
        struct.pack_into("<IIHHIIIIIII", image4, 0x220, 0, 0, 0, 0, 0, 1, 2, 2, 0x1080, 0x1100, 0x11FD)
        self.assertEqual(pe_inspect._exports(pe_inspect.PeImage(bytes(image4))), ["First", "#2"])

    def test_tls_directory_and_callbacks_straddling_section_boundary_do_not_cross(self):
        for is64 in (True, False):
            with self.subTest(is64=is64):
                width = 8 if is64 else 4
                base = IMAGE_BASE if is64 else BASE32
                dirs_at = OPT + (112 if is64 else 96)
                tls_size = 40 if is64 else 24
                cb_field = 24 if is64 else 12
                fmt = "<Q" if is64 else "<I"

                # TLS directory itself straddling `.a`/`.b`
                image, _ = self.two_section_bytes(is64=is64, a_vsize=0x200, a_raw=0x200)
                struct.pack_into("<II", image, dirs_at + pe_inspect.DIR_TLS * 8, 0x11F0, tls_size)
                struct.pack_into(fmt, image, 0x220, base + 0x1080)
                struct.pack_into(fmt, image, 0x220 + width, 0)
                struct.pack_into(fmt, image, 0x3F0 + cb_field, base + 0x1020)
                self.assertEqual(pe_inspect._tls_callbacks(pe_inspect.PeImage(bytes(image))), 0)

                # TLS callback array entry straddling `.a`/`.b`
                image2, b_raw2 = self.two_section_bytes(is64=is64, a_vsize=0x200, a_raw=0x200)
                struct.pack_into("<II", image2, dirs_at + pe_inspect.DIR_TLS * 8, 0x1020, tls_size)
                cb_rva = 0x1200 - width - 2
                struct.pack_into(fmt, image2, 0x220 + cb_field, base + cb_rva)
                struct.pack_into(fmt, image2, 0x200 + (cb_rva - 0x1000), base + 0x1080)
                straddle = struct.pack(fmt, base + 0x1090)
                image2[0x3FE:0x400] = straddle[:2]
                image2[b_raw2:b_raw2 + width - 2] = straddle[2:]
                self.assertEqual(pe_inspect._tls_callbacks(pe_inspect.PeImage(bytes(image2))), 1)


if __name__ == "__main__":
    unittest.main()
