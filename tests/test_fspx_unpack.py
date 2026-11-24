"""Unit tests for tools/fspx_unpack.py (the FSPX payloads of the FidelityFX DLL's shader containers) and for the
translator's FSPX sidecar reads (rdna2spirv.native_blob / cs_dispatch_state).

    python -I tests/test_fspx_unpack.py

Zero game data: every payload and container is built here from synthetic bytes.
"""
import contextlib
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "fspx_unpack.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import fspx_unpack  # noqa: E402

ENDPGM = struct.pack("<I", 0xBF810000)
S_NOP = struct.pack("<I", 0xBF800000)
V_MOV = struct.pack("<I", 0x7E000280)  # v_mov_b32 v0, 0 (any ordinary VOP1 word)


def prologue():
    """Three 12-byte VOP3 opcode-0x180 words with literals, the shape the DLL's programs start with."""
    out = b""
    for lo in (0x1111, 0x2222, 0x0000):
        out += struct.pack("<III", 0xD5800000 | lo, 0x03FF0000, 0x12345678)
    return out


def kind0(threads=(64, 1, 1), wave32=False, rsrc1=0x5, rsrc2=0xC):
    """Synthetic values only: words 0 and 1 are arbitrary, RSRC1/RSRC2 are built from the fields the tests check."""
    x, y, z = threads
    return struct.pack("<7I", 0xA5A5A5A5, 0x5A5A5A5A, 0, x | y << 16, z | (0x10 << 16 if wave32 else 0), rsrc1, rsrc2)


def make_fspx(entries, version=1, count=None, total=None, magic=b"FSPX"):
    """entries: list of (kind, bytes). Lays them out after the directory, 16-byte aligned."""
    n = len(entries) if count is None else count
    off = 16 + 16 * len(entries)
    directory, body = b"", b""
    for kind, data in entries:
        pad = (-off) % 16
        body += b"\0" * pad
        off += pad
        directory += struct.pack("<IIII", kind, off, len(data), 0)
        body += data
        off += len(data)
    size = 16 + len(directory) + len(body)
    return magic + struct.pack("<III", version, n, size if total is None else total) + directory + body


def make_dxbc(parts):
    """A DXBC container from (tag, payload) pairs. The hash is not checked by anything here."""
    n = len(parts)
    off = 32 + 4 * n
    offsets, blobs = [], b""
    for tag, data in parts:
        offsets.append(off + len(blobs))
        blobs += tag + struct.pack("<I", len(data)) + data
    total = 32 + 4 * n + len(blobs)
    return b"DXBC" + b"\0" * 16 + struct.pack("<III", 1, total, n) + struct.pack(f"<{n}I", *offsets) + blobs


def psv0(stage=5):
    info = bytearray(52)
    info[24] = stage
    return struct.pack("<I", len(info)) + bytes(info)


def program(with_prologue=True, endpgm=True):
    return (prologue() if with_prologue else b"") + V_MOV * 3 + (ENDPGM if endpgm else b"") + S_NOP * 4


class ParseFspx(unittest.TestCase):
    def test_ok(self):
        data = make_fspx([(0, kind0()), (1, b"\1" * 20), (6, program()), (7, b"\7" * 40)])
        status, entries = fspx_unpack.parse_fspx(data)
        self.assertEqual(status, 0)
        self.assertEqual(sorted(entries), [0, 1, 6, 7])
        off, size = entries[6]
        self.assertEqual(data[off:off + size], program())

    def test_status_codes_follow_the_dll(self):
        good = make_fspx([(0, kind0()), (6, program())])
        self.assertEqual(fspx_unpack.parse_fspx(good[:15])[0], 1)
        self.assertEqual(fspx_unpack.parse_fspx(b"FSPY" + good[4:])[0], 2)
        # a directory that does not fit the (declared or real) size
        self.assertEqual(fspx_unpack.parse_fspx(make_fspx([(0, kind0()), (6, program())], count=200))[0], 4)
        self.assertEqual(fspx_unpack.parse_fspx(make_fspx([(6, program())]))[0], 5)
        self.assertEqual(fspx_unpack.parse_fspx(make_fspx([(0, kind0())]))[0], 5)
        self.assertEqual(fspx_unpack.parse_fspx(make_fspx([(0, kind0()), (6, b"")]))[0], 5)

    def test_optional_kinds_and_unknown_kinds(self):
        status, entries = fspx_unpack.parse_fspx(make_fspx([(0, kind0()), (6, program()), (9, b"x" * 8)]))
        self.assertEqual(status, 0)
        self.assertEqual(sorted(entries), [0, 6])

    def test_entry_outside_the_payload_is_skipped(self):
        data = bytearray(make_fspx([(0, kind0()), (6, program())]))
        struct.pack_into("<I", data, 16 + 16 + 8, 1 << 20)  # kind 6 size runs past the end
        status, entries = fspx_unpack.parse_fspx(bytes(data))
        self.assertEqual(status, 5)
        self.assertNotIn(6, entries)

    def test_total_is_clamped_to_the_buffer(self):
        data = make_fspx([(0, kind0()), (6, program())], total=1 << 20)
        self.assertEqual(fspx_unpack.parse_fspx(data)[0], 0)

    def test_version_is_not_checked(self):
        self.assertEqual(fspx_unpack.parse_fspx(make_fspx([(0, kind0()), (6, program())], version=7))[0], 0)


class Kind0AndCode(unittest.TestCase):
    def test_decode_wave64(self):
        m = fspx_unpack.decode_kind0(kind0(threads=(64, 1, 1), wave32=False, rsrc1=9, rsrc2=0xC | 0x180 | 3 << 15))
        self.assertEqual(m["threads"], [64, 1, 1])
        self.assertFalse(m["wave32"])
        self.assertEqual(m["vgprs"], (9 + 1) * 4)
        self.assertEqual(m["user_sgprs"], 6)
        self.assertEqual(m["tgid_enable"], [1, 1, 0])
        self.assertEqual(m["lds_bytes"], 3 * 512)

    def test_decode_wave32(self):
        m = fspx_unpack.decode_kind0(kind0(threads=(8, 8, 1), wave32=True, rsrc1=1, rsrc2=0x18 | 0x80))
        self.assertEqual(m["threads"], [8, 8, 1])
        self.assertTrue(m["wave32"])
        self.assertEqual(m["vgprs"], (1 + 1) * 8)
        self.assertEqual(m["user_sgprs"], 12)
        self.assertEqual(m["tgid_enable"], [1, 0, 0])

    def test_short_kind0_keeps_the_words(self):
        self.assertEqual(fspx_unpack.decode_kind0(struct.pack("<2I", 1, 2)), {"words": [1, 2]})

    def test_code_range(self):
        code = program()
        self.assertEqual(fspx_unpack.code_range(code), (36, 36 + 12 + 4))
        self.assertEqual(fspx_unpack.code_range(program(with_prologue=False)), (0, 12 + 4))
        self.assertEqual(fspx_unpack.code_range(program(endpgm=False))[1], None)

    def test_unaligned_endpgm_bytes_are_not_the_end(self):
        code = prologue() + b"\0\0" + ENDPGM + b"\0\0" + V_MOV + ENDPGM
        start, end = fspx_unpack.code_range(code)
        self.assertEqual(start, 36)
        self.assertEqual(end, len(code))


class Containers(unittest.TestCase):
    def container(self, stage=5):
        payload = make_fspx([(0, kind0()), (6, program())])
        return make_dxbc([(b"PSV0", psv0(stage)), (b"RTS0", b"\0" * 24), (b"ACCC", payload)])

    def test_found_inside_other_bytes(self):
        plain = make_dxbc([(b"DXIL", b"\0" * 32)])
        data = b"MZ" + b"\x90" * 101 + self.container() + b"\xcc" * 7 + plain + self.container() + b"\0" * 5
        found = fspx_unpack.find_containers(data)
        self.assertEqual(len(found), 2)
        self.assertEqual(found[0][0], 103)
        self.assertEqual(fspx_unpack.dxbc_parts(found[0][1])[b"ACCC"][:4], b"FSPX")

    def test_psv_stage(self):
        self.assertEqual(fspx_unpack.psv_stage(fspx_unpack.dxbc_parts(self.container(5))), 5)
        self.assertIsNone(fspx_unpack.psv_stage({}))

    def test_truncated_container_has_no_parts(self):
        self.assertEqual(fspx_unpack.dxbc_parts(self.container()[:40]), {})


class Cli(unittest.TestCase):
    def run_tool(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = fspx_unpack.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def disassemble(self, body):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        src = os.path.join(tmp.name, "one.fspx")
        with open(src, "wb") as f:
            f.write(make_fspx([(0, kind0()), (6, program())]))
        mc = os.path.join(tmp.name, "fake_mc.py")
        with open(mc, "w", encoding="utf-8", newline="\n") as f:
            f.write(body)
        out = os.path.join(tmp.name, "out")
        env = dict(os.environ, LLVM_MC=f'"{sys.executable}" "{mc}"')
        result = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", "fspx_unpack.py"),
                                 src, "--out", out, "--disasm"], capture_output=True, text=True, env=env)
        return result, out

    def test_disassembly_error_writes_no_listing(self):
        result, out = self.disassemble("import sys\n"
                                       "print('\\ts_endpgm')\n"
                                       "sys.stderr.write('error: disassembly stopped\\nextra detail\\n')\n"
                                       "sys.exit(23)\n")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(result.stderr.splitlines(), ["one: error: disassembly stopped"])
        self.assertIn("1 input(s) failed", result.stdout)
        self.assertEqual(os.listdir(os.path.join(out, "disasm")), [])
        self.assertEqual(os.listdir(os.path.join(out, "fspx")), [])
        with open(os.path.join(out, "index.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f), [])

    def test_empty_disassembly_writes_no_listing(self):
        result, out = self.disassemble("print('   ')\n")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(result.stderr.splitlines(), ["one: no output"])
        self.assertEqual(os.listdir(os.path.join(out, "disasm")), [])
        self.assertEqual(os.listdir(os.path.join(out, "fspx")), [])

    def test_disassembly_with_invalid_words_keeps_listing(self):
        result, out = self.disassemble("import sys\n"
                                       "print('\\ts_endpgm')\n"
                                       "sys.stderr.write('warning: invalid instruction encoding\\n')\n")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stderr, "")
        with open(os.path.join(out, "disasm", "one.s"), encoding="utf-8") as f:
            listing = f.read()
        self.assertIn("1 invalid words", listing)
        self.assertIn("\ts_endpgm", listing)
        self.assertIn("warning: invalid instruction encoding", listing)

    def test_clean_disassembly_keeps_listing(self):
        result, out = self.disassemble("print('\\ts_endpgm')\n")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stderr, "")
        with open(os.path.join(out, "disasm", "one.s"), encoding="utf-8") as f:
            listing = f.read()
        self.assertIn("0 invalid words", listing)
        self.assertIn("\ts_endpgm", listing)

    def test_dll_like_file(self):
        c = Containers().container()
        with tempfile.TemporaryDirectory() as t:
            src = os.path.join(t, "lib.dll")
            with open(src, "wb") as f:
                f.write(b"MZ" + b"\0" * 62 + c + b"\0" * 9 + c)
            out = os.path.join(t, "out")
            code, stdout, _ = self.run_tool(src, "--out", out)
            self.assertEqual(code, 0)
            self.assertIn("2 FSPX program(s)", stdout)
            with open(os.path.join(out, "index.json"), encoding="utf-8") as f:
                index = json.load(f)
            self.assertEqual([e["file"] for e in index], ["lib_00.dxbc", "lib_01.dxbc"])
            self.assertEqual({e["stage"] for e in index}, {"cs"})
            with open(os.path.join(out, "fspx", "lib_00.json"), encoding="utf-8") as f:
                meta = json.load(f)
            self.assertEqual(meta["threads"], [64, 1, 1])
            self.assertEqual(meta["code"], {"start": 36, "end": 52, "prologue": True})
            with open(os.path.join(out, "fspx", "lib_00.code"), "rb") as f:
                self.assertEqual(f.read(), program())
            with open(os.path.join(out, "lib_01.dxbc"), "rb") as f:
                self.assertEqual(f.read(), c)

    def test_retains_programs_with_matching_input_stems(self):
        container = Containers().container()
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a", "sample.dxbc")
            b = os.path.join(tmp, "b", "sample.dll")
            for path, data in ((a, container), (b, b"MZ" + bytes(62) + container)):
                os.makedirs(os.path.dirname(path))
                with open(path, "wb") as f:
                    f.write(data)
            out = os.path.join(tmp, "out")
            code, _, err = self.run_tool(a, b, "--out", out)
            self.assertEqual(code, 0, err)
            with open(os.path.join(out, "index.json"), encoding="utf-8") as f:
                index = json.load(f)
            self.assertEqual([entry["file"] for entry in index], ["sample.dxbc", "sample_2.dxbc"])
            for name, offset in (("sample", 0), ("sample_2", 64)):
                with open(os.path.join(out, "fspx", name + ".json"), encoding="utf-8") as f:
                    self.assertEqual(json.load(f)["source"]["offset"], offset)
                self.assertTrue(os.path.isfile(os.path.join(out, name + ".dxbc")))

    def test_bare_payload_and_a_file_without_fspx(self):
        with tempfile.TemporaryDirectory() as t:
            bare = os.path.join(t, "one.fspx")
            with open(bare, "wb") as f:
                f.write(make_fspx([(0, kind0(wave32=True)), (6, program())]))
            junk = os.path.join(t, "junk.bin")
            with open(junk, "wb") as f:
                f.write(b"DXBC" + b"\0" * 100)
            out = os.path.join(t, "out")
            code, _, err = self.run_tool(bare, junk, "--out", out)
            self.assertEqual(code, 1)
            self.assertIn("junk.bin: no FSPX payload", err)
            with open(os.path.join(out, "index.json"), encoding="utf-8") as f:
                index = json.load(f)
            self.assertEqual(len(index), 1)
            self.assertTrue(index[0]["wave32"])
            self.assertFalse(os.path.exists(os.path.join(out, "one.dxbc")))

    def test_bad_payload_status_is_reported(self):
        with tempfile.TemporaryDirectory() as t:
            src = os.path.join(t, "bad.fspx")
            with open(src, "wb") as f:
                f.write(make_fspx([(6, program())]))
            code, _, err = self.run_tool(src, "--out", os.path.join(t, "out"))
            self.assertEqual(code, 1)
            self.assertIn("FSPX status 5", err)

    def test_short_kind0_unpacks_without_keyerror(self):
        with tempfile.TemporaryDirectory() as t:
            src = os.path.join(t, "short.fspx")
            with open(src, "wb") as f:
                f.write(make_fspx([(0, struct.pack("<2I", 1, 2)), (6, program())]))
            out = os.path.join(t, "out")
            code, _, err = self.run_tool(src, "--out", out)
            self.assertEqual(code, 0)
            self.assertEqual(err, "")
            with open(os.path.join(out, "index.json"), encoding="utf-8") as f:
                index = json.load(f)
            self.assertEqual(len(index), 1)
            self.assertNotIn("threads", index[0])
            self.assertNotIn("wave32", index[0])
            with open(os.path.join(out, "fspx", "short.json"), encoding="utf-8") as f:
                meta = json.load(f)
            self.assertEqual(meta["words"], [1, 2])

    @unittest.skipUnless(sys.platform == "win32", "the exclusive-lock repro needs Windows")
    def test_locked_input_is_counted_as_failed(self):
        # An exclusively locked (share-none) existing file must be a counted failed input with a one-line
        # message, not a raw PermissionError traceback after the os.path pass.
        import ctypes
        with tempfile.TemporaryDirectory() as t:
            src = os.path.join(t, "locked.dll")
            with open(src, "wb") as f:
                f.write(b"\x00" * 16)
            out = os.path.join(t, "out")
            k32 = ctypes.windll.kernel32
            handle = k32.CreateFileW(src, 0x80000000, 0, None, 3, 0x80, None)
            self.assertNotIn(handle, (0, -1), "could not lock the probe file")
            try:
                code, _, err = self.run_tool(src, "--out", out)
            finally:
                k32.CloseHandle(handle)
            self.assertEqual(code, 1)
            self.assertIn("cannot read input", err)
            self.assertIn("Permission denied", err)
            self.assertNotIn("Traceback", err)
            with open(os.path.join(out, "index.json"), encoding="utf-8") as f:
                self.assertEqual(json.load(f), [])


class CliOutCollision(unittest.TestCase):
    """--out must never be the folder the tool reads, including through another spelling of its name."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = os.path.join(self.tmp.name, "input")
        os.makedirs(self.folder)
        self.src = os.path.join(self.folder, "sample.fspx")
        with open(self.src, "wb") as f:
            f.write(make_fspx([(0, kind0()), (6, program())]))
        with open(self.src, "rb") as f:
            self.contents = f.read()

    def run_tool(self, *argv):
        return subprocess.run([sys.executable, "-I", TOOL, *argv], capture_output=True, text=True, timeout=60)

    def assert_refused(self, result):
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(result.stderr.count("ERROR:"), 1, result.stderr)
        self.assertIn("input folder", result.stderr)
        self.assertEqual(result.stdout, "")

    def assert_input_untouched(self):
        with open(self.src, "rb") as f:
            self.assertEqual(f.read(), self.contents)
        self.assertFalse(os.path.exists(os.path.join(self.folder, "index.json")))
        self.assertFalse(os.path.exists(os.path.join(self.folder, "fspx")))

    def test_out_equal_to_the_input_folder_is_refused_before_opening_it(self):
        result = self.run_tool(self.folder, "--out", self.folder)
        self.assert_refused(result)
        self.assert_input_untouched()
        self.assertEqual(os.listdir(self.folder), ["sample.fspx"])

    def test_out_inside_the_input_folder_is_refused(self):
        result = self.run_tool(self.folder, "--out", os.path.join(self.folder, "out"))
        self.assert_refused(result)
        self.assert_input_untouched()
        self.assertFalse(os.path.exists(os.path.join(self.folder, "out")))

    def test_out_equal_to_the_folder_holding_a_file_input_is_refused(self):
        result = self.run_tool(self.src, "--out", self.folder)
        self.assert_refused(result)
        self.assert_input_untouched()

    def test_an_alias_of_the_input_folder_is_refused(self):
        alias = os.path.join(self.tmp.name, "alias")
        try:
            os.symlink(self.folder, alias, target_is_directory=True)
        except (OSError, NotImplementedError, AttributeError) as e:
            self.skipTest(f"directory symlinks unavailable: {e}")
        result = self.run_tool(self.folder, "--out", alias)
        self.assert_refused(result)
        self.assert_input_untouched()

    def test_a_hard_link_to_the_input_file_is_refused(self):
        alias = os.path.join(self.tmp.name, "alias.fspx")
        try:
            os.link(self.src, alias)
        except (OSError, NotImplementedError, AttributeError) as e:
            self.skipTest(f"hard links unavailable: {e}")
        result = self.run_tool(self.src, "--out", alias)
        self.assert_refused(result)
        self.assert_input_untouched()

    def test_a_case_variant_of_the_input_folder_follows_the_volume(self):
        alias = os.path.join(self.tmp.name, "INPUT")
        insensitive = os.path.exists(alias)
        result = self.run_tool(self.src, "--out", alias)
        if insensitive:
            self.assert_refused(result)
        else:
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_input_untouched()

    def test_a_distinct_output_folder_still_works(self):
        out = os.path.join(self.tmp.name, "out")
        result = self.run_tool(self.src, "--out", out)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("1 FSPX program(s)", result.stdout)
        with open(os.path.join(out, "index.json"), encoding="utf-8") as f:
            self.assertEqual([entry["file"] for entry in json.load(f)], ["sample.dxbc"])
        with open(os.path.join(out, "fspx", "sample.code"), "rb") as f:
            self.assertEqual(f.read(), program())
        self.assert_input_untouched()


class TranslatorSidecars(unittest.TestCase):
    """rdna2spirv reads an FSPX program's code and dispatch state from fspx/ when there is no FSSM blob."""

    def setUp(self):
        import rdna2spirv
        self.tr = rdna2spirv

    def layout(self, t, fssm=None, fspx_meta=None, code=None):
        for sub in ("disasm", "fssm", "fspx"):
            os.makedirs(os.path.join(t, sub), exist_ok=True)
        s = os.path.join(t, "disasm", "cs_x.s")
        with open(s, "w", encoding="utf-8") as f:
            f.write("; cs_x\n")
        if fssm is not None:
            with open(os.path.join(t, "fssm", "cs_x.fssm"), "wb") as f:
                f.write(fssm)
        if fspx_meta is not None:
            with open(os.path.join(t, "fspx", "cs_x.json"), "w", encoding="utf-8") as f:
                json.dump(fspx_meta, f)
        if code is not None:
            with open(os.path.join(t, "fspx", "cs_x.code"), "wb") as f:
                f.write(code)
        return s

    def test_fspx_dispatch_state_and_blob(self):
        with tempfile.TemporaryDirectory() as t:
            s = self.layout(t, fspx_meta={"rsrc2": 0x98, "threads": [32, 1, 1]}, code=b"CODE")
            self.assertEqual(self.tr.cs_dispatch_state(s, "cs_x"), (0x98, 32, 1, 1))
            self.assertEqual(self.tr.native_blob(s, "cs_x"), b"CODE")

    def test_fssm_wins_when_both_exist(self):
        hdr = bytearray(0x148)
        struct.pack_into("<4I", hdr, 0x138, 0x18C, 64, 2, 1)
        with tempfile.TemporaryDirectory() as t:
            s = self.layout(t, fssm=bytes(hdr), fspx_meta={"rsrc2": 0x98, "threads": [32, 1, 1]}, code=b"CODE")
            self.assertEqual(self.tr.cs_dispatch_state(s, "cs_x"), (0x18C, 64, 2, 1))
            self.assertEqual(self.tr.native_blob(s, "cs_x"), bytes(hdr))

    def test_neither_is_still_an_error(self):
        with tempfile.TemporaryDirectory() as t:
            s = self.layout(t)
            with self.assertRaises(FileNotFoundError):
                self.tr.cs_dispatch_state(s, "cs_x")


if __name__ == "__main__":
    unittest.main()
