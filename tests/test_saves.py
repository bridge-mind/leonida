"""Tests for tools/saves.py: ASAV/JSAV/SLAYER1 parsing, the byte-exact JSAV rebuild, extract/pack, diff, and the
import into the offline XGameSave folder layout. Every container here is synthetic, built by the test itself; no real
save data is used or quoted.

    python -I tests/test_saves.py
"""
import contextlib
import errno
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "saves.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))
import saves  # noqa: E402

CODE_GROUP = "$0x11111111"
SCRIPT_GROUP = "$0x22222222"


def block(group, h, body):
    """One synthetic block in the retail shape: {"<group>":{"$0x<hash>":{...}}}, compact JSON."""
    return json.dumps({group: {"$0x%08x" % h: body}}, separators=(",", ":")).encode()


def make_jsav(n_code=3, n_script=2, seed=0):
    hashes, blocks = [], []
    for i in range(n_code + n_script):
        h = 0x10000000 + seed * 0x100 + i
        group = CODE_GROUP if i < n_code else SCRIPT_GROUP
        body = {"$0xaaaa0000": i, "$0xbbbb0000": {"$x": 1.5, "$y": -2.25, "$z": 0.0},
                "$0xcccc0000": {"$content": "int_array", "$data": list(range(i + 1))}}
        hashes.append(h)
        blocks.append(block(group, h, body))
    return saves.build_jsav(3, n_code, hashes, [1] * len(blocks), blocks), hashes, blocks


def make_asav(kind=3, title="Slot title", slot="SLOT0", payload=b"\x0a\x00\x00\x00", ts=1700000000):
    d = bytearray(saves.ASAV_HDR)
    struct.pack_into("<4sIIIIIII", d, 0, b"ASAV", 1, kind, 1, 0, saves.ASAV_HDR, 0, saves.ASAV_HDR + len(payload))
    struct.pack_into("<Q", d, 0x20, ts)
    t = title.encode("utf-16le")
    d[0x28:0x28 + len(t)] = t
    s = slot.encode("ascii")
    d[0x128:0x128 + len(s)] = s
    struct.pack_into("<IIQ", d, 0x160, 0x12345678, 0, saves.ASAV_HDR + len(payload))
    return bytes(d) + payload


def make_slayer(name, inner):
    n = name.encode("ascii")
    return b"SLAYER1\0" + struct.pack("<II", 1, len(n)) + n + struct.pack("<Q", len(inner)) + inner


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def rd(path):
    with open(path, "rb") as f:
        return f.read()


def rd_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def quiet(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        fn(*args)
    return out.getvalue()


class JsavTest(unittest.TestCase):
    def test_layout_and_framing(self):
        d, hashes, blocks = make_jsav()
        h, es = saves.parse_jsav(d)
        self.assertEqual((h["version"], h["code_blocks"], h["script_blocks"]), (3, 3, 2))
        self.assertEqual(h["body_offset"] % 16, 0)
        self.assertEqual(len(d) % 16, 0)
        self.assertTrue(d[h["body_offset"]:].startswith(b'{"CodeSaveBlocks":[\r\n'))
        self.assertIn(b'],\r\n"ScriptSaveBlocks":[\r\n', d)
        self.assertTrue(d.rstrip(b"\0").endswith(b"]\r\n}"))
        self.assertEqual([int(e["hash"], 16) for e in es], hashes)
        self.assertEqual(saves.jsav_blocks(d, es), blocks)
        self.assertEqual({e["flags"] for e in es} | {e["reserved"] for e in es}, {1, 0})

    def test_body_is_one_json_document(self):
        d, _, _ = make_jsav()
        h, _ = saves.parse_jsav(d)
        doc = json.loads(d[h["body_offset"]:h["body_end"]])
        self.assertEqual(len(doc["CodeSaveBlocks"]), 3)
        self.assertEqual(len(doc["ScriptSaveBlocks"]), 2)

    def test_rebuild_is_byte_identical(self):
        d, _, _ = make_jsav(5, 4)
        h, es = saves.parse_jsav(d)
        again = saves.build_jsav(h["version"], h["code_blocks"], [int(e["hash"], 16) for e in es],
                                 [e["flags"] for e in es], saves.jsav_blocks(d, es))
        self.assertEqual(again, d)

    def test_edge_counts(self):
        for n_code, n_script in ((0, 2), (2, 0), (1, 1)):
            d, _, blocks = make_jsav(n_code, n_script)
            h, es = saves.parse_jsav(d)
            self.assertEqual(saves.jsav_blocks(d, es), blocks)
            json.loads(d[h["body_offset"]:h["body_end"]])

    def test_verify_cli(self):
        d, _, _ = make_jsav()
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "s.jsav")
            write(p, d)
            r = subprocess.run([sys.executable, "-I", TOOL, "verify", p], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("byte-identical", r.stdout)
            write(p, d[:-16] + b"x" * 16)          # damage the body
            r = subprocess.run([sys.executable, "-I", TOOL, "verify", p], capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)

    def test_verify_batch_reports_every_input_and_keeps_the_worst_status(self):
        data, _, _ = make_jsav()
        with tempfile.TemporaryDirectory() as tmp:
            bad, missing, changed, good = (os.path.join(tmp, n) for n in
                                           ("bad.jsav", "missing.jsav", "changed.jsav", "good.jsav"))
            write(bad, b"JSAV")
            write(changed, data + bytes(16))  # excess padding: parses but does not rebuild identically
            write(good, data)
            for paths, status in [([bad, missing, changed, good], 2), ([changed, good], 1)]:
                with self.subTest(paths=paths):
                    r = subprocess.run([sys.executable, "-I", TOOL, "verify", *paths],
                                       capture_output=True, text=True)
                    self.assertEqual(r.returncode, status, r.stderr)
                    self.assertIn(f"{changed}: MISMATCH", r.stdout)
                    self.assertIn(f"{good}: OK", r.stdout)
                    errors = r.stderr.splitlines()
                    self.assertEqual(len(errors), len(paths) - 2)
                    for path, line in zip(paths[:-2], errors):
                        self.assertIn(path, line)

    def test_block_decoder_recursion_error_does_not_abort_verify_batch(self):
        data, _, blocks = make_jsav(seed=1)
        loads = json.loads

        def decode(payload, *args, **kwargs):
            if payload == blocks[1]:  # fail one block, not the whole-body parse or the next file
                raise RecursionError("decoder nesting limit")
            return loads(payload, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            bad, good = (os.path.join(tmp, n) for n in ("first.jsav", "next.jsav"))
            write(bad, data)
            write(good, make_jsav(seed=2)[0])
            with mock.patch.object(saves.json, "loads", side_effect=decode), \
                    contextlib.redirect_stderr(io.StringIO()) as err, contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(saves.main(["saves.py", "verify", bad, good]), 2)
            self.assertIn(f"{good}: OK", out.getvalue())
            self.assertEqual(err.getvalue().splitlines(), [f"ERROR: JSAV block 1 exceeds the JSON nesting limit ({bad})"])

    def test_extract_edit_pack(self):
        d, _, _ = make_jsav()
        with tempfile.TemporaryDirectory() as tmp:
            src, out, packed = os.path.join(tmp, "s.jsav"), os.path.join(tmp, "x"), os.path.join(tmp, "p.jsav")
            write(src, d)
            quiet(saves.extract, src, out)
            idx = rd_json(os.path.join(out, "index.json"))
            self.assertEqual(len(idx["entries"]), 5)
            self.assertTrue(os.path.isfile(os.path.join(out, "save.pretty.json")))
            quiet(saves.pack, out, packed)
            self.assertEqual(rd(packed), d)
            # an edited block of a different length still packs to a valid file
            f = os.path.join(out, idx["entries"][1]["file"])
            write(f, block(CODE_GROUP, 0x10000001, {"$0xaaaa0000": 123456789}))
            quiet(saves.pack, out, packed)
            h, es = saves.parse_jsav(rd(packed))
            self.assertEqual(json.loads(saves.jsav_blocks(rd(packed), es)[1]),
                             {CODE_GROUP: {"$0x10000001": {"$0xaaaa0000": 123456789}}})

    def test_diff(self):
        a, _, _ = make_jsav(seed=0)
        h, es = saves.parse_jsav(a)
        blocks = saves.jsav_blocks(a, es)
        blocks[2] = blocks[2].replace(b'"$0xaaaa0000":2', b'"$0xaaaa0000":7')
        b = saves.build_jsav(3, 3, [int(e["hash"], 16) for e in es], [1] * 5, blocks)
        with tempfile.TemporaryDirectory() as tmp:
            pa, pb = os.path.join(tmp, "a"), os.path.join(tmp, "b")
            write(pa, a)
            write(pb, b)
            out = quiet(saves.diff, pa, pb)
        self.assertIn("changed: code 10000002", out)
        self.assertIn("4 blocks identical", out)

    def test_diff_keeps_duplicate_group_hash_blocks(self):
        # Two blocks can share a (group, hash) pair; the second must not overwrite the first in diff's table.
        hashes = [0x11111111, 0x11111111]
        blocks = [b'{"$0x11111111":1}', b'{"$0x11111111":2}']
        a = saves.build_jsav(3, 2, hashes, [1, 1], blocks)
        b = saves.build_jsav(3, 2, hashes, [1, 1], [blocks[0], b'{"$0x11111111":3}'])
        with tempfile.TemporaryDirectory() as tmp:
            pa, pb = os.path.join(tmp, "a"), os.path.join(tmp, "b")
            write(pa, a)
            write(pb, b)
            out = quiet(saves.diff, pa, pb)
        self.assertIn("changed: code 11111111", out)
        self.assertIn('"$0x11111111":2', out)
        self.assertIn('"$0x11111111":3', out)
        self.assertIn("1 blocks identical", out)
        self.assertNotIn("only in", out)

    def test_diff_reports_one_sided_duplicate_with_the_same_label_format(self):
        hashes = [0x11111111, 0x11111111]
        blocks = [b'{"$0x11111111":1}', b'{"$0x11111111":2}']
        a = saves.build_jsav(3, 1, hashes[:1], [1], blocks[:1])
        b = saves.build_jsav(3, 2, hashes, [1, 1], blocks)
        with tempfile.TemporaryDirectory() as tmp:
            pa, pb = os.path.join(tmp, "a"), os.path.join(tmp, "b")
            write(pa, a)
            write(pb, b)
            out = quiet(saves.diff, pa, pb)
        self.assertIn("only in B: code 11111111 17", out)
        self.assertIn("1 blocks identical", out)
        self.assertNotIn("changed:", out)

    def test_diff_matches_identical_duplicate_blocks_by_content(self):
        hashes = [0x11111111, 0x11111111]
        blocks = [b'{"$0x11111111":1}', b'{"$0x11111111":2}']
        a = saves.build_jsav(3, 2, hashes, [1, 1], blocks)
        b = saves.build_jsav(3, 1, hashes[1:], [1], blocks[1:])
        with tempfile.TemporaryDirectory() as tmp:
            pa, pb = os.path.join(tmp, "a"), os.path.join(tmp, "b")
            write(pa, a)
            write(pb, b)
            out = quiet(saves.diff, pa, pb)
        self.assertIn("only in A: code 11111111 17", out)
        self.assertIn("1 blocks identical", out)
        self.assertNotIn("changed:", out)


class AtomicPackTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        self.source = os.path.join(self.root, "source.jsav")
        self.extracted = os.path.join(self.root, "extracted")
        self.out = os.path.join(self.root, "packed.jsav")
        self.old, _, _ = make_jsav(seed=0)
        self.new, _, _ = make_jsav(seed=1)
        write(self.source, self.new)
        quiet(saves.extract, self.source, self.extracted)
        write(self.out, self.old)

    def fail_writes(self, error, on_close=False):
        # Inject a failure after real bytes have been written, both for main's direct output and for staging.
        real_open, real_temporary = open, tempfile.NamedTemporaryFile

        class BrokenWrite:
            def __init__(self, stream):
                self.stream = stream

            def __getattr__(self, name):
                return getattr(self.stream, name)

            def __enter__(self):
                self.stream.__enter__()
                return self

            def __exit__(self, *args):
                result = self.stream.__exit__(*args)
                if on_close and args[0] is None:
                    raise error
                return result

            def write(self, data):
                if on_close:
                    return self.stream.write(data)
                self.stream.write(data[:4])
                raise error

        def opened(path, mode="r", *args, **kwargs):
            stream = real_open(path, mode, *args, **kwargs)
            return BrokenWrite(stream) if mode == "wb" else stream

        def staged(*args, **kwargs):
            return BrokenWrite(real_temporary(*args, **kwargs))

        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(saves, "open", side_effect=opened, create=True))
        stack.enter_context(mock.patch.object(tempfile, "NamedTemporaryFile", side_effect=staged))
        return stack

    def test_partial_write_preserves_previous_save_and_removes_staging(self):
        before = sorted(os.listdir(self.root))
        with self.fail_writes(OSError(errno.ENOSPC, "simulated disk full")):
            with self.assertRaises(OSError):
                quiet(saves.pack, self.extracted, self.out)
        self.assertEqual(rd(self.out), self.old)
        self.assertEqual(sorted(os.listdir(self.root)), before)

    def test_failed_new_output_leaves_no_partial_save(self):
        os.remove(self.out)
        before = sorted(os.listdir(self.root))
        with self.fail_writes(OSError(errno.ENOSPC, "simulated disk full")):
            with self.assertRaises(OSError):
                quiet(saves.pack, self.extracted, self.out)
        self.assertFalse(os.path.exists(self.out))
        self.assertEqual(sorted(os.listdir(self.root)), before)

    def test_failed_close_preserves_previous_save_and_removes_staging(self):
        before = sorted(os.listdir(self.root))
        with self.fail_writes(OSError(errno.ENOSPC, "simulated flush failure"), on_close=True):
            with self.assertRaises(OSError):
                quiet(saves.pack, self.extracted, self.out)
        self.assertEqual(rd(self.out), self.old)
        self.assertEqual(sorted(os.listdir(self.root)), before)

    def test_interruption_preserves_previous_save_and_removes_staging(self):
        before = sorted(os.listdir(self.root))
        with self.fail_writes(KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt):
                quiet(saves.pack, self.extracted, self.out)
        self.assertEqual(rd(self.out), self.old)
        self.assertEqual(sorted(os.listdir(self.root)), before)

    def test_replace_failure_preserves_previous_save_and_removes_staging(self):
        before = sorted(os.listdir(self.root))
        with mock.patch.object(saves.os, "replace", side_effect=PermissionError("destination locked")):
            with self.assertRaises(PermissionError):
                quiet(saves.pack, self.extracted, self.out)
        self.assertEqual(rd(self.out), self.old)
        self.assertEqual(sorted(os.listdir(self.root)), before)

    def test_readonly_staging_is_removed_if_replacement_fails(self):
        before = sorted(os.listdir(self.root))
        with mock.patch.object(saves, "published_mode", return_value=0o444), \
                mock.patch.object(saves.os, "replace", side_effect=PermissionError("destination locked")):
            with self.assertRaises(PermissionError):
                quiet(saves.pack, self.extracted, self.out)
        self.assertEqual(rd(self.out), self.old)
        self.assertEqual(sorted(os.listdir(self.root)), before)

    def test_new_save_is_byte_exact_and_gets_normal_creation_permissions(self):
        os.remove(self.out)
        quiet(saves.pack, self.extracted, self.out)
        self.assertEqual(rd(self.out), self.new)
        self.assertEqual(os.stat(self.out).st_mode & 0o777, os.stat(self.source).st_mode & 0o777)

    def test_hard_linked_destination_is_refused_without_splitting_aliases(self):
        alias = os.path.join(self.root, "alias.jsav")
        try:
            os.link(self.out, alias)
        except OSError as exc:
            self.skipTest(f"cannot create hard links: {exc}")
        before = sorted(os.listdir(self.root))
        with self.assertRaisesRegex(ValueError, "hard link"):
            quiet(saves.pack, self.extracted, self.out)
        result = subprocess.run([sys.executable, "-I", TOOL, "pack", self.extracted, self.out],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("hard link", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertTrue(os.path.samefile(self.out, alias))
        self.assertEqual(rd(alias), self.old)
        self.assertEqual(rd(self.out), self.old)
        self.assertEqual(sorted(os.listdir(self.root)), before)

    def symlink(self, target, link):
        try:
            os.symlink(target, link)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"cannot create symbolic links: {exc}")

    def test_destination_symlinks_update_the_target_and_remain_links(self):
        probe = os.path.join(self.root, "symlink-probe")
        self.symlink("missing-probe-target", probe)
        os.unlink(probe)
        for relative in (False, True):
            for dangling in (False, True):
                with self.subTest(relative=relative, dangling=dangling):
                    label = f"{relative}-{dangling}"
                    target_dir = os.path.join(self.root, "target-" + label)
                    os.mkdir(target_dir)
                    target = os.path.join(target_dir, "save.jsav")
                    if not dangling:
                        write(target, self.old)
                    link = os.path.join(self.root, "link-" + label)
                    spelling = os.path.relpath(target, self.root) if relative else target
                    self.symlink(spelling, link)
                    # What the link stores, read back before packing: on Windows an absolute target comes back
                    # with a \\?\ prefix, so comparing with `spelling` failed there for a link pack() left alone.
                    stored = os.readlink(link)
                    quiet(saves.pack, self.extracted, link)
                    self.assertTrue(os.path.islink(link))
                    self.assertEqual(os.readlink(link), stored)
                    self.assertEqual(rd(target), self.new)
                    self.assertEqual(os.listdir(target_dir), ["save.jsav"])

    def test_failed_publication_through_symlink_preserves_link_and_target(self):
        link = os.path.join(self.root, "link.jsav")
        self.symlink(os.path.basename(self.out), link)
        before = sorted(os.listdir(self.root))
        with mock.patch.object(saves.os, "replace", side_effect=PermissionError("destination locked")):
            with self.assertRaises(PermissionError):
                quiet(saves.pack, self.extracted, link)
        self.assertTrue(os.path.islink(link))
        self.assertEqual(os.readlink(link), os.path.basename(self.out))
        self.assertEqual(rd(self.out), self.old)
        self.assertEqual(sorted(os.listdir(self.root)), before)

    @unittest.skipUnless(os.name == "nt", "Windows directory junction")
    def test_destination_through_directory_junction_updates_the_target(self):
        target_dir = os.path.join(self.root, "target")
        os.mkdir(target_dir)
        junction = os.path.join(self.root, "junction")
        result = subprocess.run(["cmd", "/c", "mklink", "/J", junction, target_dir],
                                capture_output=True, text=True)
        if result.returncode:
            self.skipTest("cannot create junction: " + result.stderr)
        self.addCleanup(os.rmdir, junction)
        target = os.path.join(target_dir, "save.jsav")
        write(target, self.old)
        quiet(saves.pack, self.extracted, os.path.join(junction, "save.jsav"))
        self.assertTrue(os.path.isdir(junction))
        self.assertEqual(rd(target), self.new)
        self.assertEqual(os.listdir(target_dir), ["save.jsav"])

    @unittest.skipUnless(os.name == "posix", "POSIX permission bits")
    def test_replacement_preserves_existing_permission_bits(self):
        os.chmod(self.out, 0o640)
        quiet(saves.pack, self.extracted, self.out)
        self.assertEqual(rd(self.out), self.new)
        self.assertEqual(os.stat(self.out).st_mode & 0o777, 0o640)


class AsavTest(unittest.TestCase):
    def test_header_fields(self):
        d = make_asav(kind=2, title="Profile", slot="SettingsX", payload=b'{"$v": 2.0}')
        h = saves.parse_asav(d)
        self.assertEqual((h["version"], h["kind"], h["payload_offset"], h["field_1c"]), (1, 2, 0x170, len(d)))
        self.assertEqual((h["title"], h["slot_name"], h["checksum_160"]), ("Profile", "SettingsX", "12345678"))
        self.assertEqual(h["timestamp_utc"], "2023-11-14 22:13:20 UTC")
        self.assertEqual(saves.asav_payload(d, h), b'{"$v": 2.0}')

    def test_extract_text_payload(self):
        d = make_asav(kind=2, payload=b'{\r\n\t"$0x01020304": true\r\n}')
        with tempfile.TemporaryDirectory() as tmp:
            src, out = os.path.join(tmp, "a"), os.path.join(tmp, "x")
            write(src, d)
            quiet(saves.extract, src, out)
            self.assertEqual(rd_json(os.path.join(out, "payload.json")), {"$0x01020304": True})
            self.assertEqual(rd(os.path.join(out, "header.bin")), d[:0x170])

    def test_rejects_other_magic(self):
        with self.assertRaises(ValueError):
            saves.parse_asav(b"JSAV" + bytes(0x200))


class ProfileRecordTest(unittest.TestCase):
    def test_parse_synthetic_profile(self):
        # 1644-byte profile record: first 112 bytes non-zero starting u32 10, u32 3
        dwords = [10, 3, 1, 0x100, 0x12345678, 0x9abcdef0] + list(range(22))
        payload = saves.build_profile(dwords=dwords, account_id=0x9abcdef012345678)
        self.assertEqual(len(payload), 1644)
        prof = saves.parse_profile(payload)
        self.assertEqual(prof["version"], 10)
        self.assertEqual(prof["kind"], 3)
        self.assertEqual(prof["active_bytes"], 112)
        self.assertEqual(prof["zero_padding_bytes"], 1532)
        self.assertTrue(prof["zero_padding_clean"])
        self.assertEqual(prof["account_id"], 0x9abcdef012345678)
        self.assertEqual(prof["subversion"], 1)
        self.assertEqual(prof["flags"], 0x100)
        self.assertEqual(len(prof["dwords"]), 28)

    def test_profile_extract_writes_profile_json(self):
        payload = saves.build_profile(account_id=0xCAFEF00D)
        d = make_asav(kind=3, title="Player", slot="Player", payload=payload)
        with tempfile.TemporaryDirectory() as tmp:
            src, out = os.path.join(tmp, "prof_save"), os.path.join(tmp, "out")
            write(src, d)
            quiet(saves.extract, src, out)
            self.assertTrue(os.path.isfile(os.path.join(out, "profile.json")))
            pjson = rd_json(os.path.join(out, "profile.json"))
            self.assertEqual(pjson["version"], 10)
            self.assertEqual(pjson["kind"], 3)
            self.assertEqual(pjson["account_id"], 0xCAFEF00D)
            self.assertEqual(rd(os.path.join(out, "payload.bin")), payload)

    def test_profile_info_output(self):
        payload = saves.build_profile(account_id=0x11223344)
        d = make_asav(kind=3, title="Player", slot="Player", payload=payload)
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "prof_save")
            write(src, d)
            out = quiet(saves.info, [src])
            self.assertIn("profile record", out)
            self.assertIn("version=10", out)
            self.assertIn("kind=3", out)
            self.assertIn("active=112B", out)

    def test_profile_rejects_invalid_version_or_kind(self):
        with self.assertRaises(ValueError):
            saves.parse_profile(struct.pack("<II", 9, 3) + bytes(1636))
        with self.assertRaises(ValueError):
            saves.parse_profile(struct.pack("<II", 10, 2) + bytes(1636))
        with self.assertRaises(ValueError):
            saves.parse_profile(b"\x0a\x00")


class MalformedAsavTest(unittest.TestCase):
    """A truncated or lying ASAV is a ValueError with a readable message, not a struct.error (#980)."""

    def test_file_shorter_than_the_fixed_header_is_rejected(self):
        for blob in (b"ASAV", b"ASAV" + bytes(0x20)):
            with self.subTest(size=len(blob)), self.assertRaises(ValueError) as ctx:
                saves.parse_asav(blob)
            self.assertIn(f"header is only {len(blob)} bytes", str(ctx.exception))

    def test_a_short_non_asav_blob_keeps_the_not_asav_message(self):
        with self.assertRaises(ValueError) as ctx:
            saves.parse_asav(b"NOPE")
        self.assertEqual(str(ctx.exception), "not ASAV")

    def test_payload_offset_past_the_end_is_rejected(self):
        d = bytearray(make_asav(kind=2))
        struct.pack_into("<I", d, 0x14, 0xFFFF)  # payload_offset
        with self.assertRaises(ValueError) as ctx:
            saves.parse_asav(bytes(d))
        self.assertIn("payload offset 0xffff is past the end", str(ctx.exception))

    def test_cli_reports_an_error_without_a_traceback(self):
        # End to end: #923's dispatch guard plus this parser message give one line and exit 2.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "truncated.asav")
            with open(path, "wb") as f:
                f.write(b"ASAV")
            p = subprocess.run([sys.executable, "-I", TOOL, "info", path],
                               capture_output=True, text=True, check=False)
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertIn("ERROR:", p.stderr)
        self.assertIn("header is only 4 bytes, an ASAV header is 368", p.stderr)
        self.assertNotIn("Traceback", p.stderr)


class SlayerTest(unittest.TestCase):
    def test_wrapper_and_unaligned_name(self):
        inner = make_asav(kind=1, title="Packed", slot="SLOT7") + os.urandom(64)
        for name in ("ABCD0000", "memory.dat"):          # 8 and 10 bytes: the second leaves the ASAV unaligned
            d = make_slayer(name, inner)
            h = saves.parse_slayer(d)
            self.assertEqual((h["name"], h["payload_size"]), (name, len(inner)))
            self.assertEqual(d[h["payload_offset"]:], inner)
            self.assertEqual(saves.parse_asav(d[h["payload_offset"]:])["slot_name"], "SLOT7")

    def test_info_reports_without_decoding(self):
        d = make_slayer("ABCD0000", make_asav(kind=1, title="Packed") + os.urandom(64))
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "packed.dat")
            write(p, d)
            out = quiet(saves.info, [p])
        self.assertIn("SLAYER1", out)
        self.assertIn("not decoded", out)


class MalformedSlayerTest(unittest.TestCase):
    """A truncated or lying SLAYER1 header is a SLAYER1-specific ValueError, not a struct.error."""

    def test_a_short_non_slayer_blob_keeps_the_not_slayer_message(self):
        with self.assertRaises(ValueError) as ctx:
            saves.parse_slayer(b"NOPE")
        self.assertEqual(str(ctx.exception), "not SLAYER1")

    def test_bounds(self):
        # Boundary cases around the name length: a short fixed preamble, an oversized declared name, and a
        # name that fits but truncates the 8-byte payload-size field that follows it.
        name = b"ABCD0000"
        cases = [
            (b"SLAYER1\0", "header is only 8 bytes"),
            (b"SLAYER1\0" + struct.pack("<II", 1, 0x1000), "declares a 4096-byte name"),
            (b"SLAYER1\0" + struct.pack("<II", 1, len(name)) + name + b"\x05", "needs 32 bytes"),
        ]
        for blob, needle in cases:
            with self.subTest(size=len(blob)), self.assertRaises(ValueError) as ctx:
                saves.parse_slayer(blob)
            self.assertIn(needle, str(ctx.exception))

    def test_cli_reports_an_error_with_a_path_and_no_traceback(self):
        # End to end: the dispatch guard plus the parser message give one line, the path, and exit 2.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "truncated.dat")
            write(path, b"SLAYER1\0" + struct.pack("<II", 1, 0x1000))
            p = subprocess.run([sys.executable, "-I", TOOL, "info", path],
                               capture_output=True, text=True, check=False)
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertIn("ERROR:", p.stderr)
        self.assertIn(path, p.stderr)
        self.assertIn("declares a 4096-byte name", p.stderr)
        self.assertNotIn("Traceback", p.stderr)


class ImportTest(unittest.TestCase):
    def test_encoding_matches_the_shim(self):
        # Same rule as encode_name() in runtime/shims/xgameruntime_gamesave.cpp.
        self.assertEqual(saves.xgs_encode("slot 1/main"), "slot%201%2Fmain")
        self.assertEqual(saves.xgs_encode("world.dat"), "world.dat")
        self.assertEqual(saves.xgs_encode(".hidden"), "%2Ehidden")
        self.assertEqual(saves.xgs_encode("~tmp"), "%7Etmp")
        self.assertEqual(saves.xgs_encode("é"), "%C3%A9")

    def _folders(self, tmp):
        jsav, _, _ = make_jsav()
        story = os.path.join(tmp, "in", "story")
        write(os.path.join(story, "SLOT0"), jsav)
        write(os.path.join(story, "chives.dat"), make_slayer("SLOT0", make_asav(kind=1, title="T", slot="SLOT3")))
        prof = os.path.join(tmp, "in", "profile")
        write(os.path.join(prof, "Player"), make_asav(kind=3, title="Player", slot="Player"))
        return story, prof, jsav

    def test_import_rejects_flag_as_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            story, _, _ = self._folders(tmp)
            tool = os.path.join(ROOT, "tools", "saves.py")
            result = subprocess.run(
                [sys.executable, "-I", tool, "import", story, "--out", "--dry-run"],
                cwd=tmp, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("--out needs a value", result.stderr)
            self.assertFalse(os.path.exists(os.path.join(tmp, "--dry-run")))

    def test_import_map_needs_old_equals_new(self):
        with tempfile.TemporaryDirectory() as tmp:
            story, _, _ = self._folders(tmp)
            tool = os.path.join(ROOT, "tools", "saves.py")
            result = subprocess.run(
                [sys.executable, "-I", tool, "import", story, "--map", "SLOT3", "--dry-run"],
                cwd=tmp, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("--map needs OLD=CONTAINER/BLOB", result.stderr)
            self.assertIn("SLOT3", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_import_layout_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            story, prof, jsav = self._folders(tmp)
            out = os.path.join(tmp, "saves")
            quiet(saves.import_saves, [story, prof, "--out", out])
            # slot name comes from the packed copy's ASAV header when there is one
            self.assertEqual(rd(os.path.join(out, "local-user", "SLOT3", "SLOT0")), jsav)
            self.assertTrue(os.path.isfile(os.path.join(out, "local-user", "Player", "Player")))
            m = rd_json(os.path.join(out, "import-manifest.json"))
            self.assertEqual(sorted(e["container"] for e in m["entries"]), ["Player", "SLOT3"])
            self.assertFalse(any(n.startswith("~") for _, _, fs in os.walk(out) for n in fs))

    def test_map_dry_run_force_and_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            story, prof, _ = self._folders(tmp)
            out = os.path.join(tmp, "saves")
            quiet(saves.import_saves, [story, "--out", out, "--map", "SLOT3=slot 0/story", "--dry-run"])
            self.assertFalse(os.path.exists(out))
            quiet(saves.import_saves, [story, "--out", out, "--map", "SLOT3=slot 0/story"])
            self.assertTrue(os.path.isfile(os.path.join(out, "local-user", "slot%200", "story")))
            with self.assertRaises(SystemExit):        # refuses to overwrite without --force
                quiet(saves.import_saves, [story, "--out", out, "--map", "SLOT3=slot 0/story"])
            quiet(saves.import_saves, [story, "--out", out, "--map", "SLOT3=slot 0/story", "--force"])
            with self.assertRaises(SystemExit):        # two payloads onto one blob (case-insensitive on Windows)
                quiet(saves.import_saves, [story, prof, "--out", out, "--force",
                                           "--map", "SLOT3=Same/blob", "--map", "Player=same/BLOB"])

    def test_container_names_follow_the_runtime_rule(self):
        # The same rule as valid_container_name() in runtime/shims/xgameruntime_gamesave.cpp.
        for good in ("SLOT3", "slots/SAVE_00.sav-1", "a/b/c"):
            self.assertTrue(saves.valid_container_name(good), good)
        for bad in ("slot 0", "a..b", "x.", "dir/", "a/b-c/d", "", "é", "a" * 257):
            self.assertFalse(saves.valid_container_name(bad), bad)
        with tempfile.TemporaryDirectory() as tmp:
            story, _, _ = self._folders(tmp)
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                quiet(saves.import_saves, [story, "--out", os.path.join(tmp, "o"), "--map", "SLOT3=slot 0/story",
                                           "--dry-run"])
            self.assertIn("'slot 0' breaks XGameSaveCreateContainer's naming rule", err.getvalue())
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                quiet(saves.import_saves, [story, "--out", os.path.join(tmp, "o"), "--dry-run"])
            self.assertEqual(err.getvalue(), "")

    def test_folder_with_two_payloads_is_refused(self):
        # A folder holding a save and an older backup with the same signature must not import either silently.
        with tempfile.TemporaryDirectory() as tmp:
            folder = os.path.join(tmp, "ambiguous")
            jsav, _, _ = make_jsav()
            write(os.path.join(folder, "slot1"), jsav)
            write(os.path.join(folder, "slot1.bak"), jsav)
            out = os.path.join(tmp, "o")
            with self.assertRaises(SystemExit) as cm:
                quiet(saves.import_saves, [folder, "--out", out, "--dry-run"])
            msg = str(cm.exception)
            self.assertIn("slot1", msg)
            self.assertIn("slot1.bak", msg)
            self.assertFalse(os.path.exists(out))

    def test_folder_without_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = os.path.join(tmp, "empty")
            write(os.path.join(empty, "readme.txt"), b"nothing here")
            with self.assertRaises(SystemExit):
                quiet(saves.import_saves, [empty, "--out", os.path.join(tmp, "o")])

    def test_discovery_reads_at_most_eight_bytes_per_candidate(self):
        # save_slot probes each candidate with eight bytes, then reads only the chosen payload in full.
        with tempfile.TemporaryDirectory() as tmp:
            folder = os.path.join(tmp, "story")
            jsav, _, _ = make_jsav()
            write(os.path.join(folder, "SLOT0"), jsav)
            write(os.path.join(folder, "notes.txt"), b"just text" * 1000)
            with mock.patch.object(saves, "read", wraps=saves.read) as reader:
                quiet(saves.save_slot, folder)
        payload = os.path.join(folder, "SLOT0")
        other = os.path.join(folder, "notes.txt")
        self.assertEqual(reader.call_args_list,
                         [mock.call(payload, 8), mock.call(other, 8), mock.call(payload)])

    def test_info_directory_reads_only_eight_bytes_per_candidate(self):
        # The info directory loop probes each candidate with eight bytes and only reads the chosen save in full,
        # so an unrelated large text file next to it is never read in full (mirrors save_slot's probe, #1556).
        with tempfile.TemporaryDirectory() as tmp:
            folder = os.path.join(tmp, "story")
            jsav, _, _ = make_jsav()
            write(os.path.join(folder, "SLOT0"), jsav)
            write(os.path.join(folder, "notes.txt"), b"just text" * 1000)
            with mock.patch.object(saves, "read", wraps=saves.read) as reader:
                quiet(saves.info, [folder])
        payload = os.path.join(folder, "SLOT0")
        other = os.path.join(folder, "notes.txt")
        # read(cj) opens the chives.json path only when it exists (it does not here); each candidate is an
        # eight-byte probe and the selected save is then read in full by info_file.
        self.assertEqual(reader.call_args_list,
                         [mock.call(payload, 8), mock.call(payload), mock.call(other, 8)])


class MalformedJsavTest(unittest.TestCase):
    """A truncated or lying JSAV is one error line, not a struct.error traceback."""

    def test_excessive_json_body_nesting_is_a_validation_error(self):
        # Linux CPython 3.12 can accept 6,000 levels; exercise the C decoder's higher limit too.
        block = b'{"nested":' + b'[' * 100_000 + b'0' + b']' * 100_000 + b'}'
        data = saves.build_jsav(3, 1, [1], [1], [block])
        with self.assertRaisesRegex(ValueError, r"JSAV body is not valid UTF-8 JSON"):
            saves.parse_jsav(data)

    def test_all_read_commands_refuse_excessive_nesting_without_changing_exports(self):
        block = b'{"nested":' + b'[' * 100_000 + b'0' + b']' * 100_000 + b'}'
        data = saves.build_jsav(3, 1, [1], [1], [block])
        with tempfile.TemporaryDirectory() as tmp:
            source, out = os.path.join(tmp, "deep.jsav"), os.path.join(tmp, "export")
            index = os.path.join(out, "index.json")
            prior_block = os.path.join(out, "blocks", "previous.json")
            write(source, data)
            write(index, b"previous complete index")
            write(prior_block, b"previous complete block")
            for args in (("info", source), ("verify", source),
                         ("diff", source, source), ("extract", source, out)):
                with self.subTest(command=args[0]):
                    result = subprocess.run([sys.executable, "-I", TOOL, *args],
                                            capture_output=True, text=True, timeout=10)
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertIn("ERROR: JSAV body is not valid UTF-8 JSON", result.stderr)
                    self.assertEqual(len(result.stderr.splitlines()), 1)
                    self.assertNotIn("Traceback", result.stderr)
                    self.assertEqual(rd(source), data)
                    self.assertEqual(rd(index), b"previous complete index")
                    self.assertEqual(rd(prior_block), b"previous complete block")
                    self.assertEqual(sorted(os.listdir(out)), ["blocks", "index.json"])
                    self.assertEqual(os.listdir(os.path.join(out, "blocks")), ["previous.json"])

    def test_entry_ranges_must_stay_inside_the_json_body(self):
        original, _, _ = make_jsav()
        header, entries = saves.parse_jsav(original)
        cases = {
            "header": (0, 2),
            "entry table": (saves.JSAV_HDR, 2),
            "alignment padding": (header["table_end"], 1),
            "past EOF": (len(original) + 16, 1),
            "truncated block": (entries[0]["offset"], len(original)),
            "trailing padding": (header["body_end"] - 1, 2),
            "uint32 extent": (0xFFFFFFFF, 0xFFFFFFFF),
        }
        for name, (offset, size) in cases.items():
            with self.subTest(name=name):
                data = bytearray(original)
                struct.pack_into("<II", data, saves.JSAV_HDR + 12, offset, size)
                with self.assertRaisesRegex(ValueError, r"JSAV entry 0.*outside.*body"):
                    saves.parse_jsav(bytes(data))

    def test_all_read_commands_refuse_a_bad_entry_before_using_blocks(self):
        data, _, _ = make_jsav()
        bad = bytearray(data)
        struct.pack_into("<I", bad, saves.JSAV_HDR + 12, len(data) + 16)
        with tempfile.TemporaryDirectory() as tmp:
            source, out = os.path.join(tmp, "bad.jsav"), os.path.join(tmp, "extract")
            write(source, bad)
            for args in (("info", source), ("extract", source, out),
                         ("diff", source, source), ("verify", source)):
                with self.subTest(command=args[0]):
                    result = subprocess.run([sys.executable, "-I", TOOL, *args],
                                            capture_output=True, text=True, timeout=10)
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertIn("ERROR: JSAV entry 0", result.stderr)
                    self.assertNotIn("Traceback", result.stderr)
                    self.assertNotIn("blocks identical", result.stdout)
            self.assertFalse(os.path.exists(os.path.join(out, "blocks")))
            self.assertEqual(rd(source), bytes(bad))

    def test_appended_block_is_not_part_of_the_outer_json_body(self):
        original, _, blocks = make_jsav()
        data = bytearray(original + blocks[0])
        struct.pack_into("<I", data, saves.JSAV_HDR + 12, len(original))
        with self.assertRaisesRegex(ValueError, r"JSAV entry 0.*outside.*body"):
            saves.parse_jsav(bytes(data))

    def test_invalid_json_and_trailing_data_produce_no_partial_exports(self):
        original, _, _ = make_jsav()
        body_end = len(original.rstrip(b"\0"))
        cases = {
            "second document": original + b'{"extra":1}',
            "malformed document": original[:body_end - 1] + b"]",
            "invalid UTF-8": original + b"\xff",
        }
        with tempfile.TemporaryDirectory() as tmp:
            for name, data in cases.items():
                with self.subTest(name=name):
                    source = os.path.join(tmp, "bad.jsav")
                    out = os.path.join(tmp, name)
                    index = os.path.join(out, "index.json")
                    write(source, data)
                    write(index, b"previous complete export")
                    for command in (("extract", source, out), ("diff", source, source)):
                        result = subprocess.run([sys.executable, "-I", TOOL, *command],
                                                capture_output=True, text=True, timeout=10)
                        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                        self.assertNotIn("Traceback", result.stderr)
                    self.assertEqual(rd(index), b"previous complete export")
                    self.assertFalse(os.path.exists(os.path.join(out, "blocks")))

    def test_json_end_uses_utf8_bytes_and_allows_whitespace_before_padding(self):
        block = json.dumps({"message": "\u0105\u6f22"}, ensure_ascii=False).encode("utf-8")
        original = saves.build_jsav(3, 1, [1], [1], [block])
        end = len(original.rstrip(b"\0"))
        data = original[:end] + b"\r\n\t " + b"\0" * 16
        header, entries = saves.parse_jsav(data)
        self.assertEqual(header["body_end"], end)
        self.assertEqual(saves.jsav_blocks(data, entries), [block])

    def test_header_shorter_than_16_bytes_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            saves.parse_jsav(b"JSAV")
        self.assertIn("header is only 4 bytes", str(ctx.exception))

    def test_a_short_non_jsav_blob_keeps_the_not_jsav_message(self):
        with self.assertRaises(ValueError) as ctx:
            saves.parse_jsav(b"NOPE")
        self.assertEqual(str(ctx.exception), "not JSAV")

    def test_entry_table_past_end_is_rejected(self):
        data = b"JSAV" + struct.pack("<III", 1, 4, 0)
        with self.assertRaises(ValueError) as ctx:
            saves.parse_jsav(data)
        self.assertIn("declares 4 entries but the file has 16 bytes", str(ctx.exception))

    def test_missing_body_marker_is_rejected(self):
        # A complete entry table with no JSON body marker after it.
        data = bytearray(b"JSAV" + struct.pack("<III", 3, 1, 0))
        data += struct.pack("<IIIII", 0x11111111, 1, 0, 36, 0)
        with self.assertRaises(ValueError) as ctx:
            saves.parse_jsav(bytes(data))
        self.assertIn("body marker not found", str(ctx.exception))

    def test_cli_reports_an_error_without_a_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "truncated.jsav")
            with open(path, "wb") as f:
                f.write(b"JSAV" + struct.pack("<III", 1, 4, 0))
            p = subprocess.run([sys.executable, "-I", TOOL, "info", path],
                               capture_output=True, text=True)
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertIn("ERROR:", p.stderr)
        self.assertIn("declares 4 entries", p.stderr)
        self.assertNotIn("Traceback", p.stderr)


class PackIndexValidationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = os.path.join(self.tmp.name, "source.jsav")
        self.extracted = os.path.join(self.tmp.name, "extracted")
        self.index_path = os.path.join(self.extracted, "index.json")
        self.original, _, _ = make_jsav(1, 1)
        write(self.source, self.original)
        quiet(saves.extract, self.source, self.extracted)
        self.index = rd_json(self.index_path)

    def invalid_indexes(self):
        yield None, "object"
        yield [], "object"
        fields = [
            (("header",), (None, [], 42), "header"),
            (("entries",), (None, {}, 42), "entries"),
            (("header", "version"), (None, "3", True, -1, 0x100000000), "header.version"),
            (("header", "code_blocks"), (None, "1", True, -1, 3), "header.code_blocks"),
            (("entries", 0), (None, [], 42), "entries[0]"),
            (("entries", 0, "file"), (None, [], 42, ""), "entries[0].file"),
            (("entries", 0, "hash"), (None, 1, True, "not-hex", "-1", "100000000"), "entries[0].hash"),
            (("entries", 0, "flags"), (None, "1", True, -1, 0x100000000), "entries[0].flags"),
        ]
        for path, values, field in fields:
            for value in values:
                doc = json.loads(json.dumps(self.index))
                node = doc
                for key in path[:-1]:
                    node = node[key]
                node[path[-1]] = value
                yield doc, field
        for path, field in [(("header",), "header"), (("entries",), "entries"),
                            (("header", "version"), "header.version"),
                            (("header", "code_blocks"), "header.code_blocks"),
                            (("entries", 0, "file"), "entries[0].file"),
                            (("entries", 0, "hash"), "entries[0].hash"),
                            (("entries", 0, "flags"), "entries[0].flags")]:
            doc = json.loads(json.dumps(self.index))
            node = doc
            for key in path[:-1]:
                node = node[key]
            del node[path[-1]]
            yield doc, field

    def test_malformed_indexes_name_the_index_and_invalid_field(self):
        for doc, field in self.invalid_indexes():
            with self.subTest(field=field, doc=doc):
                write(self.index_path, json.dumps(doc).encode("utf-8"))
                with self.assertRaises(ValueError) as error:
                    saves.pack_from_dir(self.extracted)
                self.assertIn(self.index_path, str(error.exception))
                self.assertIn(field, str(error.exception))

    def test_cli_errors_leave_existing_output_untouched(self):
        out = os.path.join(self.tmp.name, "packed.jsav")
        write(out, b"previous complete output")
        for doc in (None, {}, {"header": None, "entries": []},
                    {"header": {"version": 3, "code_blocks": 0}, "entries": [None]}):
            with self.subTest(doc=doc):
                write(self.index_path, json.dumps(doc).encode("utf-8"))
                result = subprocess.run([sys.executable, "-I", TOOL, "pack", self.extracted, out],
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("ERROR:", result.stderr)
                self.assertIn(self.index_path, result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertEqual(rd(out), b"previous complete output")

    def test_json_syntax_and_encoding_errors_name_the_index(self):
        for data in (b"{", b"\xff"):
            with self.subTest(data=data):
                write(self.index_path, data)
                with self.assertRaises(ValueError) as error:
                    saves.pack_from_dir(self.extracted)
                self.assertIn(self.index_path, str(error.exception))
                self.assertIn("invalid JSON", str(error.exception))

    def test_excessive_json_nesting_names_the_index(self):
        # The C decoder's limit differs by platform; Linux CPython 3.12 accepts 6,000 levels.
        write(self.index_path, b"[" * 100_000 + b"0" + b"]" * 100_000)
        with self.assertRaises(ValueError) as error:
            saves.pack_from_dir(self.extracted)
        self.assertIn(self.index_path, str(error.exception))
        self.assertIn("invalid JSON", str(error.exception))

    def test_cli_excessive_json_nesting_preserves_existing_output(self):
        write(self.index_path, b"[" * 100_000 + b"0" + b"]" * 100_000)
        out = os.path.join(self.tmp.name, "packed.jsav")
        write(out, b"previous complete output")
        result = subprocess.run([sys.executable, "-I", TOOL, "pack", self.extracted, out],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("ERROR:", result.stderr)
        self.assertIn(self.index_path, result.stderr)
        self.assertIn("invalid JSON", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(rd(out), b"previous complete output")

    def test_pack_refuses_an_out_that_names_a_source_file(self):
        """pack reads the index and every block; an --out naming one of them replaced the file it had just read."""
        block_path = os.path.join(self.extracted, self.index["entries"][0]["file"])
        for target in (self.index_path, block_path):
            with self.subTest(target=os.path.relpath(target, self.extracted)):
                before = rd(target)
                with self.assertRaises(ValueError) as error:
                    saves.pack(self.extracted, target)
                self.assertIn(target, str(error.exception))
                self.assertIn("would replace it", str(error.exception))
                self.assertNotIn("invalid JSON", str(error.exception))
                self.assertEqual(rd(target), before, "the source file changed")

    def test_cli_pack_refuses_an_out_that_names_the_index(self):
        before = rd(self.index_path)
        result = subprocess.run([sys.executable, "-I", TOOL, "pack", self.extracted, self.index_path],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("ERROR:", result.stderr)
        self.assertIn("would replace it", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(rd(self.index_path), before)

    def test_valid_editable_metadata_still_round_trips(self):
        self.index["header"]["note"] = "extra metadata is ignored"
        self.index["entries"][0]["hash"] = "0X" + self.index["entries"][0]["hash"]
        write(self.index_path, json.dumps(self.index).encode("utf-8"))
        self.assertEqual(saves.pack_from_dir(self.extracted), self.original)
        write(self.index_path, json.dumps({"header": {"version": 3, "code_blocks": 0},
                                         "entries": []}).encode("utf-8"))
        self.assertEqual(saves.pack_from_dir(self.extracted), saves.build_jsav(3, 0, [], [], []))


if __name__ == "__main__":
    unittest.main()
