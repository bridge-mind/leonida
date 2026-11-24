"""Tests for tools/awc_names.py, built from synthetic AWC stream tables and words (no game data).

    python -I tests/test_awc_names.py
"""
import contextlib
import csv
import importlib.util
import io
import json
import os
import struct
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_module():
    path = os.path.join(ROOT, "tools", "awc_names.py")
    spec = importlib.util.spec_from_file_location("awc_names_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A = load_module()


def make_bank(names, flags=0, extra_ids=()):
    """A minimal AWC: header, the optional tables ``flags`` selects, and a stream table without tags."""
    ids = [A.joaat(n) & A.M29 for n in names] + list(extra_ids)
    n = len(ids)
    out = bytearray(b"ADAT" + struct.pack("<III", flags, n, 0))
    if flags & 0x400000:
        out += bytes(4)
    if flags & 0x10000:
        out += bytes(2 * n)
    if flags & 0x200000:
        out += bytes(4 * (n + 1))
    # a non-zero tag count in bits 31:29 must not leak into the id
    out += b"".join(struct.pack("<I", (1 << 29) | i) for i in ids)
    struct.pack_into("<I", out, 12, len(out))
    return bytes(out)


def write(path, data):
    with open(path, "wb" if isinstance(data, bytes) else "w") as fh:
        fh.write(data)
    return path


class TestJoaat(unittest.TestCase):
    def test_reference_values(self):
        # Jenkins one-at-a-time, lowercased (the same function as tools/hashdict.py)
        self.assertEqual(A.joaat("a"), 0xCA2E9442)
        self.assertEqual(A.joaat("ABC"), A.joaat("abc"))

    def test_incremental_state(self):
        base = A.joaat_state(b"foo_bar")
        self.assertEqual(A.joaat_finish(A.joaat_state(b"_01", base)), A.joaat("foo_bar_01"))

    def test_matches_hashdict(self):
        import importlib.util as u
        spec = u.spec_from_file_location("hashdict_ref", os.path.join(ROOT, "tools", "hashdict.py"))
        hd = u.module_from_spec(spec)
        spec.loader.exec_module(hd)
        for w in ("x", "Some_Name_01", "path/to/thing", "caf\xe9"):
            self.assertEqual(A.joaat(w), hd.joaat(w))


class TestBanks(unittest.TestCase):
    def test_flags_skip_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            for flags in (0, 0x400000, 0x10000, 0x200000, 0x610000):
                path = write(os.path.join(tmp, f"{flags:x}.awc"), make_bank(["a_01", "b_02"], flags))
                self.assertEqual(A.bank_ids(path), [A.joaat("a_01") & A.M29, A.joaat("b_02") & A.M29])

    def test_directory_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(os.path.join(tmp, "sub"))
            write(os.path.join(tmp, "sub", "one.awc"), make_bank(["x"]))
            write(os.path.join(tmp, "readme.txt"), "not a bank")
            self.assertEqual(list(A.load_banks(tmp)), ["sub/one"])
            manifest = write(os.path.join(tmp, "audio_manifest.json"), json.dumps(
                {"src": "audio.rpf", "Count": 2, "Banks": [{"Path": "p/b.awc", "Streams": [{"Id": "%08X" % (A.joaat("x") & A.M29)}]},
                                {"Path": "p/unreadable.awc", "Streams": None}]}))
            self.assertEqual(A.load_banks(manifest), {"p/b.awc": [A.joaat("x") & A.M29], "p/unreadable.awc": []})

    def test_truncated_bank(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(os.path.join(tmp, "t.awc"), make_bank(["a", "b"])[:-3])
            with self.assertRaises(ValueError):
                A.bank_ids(path)

    def test_stream_table_outside_declared_header(self):
        # The file is long enough, but the declared header cannot hold the stream table; the canonical
        # rsc8_audio parser rejects the same bytes.
        table = struct.pack("<II", 0x11223344, 0x55667788)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.awc")
            for header_size in (8, 16, 23):
                with self.subTest(header_size=header_size):
                    with open(path, "wb") as fh:
                        fh.write(b"ADAT" + struct.pack("<III", 0, 2, header_size) + table)
                    with self.assertRaisesRegex(ValueError, "declared header"):
                        A.bank_ids(path)

    def test_declared_header_past_the_file(self):
        # The stream table fits in the file and in the declared header, but the header itself does not fit in
        # the file; the canonical rsc8_audio parser rejects the same bytes.
        raw = b"ADAT" + struct.pack("<IIII", 0, 1, 21, 7)
        with tempfile.TemporaryDirectory() as tmp:
            path = write(os.path.join(tmp, "bad.awc"), raw)
            with self.assertRaisesRegex(ValueError, "declared header"):
                A.bank_ids(path)

    def test_truncated_stream_table_reads_only_header(self):
        # A truncated ordinary bank with a stream count larger than the file holds must fail from the header
        # and file size alone: os.fstat reports the real size without a seek, so only the 16-byte header is
        # ever read before the size check fails.
        n = 12
        header = b"ADAT" + struct.pack("<III", 0, n, 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.awc")
            with open(path, "wb") as f:
                f.write(header + bytes(3))

            with open(path, "rb") as source:
                spy = mock.Mock(wraps=source)
                with mock.patch.object(A, "open", create=True, return_value=contextlib.nullcontext(spy)):
                    with self.assertRaisesRegex(ValueError, "stream table extends past"):
                        A.bank_ids(path)
            self.assertEqual(spy.read.call_args_list, [mock.call(16)])


class TestRules(unittest.TestCase):
    def index(self, banks):
        return A.Index({k: [A.joaat(n) & A.M29 for n in v] for k, v in banks.items()})

    def common_words(self, *tokens):
        # each token in MIN_TOKEN words so family bases built from them count as "wordy"
        return [f"{t}_filler{k}" for t in tokens for k in range(A.MIN_TOKEN)]

    def test_dialogue_rule_exact_word(self):
        d = A.Dictionary(["zz_rootau_abcd_01", "zz_rootau_abcd"])
        found = A.attack(d, d.bases(A.MIN_TOKEN), self.index({"b": ["zz_rootau_abcd_01"]}), "")
        self.assertEqual([v[:2] for v in found.values()], [("zz_rootau_abcd_01", "dialogue")])

    def test_family_needs_members_in_one_bank_and_a_verbatim_word(self):
        words = self.common_words("horn", "loop") + ["horn_loop_01"]
        d = A.Dictionary(words)
        bases = d.bases(A.MIN_TOKEN)
        self.assertIn("horn_loop", bases)
        # three members in one bank -> accepted (horn_loop_02/_03 are not words but extend the family)
        found = A.attack(d, bases, self.index({"b": ["horn_loop_01", "horn_loop_02", "horn_loop_03"]}), "")
        self.assertEqual(sorted(v[0] for v in found.values()), ["horn_loop_01", "horn_loop_02", "horn_loop_03"])
        # spread over banks -> no family anywhere
        found = A.attack(d, bases, self.index({"b1": ["horn_loop_01"], "b2": ["horn_loop_02"],
                                               "b3": ["horn_loop_03"]}), "")
        self.assertEqual(found, {})

    def test_family_without_verbatim_member_is_rejected(self):
        d = A.Dictionary(self.common_words("horn", "loop") + ["horn_loop_77"])
        found = A.attack(d, d.bases(A.MIN_TOKEN), self.index({"b": ["horn_loop_01", "horn_loop_02",
                                                                     "horn_loop_03"]}), "")
        self.assertEqual(found, {})

    def test_rare_token_base_is_not_wordy(self):
        d = A.Dictionary(self.common_words("horn") + ["horn_qqq_01"])
        self.assertNotIn("horn_qqq", d.bases(A.MIN_TOKEN))
        self.assertIn("horn_qqq", d.bases(1))

    def test_family_ok(self):
        self.assertTrue(A.family_ok(["a_01", "a_02"]))
        self.assertFalse(A.family_ok(["a_01", "a_03"]))
        self.assertTrue(A.family_ok(["a_01", "a_05", "a_09"]))
        self.assertFalse(A.family_ok(["a_01"]))

    def test_control_salt_changes_every_name(self):
        d = A.Dictionary(["zz_rootau_abcd_01"])
        found = A.attack(d, [], self.index({"b": ["zz_rootau_abcd_01"]}), A.SALT)
        self.assertEqual(found, {})

    def test_name_source_keeps_rules_with_quiet_controls(self):
        words = self.common_words("horn", "loop") + ["horn_loop_01", "zz_rootau_abcd_01"]
        d = A.Dictionary(words)
        index = self.index({"b": ["horn_loop_01", "horn_loop_02", "zz_rootau_abcd_01"]})
        found, stats = A.name_source(d, d.bases(A.MIN_TOKEN), index)
        self.assertEqual(stats["named"], 3)
        self.assertTrue(stats["dialogue_rule"])
        self.assertEqual(stats["family_min_size"], 2)
        self.assertEqual(stats["control_dialogue"] + stats["control_family"], 0)

    def test_rejected_dialogue_hit_does_not_shadow_accepted_family_member(self):
        d = A.Dictionary(["zz_fooau_abcd_01"])
        bases = ["zz_fooau_abcd"]
        # Including SALT + zz_fooau_abcd_01 makes the salted control reject the dialogue rule;
        # zz_fooau_abcd_01 must still be named via its 3-member family alongside _02 and _03.
        index = self.index({
            "b0": [
                "zz_fooau_abcd_01",
                "zz_fooau_abcd_02",
                "zz_fooau_abcd_03",
                A.SALT + "zz_fooau_abcd_01",
            ]
        })
        real, stats = A.name_source(d, bases, index)
        self.assertFalse(stats["dialogue_rule"])
        self.assertEqual(stats["family_min_size"], 2)
        self.assertEqual(stats["named"], 3)
        self.assertEqual(
            sorted(real.values()),
            [
                ("zz_fooau_abcd_01", "family", 3),
                ("zz_fooau_abcd_02", "family", 3),
                ("zz_fooau_abcd_03", "family", 3),
            ],
        )


class TestConversations(unittest.TestCase):
    def test_waves_per_label(self):
        blob = (b"junk\x00ZZ_ROOTAU_ABCD\x00ZZ_LINE_A\x00ZZ_ROOTAU_ABCD\x00ZZ_LINE_B\x00"
                b"ZZ_ROOTAU_ABCD\x00ZZ_LINE_A\x00YY_XAU_EFGH\x00YY_ONLY\x00")
        with tempfile.TemporaryDirectory() as tmp:
            write(os.path.join(tmp, "c.pso"), blob)
            waves = sorted(A.conversation_waves([tmp]))
        # a repeated label is one variation, not two
        self.assertEqual(waves, ["yy_xau_efgh_01", "zz_rootau_abcd_01", "zz_rootau_abcd_02"])

    @staticmethod
    def pairs(*labels):
        return [b"ZZ_ROOTAU_ABCD\x00" + label + b"\x00" for label in labels]

    def carve(self, tmp, streams):
        """A carve folder: one `<offset>.bin` per stream, laid out at 64 KiB steps as rpf9_carve.py names them."""
        folder = os.path.join(tmp, "story000")
        os.makedirs(folder)
        for k, data in enumerate(streams):
            write(os.path.join(folder, f"{k * 0x10000:012x}.bin"), data)
        return folder

    def full_stream(self, tail, head=b"PSIN" + bytes(12), size=65536):
        """`size` bytes that start with `head` and end with `tail`."""
        return head + b"\x01" * (size - len(head) - len(tail)) + tail

    def test_pair_across_a_continuation_boundary(self):
        # A carved stream of a whole number of 64 KiB blocks continues in the next file (docs/page_directory.md):
        # the third pair is cut after 10 bytes and must still count, so the key has four waves (docs/audio.md).
        one, two, three, four = self.pairs(b"ZZ_LINE_A", b"ZZ_LINE_B", b"ZZ_LINE_C", b"ZZ_LINE_D")
        for size in (65536, 2 * 65536):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as tmp:
                folder = self.carve(tmp, [self.full_stream(one + two + three[:10], size=size),
                                          three[10:] + b"\x02" * 8 + four])
                waves = sorted(A.conversation_waves([folder]))
                self.assertEqual(waves, [f"zz_rootau_abcd_{n:02d}" for n in range(1, 5)])

    def test_streams_that_do_not_continue_are_matched_apart(self):
        # A stream short of a whole block ends its blob, and a new PSIN stream starts one: no pair is made across.
        one, two = self.pairs(b"ZZ_LINE_A", b"ZZ_LINE_B")
        for first, second in ((self.full_stream(one + two[:10], size=65535), two[10:]),
                              (self.full_stream(one + two[:10]), b"PSIN" + two[10:])):
            with self.subTest(first=len(first)), tempfile.TemporaryDirectory() as tmp:
                folder = self.carve(tmp, [first, second])
                self.assertEqual(A.conversation_waves([folder]), ["zz_rootau_abcd_01"])


class TestCli(unittest.TestCase):
    def test_sources_with_same_basename_keep_separate_summaries_and_streams(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources = []
            for parent, count in (("first", 1), ("second", 2)):
                folder = os.path.join(tmp, parent, "banks")
                os.makedirs(folder)
                sources.append(folder)
                write(os.path.join(folder, "one.awc"), make_bank(["zz_rootau_abcd_01"], extra_ids=range(count - 1)))
            words = write(os.path.join(tmp, "words.txt"), "zz_rootau_abcd_01\n")
            out = os.path.join(tmp, "names.json")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(A.main([*sources, "--words", words, "--out", out]), 0)
            with open(out, encoding="utf-8") as fh:
                doc = json.load(fh)
            self.assertEqual({k: v["streams"] for k, v in doc["summary"].items()},
                             {"first/banks": 1, "second/banks": 2})
            self.assertEqual([row["source"] for row in doc["streams"]], ["first/banks", "second/banks"])

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            banks = os.path.join(tmp, "banks")
            os.mkdir(banks)
            write(os.path.join(banks, "one.awc"), make_bank(["zz_rootau_abcd_01", "zz_rootau_abcd_02"],
                                                            extra_ids=[12345, 12345]))
            conv = write(os.path.join(tmp, "c.pso"),
                         b"ZZ_ROOTAU_ABCD\x00ZZ_A\x00ZZ_ROOTAU_ABCD\x00ZZ_B\x00")
            words = write(os.path.join(tmp, "words.txt"), "unrelated\nother_word\n")
            out = os.path.join(tmp, "names.csv")
            with contextlib.redirect_stdout(io.StringIO()) as log:
                self.assertEqual(A.main([banks, "--words", words, "--conversations", conv, "--out", out]), 0)
            self.assertIn("3 distinct bank/ID pairs, 2 named", log.getvalue())
            with open(out, encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual([(r["bank"], r["name"], r["rule"]) for r in rows],
                             [("one", "zz_rootau_abcd_01", "dialogue"), ("one", "zz_rootau_abcd_02", "dialogue")]
                             if A.joaat("zz_rootau_abcd_01") & A.M29 < A.joaat("zz_rootau_abcd_02") & A.M29 else
                             [("one", "zz_rootau_abcd_02", "dialogue"), ("one", "zz_rootau_abcd_01", "dialogue")])
            json_out = os.path.join(tmp, "names.json")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(A.main([banks, "--conversations", conv, "--out", json_out]), 0)
            with open(json_out, encoding="utf-8") as fh:
                self.assertEqual(json.load(fh)["summary"]["banks"]["streams"], 3)

    def test_rejects_output_aliasing_conversation(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank = write(os.path.join(tmp, "one.awc"), make_bank(["zz_rootau_abcd_01"]))
            original = b"ZZ_ROOTAU_ABCD\x00ZZ_A\x00"
            conv = write(os.path.join(tmp, "c.pso"), original)
            alias = os.path.join(tmp, ".", "c.pso")
            err = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                with self.assertRaises(SystemExit) as cm:
                    A.main([bank, "--conversations", conv, "--out", alias])
            self.assertEqual(cm.exception.code, 2)
            self.assertIn("--out must not overwrite an input", err.getvalue())
            with open(conv, "rb") as f:
                self.assertEqual(f.read(), original)

    def test_rejects_output_aliasing_recursive_inputs(self):
        for source in ("bank", "conversation"):
            for alias in ("path", "parent", "hardlink"):
                with self.subTest(source=source, alias=alias), tempfile.TemporaryDirectory() as tmp:
                    banks = os.path.join(tmp, "banks")
                    conversations = os.path.join(tmp, "conversations")
                    os.makedirs(os.path.join(banks, "nested"))
                    os.makedirs(os.path.join(conversations, "nested"))
                    bank_data = make_bank(["zz_rootau_abcd_01"])
                    conv_data = b"ZZ_ROOTAU_ABCD\x00ZZ_A\x00"
                    bank = write(os.path.join(banks, "nested", "one.AWC"), bank_data)
                    conv = write(os.path.join(conversations, "nested", "labels.bin"), conv_data)
                    target = bank if source == "bank" else conv
                    if alias == "parent":
                        out = os.path.join(os.path.dirname(target), "..", "nested", os.path.basename(target))
                    elif alias == "hardlink":
                        out = os.path.join(tmp, "names.csv")
                        try:
                            os.link(target, out)
                        except (OSError, NotImplementedError) as e:
                            self.skipTest(f"hard links unavailable: {e}")
                    else:
                        out = target
                    err = io.StringIO()
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                        with self.assertRaises(SystemExit) as cm:
                            A.main([banks, "--conversations", conversations, "--out", out])
                    self.assertEqual(cm.exception.code, 2)
                    self.assertIn("--out must not overwrite an input", err.getvalue())
                    for path, original in ((bank, bank_data), (conv, conv_data)):
                        with open(path, "rb") as fh:
                            self.assertEqual(fh.read(), original)

    def test_allows_report_inside_bank_directory(self):
        for existing in (False, True):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as tmp:
                banks = os.path.join(tmp, "banks")
                os.makedirs(os.path.join(banks, "nested"))
                bank_data = make_bank(["zz_rootau_abcd_01"])
                bank = write(os.path.join(banks, "nested", "one.awc"), bank_data)
                conv = write(os.path.join(tmp, "labels.bin"), b"ZZ_ROOTAU_ABCD\x00ZZ_A\x00")
                out = os.path.join(banks, "nested", "names.csv")
                if existing:
                    write(out, "old report\n")
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(A.main([banks, "--conversations", conv, "--out", out]), 0)
                with open(out, encoding="utf-8", newline="") as fh:
                    self.assertEqual(len(list(csv.DictReader(fh))), 1)
                with open(bank, "rb") as fh:
                    self.assertEqual(fh.read(), bank_data)

    def test_hashdict_json_words(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(os.path.join(tmp, "hashes.json"), json.dumps({"0000abcd": "Some_Word"}))
            self.assertEqual(A.read_words(path), ["Some_Word"])

    def test_needs_a_dictionary(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            A.main([ROOT])

    def test_malformed_banks_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            words = write(os.path.join(tmp, "words.json"), json.dumps(["x"]))
            for doc in ([1], [["Path"]]):
                banks = write(os.path.join(tmp, "banks.json"), json.dumps(doc))
                err = io.StringIO()
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                    self.assertEqual(A.main(["--words", words, banks]), 2)
                self.assertEqual(err.getvalue().count("\n"), 1, err.getvalue())
                self.assertIn("awc_names: error:", err.getvalue())

    def test_missing_input(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(A.main(["/nonexistent", "--words", "/nonexistent.txt"]), 2)

    def test_min_token_below_one_is_rejected(self):
        # --min-token counts the dictionary words a token must appear in. 0 or a negative used to pass
        # type=int and silently disable the common-token filter (every token passes tokens[t] >= min_token).
        with tempfile.TemporaryDirectory() as tmp:
            bank = write(os.path.join(tmp, "b.awc"), make_bank(["abc_01", "abc_02", "abc_03"]))
            words = write(os.path.join(tmp, "w.txt"), "abc_01\nabc_02\nabc_03\n")
            for bad in ("0", "-1", "-5"):
                with self.subTest(min_token=bad):
                    err = io.StringIO()
                    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaises(SystemExit) as cm:
                            A.main([bank, "--words", words, "--min-token", bad])
                    self.assertEqual(cm.exception.code, 2)
                    self.assertIn("at least 1", err.getvalue())
                    self.assertNotIn("Traceback", err.getvalue())
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as cm:
                    A.main([bank, "--words", words, "--min-token", "x"])
            self.assertEqual(cm.exception.code, 2)
            self.assertNotIn("Traceback", err.getvalue())
            # 1 is the smallest meaningful value and still runs.
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(A.main([bank, "--words", words, "--min-token", "1"]), 0)

    @unittest.skipUnless(sys.platform == "win32", "the exclusive-lock repro needs Windows")
    def test_locked_words_input_reports_rc_2_without_traceback(self):
        # An exclusively locked (share-none) existing file passes the os.path.exists check, so the
        # read_words() open must turn PermissionError into the one-line rc-2 error, not a traceback.
        import ctypes
        with tempfile.TemporaryDirectory() as tmp:
            blob = os.path.join(tmp, "locked.awc")
            with open(blob, "wb") as fh:
                fh.write(b"\x00" * 16)
            k32 = ctypes.windll.kernel32
            handle = k32.CreateFileW(blob, 0x80000000, 0, None, 3, 0x80, None)
            self.assertNotIn(handle, (0, -1), "could not lock the probe file")
            try:
                err = io.StringIO()
                with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                    rc = A.main([blob, "--words", blob])
            finally:
                k32.CloseHandle(handle)
            self.assertEqual(rc, 2)
            self.assertIn("cannot read", err.getvalue())
            self.assertIn("Permission denied", err.getvalue())
            self.assertNotIn("Traceback", err.getvalue())

    def test_unreadable_json_words_report_rc_2_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            words = write(os.path.join(tmp, "words.json"), "{not json")
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                rc = A.main([tmp, "--words", words])
            self.assertEqual(rc, 2)
            self.assertIn("cannot read", err.getvalue())
            self.assertNotIn("Traceback", err.getvalue())

    def test_scalar_json_words_report_rc_2_without_traceback(self):
        # A JSON scalar is neither a hashdict object nor a word list: one error line, exit 2. A string
        # must not be read as one word per character.
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "o.csv")
            for doc in ("42", "null", "true", '"abc"'):
                with self.subTest(doc=doc):
                    words = write(os.path.join(tmp, "words.json"), doc)
                    err = io.StringIO()
                    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                        rc = A.main([tmp, "--words", words, "--out", out])
                    self.assertEqual(rc, 2)
                    self.assertEqual(err.getvalue().count("\n"), 1, err.getvalue())
                    self.assertIn("awc_names: error: cannot read", err.getvalue())
                    self.assertFalse(os.path.exists(out))

    def test_json_word_list_still_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(os.path.join(tmp, "words.json"), json.dumps(["a_word", 2]))
            self.assertEqual(A.read_words(path), ["a_word", "2"])


if __name__ == "__main__":
    unittest.main()
