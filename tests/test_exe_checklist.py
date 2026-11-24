"""tools/exe_checklist.py: the ASSUMPTION / UNVERIFIED / VERIFY notes, gathered for the executable's first look.

The repository here is a small tree built in a temporary folder (one shim, one doc), so the counts are exact; the
last test runs the tool on this repository to make sure every note it finds has a file, a line and some text.
Standard library only; no game files.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import exe_checklist as ec  # noqa: E402

SHIM = """// demo: a shim with two notes.
//
// ASSUMPTION: the quota is 1 GiB until the executable
// shows its own.
#include <x>
constexpr GUID kDemoClass = {0x704c3f58, 0xe629, 0x4cc2, {0xb1, 0x97, 0x30, 0x51, 0x1b, 0x99, 0x6f, 0xe2}};
constexpr size_t kSize = 18;   // UNVERIFIED, see the header
// The SKU. UNVERIFIED: nothing
// compares it.
int f();
"""
OTHER = """// no class here
// ASSUMPTION: a note in a file that serves no runtime class.
"""
DOC = """# Demo

Some text.

- **Names:** fine.
- **VERIFY:** the container names, from the
  executable's calls.

**ASSUMPTION:** a paragraph note
that spans two lines.

- **Unverified:** the same note in the docs' prose spelling.
- **Assumption** — the term's definition, not a note.
- **Assumptions to confirm:** the plural spelling is a note too.
"""
SAVE_CLASS = "704c3f58-e629-4cc2-b197-30511b996fe2"


class ExeChecklist(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        os.makedirs(os.path.join(self.root, "runtime", "shims"))
        os.makedirs(os.path.join(self.root, "docs"))
        for rel, text in (("runtime/shims/demo.cpp", SHIM), ("runtime/shims/other.cpp", OTHER), ("docs/demo.md", DOC)):
            with open(os.path.join(self.root, rel), "w", encoding="utf-8", newline="\n") as f:
                f.write(text)

    def tearDown(self):
        self.tmp.cleanup()

    def test_notes_start_at_their_marker_and_skip_trailing_comments(self):
        notes = ec.collect(self.root)
        demo = [n for n in notes if n["file"] == "runtime/shims/demo.cpp"]
        self.assertEqual([(n["line"], n["kind"]) for n in demo], [(3, "ASSUMPTION"), (8, "UNVERIFIED")])
        self.assertEqual(demo[0]["text"], "the quota is 1 GiB until the executable shows its own.")
        self.assertEqual(demo[1]["text"], "nothing compares it.")
        self.assertEqual(demo[0]["classes"], [SAVE_CLASS])

    def test_one_class_scan_per_file_with_several_notes(self):
        many = "".join(f"// ASSUMPTION: note {i}\nint value;\n" for i in range(4))
        with open(os.path.join(self.root, "runtime/shims/many.cpp"), "w", encoding="utf-8", newline="\n") as f:
            f.write(SHIM + many)
        seen, real = [], ec.served_classes

        def counting(source):
            seen.append(source)
            return real(source)

        ec.served_classes = counting
        try:
            notes = [n for n in ec.collect(self.root) if n["file"] == "runtime/shims/many.cpp"]
        finally:
            ec.served_classes = real
        self.assertEqual(len(seen), 3)      # one scan per file with notes (demo.cpp, other.cpp, many.cpp)
        self.assertEqual(sum(1 for s in seen if "note 0" in s), 1)   # not once per note
        self.assertEqual(len(notes), 6)     # two in the header, four added
        self.assertEqual({tuple(n["classes"]) for n in notes}, {(SAVE_CLASS,)})

    def test_doc_notes_are_the_list_item_or_paragraph(self):
        docs = [n for n in ec.collect(self.root) if n["file"] == "docs/demo.md"]
        self.assertEqual([(n["line"], n["kind"]) for n in docs],
                         [(6, "VERIFY"), (9, "ASSUMPTION"), (12, "UNVERIFIED"), (14, "ASSUMPTION")])
        self.assertEqual(docs[0]["text"], "the container names, from the executable's calls.")
        self.assertEqual(docs[1]["text"], "a paragraph note that spans two lines.")
        self.assertEqual(docs[2]["text"], "the same note in the docs' prose spelling.")
        self.assertEqual(docs[3]["text"], "the plural spelling is a note too.")

    def test_a_marker_inside_a_sentence_or_a_table_row_is_prose(self):
        with open(os.path.join(self.root, "docs", "demo.md"), "a", encoding="utf-8", newline="\n") as f:
            f.write("| `v_demo` | written without the console (**Assumption**: not a note) |\n"
                    "| **Unverified:** nothing | `v_other` |\n"
                    "\n"
                    "A sentence that carries an **Assumption:** after its start is not a note either.\n"
                    "\n"
                    "- **A list item** whose next sentence says **Unverified:** this is not a note of its own.\n")
        docs = [n for n in ec.collect(self.root) if n["file"] == "docs/demo.md"]
        self.assertEqual([(n["line"], n["kind"]) for n in docs],
                         [(6, "VERIFY"), (9, "ASSUMPTION"), (12, "UNVERIFIED"), (14, "ASSUMPTION")])

    def test_called_notes_come_first(self):
        report = os.path.join(self.root, "x.gdk_calls.json")
        with open(report, "w", encoding="utf-8") as f:
            json.dump({"images": [{"classes": [{"class_id": SAVE_CLASS.upper()}]}]}, f)
        out = os.path.join(self.root, "checklist.md")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ec.main(["--root", self.root, "--gdk-calls", report, "--out", out]), 0)
        with open(out, "rb") as f:
            raw = f.read()
        self.assertNotIn(b"\r\n", raw)
        lines = [l for l in raw.decode("utf-8").split("\n") if l.startswith("- [ ]")]
        self.assertIn("demo.cpp:3` ASSUMPTION **called**", lines[0])
        self.assertIn("demo.cpp:8` UNVERIFIED **called**", lines[1])
        self.assertIn("other.cpp:2` ASSUMPTION:", lines[2])   # serves no class: no tag, after the called ones
        self.assertEqual(len(lines), 7)

    def test_a_class_the_executable_does_not_call(self):
        report = os.path.join(self.root, "y.json")
        with open(report, "w", encoding="utf-8") as f:
            json.dump({"classes": [{"class_id": "00000000-0000-0000-0000-000000000001"}]}, f)   # one image, no "images"
        text = ec.render(ec.collect(self.root), ec.called_classes([report]))
        self.assertIn("demo.cpp:3` ASSUMPTION (not called)", text)

    def test_verify_notes_in_code_are_collected(self):
        verify = "// pad demo.\n//\n// The pads (VERIFY: against the game).\n// VERIFY: the timeout is not honoured.\n"
        with open(os.path.join(self.root, "runtime", "shims", "verify.cpp"), "w", encoding="utf-8", newline="\n") as f:
            f.write(verify)
        notes = [n for n in ec.collect(self.root) if n["file"] == "runtime/shims/verify.cpp"]
        self.assertEqual([(n["line"], n["kind"]) for n in notes], [(3, "VERIFY"), (4, "VERIFY")])
        self.assertEqual(notes[0]["text"], "against the game).")
        self.assertEqual(notes[1]["text"], "the timeout is not honoured.")

    def test_a_root_that_is_not_a_directory_is_exit_2(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ec.main(["--root", os.path.join(self.root, "no_such")]), 2)
        self.assertIn("not a directory", stderr.getvalue())

    def test_an_out_that_is_a_directory_is_exit_2(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ec.main(["--root", self.root, "--out", os.path.join(self.root, "docs")]), 2)
        self.assertIn("is a directory", stderr.getvalue())

    def test_an_unreadable_report_is_exit_1(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.assertEqual(ec.main(["--root", self.root, "--gdk-calls", os.path.join(self.root, "missing.json")]), 1)
        self.assertIn("cannot read", stderr.getvalue())

    def assert_report_output_rejected(self, reports, output):
        original = [report.read_bytes() for report in reports]
        args = ["--root", self.root, "--out", str(output)]
        for report in reports:
            args.extend(["--gdk-calls", str(report)])
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                ec.main(args)
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("would overwrite", stderr.getvalue())
        self.assertEqual([report.read_bytes() for report in reports], original)

    def test_an_out_that_cannot_be_written_is_exit_1(self):
        # --out under a folder that does not exist was a FileNotFoundError traceback.
        out = os.path.join(self.root, "missing_dir", "notes.md")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ec.main(["--root", self.root, "--out", out]), 1)
        self.assertIn("cannot write the checklist", stderr.getvalue())

    def test_out_cannot_overwrite_any_input_report(self):
        first = Path(self.root, "first.json")
        second = Path(self.root, "second.json")
        first.write_bytes(b'{"classes": []}')
        second.write_bytes(b'{"images": []}')
        self.assert_report_output_rejected([first, second], second)

    def test_out_cannot_overwrite_a_hard_link_to_an_input_report(self):
        report = Path(self.root, "report.json")
        alias = Path(self.root, "output.md")
        report.write_bytes(b'{"classes": []}')
        try:
            os.link(report, alias)
        except (OSError, NotImplementedError) as err:
            self.skipTest(f"hard links unavailable: {err}")
        self.assert_report_output_rejected([report], alias)

    def test_this_repository(self):
        notes = ec.collect(ROOT)
        self.assertTrue(notes)
        for n in notes:
            self.assertTrue(n["text"].strip(), n)
            self.assertGreater(n["line"], 0)
        # the store shim's remaining unverified sizes (SKU_ID_SIZE, trial-id bound) are on the list
        self.assertTrue(any(n["file"] == "runtime/shims/xgameruntime_store.cpp" and "SKU_ID_SIZE" in n["text"]
                            for n in notes))
        self.assertTrue(any(n["file"] == "docs/runtime_imports.md" and n["kind"] == "UNVERIFIED" for n in notes))
        # d3d12_vulkan.md's console assumptions carry the page-wide uppercase marker, so the first look lists them
        self.assertTrue(any(n["file"] == "docs/d3d12_vulkan.md" and n["kind"] == "ASSUMPTION" for n in notes))
        # so does modding.md's note on the three settings only the executable can confirm
        self.assertTrue(any(n["file"] == "docs/modding.md" and n["kind"] == "ASSUMPTION" for n in notes))


if __name__ == "__main__":
    unittest.main()
