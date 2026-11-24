"""Unit tests for tools/boot_trace.py on synthetic traces (no game data, no Windows, no GPU)."""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import boot_trace as bt

STARTUP = [
    {"seq": 1, "t": 0.0, "kind": "trace_open", "pid": 123},
    {
        "seq": 2,
        "kind": "map",
        "module": "game.exe",
        "base": "0x140000000",
        "size": "0x1000",
        "relocated": False,
    },
    {
        "seq": 3,
        "kind": "map",
        "module": "x.dll",
        "base": "0x180000000",
        "size": "0x2000",
        "relocated": True,
    },
    {
        "seq": 4,
        "kind": "tls_callback",
        "module": "x.dll",
        "at": "x.dll+0x10",
        "reason": 1,
    },
    {"seq": 5, "kind": "dllmain", "module": "x.dll", "result": 1},
    {"seq": 6, "kind": "dllmain", "module": "y.dll", "result": 0},
    {
        "seq": 7,
        "kind": "unimplemented_call",
        "import": "x.dll!Foo",
        "caller": "game.exe+0x10",
    },
    {
        "seq": 8,
        "kind": "unimplemented_call",
        "import": "x.dll!Foo",
        "caller": "game.exe+0x20",
    },
    {
        "seq": 9,
        "kind": "unimplemented_call",
        "import": "y.dll!Bar",
        "caller": "game.exe+0x30",
    },
    {"seq": 10, "kind": "debug_string", "text": "secret game text"},
    {
        "seq": 11,
        "kind": "exception",
        "code": "0xc0000005",
        "at": "game.exe+0x40",
        "access": "read",
        "address": "0x0",
        "stack": ["game.exe+0x40", "game.exe+0x120"],
    },
    {
        "seq": 12,
        "kind": "exception",
        "code": "0xe06d7363",
        "at": "game.exe+0x30",
        "cpp": True,
    },
    {"seq": 13, "kind": "future_event", "detail": "a kind this version does not know"},
    {"seq": 14, "kind": "exit", "code": 0},
]

D3D12 = [
    "[rt] ordinary diagnostic",
    "[d3d12-trace] #1 D3D12XboxCreateDevice ok",
    "[d3d12-trace] #2 CreateCommandQueue failed",
    "[d3d12-trace] #3 Foo unsupported",
    "[d3d12-trace] #4 D3D12XboxCreateDevice ok",
    "[d3d12-trace] #5 SetMarker/BeginEvent with unknown metadata unsupported",
    "[d3d12-trace] #6 MapTiles refused",
    "[d3d12-trace] #7 CreateRootSignature called",
    "[d3d12-trace] #8 SetMarker ignored",
    "d3d12_test.exe : [d3d12-trace] #9 ID3D12CommandQueue::ExecuteCommandLists called",
]


class BootTraceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.startup = os.path.join(self.tmp.name, "boot.jsonl")
        self.d3d12 = os.path.join(self.tmp.name, "d3d12.log")

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, path, events, newline="\n"):
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.writelines(json.dumps(event) + newline for event in events)

    def write_lines(self, path, lines):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def test_startup_summary(self):
        self.write(self.startup, STARTUP)
        trace = bt.parse_startup(self.startup)
        self.assertEqual(trace["pid"], 123)
        self.assertEqual(trace["events"], len(STARTUP))
        self.assertEqual(trace["bad_lines"], 0)
        self.assertEqual(trace["kinds"]["map"], 2)
        self.assertEqual(trace["kinds"]["future_event"], 1)
        self.assertEqual([m["module"] for m in trace["modules"]], ["game.exe", "x.dll"])
        self.assertEqual(trace["tls_callbacks"], 1)
        self.assertEqual(
            trace["dllmain"], {"ok": 1, "failed": 1, "failed_modules": ["y.dll"]}
        )
        self.assertEqual(
            [u["import"] for u in trace["unimplemented"]], ["x.dll!Foo", "y.dll!Bar"]
        )
        self.assertEqual(trace["unimplemented"][0]["caller"], "game.exe+0x10")
        self.assertEqual(trace["unimplemented_by_dll"], {"x.dll": 1, "y.dll": 1})
        self.assertEqual(trace["debug_strings"], 1)
        self.assertEqual(trace["exception_count"], 2)
        self.assertEqual(
            trace["exceptions"][0]["stack"], ["game.exe+0x40", "game.exe+0x120"]
        )
        self.assertIs(trace["exceptions"][1]["cpp"], True)
        self.assertEqual(trace["exit"], 0)

    def test_exception_limit_keeps_final_observation(self):
        early = [
            {"kind": "exception", "code": "0xe06d7363", "at": f"synthetic.exe+0x{0x1100 + i:x}", "cpp": True}
            for i in range(10)
        ]
        final = {
            "kind": "exception",
            "code": "0xc0000005",
            "at": "synthetic.exe+0x2010",
            "access": "read",
            "address": "0x0",
            "stack": ["synthetic.exe+0x2010", "synthetic.exe+0x1000"],
        }
        for limit, preceding in ((1, early), (10, early), (10, early[:-1])):
            with self.subTest(limit=limit, count=len(preceding) + 1):
                self.write(self.startup, [STARTUP[0], *preceding, final])
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(bt.main([self.startup, "--limit", str(limit), "--json"]), 0)
                summary = json.loads(output.getvalue())
                startup = summary["startup"]
                self.assertEqual(startup["exception_count"], len(preceding) + 1)
                self.assertEqual(
                    [exc["at"] for exc in startup["exceptions"]],
                    [exc["at"] for exc in preceding[:limit - 1]] + [final["at"]],
                )
                self.assertEqual(startup["exceptions"][-1]["stack"], final["stack"])
                self.assertIn("synthetic.exe+0x2010 (read 0x0)", bt.format_text(summary, limit))

    def test_debug_text_is_not_printed(self):
        self.write(self.startup, STARTUP)
        text = bt.format_text(bt.summarize(bt.parse_startup(self.startup), None), 10)
        self.assertIn("debug strings: 1", text)
        self.assertNotIn("secret game text", text)

    def test_truncated_line_is_counted_not_fatal(self):
        self.write(self.startup, STARTUP)
        with open(self.startup, "a", encoding="utf-8") as f:
            f.write('{"seq": 15, "kind":')
        trace = bt.parse_startup(self.startup)
        self.assertEqual(trace["events"], len(STARTUP))
        self.assertEqual(trace["bad_lines"], 1)
        self.assertEqual(trace["bad_line_numbers"], [len(STARTUP) + 1])

    def test_crlf_and_non_object_lines_are_counted(self):
        self.write(
            self.startup,
            [
                {"seq": 1, "kind": "trace_open", "pid": 7},
                {"seq": 2, "kind": "exit", "code": 0},
            ],
            newline="\r\n",
        )
        with open(self.startup, "a", encoding="utf-8") as f:
            f.write("null\n")
        trace = bt.parse_startup(self.startup)
        self.assertEqual(trace["events"], 2)
        self.assertEqual(trace["bad_lines"], 1)

    def test_malformed_exit_records_preserve_the_last_valid_code(self):
        for code in (0, -7):
            with self.subTest(code=code):
                self.write(self.startup, [
                    {"kind": "trace_open", "pid": 123},
                    {"kind": "exit", "code": code},
                    {"kind": "exit"},
                    {"kind": "exit", "code": "0"},
                    {"kind": "exit", "code": True},
                    {"kind": "exit", "code": None},
                    {"kind": "exit", "code": 0.0},
                    {"kind": "debug_string", "text": "synthetic"},
                ])
                trace = bt.parse_startup(self.startup)
                self.assertEqual(trace["bad_lines"], 5)
                self.assertEqual(trace["bad_line_numbers"], [3, 4, 5, 6, 7])
                self.assertEqual(trace["events"], 3)
                self.assertEqual(trace["kinds"]["exit"], 1)
                self.assertIs(type(trace["exit"]), int)
                self.assertEqual(trace["exit"], code)
                self.assertEqual(trace["debug_strings"], 1)

    def test_d3d12_summary(self):
        self.write_lines(self.d3d12, D3D12)
        trace = bt.parse_d3d12(self.d3d12)
        self.assertEqual(trace["calls"], 9)
        self.assertEqual(trace["ignored_lines"], 1)
        self.assertEqual(
            trace["outcomes"],
            {
                "ok": 2,
                "failed": 1,
                "unsupported": 2,
                "refused": 1,
                "called": 2,
                "ignored": 1,
            },
        )
        self.assertEqual(
            trace["methods"],
            [
                "D3D12XboxCreateDevice",
                "CreateCommandQueue",
                "Foo",
                "SetMarker/BeginEvent with unknown metadata",
                "MapTiles",
                "CreateRootSignature",
                "SetMarker",
                "ID3D12CommandQueue::ExecuteCommandLists",
            ],
        )
        self.assertEqual(
            trace["first"]["D3D12XboxCreateDevice"], {"seq": 1, "outcome": "ok"}
        )
        self.assertEqual(
            trace["first"]["ID3D12CommandQueue::ExecuteCommandLists"],
            {"seq": 9, "outcome": "called"},
        )
        self.assertEqual(trace["failed"], [{"seq": 2, "method": "CreateCommandQueue"}])
        self.assertEqual(
            trace["unsupported"],
            [
                {"seq": 3, "method": "Foo"},
                {"seq": 5, "method": "SetMarker/BeginEvent with unknown metadata"},
            ],
        )
        self.assertEqual(trace["refused"], [{"seq": 6, "method": "MapTiles"}])
        self.assertEqual(trace["last_seq"], 9)
        self.assertEqual(trace["sequence_break_count"], 0)
        self.assertEqual(trace["sequence_breaks"], [])

    def test_sequence_breaks_count_boundaries_and_limit_examples(self):
        self.write_lines(self.d3d12, [
            "[d3d12-trace] #4 Foo called",
            "[rt] ordinary diagnostic",
            "[d3d12-trace] #5 Foo called",
            "[d3d12-trace] #12 Foo called",
            "[d3d12-trace] #1 Foo called",
            "[d3d12-trace] #1 Foo failed",
            "[d3d12-trace] #2 Foo called",
        ])
        examples = [
            {"expected": 1, "observed": 4},
            {"expected": 6, "observed": 12},
            {"expected": 13, "observed": 1},
            {"expected": 2, "observed": 1},
        ]
        for limit in (2, 4):
            with self.subTest(limit=limit):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(bt.main(["--d3d12", self.d3d12, "--json", "--limit", str(limit)]), 0)
                summary = json.loads(output.getvalue())
                trace = summary["d3d12"]
                self.assertEqual(trace["sequence_break_count"], 4)
                self.assertEqual(trace["sequence_breaks"], examples[:limit])
                self.assertEqual(trace["calls"], 6)
                self.assertEqual(trace["outcomes"], {"called": 5, "failed": 1})
                self.assertEqual(trace["last_seq"], 2)
                self.assertEqual(trace["ignored_lines"], 1)
                text = bt.format_text(summary, limit)
                self.assertIn("sequence breaks: 4", text)
                self.assertIn("expected #6, observed #12", text)
                self.assertEqual(text.count("observed #"), limit)

    def test_powershell_error_record_prefix_keeps_the_first_call(self):
        self.write_lines(
            self.d3d12,
            [
                "d3d12_test.exe : [d3d12-trace] #1 ID3D12GraphicsCommandList::SetPipelineState called",
                "At line:1 char:1",
                "+ & '.\\d3d12_test.exe' --trace-d3d12 2> log",
            ],
        )
        trace = bt.parse_d3d12(self.d3d12)
        self.assertEqual(trace["calls"], 1)
        self.assertEqual(
            trace["first"]["ID3D12GraphicsCommandList::SetPipelineState"],
            {"seq": 1, "outcome": "called"},
        )

    def test_d3d12_text_report_names_refused_and_spaced_methods(self):
        self.write_lines(self.d3d12, D3D12)
        text = bt.format_text(bt.summarize(None, bt.parse_d3d12(self.d3d12)), 10)
        self.assertIn("refused: #6 MapTiles", text)
        self.assertIn("#5 SetMarker/BeginEvent with unknown metadata", text)
        self.assertIn("ID3D12CommandQueue::ExecuteCommandLists", text)
        self.assertIn("[rt] diagnostics are not summarized", text)
        self.assertNotIn("sequence breaks", text)

    def test_partial_submissions_are_named_in_text_and_json(self):
        method = "ID3D12CommandQueue::ExecuteCommandLists"
        self.write_lines(self.d3d12, [
            f"[d3d12-trace] #1 {method} ok",
            f"[d3d12-trace] #2 {method} partial",
            f"[d3d12-trace] #3 {method} partial",
        ])
        trace = bt.parse_d3d12(self.d3d12, limit=1)
        self.assertEqual(trace["first"][method], {"seq": 1, "outcome": "ok"})
        self.assertEqual(trace["outcomes"]["partial"], 2)
        text = bt.format_text(bt.summarize(None, trace), 1)
        self.assertIn(f"  partial: #2 {method}", text)
        self.assertNotIn(f"#3 {method}", text)

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(bt.main(["--d3d12", self.d3d12, "--limit", "1", "--json"]), 0)
        parsed = json.loads(stdout.getvalue())["d3d12"]
        self.assertEqual(parsed["partial"], [{"seq": 2, "method": method}])
        self.assertEqual(parsed["outcomes"]["partial"], 2)

    def test_json_cli(self):
        self.write(self.startup, STARTUP)
        self.write_lines(self.d3d12, D3D12)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(
                bt.main([self.startup, "--d3d12", self.d3d12, "--json"]), 0
            )
        data = json.loads(out.getvalue())
        self.assertEqual(data["startup"]["pid"], 123)
        self.assertEqual(data["d3d12"]["calls"], 9)

    def test_missing_and_empty_files_are_errors(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(bt.main([os.path.join(self.tmp.name, "missing.jsonl")]), 2)
            open(self.startup, "w").close()
            self.assertEqual(bt.main([self.startup]), 2)
            self.assertEqual(bt.main(["--limit", "0", self.startup]), 2)
        self.assertIn("cannot read", err.getvalue())

    def test_attach_failure_is_reported_before_any_d3d12_call(self):
        self.write(self.startup, [{"kind": "error", "what": "guest.dll: DllMain returned FALSE"}])
        self.write_lines(self.d3d12, ["error: guest.dll: DllMain returned FALSE"])
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(bt.main([self.startup, "--d3d12", self.d3d12]), 2)
        self.assertIn("error: guest.dll: DllMain returned FALSE", out.getvalue())
        self.assertIn("no [d3d12-trace] line", err.getvalue())

    def test_missing_startup_preserves_d3d12_json(self):
        self.write_lines(self.d3d12, ["[d3d12-trace] #1 CreateCommandQueue failed"])
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(bt.main([self.startup, "--d3d12", self.d3d12, "--json"]), 2)
        self.assertIn("cannot read", err.getvalue())
        self.assertEqual(json.loads(out.getvalue()), bt.summarize(None, bt.parse_d3d12(self.d3d12)))

    def test_no_report_when_neither_input_can_be_read(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(bt.main([self.startup, "--d3d12", self.d3d12, "--json"]), 2)
        self.assertEqual(out.getvalue(), "")
        self.assertIn(self.startup, err.getvalue())
        self.assertIn(self.d3d12, err.getvalue())

    def test_d3d12_log_without_trace_lines_is_an_error(self):
        self.write_lines(self.d3d12, ["[rt] one", "[rt] two"])
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(bt.main(["--d3d12", self.d3d12]), 2)
        self.assertIn("no [d3d12-trace] line", err.getvalue())

    def test_utf16_logs_keep_both_wrapped_and_unwrapped_calls(self):
        lines = ("synthetic.exe : [d3d12-trace] #1 First called\r\n"
                 "[d3d12-trace] #2 Second ok\r\n")
        for encoding, bom in (("utf-16-le", b"\xff\xfe"), ("utf-16-be", b"\xfe\xff")):
            with self.subTest(encoding=encoding):
                with open(self.d3d12, "wb") as f:
                    f.write(bom + lines.encode(encoding))
                output = io.StringIO()
                errors = io.StringIO()
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                    result = bt.main(["--d3d12", self.d3d12, "--json"])
                self.assertEqual(result, 0, errors.getvalue())
                trace = json.loads(output.getvalue())["d3d12"]
                self.assertEqual(trace["calls"], 2)
                self.assertEqual(trace["methods"], ["First", "Second"])

    def test_utf8_bom_is_stripped(self):
        with open(self.d3d12, "wb") as f:
            f.write(b"\xef\xbb\xbf" + b"[d3d12-trace] #1 Foo ok\n")
        trace = bt.parse_d3d12(self.d3d12)
        self.assertEqual(trace["calls"], 1)
        self.assertEqual(trace["methods"], ["Foo"])

    def test_text_report_has_the_triage_units(self):
        self.write(self.startup, STARTUP)
        text = bt.format_text(bt.summarize(bt.parse_startup(self.startup), None), 10)
        for needle in (
            "pid 123",
            "modules (2)",
            "dllmain: 1 ok, 1 failed",
            "unimplemented: 2 unique",
            "x.dll!Foo from game.exe+0x10",
            "by dll (unique imports): x.dll 1, y.dll 1",
            "exceptions: 2",
            "code 0xc0000005 at game.exe+0x40 (read 0x0)",
            "code 0xe06d7363 at game.exe+0x30 (C++ throw)",
            "stack: game.exe+0x40, game.exe+0x120",
            "exit: 0",
        ):
            self.assertIn(needle, text)

    def test_failed_file_opens_and_skipped_int3_are_listed(self):
        self.write(
            self.startup,
            [
                {"seq": 1, "kind": "trace_open", "pid": 1},
                {"seq": 2, "kind": "file_open_failed", "api": "CreateFileW", "path": "GAME:/probe/missing.bin", "error": 2, "caller": "game.exe+0x20"},
                {"seq": 3, "kind": "int3_skipped", "at": "game.exe+0x1330"},
            ],
        )
        trace = bt.parse_startup(self.startup)
        self.assertEqual(trace["file_open_failed"], [{"path": "GAME:/probe/missing.bin", "error": 2, "caller": "game.exe+0x20"}])
        self.assertEqual(trace["int3_skipped"], ["game.exe+0x1330"])
        text = bt.format_text(bt.summarize(trace, None), 10)
        self.assertIn("failed file opens: 1", text)
        self.assertIn("GAME:/probe/missing.bin error 2 from game.exe+0x20", text)
        self.assertIn("game.exe+0x1330", text)

    def test_last_observed_tls_callback_in_incomplete_startup(self):
        first = {"kind": "tls_callback", "module": "first.dll", "at": "first.dll+0x1100", "reason": 1}
        for fields, reason, phase in (
            ({"reason": 1}, 1, "process attach"),
            ({"reason": 0}, 0, "process detach"),
            ({}, None, "phase unknown"),
        ):
            with self.subTest(phase=phase):
                last = {"kind": "tls_callback", "module": "fixture.exe", "at": "fixture.exe+0x2200", **fields}
                self.write(self.startup, [STARTUP[0], first, last])
                with open(self.startup, "a", encoding="utf-8", newline="\n") as f:
                    f.write('{"kind":"tls_callback","at":"truncated')

                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(bt.main([self.startup, "--json"]), 0)
                summary = json.loads(output.getvalue())
                startup = summary["startup"]
                self.assertEqual(startup["tls_callbacks"], 2)
                self.assertEqual(startup["bad_lines"], 1)
                self.assertIsNone(startup["exit"])
                self.assertEqual(
                    startup["last_tls_callback"],
                    {"module": "fixture.exe", "at": "fixture.exe+0x2200", "reason": reason},
                )
                text = bt.format_text(summary, 10)
                self.assertIn("tls callbacks: 2", text)
                self.assertIn(f"last observed TLS callback: fixture.exe+0x2200 ({phase})", text)


if __name__ == "__main__":
    unittest.main()
