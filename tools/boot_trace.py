"""Summarize `gta6rt`'s two first-boot traces: the startup JSONL and the `[d3d12-trace]` lines.

The loader writes the startup trace with `gta6rt --trace FILE` (one JSON object per line, flushed per event so it
survives a crash; `runtime/README.md`, "Startup trace"), and the D3D12 proxy writes one `[d3d12-trace] #<seq>
<method> <outcome>` line to stderr with `--trace-d3d12`. This tool turns both into the small report a first-boot
triage needs: modules, unimplemented calls and their callers, DllMain/TLS results, exceptions and the exit, then
the D3D12 calls by outcome and the first call per method.

The method field can contain spaces (`SetMarker/BeginEvent with unknown metadata`), so the parser takes the
outcome as the last field instead of assuming a single-word method. PowerShell 5.1 wraps the first stderr line of
a native program in an error record (`program.exe : [d3d12-trace] #1 ...`), so a `<name> : ` prefix before the
tag is accepted too. The reader accepts UTF-8 and BOM-marked UTF-16, including PowerShell 5.1 stderr redirection.

Usage:
    python -I boot_trace.py boot.jsonl [--d3d12 d3d12.log] [--json] [--limit N]

The traces can quote what the game prints, so they stay on the player's machine: the text report counts debug
strings instead of printing them, and only the tests' synthetic fixtures go into the repository. Share counts and
error classes from a real run, not its trace (AGENTS.md, "No game data").

Standard library only; reads files, never runs them. A truncated last line (the expected shape after a crash) is
counted, not fatal; a file with no valid event is an error (exit 2). When one of two requested traces fails to
parse, the report still includes the other trace.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

D3D12_LINE = re.compile(r"^(?:.+ : )?\[d3d12-trace\] #(\d+) (.+) (\S+)\s*$")


class TraceError(ValueError):
    """The file is unreadable or holds no valid trace event."""


def _open_text(path: str):
    """Read UTF-8 or use the byte-order mark to decode a UTF-16 trace."""
    try:
        with open(path, "rb") as raw:
            head = raw.read(2)
        encoding = "utf-16" if head in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
        return open(path, encoding=encoding, errors="replace")
    except OSError as e:
        raise TraceError(f"cannot read {path}: {e.strerror or e}") from e


def parse_startup(path: str, limit: int = 10) -> dict:
    """Read the `--trace` JSONL into the units the first-boot triage uses."""
    trace = {
        "path": path,
        "pid": None,
        "events": 0,
        "bad_lines": 0,
        "bad_line_numbers": [],
        "kinds": {},
        "modules": [],
        "tls_callbacks": 0,
        "dllmain": {"ok": 0, "failed": 0, "failed_modules": []},
        "unimplemented": [],
        "unimplemented_by_dll": {},
        "debug_strings": 0,
        "file_open_failed": [],
        "file_open_failed_count": 0,
        "int3_skipped": [],
        "exceptions": [],
        "exception_count": 0,
        "exit": None,
        "error": None,
    }
    seen_imports = set()
    with _open_text(path) as fh:
        for number, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                if isinstance(event, dict) and event.get("kind") == "exit" and type(event.get("code")) is not int:
                    event = None
            except json.JSONDecodeError:
                trace["bad_lines"] += 1
                if len(trace["bad_line_numbers"]) < limit:
                    trace["bad_line_numbers"].append(number)
                continue
            if not isinstance(event, dict) or "kind" not in event:
                trace["bad_lines"] += 1
                if len(trace["bad_line_numbers"]) < limit:
                    trace["bad_line_numbers"].append(number)
                continue
            trace["events"] += 1
            kind = str(event.get("kind"))
            trace["kinds"][kind] = trace["kinds"].get(kind, 0) + 1
            if kind == "trace_open":
                trace["pid"] = event.get("pid")
            elif kind == "map":
                trace["modules"].append(
                    {
                        "module": event.get("module"),
                        "base": event.get("base"),
                        "size": event.get("size"),
                        "relocated": event.get("relocated"),
                    }
                )
            elif kind == "tls_callback":
                trace["tls_callbacks"] += 1
                trace["last_tls_callback"] = {
                    "module": event.get("module"),
                    "at": event.get("at"),
                    "reason": event.get("reason"),
                }
            elif kind == "dllmain":
                if event.get("result"):
                    trace["dllmain"]["ok"] += 1
                else:
                    trace["dllmain"]["failed"] += 1
                    if len(trace["dllmain"]["failed_modules"]) < limit:
                        trace["dllmain"]["failed_modules"].append(event.get("module"))
            elif kind == "unimplemented_call":
                name = str(event.get("import", "?"))
                if name not in seen_imports:
                    seen_imports.add(name)
                    trace["unimplemented"].append(
                        {"import": name, "caller": event.get("caller")}
                    )
                    dll = name.split("!", 1)[0]
                    trace["unimplemented_by_dll"][dll] = (
                        trace["unimplemented_by_dll"].get(dll, 0) + 1
                    )
            elif kind == "debug_string":
                trace["debug_strings"] += 1
            elif kind == "file_open_failed":
                trace["file_open_failed_count"] += 1
                if len(trace["file_open_failed"]) < limit:
                    trace["file_open_failed"].append(
                        {
                            "path": event.get("path"),
                            "error": event.get("error"),
                            "caller": event.get("caller"),
                        }
                    )
            elif kind == "int3_skipped":
                trace["int3_skipped"].append(event.get("at"))
            elif kind == "exception":
                trace["exception_count"] += 1
                if limit <= 0:
                    continue
                stack = event.get("stack")
                detail = {
                    "code": event.get("code"),
                    "at": event.get("at"),
                    "cpp": event.get("cpp"),
                    "access": event.get("access"),
                    "address": event.get("address"),
                    "stack": stack[:8] if isinstance(stack, list) else [],
                }
                if len(trace["exceptions"]) < limit:
                    trace["exceptions"].append(detail)
                else:
                    trace["exceptions"][-1] = detail
            elif kind == "exit":
                trace["exit"] = event.get("code")
            elif kind == "error":
                trace["error"] = event.get("what")
    if trace["events"] == 0:
        raise TraceError(f"{path} has no valid startup event")
    return trace


def parse_d3d12(path: str, limit: int = 10) -> dict:
    """Read the `--trace-d3d12` stderr lines, ignoring the runtime's ordinary diagnostics."""
    trace = {
        "path": path,
        "calls": 0,
        "ignored_lines": 0,
        "outcomes": {},
        "methods": [],
        "first": {},
        "failed": [],
        "unsupported": [],
        "refused": [],
        "partial": [],
        "last_seq": 0,
        "sequence_break_count": 0,
        "sequence_breaks": [],
    }
    with _open_text(path) as fh:
        for line in fh:
            match = D3D12_LINE.match(line.rstrip("\n"))
            if not match:
                if line.strip():
                    trace["ignored_lines"] += 1
                continue
            seq, method, outcome = int(match.group(1)), match.group(2), match.group(3)
            trace["calls"] += 1
            expected = trace["last_seq"] + 1
            if seq != expected:
                trace["sequence_break_count"] += 1
                if len(trace["sequence_breaks"]) < limit:
                    trace["sequence_breaks"].append({"expected": expected, "observed": seq})
            trace["last_seq"] = seq
            trace["outcomes"][outcome] = trace["outcomes"].get(outcome, 0) + 1
            if method not in trace["first"]:
                trace["methods"].append(method)
                trace["first"][method] = {"seq": seq, "outcome": outcome}
            if outcome in ("failed", "unsupported", "refused", "partial"):
                bucket = trace[outcome]
                if len(bucket) < limit:
                    bucket.append({"seq": seq, "method": method})
    if trace["calls"] == 0:
        raise TraceError(
            f"{path} has no [d3d12-trace] line among {trace['ignored_lines']} line(s); "
            "is it the stderr log of a --trace-d3d12 run, with the Vulkan device enabled?"
        )
    return trace


def summarize(startup: dict | None, d3d12: dict | None) -> dict:
    return {"startup": startup, "d3d12": d3d12}


def _tail(items, limit: int) -> str:
    shown = ", ".join(str(x) for x in items[:limit])
    return shown + (f", ... {len(items) - limit} more" if len(items) > limit else "")


def format_text(summary: dict, limit: int) -> str:
    out = []
    startup = summary.get("startup")
    if startup:
        out.append(f"startup trace: {startup['path']}")
        if startup["events"]:
            out.append(
                f"  pid {startup['pid']}, {startup['events']} event(s), {startup['bad_lines']} bad line(s)"
            )
            kinds = ", ".join(f"{k} {v}" for k, v in sorted(startup["kinds"].items()))
            out.append(f"  kinds: {kinds}")
        if startup["modules"]:
            names = [m["module"] for m in startup["modules"]]
            out.append(f"  modules ({len(names)}): {_tail(names, limit)}")
        out.append(f"  tls callbacks: {startup['tls_callbacks']}")
        if callback := startup.get("last_tls_callback"):
            reason = callback["reason"]
            phase = "phase unknown"
            if type(reason) is int:
                phase = {0: "process detach", 1: "process attach"}.get(reason, phase)
            at = callback["at"] or "address unknown"
            out.append(f"    last observed TLS callback: {at} ({phase})")
        dm = startup["dllmain"]
        line = f"  dllmain: {dm['ok']} ok, {dm['failed']} failed"
        if dm["failed_modules"]:
            line += f" ({_tail(dm['failed_modules'], limit)})"
        out.append(line)
        if startup["unimplemented"]:
            out.append(f"  unimplemented: {len(startup['unimplemented'])} unique")
            for item in startup["unimplemented"][:limit]:
                out.append(f"    {item['import']} from {item['caller']}")
            if len(startup["unimplemented"]) > limit:
                out.append(f"    ... {len(startup['unimplemented']) - limit} more")
            by_dll = ", ".join(
                f"{dll} {count}"
                for dll, count in sorted(startup["unimplemented_by_dll"].items())
            )
            out.append(f"  by dll (unique imports): {by_dll}")
        else:
            out.append("  unimplemented: none")
        if startup["debug_strings"]:
            out.append(f"  debug strings: {startup['debug_strings']} (text not shown)")
        if startup["file_open_failed_count"]:
            out.append(f"  failed file opens: {startup['file_open_failed_count']}")
            for item in startup["file_open_failed"][:limit]:
                out.append(f"    {item['path']} error {item['error']} from {item['caller']}")
            if startup["file_open_failed_count"] > limit:
                out.append(f"    ... {startup['file_open_failed_count'] - limit} more")
        if startup["int3_skipped"]:
            out.append(f"  int3 skipped (LEONIDA_SKIP_INT3, one per address): {_tail(startup['int3_skipped'], limit)}")
        if startup["exception_count"]:
            out.append(f"  exceptions: {startup['exception_count']}")
            for exc in startup["exceptions"][:limit]:
                cpp = exc.get("cpp")
                detail = " (C++ throw)" if cpp is True else f" ({cpp})" if cpp else ""
                if exc.get("access"):
                    detail += f" ({exc['access']}"
                    if exc.get("address"):
                        detail += f" {exc['address']}"
                    detail += ")"
                out.append(f"    code {exc['code']} at {exc['at']}{detail}")
                if exc.get("stack"):
                    out.append(f"      stack: {_tail(exc['stack'], limit)}")
        if startup["error"]:
            out.append(f"  error: {startup['error']}")
        if startup["exit"] is not None:
            out.append(f"  exit: {startup['exit']}")
    d3d12 = summary.get("d3d12")
    if d3d12:
        out.append(f"d3d12 trace: {d3d12['path']}")
        outcomes = ", ".join(f"{k} {v}" for k, v in sorted(d3d12["outcomes"].items()))
        out.append(
            f"  {d3d12['calls']} call(s), {d3d12['ignored_lines']} other line(s)"
            " (the [rt] diagnostics are not summarized)"
            + (f": {outcomes}" if outcomes else "")
        )
        if d3d12["sequence_break_count"]:
            out.append(f"  sequence breaks: {d3d12['sequence_break_count']}")
            for item in d3d12["sequence_breaks"][:limit]:
                out.append(f"    expected #{item['expected']}, observed #{item['observed']}")
        if d3d12["methods"]:
            out.append(f"  first per method: {_tail(d3d12['methods'], limit)}")
        if d3d12["unsupported"]:
            unsupported = [f"#{c['seq']} {c['method']}" for c in d3d12["unsupported"]]
            out.append(f"  unsupported: {_tail(unsupported, limit)}")
        if d3d12["failed"]:
            failed = [f"#{c['seq']} {c['method']}" for c in d3d12["failed"]]
            out.append(f"  failed: {_tail(failed, limit)}")
        if d3d12["refused"]:
            refused = [f"#{c['seq']} {c['method']}" for c in d3d12["refused"]]
            out.append(f"  refused: {_tail(refused, limit)}")
        if d3d12.get("partial"):
            partial = [f"#{c['seq']} {c['method']}" for c in d3d12["partial"]]
            out.append(f"  partial: {_tail(partial, limit)}")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Summarize gta6rt's startup and D3D12 first-boot traces."
    )
    ap.add_argument("startup", nargs="?", help="the JSONL written by gta6rt --trace")
    ap.add_argument("--d3d12", help="the stderr log written with --trace-d3d12")
    ap.add_argument("--json", action="store_true", help="print the summary as JSON")
    ap.add_argument(
        "--limit", type=int, default=10, help="items listed per section (default 10)"
    )
    a = ap.parse_args(argv)
    if not a.startup and not a.d3d12:
        print("error: give the startup trace, --d3d12, or both", file=sys.stderr)
        return 2
    if a.limit < 1:
        print("error: --limit must be at least 1", file=sys.stderr)
        return 2
    summary = summarize(None, None)
    exit_code = 0
    for section, path, parse in (("startup", a.startup, parse_startup), ("d3d12", a.d3d12, parse_d3d12)):
        if path:
            try:
                summary[section] = parse(path, a.limit)
            except TraceError as e:
                print(f"error: {e}", file=sys.stderr)
                exit_code = 2
    if not any(summary.values()):
        return 2
    if a.json:
        print(json.dumps(summary, indent=1))
    else:
        print(format_text(summary, a.limit))
    return exit_code


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
