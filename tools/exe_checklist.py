"""The things to confirm against the game executable: every ASSUMPTION, UNVERIFIED and VERIFY note, in one list.

    python -I tools/exe_checklist.py [--gdk-calls REPORT.json]... [--out checklist.md] [--root DIR]

The runtime and the docs mark what was written without the executable: `// ASSUMPTION ...` or `// ... UNVERIFIED`
comments in runtime/src and runtime/shims (C++), and `**VERIFY:**`, `**ASSUMPTION:**` or `**UNVERIFIED:**` notes in
docs/*.md, in any case (the prose spelling `**Assumption**`, `**Unverified:**` counts too; a bold term followed by
a dash defines it and is not a note, and a marker that does not start its line or that sits in a table row is prose,
not a note). This collects them with their file and line, the whole comment block or list item each starts, so the
first look at the executable (scripts/first_look.ps1) has one list to work through.

--gdk-calls takes gdk_runtime_calls.py's JSON for the executable (its --out file or its --json output, one image or
several). A shim note then says whether the executable calls a runtime class that file serves: "called" items are
the ones a boot can reach, so they come first; "not called" ones can wait. The classes of a file are its
`constexpr GUID k...Class` ids, as gdk_runtime_calls reads them.

Output: Markdown, to stdout or --out (LF). Exit status 0 on success; 1 when a --gdk-calls file can't be read or
the checklist cannot be written to --out; 2 when --out refers to an input report.
Standard library only.
Usage: `python -I tools/exe_checklist.py --help` lists the options.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gdk_runtime_calls import served_classes  # noqa: E402
from report_paths import same_report_path  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.normpath(os.path.join(HERE, ".."))
SOURCE_DIRS = ("runtime/src", "runtime/shims")
SOURCE_EXTS = (".cpp", ".h", ".hpp", ".inl")
CODE_MARK = re.compile(r"\b(ASSUMPTION|UNVERIFIED|VERIFY)\b")
# The docs spell the marker in caps (`**VERIFY:**`) and in prose (`**Assumption**`, `**Unverified:**`, `**Assumptions
# to confirm:**`); all of them are notes.
DOC_MARK = re.compile(r"\*\*(VERIFY|ASSUMPTION|UNVERIFIED)S?\b[^*]*:?\*\*", re.IGNORECASE)
# A bold marker followed by a dash defines the term (`- **Assumption** — a design choice ...`), it is not a note.
DEFINITION = re.compile(r"[\u2014\u2013-]\s")
# A note is what a list item starts, so the marker after this prefix is at the item's start.
LIST_ITEM = re.compile(r"^(?:[-*+]|\d+[.)])\s+")


def comment_text(line: str) -> str | None:
    """The text of a whole-line `//` comment, or None (code, or code with a trailing comment)."""
    stripped = line.lstrip()
    return stripped[2:].strip() if stripped.startswith("//") else None


def without_marker(text: str) -> str:
    """The note without its leading marker word ("ASSUMPTION: x" -> "x"), which the list already shows."""
    return re.sub(r"^\**\s*(ASSUMPTION|UNVERIFIED|VERIFY)S?\b[^:]*:\**\s*", "", text, count=1, flags=re.IGNORECASE)


def code_notes(path: str, rel: str) -> list[dict]:
    """Each whole-line comment run holding a marker, as one note from the run's first line to its last. Trailing
    comments (`x = 18;  // UNVERIFIED, see the header`) point at such a run and are not notes of their own."""
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().split("\n")
    notes, i = [], 0
    while i < len(lines):
        if comment_text(lines[i]) is None:
            i += 1
            continue
        start = i
        parts = []
        while i < len(lines) and comment_text(lines[i]) is not None:
            parts.append(comment_text(lines[i]))
            i += 1
        # A run can hold several notes, and a note starts at its marker (a file's header describes the file first):
        # each one runs from its marker to the next marker or the end of the run.
        marks = [k for k, p in enumerate(parts) if CODE_MARK.search(p)]
        for n, at in enumerate(marks):
            stop = marks[n + 1] if n + 1 < len(marks) else len(parts)
            first = parts[at][CODE_MARK.search(parts[at]).start():]
            text = " ".join(p for p in [first] + parts[at + 1:stop] if p)
            notes.append({"file": rel, "line": start + at + 1, "kind": CODE_MARK.search(first).group(1),
                          "text": without_marker(text)})
    return notes


def doc_notes(path: str, rel: str) -> list[dict]:
    """Each note whose bold marker starts its line (after the indentation and an optional list marker), from the line
    it starts on to the next blank line or item. A marker further into a sentence (`... work (**Assumption**);`) or a
    table row (`| ... |`) is prose: the note does not start there, so it is not a note."""
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().split("\n")
    notes = []
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("|"):
            continue
        body = LIST_ITEM.sub("", stripped)
        m = DOC_MARK.match(body)
        if not m:
            continue
        if DEFINITION.match(body[m.end():].lstrip()):
            continue
        end = i + 1
        while end < len(lines) and lines[end].strip() and not lines[end].lstrip().startswith(("- ", "* ", "#", "|")):
            end += 1
        text = " ".join(re.sub(r"^[-*] ", "", l.strip()) for l in lines[i:end])
        notes.append({"file": rel, "line": i + 1, "kind": m.group(1).upper(), "text": without_marker(text)})
    return notes


def collect(root: str) -> list[dict]:
    notes = []
    for d in SOURCE_DIRS:
        base = os.path.join(root, d)
        for dirpath, _dirs, files in sorted(os.walk(base)):
            for name in sorted(files):
                if name.endswith(SOURCE_EXTS):
                    path = os.path.join(dirpath, name)
                    rel = os.path.relpath(path, root).replace(os.sep, "/")
                    found = code_notes(path, rel)
                    if not found:
                        continue
                    # The classes belong to the file, so parse them once and give every note from it the same list.
                    with open(path, encoding="utf-8", errors="replace") as f:
                        classes = sorted(served_classes(f.read()))
                    for n in found:
                        n["classes"] = classes
                    notes += found
    docs = os.path.join(root, "docs")
    if os.path.isdir(docs):
        for name in sorted(os.listdir(docs)):
            if name.endswith(".md"):
                notes += doc_notes(os.path.join(docs, name), f"docs/{name}")
    return notes


def called_classes(paths: list[str]) -> set[str]:
    """The class ids the executable's gdk_runtime_calls reports list (any image in them)."""
    ids = set()
    for p in paths:
        with open(p, encoding="utf-8") as f:
            doc = json.load(f)
        for image in doc.get("images", [doc]):
            for c in image.get("classes", []):
                if c.get("class_id"):
                    ids.add(str(c["class_id"]).lower())
    return ids


def render(notes: list[dict], called: set[str] | None) -> str:
    def status(n):
        if called is None or not n.get("classes"):
            return None
        return any(c in called for c in n["classes"])

    out = ["# Checklist: what to confirm against the executable", ""]
    out.append(f"{len(notes)} notes from the runtime sources and the docs (tools/exe_checklist.py).")
    if called is not None:
        out.append("Runtime notes are marked by whether the executable calls a class their file serves "
                   "(gdk_runtime_calls); the called ones come first.")
    out.append("")
    groups = [("Runtime", [n for n in notes if not n["file"].startswith("docs/")]),
              ("Docs", [n for n in notes if n["file"].startswith("docs/")])]
    for title, group in groups:
        if not group:
            continue
        out.append(f"## {title}")
        out.append("")
        order = {True: 0, None: 1, False: 2}
        for n in sorted(group, key=lambda n: (order[status(n)], n["file"], n["line"])):
            s = status(n)
            tag = "" if s is None else (" **called**" if s else " (not called)")
            out.append(f"- [ ] `{n['file']}:{n['line']}` {n['kind']}{tag}: {n['text']}")
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--gdk-calls", action="append", default=[], help="gdk_runtime_calls.py JSON for the executable")
    ap.add_argument("--out", help="write the checklist here instead of stdout")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="the repository (default: %(default)s)")
    a = ap.parse_args(argv)
    if not os.path.isdir(a.root):
        print(f"error: --root is not a directory: {a.root}", file=sys.stderr)
        return 2
    if a.out and os.path.isdir(a.out):
        print(f"error: --out path is a directory: {a.out}", file=sys.stderr)
        return 2
    if a.out:
        for report in a.gdk_calls:
            if same_report_path(a.out, report):
                ap.error(f"--out {a.out} would overwrite the --gdk-calls report {report}")
    called = None
    if a.gdk_calls:
        try:
            called = called_classes(a.gdk_calls)
        except (OSError, ValueError, AttributeError) as e:
            print(f"cannot read the gdk_runtime_calls report: {e}", file=sys.stderr)
            return 1
    text = render(collect(a.root), called)
    if a.out:
        try:
            with open(a.out, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
        except OSError as e:
            print(f"cannot write the checklist to {a.out}: {e.strerror or e}", file=sys.stderr)
            return 1
        print(f"wrote {a.out}")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
