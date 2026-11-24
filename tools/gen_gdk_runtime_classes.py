"""Build tools/gdk_runtime_classes.json from a clone of xodus-gaming/xgameruntime-docs.

Usage:
    python -I tools/gen_gdk_runtime_classes.py <xgameruntime-docs clone> [--out FILE]

The docs (GPL-3.0) describe the COM classes the GDK runtime's QueryApiImpl hands out: one folder per class under
COM/, a README.md with the class id and the interfaces it implements, and one page per interface with its id, the
interface it inherits from and its "Layout" (the method order after IUnknown's three slots). This keeps the facts
only: ids, inheritance and method names in order. No text is copied.

- A method the docs list as *unknown* stays in the list as null, so the methods after it keep their slot.
- An interface whose page has no "Layout" ("Layout changes unknown, all methods listed under ...", "No known
  methods") gets "methods": null; its own methods, if any, are in its base's list.
- A class can implement more than one interface with its own vtable (XUserImpl: IXUserImpl6 and
  IXUserGamertagImpl); each interface lists its own chain, so each starts at slot 3.

tools/gdk_runtime_calls.py reads the output. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

ID = re.compile(r"(?:Interface|Class) Id: `([0-9a-fA-F-]{36})`")
INHERITS = re.compile(r"Inherits: (?:\[(I\w+)\]|IUnknown\b)")
LINKED_INTERFACE = re.compile(r"\[(I\w+)\]\(\w+\.md\)")
METHOD = re.compile(r"\[(\w+)\]|(\w+)")


def read_utf8(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def parse_layout(text: str) -> list[str | None] | None:
    """The method names of a "Layout:" list in order (None for an *unknown* entry), or None without a list."""
    if "Layout:" not in text:
        return None
    methods: list[str | None] = []
    for line in text.split("Layout:", 1)[1].splitlines():
        line = line.strip()
        if not line.startswith("- "):
            if methods and line:
                break            # the paragraph after the list
            continue
        item = line[2:].strip()
        m = METHOD.match(item)
        methods.append(None if item.startswith("*unknown*") or not m else (m.group(1) or m.group(2)))
    return methods


def parse_docs(root: str) -> tuple[dict, dict]:
    com = os.path.join(root, "COM")
    if not os.path.isdir(com):
        raise FileNotFoundError(f"{root} has no COM folder; is it a clone of xgameruntime-docs?")
    classes, interfaces = {}, {}
    for cls in sorted(os.listdir(com)):
        folder = os.path.join(com, cls)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if not name.endswith(".md"):
                continue
            text = read_utf8(os.path.join(folder, name))
            m = ID.search(text)
            if not m:
                continue
            guid = m.group(1).lower()
            if name == "README.md":
                implements = text.split("Implements:", 1)[1].splitlines()[0] if "Implements:" in text else ""
                classes[cls] = {"id": guid, "implements": LINKED_INTERFACE.findall(implements)}
                continue
            inherits = INHERITS.search(text)
            interfaces[name[:-3]] = {
                "id": guid,
                "class": cls,
                "inherits": inherits.group(1) if inherits and inherits.group(1) else None,
                "methods": parse_layout(text),
            }
    return classes, interfaces


def check(classes: dict, interfaces: dict) -> list[str]:
    """Links the docs make that the table cannot follow."""
    problems = []
    for name, cls in classes.items():
        for iface in cls["implements"]:
            if iface not in interfaces:
                problems.append(f"{name} implements {iface}, which has no page")
    for name, iface in interfaces.items():
        if iface["inherits"] and iface["inherits"] not in interfaces:
            problems.append(f"{name} inherits {iface['inherits']}, which has no page")
    return problems


def revision(root: str) -> str:
    try:
        out = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip() or "unknown revision"
    except (OSError, subprocess.CalledProcessError):
        return "unknown revision"


def build(root: str) -> dict:
    classes, interfaces = parse_docs(root)
    return {
        "source": f"https://github.com/xodus-gaming/xgameruntime-docs at {revision(root)}",
        "licence": "The source is GPL-3.0. Facts only: ids, inheritance and method order; no text is copied.",
        "note": ("The runtime classes QueryApiImpl hands out. An interface's vtable is IUnknown's three slots, then "
                 "the methods of its inheritance chain, base first. A null method is one the docs list as unknown; "
                 "null 'methods' means the docs give no layout for that interface (its methods, if any, are listed "
                 "under its base). Made by tools/gen_gdk_runtime_classes.py."),
        "classes": classes,
        "interfaces": interfaces,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("docs", help="a clone of xodus-gaming/xgameruntime-docs")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "gdk_runtime_classes.json"))
    a = ap.parse_args(argv)
    try:
        table = build(a.docs)
    except OSError as e:
        print(f"cannot read the docs: {e}", file=sys.stderr)
        return 1
    problems = check(table["classes"], table["interfaces"])
    for p in problems:
        print(f"warning: {p}", file=sys.stderr)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(table, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"{len(table['classes'])} classes, {len(table['interfaces'])} interfaces -> {a.out}")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
