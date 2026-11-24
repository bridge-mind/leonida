"""Generate logging stubs for D3D12 COM interfaces from the Windows SDK's d3d12.h.

For each requested interface, writes a class `<Interface>Stubs` deriving from it that overrides every method of
its chain (except IUnknown's) with a body that reports the call once through rt::d3d12::stub() and returns
E_NOTIMPL, nothing or a value-initialised result. The runtime's COM objects derive from these and override the
methods they implement, so their vtables have exactly the SDK's method order and the compiler rejects a missing
method. Run at build time (runtime/CMakeLists.txt); the output is not checked in.

    python -I tools/gen_d3d12_stubs.py --out <file> [--header <d3d12.h>] Interface [Interface ...]
    python -I tools/gen_d3d12_stubs.py --report <json> --source <cpp>... [--header <d3d12.h>] Interface ...

Without --header, d3d12.h is looked up in the directories of the INCLUDE environment variable (set by vcvars).
The report mode lists each interface's methods and which names are not overridden as `STDMETHODCALLTYPE` in the
given sources; it is a name-based coverage aid for the D3D12 layer, not a compiler check.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_paths import same_report_path  # noqa: E402  one spelling comparison for every report pair

INTERFACE = re.compile(
    r'MIDL_INTERFACE\("[0-9a-fA-F-]+"\)\s*\n\s*(\w+)\s*:\s*public\s+(\w+)\s*\{\s*public:(.*?)\n\s*\};', re.S)
METHOD = re.compile(r'virtual\s+(.*?)\s*STDMETHODCALLTYPE\s+(\w+)\s*\((.*)\)\s*(const)?\s*=\s*0\s*;\s*$', re.S)
COMMENTS_OR_LITERALS = re.compile(r"""("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|/\*.*?\*/|//[^\r\n]*""", re.S)


def find_header():
    for directory in os.environ.get("INCLUDE", "").split(os.pathsep):
        path = os.path.join(directory, "d3d12.h")
        if directory and os.path.isfile(path):
            return path
    return None


def parse_interfaces(text):
    """name -> (base, [lines]) where a line is ('#', text) or ('m', return, name, params, const)."""
    text = COMMENTS_OR_LITERALS.sub(lambda match: match.group(1) or " ", text)
    out = {}
    for match in INTERFACE.finditer(text):
        name, base, body = match.groups()
        items, pending = [], []
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#") and not pending:
                items.append(("#", stripped))
                continue
            pending.append(stripped)
            if re.search(r"=\s*0\s*;\s*$", stripped):
                declaration = " ".join(pending)
                pending = []
                method = METHOD.match(declaration)
                if not method:
                    raise SystemExit(f"{name}: cannot parse {declaration!r}")
                ret, method_name, params, const = method.groups()
                items.append(("m", " ".join(ret.split()), method_name, " ".join(params.split()), const or ""))
        if pending:
            raise SystemExit(f"{name}: unterminated declaration {' '.join(pending)!r}")
        out[name] = (base, items)
    return out


def chain(interfaces, leaf):
    names = []
    seen = set()
    name = leaf
    while name != "IUnknown":
        if name not in interfaces:
            raise SystemExit(f"interface {name} not found in d3d12.h")
        if name in seen:
            raise SystemExit(f"circular inheritance in d3d12.h: {' -> '.join([*names, name])}")
        seen.add(name)
        names.append(name)
        name = interfaces[name][0]
    return list(reversed(names))


def emit_class(interfaces, leaf):
    lines = [f"class {leaf}Stubs : public {leaf} {{", "public:"]
    for name in chain(interfaces, leaf):
        lines.append(f"    // {name}")
        for item in interfaces[name][1]:
            if item[0] == "#":
                lines.append(item[1])
                continue
            _, ret, method, params, const = item
            if ret == "void":
                result = ""
            elif ret == "HRESULT":
                result = " return E_NOTIMPL;"
            else:
                result = " return {};"
            signature = f"{ret} STDMETHODCALLTYPE {method}({params}){' const' if const else ''}"
            lines.append(f'    {signature} override {{ ::rt::d3d12::stub("{name}::{method}");{result} }}')
    lines.append("};")
    return "\n".join(lines)


OVERRIDE = re.compile(r"\bSTDMETHODCALLTYPE\s+(\w+)\s*\(")


def implemented_names(paths):
    """Method names overridden as STDMETHODCALLTYPE in the given sources."""
    names = set()
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            source = COMMENTS_OR_LITERALS.sub(" ", f.read())
            names.update(OVERRIDE.findall(source))
    return names


def coverage(interfaces, leaves, sources):
    """A JSON-serialisable method-coverage report for the requested leaf interfaces."""
    implemented = implemented_names(sources)
    report = []
    for leaf in dict.fromkeys(leaves):
        methods = []
        for name in chain(interfaces, leaf):
            methods.extend((name, item[2]) for item in interfaces[name][1] if item[0] == "m")
        # d3d12.h declares a struct-returning method twice, once in each branch of
        # `#if defined(_MSC_VER) || !defined(_WIN32)` / `#else`; it is one vtable slot, so count it once.
        methods = list(dict.fromkeys(methods))
        open_methods = [f"{owner}::{method}" for owner, method in methods if method not in implemented]
        report.append({
            "interface": leaf,
            "methods": len(methods),
            "implemented": len(methods) - len(open_methods),
            "open": open_methods,
        })
    return report


def write_if_changed(path, text):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        with open(path, encoding="utf-8") as f:
            if f.read() == text:
                return
    except OSError:
        pass
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def same_path(first, second):
    """True when writing first would replace second (same file, or a case/symlink alias of it). The comparison is
    report_paths.same_report_path, which also folds case on a case-insensitive volume: os.path.normcase does that
    only on Windows, and macOS's default APFS volume would let `R.json` and `r.json` pass as two artifacts."""
    return same_report_path(first, second)


def write_report(path, text):
    """write_if_changed with a one-line ERROR instead of a traceback; False when it could not write."""
    try:
        write_if_changed(path, text)
    except OSError as e:
        print(f"ERROR: cannot write {path}: {e}", file=sys.stderr)
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", help="generated stub header (required unless --report is given)")
    parser.add_argument("--report", help="write a JSON method-coverage report here")
    parser.add_argument("--source", action="append", default=[], metavar="CPP",
                        help="source file with overrides, repeatable (for --report)")
    parser.add_argument("--header")
    parser.add_argument("interfaces", nargs="+")
    args = parser.parse_args()
    if not args.out and not args.report:
        parser.error("--out is required unless --report is given")
    if args.report and not args.source:
        parser.error("--report needs at least one --source")
    for path in args.source:
        if not os.path.isfile(path):
            parser.error(f"--source {path}: no such file")
    for path in (args.out, args.report):
        if path and os.path.isdir(path):
            parser.error(f"{path} is a folder, not a file")
    if args.out and args.report and same_path(args.out, args.report):
        parser.error("--out and --report must use different paths")
    header = args.header or find_header()
    if not header:
        raise SystemExit("d3d12.h not found: pass --header or run from a vcvars environment")
    if not os.path.isfile(header):
        parser.error(f"--header {header}: no such file")
    for path in [header] + list(args.source):
        for flag, target in (("--out", args.out), ("--report", args.report)):
            if target and same_path(target, path):
                parser.error(f"{flag} {target} would overwrite the input {path}")
    with open(header, encoding="utf-8", errors="replace") as f:
        interfaces = parse_interfaces(f.read())

    if args.report:
        report = coverage(interfaces, args.interfaces, args.source)
        if not write_report(args.report, json.dumps(report, indent=2) + "\n"):
            return 2
    if not args.out:
        return 0

    parts = [
        "// Generated by tools/gen_d3d12_stubs.py from " + os.path.basename(header) + ". Do not edit.",
        "#pragma once",
        "#include <d3d12.h>",
        "namespace rt::d3d12 { void stub(const char* method); }",
        "#pragma warning(push)",
        "#pragma warning(disable : 4100)  // unreferenced parameters",
    ]
    parts += [emit_class(interfaces, leaf) for leaf in dict.fromkeys(args.interfaces)]
    parts.append("#pragma warning(pop)")
    if not write_report(args.out, "\n".join(parts) + "\n"):
        return 2
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
