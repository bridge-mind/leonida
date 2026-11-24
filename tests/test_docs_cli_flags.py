"""Every `--flag` the docs name exists in the code, and a command line that runs one tool uses that tool's flags.

    python -I tests/test_docs_cli_flags.py

`tests/test_docs_repo_paths.py` checks the paths the docs name; this checks the options. A renamed or removed
argparse option leaves `python -I tools/x.py --old` in a README or guide that no longer runs. Three checks:

1. a `--flag` anywhere in a Markdown page appears as a string in some tracked non-Markdown source file;
2. in a fenced command that starts `python [-I] tools/x.py` (or `scripts/`, or `make_game.py`), every flag appears in
   that script (a script that builds flags as `"--" + name` is skipped; `--help` is argparse's own);
3. in a fenced command running a PowerShell script (`scripts/` or `tools/`), every flag appears in that script's
   `param()` block.

Reads tracked files only; no game files.
"""
import os
import re
import subprocess
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLAG = re.compile(r"(?<![\w-])(--[a-z][\w-]*)")
SOURCE_SUFFIXES = (".py", ".ps1", ".cpp", ".cc", ".c", ".h", ".hpp", ".cmake", ".txt", ".cs", ".cjs", ".js", ".html",
                   ".bat", ".sh", ".yml", ".yaml", ".csproj")
# Flags of other programs the docs quote: gh (AGENTS.md), pip (docs/shader_validation.md), argparse itself.
FOREIGN = {"--search", "--state", "--index-url", "--help"}
# Pages that quote another project's tree (SCARAB, americas-recomp).
OTHER_PROJECTS = {"docs/fssm_crosscheck.md", "docs/d3d12_xs_surface.md"}
COMMAND = re.compile(r"\bpython3?(?: -I)? ((?:tools|scripts)/[\w/]+\.py|make_game\.py)\b")
PS1_COMMAND = re.compile(
    r"(?:powershell(?:\.exe)?\s+.*?-File\s+|(?:\./|\.\\|&|\.)\s*|\b)((?:tools|scripts)[\\/][\w./\\-]+\.ps1)\b",
    re.IGNORECASE
)
PS1_FLAG = re.compile(r"(?<![\w-])(-[a-zA-Z][\w-]*)")


def tracked_files():
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        return [p for p in out.splitlines() if p]
    except (OSError, subprocess.CalledProcessError):
        found = []
        for root, dirs, files in os.walk(ROOT):
            dirs[:] = [d for d in dirs if d not in (".git", "build", "__pycache__", "node_modules")]
            found += [os.path.relpath(os.path.join(root, f), ROOT).replace(os.sep, "/") for f in files]
        return found


def read(path):
    with open(os.path.join(ROOT, path), encoding="utf-8", errors="replace") as f:
        return f.read()


def strip_ps_comments(src):
    """Remove PowerShell block comments (<# ... #>) and line comments (# ...) outside strings."""
    src = re.sub(r"<#.*?#>", "", src, flags=re.DOTALL)
    lines = []
    for line in src.splitlines():
        in_sgl = in_dbl = False
        res = []
        for ch in line:
            if ch == "'" and not in_dbl:
                in_sgl = not in_sgl
            elif ch == '"' and not in_sgl:
                in_dbl = not in_dbl
            elif ch == "#" and not in_sgl and not in_dbl:
                break
            res.append(ch)
        lines.append("".join(res))
    return "\n".join(lines)


def extract_ps1_params(content):
    """Set of lowercase parameter names from the first param(...) block in a PowerShell script."""
    cleaned = strip_ps_comments(content)
    m = re.search(r"\bparam\s*\(", cleaned, re.IGNORECASE)
    if not m:
        return set()
    start = m.end()
    depth = 1
    i = start
    n = len(cleaned)
    in_sgl = in_dbl = False
    while i < n and depth > 0:
        ch = cleaned[i]
        if ch == "'" and not in_dbl:
            in_sgl = not in_sgl
        elif ch == '"' and not in_sgl:
            in_dbl = not in_dbl
        elif not in_sgl and not in_dbl:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
        i += 1
    param_body = cleaned[start:i - 1 if depth == 0 else i]

    parts, cur, nest = [], [], 0
    in_sgl = in_dbl = False
    for ch in param_body:
        if ch == "'" and not in_dbl:
            in_sgl = not in_sgl
        elif ch == '"' and not in_sgl:
            in_dbl = not in_dbl
        elif not in_sgl and not in_dbl:
            if ch in "([{":
                nest += 1
            elif ch in ")]}":
                if nest > 0:
                    nest -= 1
            elif ch == "," and nest == 0:
                parts.append("".join(cur).strip())
                cur = []
                continue
        cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())

    params = set()
    for part in parts:
        decl = part.split("=", 1)[0]
        vars_found = re.findall(r"\$([a-zA-Z_][a-zA-Z0-9_]*)", decl)
        if vars_found:
            params.add(vars_found[-1].lower())
    return params


def unknown_flags(pages, code):
    """[(page:line, flag)] for flags in `pages` ({name: text}) that do not occur in `code` (one string)."""
    out = []
    for page, text in sorted(pages.items()):
        for n, line in enumerate(text.split("\n"), 1):
            for flag in sorted(set(FLAG.findall(line))):
                if flag not in FOREIGN and flag not in code:
                    out.append((f"{page}:{n}", flag))
    return out


def fenced_commands(text):
    """[(line number, one command)] for the fenced code lines, with backslash/backtick continuations joined."""
    out, fence, buf, start = [], False, "", 0
    for n, line in enumerate(text.split("\n"), 1):
        if line.lstrip().startswith("```"):
            fence, buf = not fence, ""
            continue
        if not fence:
            continue
        if not buf:
            start = n
        stripped = line.rstrip()
        if stripped.endswith(("\\", "`")) and not stripped.endswith("``"):
            buf += " " + stripped[:-1]
            continue
        out.append((start, buf + " " + stripped))
        buf = ""
    return out


# These dev.py subcommands pass their arguments through to the producer.
DEV_PRODUCERS = {
    "boot-trace": "tools/boot_trace.py",
    "guest-scene": "scripts/run_guest_scene.py",
}


def tool_flag_mismatches(pages, sources):
    """[(page:line, tool, flag)] for fenced commands running one tool with a flag its source lacks.
    `sources` maps a tool path to its text."""
    out = []
    for page, text in sorted(pages.items()):
        for n, cmd in fenced_commands(text):
            m = COMMAND.search(cmd)
            if not m or m.group(1) not in sources:
                continue
            tool = m.group(1)
            args = cmd[m.end():].split(maxsplit=1)
            if tool == "scripts/dev.py" and args:
                tool = DEV_PRODUCERS.get(args[0], tool)
            source = sources.get(tool, "")
            if '"--" +' in source or "'--' +" in source:
                continue
            for flag in sorted(set(FLAG.findall(cmd))):
                if flag not in FOREIGN and flag not in source:
                    out.append((f"{page}:{n}", tool, flag))
    return out


def ps1_flag_mismatches(pages, sources):
    """[(page:line, tool, flag)] for fenced commands running a .ps1 script with a flag its param() lacks.
    `sources` maps a script path to its text or parameter set."""
    param_cache = {}
    for path, src in sources.items():
        norm = path.replace("\\", "/")
        param_cache[norm] = src if isinstance(src, set) else extract_ps1_params(src)
    out = []
    for page, text in sorted(pages.items()):
        for n, cmd in fenced_commands(text):
            for m in PS1_COMMAND.finditer(cmd):
                tool = m.group(1).replace("\\", "/")
                if tool not in param_cache:
                    continue
                tail = cmd[m.end():].split("#", 1)[0]
                allowed = param_cache[tool]
                for flag in sorted(set(PS1_FLAG.findall(tail))):
                    flag_name = flag.lstrip("-").lower()
                    if flag_name not in allowed:
                        out.append((f"{page}:{n}", tool, flag))
    return out


class DocsCliFlags(unittest.TestCase):
    def test_dev_inherits_only_the_selected_producer_flags(self):
        pages = {"docs/dev.md": ("```\n"
                                 "python -I scripts/dev.py boot-trace --d3d12 --negative-controls\n"
                                 "python -I scripts/dev.py guest-scene --negative-controls --d3d12\n"
                                 "python -I scripts/dev.py discover --json --d3d12\n"
                                 "```\n")}
        sources = {
            "scripts/dev.py": "--json",
            "tools/boot_trace.py": "--d3d12",
            "scripts/run_guest_scene.py": "--negative-controls",
        }
        self.assertEqual(tool_flag_mismatches(pages, sources), [
            ("docs/dev.md:2", "tools/boot_trace.py", "--negative-controls"),
            ("docs/dev.md:3", "scripts/run_guest_scene.py", "--d3d12"),
            ("docs/dev.md:4", "scripts/dev.py", "--d3d12"),
        ])

    def test_checkers_report_only_the_missing_flag(self):
        pages = {"docs/a.md": "Use `--real` or `--gone`, and `gh --search`.\n"
                              "```\npython -I tools/x.py --real --old \\\n    --more\npython -I tools/y.py --old\n```\n"}
        self.assertEqual(unknown_flags(pages, '"--real" "--more" "--old"'), [("docs/a.md:1", "--gone")])
        sources = {"tools/x.py": 'add("--real") add("--more")', "tools/y.py": 'add("--old")'}
        self.assertEqual(tool_flag_mismatches(pages, sources), [("docs/a.md:3", "tools/x.py", "--old")])
        sources["tools/x.py"] += ' "--" + name'
        self.assertEqual(tool_flag_mismatches(pages, sources), [])

        ps1_pages = {"docs/b.md": "```\n.\\scripts\\test.ps1 -ValidParam -BadParam\n```\n"}
        ps1_sources = {"scripts/test.ps1": "param([string]$ValidParam)"}
        self.assertEqual(ps1_flag_mismatches(ps1_pages, ps1_sources), [("docs/b.md:2", "scripts/test.ps1", "-BadParam")])

    def test_docs_name_existing_flags(self):
        files = tracked_files()
        code = "\n".join(read(p) for p in files if p.endswith(SOURCE_SUFFIXES) and not p.startswith("docs/"))
        pages = {p: read(p) for p in files if p.endswith(".md") and p not in OTHER_PROJECTS}
        self.assertTrue(code and pages)
        missing = unknown_flags(pages, code)
        self.assertEqual(missing, [], "flags in the docs that no source file mentions: " +
                         "; ".join(f"{where} {flag}" for where, flag in missing))

    def test_fenced_commands_use_the_flags_of_their_tool(self):
        files = tracked_files()
        sources = {p: read(p) for p in files if p.endswith(".py") and (p.startswith(("tools/", "scripts/")) or p == "make_game.py")}
        pages = {p: read(p) for p in files if p.endswith(".md") and p not in OTHER_PROJECTS}
        bad = tool_flag_mismatches(pages, sources)
        self.assertEqual(bad, [], "fenced commands using a flag their script lacks: " +
                         "; ".join(f"{where} {tool} {flag}" for where, tool, flag in bad))

        ps1_sources = {p: read(p) for p in files if p.endswith(".ps1") and p.startswith(("tools/", "scripts/"))}
        ps1_bad = ps1_flag_mismatches(pages, ps1_sources)
        self.assertEqual(ps1_bad, [], "fenced commands using a flag their PowerShell script lacks: " +
                         "; ".join(f"{where} {tool} {flag}" for where, tool, flag in ps1_bad))


if __name__ == "__main__":
    unittest.main()
