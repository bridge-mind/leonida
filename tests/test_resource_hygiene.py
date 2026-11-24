"""Regression tests ensuring tools close file handles cleanly under -W error::ResourceWarning."""

import ast
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _bare_open_calls(path: pathlib.Path) -> list[tuple[int, str]]:
    """Return (lineno, source_line) for any open(...) call not managed by a with statement."""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))
    managed: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if isinstance(item.context_expr, ast.Call):
                    managed.add(id(item.context_expr))
    lines = src.splitlines()
    bare: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
            if id(node) not in managed:
                bare.append((node.lineno, lines[node.lineno - 1].strip()))
    return bare


class ResourceHygieneTest(unittest.TestCase):
    def test_no_unmanaged_open_calls(self):
        for rel in (
            "tools/rdna2spirv.py",
            "tools/amd_swizzle.py",
            "tools/awc_opus.py",
            "tools/wave64_graphics_scan.py",
            "tools/tex_export.py",
            "tools/tex_mode_probe.py",
            "tools/shader_disasm.py",
            "tools/hashdict.py",
            "tools/d3d12_xs_surface.py",
        ):
            with self.subTest(file=rel):
                bare = _bare_open_calls(ROOT / rel)
                self.assertEqual(bare, [], f"unmanaged open() calls in {rel}: {bare}")

    def test_import_rdna2spirv_no_resource_warning(self):
        r = subprocess.run(
            [
                sys.executable,
                "-W",
                "error::ResourceWarning",
                "-c",
                f"import sys; sys.path.insert(0, {str(ROOT / 'tools')!r}); import rdna2spirv",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
