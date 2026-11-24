"""Regression checks for the Windows pseudo-handle passed to SetPriorityClass."""

import ctypes
import importlib
import subprocess
import sys
import unittest
from ctypes import wintypes
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

MODULES = ("bvh_index", "manifest_verifier", "navmesh_index", "rpf9_carve")
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000


class WindowsProcessPriorityTest(unittest.TestCase):
    def test_win32_signatures(self):
        for name in MODULES:
            with self.subTest(module=name):
                module = importlib.import_module(name)
                kernel32 = mock.Mock()
                with (
                    mock.patch.object(module, "os", SimpleNamespace(name="nt")),
                    mock.patch.object(ctypes, "windll", SimpleNamespace(kernel32=kernel32), create=True),
                ):
                    module.below_normal()
                self.assertEqual(kernel32.GetCurrentProcess.argtypes, ())
                self.assertIs(kernel32.GetCurrentProcess.restype, wintypes.HANDLE)
                self.assertEqual(kernel32.SetPriorityClass.argtypes, (wintypes.HANDLE, wintypes.DWORD))
                self.assertIs(kernel32.SetPriorityClass.restype, wintypes.BOOL)
                kernel32.SetPriorityClass.assert_called_once_with(
                    kernel32.GetCurrentProcess.return_value, BELOW_NORMAL_PRIORITY_CLASS
                )

    @unittest.skipUnless(sys.platform == "win32", "requires Windows process priority classes")
    def test_child_process_reaches_below_normal_priority(self):
        # Start every child at normal priority so a swallowed error cannot pass the test.
        probe = """
import ctypes
import importlib
import sys
from ctypes import wintypes

sys.path.insert(0, sys.argv[1])
importlib.import_module(sys.argv[2]).below_normal()
kernel32 = ctypes.WinDLL("kernel32")
kernel32.GetCurrentProcess.argtypes = ()
kernel32.GetCurrentProcess.restype = wintypes.HANDLE
kernel32.GetPriorityClass.argtypes = (wintypes.HANDLE,)
kernel32.GetPriorityClass.restype = wintypes.DWORD
print(kernel32.GetPriorityClass(kernel32.GetCurrentProcess()))
"""
        for name in MODULES:
            with self.subTest(module=name):
                result = subprocess.run(
                    [sys.executable, "-I", "-B", "-c", probe, str(TOOLS), name],
                    creationflags=subprocess.NORMAL_PRIORITY_CLASS,
                    capture_output=True, encoding="utf-8", errors="replace", timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), str(BELOW_NORMAL_PRIORITY_CLASS), result.stderr)


if __name__ == "__main__":
    unittest.main()
