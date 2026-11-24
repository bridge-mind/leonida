"""Unit tests for tools/inline_scan.py on hand-written dumpbin-style listings.

The listings are synthetic: the instruction shapes of a checked and an unchecked inline command-list write, written
here line by line, so the test needs no game files, no dumpbin and no third-party packages.

    python -I tests/test_inline_scan.py
"""
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import inline_scan  # noqa: E402


def listing(*rows):
    """Rows are 'mnemonic operands'; '@ADDR mnemonic operands' pins an address (a branch target)."""
    out = []
    for i, row in enumerate(rows):
        mnem, _, ops = row.partition(" ")
        addr = 0x140001000 + 4 * i
        if mnem.startswith("@"):
            addr = int(mnem[1:], 16)
            mnem, _, ops = ops.partition(" ")
        out.append(f"  {addr:016X}: {mnem:<11} {ops}\n")
    return out


def sites(rows):
    return list(inline_scan.scan(listing(*rows)))


# A four-word dispatch with its limit check and the vtable fallback (slot 14 at +0x70) when the buffer is full.
GUARDED = (
    "mov rax,qword ptr [rbx+18h]",
    "cmp rax,qword ptr [rbx+28h]",
    "jae 0000000140002000",
    "mov dword ptr [rax],4002C400h",
    "mov dword ptr [rax+4],ecx",
    "add rax,10h",
    "mov qword ptr [rbx+18h],rax",
    "mov byte ptr [rbx+0000000000000140h],0",
    "jmp 0000000140003000",
    "@0000000140002000 mov rax,qword ptr [rbx]",
    "mov rcx,rbx",
    "call qword ptr [rax+0000000000000070h]",
)
UNGUARDED = tuple(r for r in GUARDED if not r.startswith(("cmp", "jae")))


class InlineScanTests(unittest.TestCase):
    def test_guarded_dispatch_is_checked_with_limit_header_and_fallback_slot(self):
        (site,) = sites(GUARDED)
        self.assertTrue(site["checked"])
        self.assertTrue(site["list"])
        self.assertEqual(site["limit"], 0x28)
        self.assertEqual(site["size"], 0x10)
        self.assertEqual(site["header"], 0x4002C400)
        self.assertEqual(site["fallback_slot"], 14)

    def test_the_same_write_without_its_compare_is_unchecked(self):
        # The only list field (+140h) is written after the store-back.
        (site,) = sites(UNGUARDED)
        self.assertTrue(site["list"])
        self.assertFalse(site["checked"])

    def test_unchecked_with_the_root_header_table_read_before_the_write(self):
        rows = list(UNGUARDED)
        rows.insert(1, "mov edx,dword ptr [rbx+rdx*4+0000000000000344h]")
        (site,) = sites(rows)
        self.assertTrue(site["list"])
        self.assertFalse(site["checked"])

    def test_limit_held_in_a_register_and_cursor_advanced_with_lea(self):
        rows = (
            "mov rcx,qword ptr [rbx+18h]",
            "mov rdx,qword ptr [rbx+30h]",
            "cmp rcx,rdx",
            "jae 0000000140002000",
            "mov eax,dword ptr [rbx+0000000000000344h]",
            "mov dword ptr [rcx],eax",
            "lea rax,[rcx+0Ch]",
            "mov qword ptr [rcx+4],r8",
            "mov qword ptr [rbx+18h],rax",
        )
        (site,) = sites(rows)
        self.assertTrue(site["checked"])
        self.assertEqual(site["limit"], 0x30)
        self.assertEqual(site["size"], 0xC)

    def test_limit_register_loaded_in_another_block_still_counts_as_checked(self):
        rows = (
            "cmp rcx,rdx",
            "jae 0000000140002000",
            "mov eax,dword ptr [rbx+0000000000000344h]",
            "mov dword ptr [rcx],eax",
            "lea rax,[rcx+0Ch]",
            "mov qword ptr [rbx+18h],rax",
        )
        (site,) = sites(rows)
        self.assertTrue(site["checked"])
        self.assertEqual(site["limit"], inline_scan.REG_LIMIT)

    def test_list_copied_to_another_register_is_followed(self):
        rows = (
            "mov rbx,rcx",
            "mov rax,qword ptr [rcx+18h]",
            "cmp rax,qword ptr [rcx+30h]",
            "jae 0000000140002000",
            "mov dword ptr [rax],edx",
            "add rax,8",
            "mov qword ptr [rbx+18h],rax",
        )
        (site,) = sites(rows)
        self.assertTrue(site["checked"])
        self.assertEqual(site["limit"], 0x30)

    def test_stack_frame_slot_is_not_a_list(self):
        rows = (
            "mov qword ptr [rbp+0000000000000140h],rcx",
            "lea rax,[0000000140090000h]",
            "mov qword ptr [rax],rcx",
            "mov qword ptr [rbp+18h],rax",
        )
        self.assertTrue(all(not s["list"] for s in sites(rows)))

    def test_plain_structure_pointer_bump_is_not_a_list(self):
        rows = (
            "mov rax,qword ptr [rdi+18h]",
            "mov qword ptr [rax],rdx",
            "add rax,8",
            "mov qword ptr [rdi+18h],rax",
        )
        (site,) = sites(rows)
        self.assertFalse(site["list"])

    def test_listing_with_byte_column_parses_the_same(self):
        rows = [line.replace(": ", ": 48 8B 43 18        ", 1) for line in listing(*GUARDED)]
        (site,) = list(inline_scan.scan(rows))
        self.assertTrue(site["checked"])
        self.assertEqual(site["fallback_slot"], 14)

    def test_summary_counts_checked_and_unchecked(self):
        text = inline_scan.summarize(sites(GUARDED) + sites(UNGUARDED))
        self.assertIn("checked 1, UNCHECKED 1", text)
        self.assertIn("+0x28 x1", text)

    def test_scan_counts_the_instructions_it_read(self):
        stats = {}
        sites = list(inline_scan.scan(listing(*GUARDED), stats))
        self.assertEqual(stats["instructions"], len(GUARDED))
        self.assertIn(f"instructions scanned: {len(GUARDED)}", inline_scan.summarize(sites, stats["instructions"]))


class InlineScanCliTests(unittest.TestCase):
    """An input with no instructions must not read as a clean image ("UNCHECKED 0")."""

    def run_cli(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "listing.txt")
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(text)
            return subprocess.run([sys.executable, "-I", os.path.join(TOOLS_DIR, "inline_scan.py"), path],
                                  capture_output=True, text=True)

    def test_empty_input_exits_2_without_a_summary(self):
        proc = self.run_cli("")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("no instructions found", proc.stderr)
        self.assertNotIn("UNCHECKED", proc.stdout)

    def test_text_that_is_not_a_listing_exits_2(self):
        proc = self.run_cli("Microsoft (R) COFF/PE Dumper\nnot a disassembly\n")
        self.assertEqual(proc.returncode, 2)
        self.assertNotIn("UNCHECKED", proc.stdout)

    def test_a_listing_with_instructions_but_no_site_still_passes(self):
        proc = self.run_cli("".join(listing("mov rax,rcx", "ret")))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("instructions scanned: 2", proc.stdout)
        self.assertIn("checked 0, UNCHECKED 0", proc.stdout)


if __name__ == "__main__":
    unittest.main()
