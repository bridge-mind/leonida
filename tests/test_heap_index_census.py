"""tools/heap_index_census.py on hand-written listings and containers (no game files, no GPU)."""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import heap_index_census as hic  # noqa: E402

SFI0_RES = 1 << 25
SFI0_SMP = 1 << 26


def enc(n):
    """A placeholder `; encoding:` of n bytes: the census only needs the length (for code offsets)."""
    return "; encoding: [" + ",".join(["0x00"] * n) + "]"


def listing(name, lines):
    """lines: (text, bytes) pairs -> a tab-indented listing as tools/shader_disasm.py writes it."""
    end = 0x100 + sum(n for _, n in lines)
    body = "".join(f"\t{text:<40}{enc(n)}\n" for text, n in lines)
    return f"; {name}.fssm code 0x100..0x{end:x}\n{body}"


def dxbc(flags):
    """A DXBC container holding only an SFI0 part with the given feature flags."""
    sfi0 = b"SFI0" + struct.pack("<I", 8) + struct.pack("<Q", flags)
    head = b"DXBC" + bytes(16) + struct.pack("<II", 1, 36 + len(sfi0)) + struct.pack("<II", 1, 36)
    return head + sfi0


def fssm(rsrc2):
    return bytes(0x138) + struct.pack("<I", rsrc2) + bytes(12)


class Work:
    """A temporary <work>/shaders/dxbc folder: carve index, containers, FSSM headers and disasm listings."""

    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.disasm = os.path.join(self.root, "disasm")
        os.makedirs(self.disasm)
        os.makedirs(os.path.join(self.root, "fssm"))
        self.index = []

    def add(self, name, stage, flags, lines, rsrc2=None):
        self.index.append({"file": name + ".dxbc", "stage": stage})
        with open(os.path.join(self.root, name + ".dxbc"), "wb") as f:
            f.write(dxbc(flags))
        with open(os.path.join(self.disasm, name + ".s"), "w", encoding="utf-8", newline="\n") as f:
            f.write(listing(name, lines))
        if rsrc2 is not None:
            with open(os.path.join(self.root, "fssm", name + ".fssm"), "wb") as f:
                f.write(fssm(rsrc2))
        with open(os.path.join(self.root, "index.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(self.index, f)

    def close(self):
        self.tmp.cleanup()


# A heap-indexing PS: the resource heap is a 32-bit pointer in s8 (s9 forced to 0), a T# at a static offset;
# the sampler heap is a 32-bit pointer in s11 indexed by s5 (dynamic offset).
PS_HEAP = [
    ("s_mov_b32 s9, 0", 4),
    ("s_load_dwordx8 s[20:27], s[8:9], 0x1660", 8),
    ("s_mov_b32 s12, 0", 4),
    ("s_load_dwordx4 s[28:31], s[11:12], s5", 8),
    ("s_waitcnt lgkmcnt(0)", 4),
    ("image_sample v[0:3], v[4:5], s[20:27], s[28:31] dmask:0xf dim:SQ_RSRC_IMG_2D", 8),
    ("s_endpgm", 4),
]


class HeapIndexCensus(unittest.TestCase):
    def setUp(self):
        self.work = Work()

    def tearDown(self):
        self.work.close()

    def test_user_ptr32_heap_bases_and_roles(self):
        self.work.add("ps_heap", "ps", SFI0_RES | SFI0_SMP, PS_HEAP)
        out = hic.census(self.work.disasm)
        g = out["groups"]["ps sfi0=res+smp"]
        self.assertEqual(g["shaders"], 1)
        self.assertEqual(g["shaders_per_base"], {"user s8": 1, "user s11": 1})
        rows = {(r["base"], r["ptr"], r["role"], r["offset"]): r["count"] for r in g["loads"]}
        self.assertEqual(rows, {("user s8", "ptr32", "T#", "static"): 1,
                                ("user s11", "ptr32", "S#", "dynamic"): 1})
        self.assertEqual(g["static_offsets"]["T#"], {"aligned32": 1, "max": 0x1660})

    def test_compute_sgpr_past_user_count_is_system(self):
        # COMPUTE_PGM_RSRC2 USER_SGPR = 2: s0..s1 are user data, s2 is the first system SGPR.
        self.work.add("cs_sys", "cs", SFI0_RES, [
            ("s_load_dwordx8 s[20:27], s[2:3], 0x40", 8),
            ("s_load_dwordx4 s[28:31], s[0:1], 0x0", 8),
            ("s_endpgm", 4),
        ], rsrc2=2 << 1)
        g = hic.census(self.work.disasm)["groups"]["cs sfi0=res"]
        bases = {(r["base"], r["ptr"]) for r in g["loads"]}
        self.assertEqual(bases, {("system s[2:3]", "ptr64"), ("user s[0:1]", "ptr64")})

    def test_pointer_loaded_from_memory(self):
        self.work.add("ps_ld", "ps", 0, [
            ("s_mov_b32 s9, 0", 4),
            ("s_load_dwordx2 s[0:1], s[8:9], 0x10", 8),
            ("s_waitcnt lgkmcnt(0)", 4),
            ("s_load_dwordx8 s[20:27], s[0:1], 0x0", 8),
            ("s_endpgm", 4),
        ])
        g = hic.census(self.work.disasm)["groups"]["ps sfi0=none"]
        self.assertEqual(g["shaders_per_base"], {"loaded pointer": 1})
        self.assertEqual(g["loads"][0]["role"], "unused")

    def test_paths_that_disagree_drop_the_high_word(self):
        # pc 0x100 cmp, 0x104 cbranch to 0x110; 0x108 s9 = 0, 0x10c branch to 0x114; 0x110 s9 = s10; 0x114 load.
        self.work.add("ps_merge", "ps", SFI0_RES, [
            ("s_cmp_eq_u32 s5, 0", 4),
            ("s_cbranch_scc0 2", 4),
            ("s_mov_b32 s9, 0", 4),
            ("s_branch 1", 4),
            ("s_mov_b32 s9, s10", 4),
            ("s_load_dwordx8 s[20:27], s[8:9], 0x20", 8),
            ("s_endpgm", 4),
        ])
        g = hic.census(self.work.disasm)["groups"]["ps sfi0=res"]
        self.assertEqual([(r["base"], r["ptr"]) for r in g["loads"]], [("user s8", "ptr64_lo_only")])

    def test_same_paths_keep_the_ptr32(self):
        self.work.add("ps_same", "ps", SFI0_RES, [
            ("s_cmp_eq_u32 s5, 0", 4),
            ("s_cbranch_scc0 2", 4),
            ("s_mov_b32 s9, 0", 4),
            ("s_branch 1", 4),
            ("s_mov_b32 s9, 0", 4),
            ("s_load_dwordx8 s[20:27], s[8:9], 0x20", 8),
            ("s_endpgm", 4),
        ])
        g = hic.census(self.work.disasm)["groups"]["ps sfi0=res"]
        self.assertEqual([(r["base"], r["ptr"]) for r in g["loads"]], [("user s8", "ptr32")])

    def test_branch_past_the_last_instruction_is_skipped(self):
        # 0x104 s_cbranch_scc0 0 targets 0x108 = the end of the code: counted, not a KeyError.
        self.work.add("ps_end", "ps", SFI0_RES, [
            ("s_cmp_eq_u32 s5, 0", 4),
            ("s_cbranch_scc0 0", 4),
        ])
        out = hic.census(self.work.disasm)
        self.assertEqual(out["skipped"], {"branch target outside code": 1})

    def test_embedded_data_is_counted_as_skipped(self):
        self.work.add("ps_heap", "ps", SFI0_RES, PS_HEAP)
        path = os.path.join(self.work.disasm, "ps_heap.s")
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            f.write("\t; invalid instruction encoding\n")
        out = hic.census(self.work.disasm)
        self.assertEqual(out["skipped"], {"embedded data words": 1})
        self.assertEqual(out["groups"], {})

    def test_cli_json_and_missing_folder(self):
        self.work.add("ps_heap", "ps", SFI0_RES | SFI0_SMP, PS_HEAP)
        js = os.path.join(self.work.root, "out", "census.json")
        p = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", "heap_index_census.py"),
                            self.work.disasm, "--json", js], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("ps sfi0=res+smp: 1 shaders", p.stdout)
        with open(js, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["shaders_read"], 1)
        p = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", "heap_index_census.py"),
                            os.path.join(self.work.root, "nope")], capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertTrue(p.stderr.startswith("ERROR:"), p.stderr)

    def test_cli_json_over_the_carve_index_is_refused(self):
        self.work.add("ps_heap", "ps", SFI0_RES | SFI0_SMP, PS_HEAP)
        index = os.path.join(self.work.root, "index.json")
        with open(index, "rb") as f:
            before = f.read()
        p = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", "heap_index_census.py"),
                            self.work.disasm, "--json", index], capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn("--json would overwrite the carve index", p.stderr)
        with open(index, "rb") as f:
            self.assertEqual(f.read(), before)

    def test_cli_json_inside_the_disasm_folder_is_refused(self):
        self.work.add("ps_heap", "ps", SFI0_RES | SFI0_SMP, PS_HEAP)
        listing_path = os.path.join(self.work.disasm, "ps_heap.s")
        with open(listing_path, "rb") as f:
            before = f.read()
        tool = os.path.join(ROOT, "tools", "heap_index_census.py")
        p = subprocess.run([sys.executable, "-I", tool, self.work.disasm, "--json", listing_path],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn("must not write inside the disasm folder", p.stderr)
        with open(listing_path, "rb") as f:
            self.assertEqual(f.read(), before)
        # A name that does not exist yet is refused too: the folder itself is the input.
        p = subprocess.run([sys.executable, "-I", tool, self.work.disasm, "--json",
                            os.path.join(self.work.disasm, "report.json")], capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertFalse(os.path.exists(os.path.join(self.work.disasm, "report.json")))

    def test_cli_json_over_a_dxbc_container_is_refused(self):
        self.work.add("ps_heap", "ps", SFI0_RES | SFI0_SMP, PS_HEAP)
        container = os.path.join(self.work.root, "ps_heap.dxbc")
        with open(container, "rb") as f:
            before = f.read()
        p = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", "heap_index_census.py"),
                            self.work.disasm, "--json", container], capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn("--json would overwrite an input the census reads", p.stderr)
        with open(container, "rb") as f:
            self.assertEqual(f.read(), before)

    def test_cli_json_over_an_fssm_header_is_refused(self):
        self.work.add("cs_sys", "cs", SFI0_RES, [
            ("s_load_dwordx8 s[20:27], s[2:3], 0x40", 8),
            ("s_endpgm", 4),
        ], rsrc2=2 << 1)
        header = os.path.join(self.work.root, "fssm", "cs_sys.fssm")
        with open(header, "rb") as f:
            before = f.read()
        p = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", "heap_index_census.py"),
                            self.work.disasm, "--json", header], capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn("--json would overwrite an input the census reads", p.stderr)
        with open(header, "rb") as f:
            self.assertEqual(f.read(), before)

    def test_cli_json_over_a_hard_link_to_a_container_is_refused(self):
        self.work.add("ps_heap", "ps", SFI0_RES | SFI0_SMP, PS_HEAP)
        container = os.path.join(self.work.root, "ps_heap.dxbc")
        link = os.path.join(self.work.root, "report.json")
        os.link(container, link)
        with open(container, "rb") as f:
            before = f.read()
        p = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", "heap_index_census.py"),
                            self.work.disasm, "--json", link], capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn("--json would overwrite an input the census reads", p.stderr)
        with open(container, "rb") as f:
            self.assertEqual(f.read(), before)


if __name__ == "__main__":
    unittest.main()
