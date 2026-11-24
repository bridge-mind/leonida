"""tools/ptr_truncation_census.py on hand-written listings (standard library only, no game data, no GPU).

    python -I tests/test_ptr_truncation_census.py
"""
import gc
import io
import json
import os
import struct
import sys
import tempfile
import unittest
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import ptr_truncation_census as P  # noqa: E402
import shader_meta  # noqa: E402

TAIL = ["v_mov_b32_e32 v0, s8", "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]

LISTINGS = {
    # One descriptor table in s2, high dword forced to zero: the case shader_meta calls ptr32_lo.
    "ps_table32": ["s_mov_b32 s3, 0", "s_load_dwordx4 s[8:11], s[2:3], 0x0"] + TAIL,
    # Two truncated pointers: a table (s4, indexed) and a scalar constant load (s6).
    "ps_two32": ["s_mov_b32 s5, 0", "s_add_u32 s10, s4, 16", "s_mov_b32 s11, s5",
                 "s_load_dwordx8 s[12:19], s[10:11], 0x0", "s_mov_b32 s7, 0",
                 "s_load_dword s8, s[6:7], 0x0"] + TAIL,
    # A 64-bit pair (s0:s1) through global_load next to a truncated table (s2): "some".
    "ps_mixed": ["v_mov_b32_e32 v1, 0", "global_load_dword v2, v1, s[0:1]", "s_mov_b32 s3, 0",
                 "s_load_dwordx4 s[8:11], s[2:3], 0x0", "s_waitcnt vmcnt(0)", "exp mrt0 v2, v2, v2, v2 done vm",
                 "s_endpgm"],
    # Only a 64-bit pointer: "none truncated".
    "ps_ptr64": ["s_load_dwordx4 s[8:11], s[0:1], 0x0"] + TAIL,
    # A pointer built from a constant (not a user SGPR) with the high dword zero: not a truncated pointer,
    # but counted as a site the classifier cannot attribute.
    "ps_const_base": ["s_mov_b32 s4, 0x1000", "s_mov_b32 s5, 0", "s_load_dwordx4 s[8:11], s[4:5], 0x0"] + TAIL,
    # A global_load addressed through a VGPR pair: not a pointer to shader_meta, counted as unseen.
    "ps_global_vgpr": ["v_mov_b32_e32 v0, 0", "v_mov_b32_e32 v1, 0", "global_load_dword v2, v[0:1], off",
                       "s_waitcnt vmcnt(0)", "exp mrt0 v2, v2, v2, v2 done vm", "s_endpgm"],
    # No pointer at all.
    "ps_nopointer": ["v_mov_b32_e32 v0, s2", "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"],
}


EMPTY_DXBC = b"DXBC" + bytes(16) + struct.pack("<III", 1, 32, 0)


def write_listing(tmp, name, lines):
    """<tmp>/disasm/<name>.s plus the empty <tmp>/<name>.dxbc container translate() reads beside it."""
    with open(os.path.join(tmp, "disasm", name + ".s"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"; {name} code 0x100..0x200\n" + "".join("\t" + ln + "\n" for ln in lines))
    with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
        f.write(EMPTY_DXBC)


def scan(lines_by_name):
    with tempfile.TemporaryDirectory() as tmp:
        disasm = os.path.join(tmp, "disasm")
        os.mkdir(disasm)
        for name, lines in lines_by_name.items():
            write_listing(tmp, name, lines)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            records = {os.path.splitext(os.path.basename(p))[0]: P.scan_one(p) for p in P.listings(disasm)}
            gc.collect()
    return records


class ScanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = scan(LISTINGS)

    def test_every_listing_reaches_the_metadata_call(self):
        for name, r in self.records.items():
            self.assertIsNone(r["error"], name)
            self.assertEqual(r["stage"], "ps")

    def test_sites_name_access_low_and_high(self):
        self.assertEqual(self.records["ps_table32"]["sites"], [[2, "user", "zero", "table"]])
        self.assertEqual(sorted(map(tuple, self.records["ps_two32"]["sites"])),
                         [(4, "user_off", "zero", "table"), (6, "user", "zero", "scalar")])
        self.assertEqual(sorted(map(tuple, self.records["ps_mixed"]["sites"])),
                         [(0, "user", "user", "global"), (2, "user", "zero", "table")])
        self.assertEqual(self.records["ps_const_base"]["sites"], [[None, "other", "zero", "table"]])

    def test_flag_is_shader_meta_truncates_high(self):
        for name, r in self.records.items():
            self.assertEqual(r["truncates_high"], r["instrumented_truncates_high"], name)
        self.assertTrue(self.records["ps_table32"]["truncates_high"])
        self.assertFalse(self.records["ps_ptr64"]["truncates_high"])
        self.assertFalse(self.records["ps_const_base"]["truncates_high"])

    def test_instrumentation_is_removed_afterwards(self):
        self.assertEqual(shader_meta._record_ptr_pair.__module__, "shader_meta")
        self.assertEqual(shader_meta._step_state.__module__, "shader_meta")
        self.assertEqual(shader_meta.collect.__module__, "shader_meta")

    def test_summary(self):
        s = P.summarise(list(self.records.values()))
        self.assertEqual(s["total"], 7)
        self.assertEqual(s["truncating_shaders"], {"ps": 3})
        self.assertEqual(s["pointer_mode"]["ps"], {"all truncated": 2, "some truncated": 1, "none truncated": 1,
                                                   "no pointer": 3})
        self.assertEqual(s["truncated_pointers_per_shader"]["ps"], {"1": 2, "2": 1})
        self.assertEqual(s["truncated_pointer_access"]["ps"], {"scalar": 1, "table": 3})
        self.assertEqual(s["truncated_pointer_low_half"]["ps"], {"user": 3, "user_off": 1})
        self.assertEqual(s["truncated_pointer_sgpr"]["ps"], {"s2": 2, "s4": 1, "s6": 1})
        self.assertEqual(s["truncated_pointer_kind_in_index"]["ps"], {"ptr32_lo": 4})
        self.assertEqual(s["untruncated_site_high_half"]["ps"], {"user (global)": 1, "user (table)": 1})
        self.assertEqual(s["not_seen"]["ps"], {"shaders with lo not a user SGPR, hi zero (table)": 1,
                                             "shaders with a VGPR-addressed global_*": 1,
                                             "VGPR-addressed global_* instructions": 1})
        self.assertEqual(s["flag_mismatches"], 0)
        out = io.StringIO()
        P.print_summary(s, out)
        self.assertIn("truncate a pointer's high dword: 3: ps 3", out.getvalue())
        self.assertNotIn("ps_table32", out.getvalue())


class SummaryTests(unittest.TestCase):
    def test_not_reached_and_mismatch_are_counted(self):
        records = [
            {"stage": "cs", "error": "Unsupported: x"},
            {"stage": "cs", "error": None, "sites": [[4, "user", "zero", "table"]], "truncates_high": False,
             "instrumented_truncates_high": False, "user_sgpr_reads": [{"sgpr": 4, "kind": "ptr32_lo"}]},
        ]
        s = P.summarise(records)
        self.assertEqual(s["shaders"], {"cs": 2})
        self.assertEqual(s["reached_metadata"], {"cs": 1})
        self.assertEqual(s["not_reached"], {"Unsupported: x": 1})
        self.assertEqual(s["flag_mismatches"], 1)

    def test_pointer_used_both_ways(self):
        records = [{"stage": "ms", "error": None, "truncates_high": True, "instrumented_truncates_high": True,
                    "sites": [[16, "user", "zero", "table"], [16, "user", "unknown", "scalar"]],
                    "user_sgpr_reads": [{"sgpr": 16, "kind": "ptr64_lo"}]}]
        s = P.summarise(records)
        self.assertEqual(s["pointer_mode"]["ms"], {"some truncated": 1})
        self.assertEqual(s["truncated_also_used_untruncated"], {"ms": 1})
        self.assertEqual(s["truncated_pointer_kind_in_index"]["ms"], {"ptr64_lo": 1})

    def test_cli_writes_lf_json_without_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            disasm = os.path.join(tmp, "disasm")
            os.mkdir(disasm)
            write_listing(tmp, "ps_table32", LISTINGS["ps_table32"])
            out, idx = os.path.join(tmp, "s.json"), os.path.join(tmp, "i.json")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                stdout = sys.stdout
                sys.stdout = io.StringIO()
                try:
                    rc = P.main([disasm, "--jobs", "1", "--json", out, "--index-out", idx])
                finally:
                    sys.stdout = stdout
            self.assertEqual(rc, 0)
            for path in (out, idx):
                with open(path, "rb") as f:
                    data = f.read()
                self.assertNotIn(b"\r\n", data)
                self.assertNotIn(b"ps_table32", data)
            with open(idx, encoding="utf-8") as f:
                self.assertEqual(json.load(f), {"0": {"stage": "ps", "user_sgpr_reads": [{"sgpr": 2, "kind": "ptr32_lo"}]}})

    def test_json_and_index_out_name_two_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            disasm = os.path.join(tmp, "disasm")
            os.mkdir(disasm)
            write_listing(tmp, "ps_table32", LISTINGS["ps_table32"])
            one = os.path.join(tmp, "r.json")
            stderr = sys.stderr
            sys.stderr = io.StringIO()
            try:
                with self.assertRaises(SystemExit) as cm:
                    P.main([disasm, "--jobs", "1", "--json", one, "--index-out", one])
            finally:
                message = sys.stderr.getvalue()
                sys.stderr = stderr
            self.assertEqual(cm.exception.code, 2)
            self.assertIn("one file", message)
            self.assertFalse(os.path.exists(one))

    def test_limit_below_one_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            disasm = os.path.join(tmp, "disasm")
            os.mkdir(disasm)
            write_listing(tmp, "ps_table32", LISTINGS["ps_table32"])
            for bad in ("0", "-1"):
                stderr = sys.stderr
                sys.stderr = io.StringIO()
                try:
                    with self.assertRaises(SystemExit) as cm:
                        P.main([disasm, "--limit", bad])
                finally:
                    message = sys.stderr.getvalue()
                    sys.stderr = stderr
                self.assertEqual(cm.exception.code, 2)
                self.assertIn("--limit", message)
                self.assertIn("at least 1", message)

    def test_missing_folder(self):
        stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            self.assertEqual(P.main([os.path.join(ROOT, "no-such-folder-here")]), 2)
        finally:
            sys.stderr = stderr

    def test_limit_must_be_positive(self):
        for value in ("0", "-1"):
            with self.subTest(limit=value):
                stderr = sys.stderr
                sys.stderr = io.StringIO()
                try:
                    with self.assertRaises(SystemExit) as cm:
                        P.main([os.path.join(ROOT, "tools"), "--limit", value])
                finally:
                    message = sys.stderr.getvalue()
                    sys.stderr = stderr
                self.assertEqual(cm.exception.code, 2)
                self.assertIn("--limit must be at least 1", message)


if __name__ == "__main__":
    unittest.main()
