"""gamedata_sps1.py must decode the SPS1 shader-preset table without game data.

    python -I tests/test_gamedata_sps1.py

Every fixture is synthetic: SPS1 blobs with length-prefixed NUL-terminated preset/effect pairs and render buckets,
multi-stream 64 KiB carve directories, and minimal SGD5 databases for cross-reference and hash-lookup checks.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
TOOL = os.path.join(TOOLS, "gamedata_sps1.py")
sys.path.insert(0, TOOLS)

import gamedata_sps1  # noqa: E402
from hashdict import joaat  # noqa: E402
import sgd5_effects  # noqa: E402


def sps1_payload(entries: list[tuple[str, str, int]], count: int | None = None) -> bytes:
    """Build a synthetic SPS1 payload from (preset, effect, render_bucket) tuples."""
    n = len(entries) if count is None else count
    buf = bytearray(b"SPS1" + struct.pack("<I", n))
    for preset, effect, bucket in entries:
        p = preset.encode("ascii") + b"\x00"
        e = effect.encode("ascii") + b"\x00"
        buf += bytes([len(p)]) + p + bytes([len(e)]) + e + bytes([bucket])
    return bytes(buf)


def sgd5_effect_record(name: str) -> bytes:
    """Minimal valid SGD5 effect record (0 lists/passes, 2-byte NUL string pool)."""
    nbytes = name.encode("ascii") + b"\x00"
    out = bytearray([len(nbytes)]) + nbytes
    out += bytes(4)                             # opaque u32
    out += bytes(6 * 4)                         # 6 empty stage ordinal lists
    out += bytes(80)                            # fixed block
    out += bytes(4 * 4)                         # 4 empty subtables (16/24/68/36 stride)
    out += bytes(4)                             # 0 technique rows
    out += bytes(4)                             # 0 pass rows
    out += bytes(4)                             # 0 post-pass u32s
    out += struct.pack("<I", 0x80000000)        # post-pass marker
    out += bytes(8)                             # 2 empty length-prefixed post-pass blocks
    out += bytes(4)                             # packed post-pass count = 0
    out += struct.pack("<I", 2) + b"\x00\x00"   # string pool starting and ending with NUL
    return bytes(out)


def synthetic_sgd5_archive(db0_effects: list[str], db1_effects: list[str]):
    """Build two contiguous minimal SGD5 databases inside one synthetic archive buffer."""
    def make_db(names: list[str], start: int):
        buf = bytearray(b"SGD5" + bytes(4))
        first_group = start + len(buf)
        buf += bytes(6 * 4)                     # 6 empty program groups
        buf += struct.pack("<I", len(names))
        for name in names:
            buf += sgd5_effect_record(name)
        buf += struct.pack("<I", 1)             # database terminator
        end = start + len(buf)
        return bytes(buf), (start, first_group, end)

    db0_bytes, layout0 = make_db(db0_effects, 0)
    db1_bytes, layout1 = make_db(db1_effects, len(db0_bytes))
    return db0_bytes + db1_bytes, (layout0, layout1)


class GamedataSps1Test(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.carve = os.path.join(self.root, "carve")
        os.makedirs(self.carve)
        self.rows = []
        self.next_off = 0x1000

    def tearDown(self):
        self.tmp.cleanup()

    def add(self, payload: bytes, chunks=None):
        parts = chunks if chunks is not None else [payload]
        for part in parts:
            off = self.next_off
            self.next_off += 0x1000
            with open(os.path.join(self.carve, f"{off:012x}.bin"), "wb") as f:
                f.write(part)
            self.rows.append({
                "off": off,
                "stored": len(part),
                "size": len(part),
                "head": part[:16].hex(),
                "pre": "",
            })

    def finish(self):
        with open(os.path.join(self.carve, "index.json"), "w", encoding="utf-8") as f:
            json.dump({"summary": {"archive": "synthetic", "files": len(self.rows)}, "files": self.rows}, f)

    def test_parses_presets_buckets_identity_and_low_count_byte(self):
        entries = [
            ("mat_std.sps", "mat_std", 0),
            ("mat_cutout.sps", "mat_std", 3),
            ("folder/mat_water.sps", "shader_water", 6),
            ("no_splash.sps", "shared_fx", 4),
            ("no_water.sps", "shared_fx", 5),
        ]
        # Set count = 858? For 5 entries count must be 5; also verify ascii detection with 90 ('Z') via a 90-entry table.
        parsed = gamedata_sps1.parse_sps1(sps1_payload(entries))
        self.assertEqual(parsed["type"], "SPS1")
        self.assertEqual(parsed["header"]["entry_count"], 5)
        self.assertIsNone(parsed["header"]["low_count_byte_ascii"])
        t = parsed["table"]
        self.assertEqual(t["records"], 5)
        self.assertEqual(t["distinct_presets"], 5)
        self.assertEqual(t["distinct_effects"], 3)
        self.assertEqual(t["sps_extension_count"], 5)
        self.assertEqual((t["root_presets"], t["subfolder_presets"], t["subfolder_stem_collisions"]), (4, 1, 0))
        self.assertEqual((t["identity_presets"], t["aliased_presets"]), (1, 4))
        self.assertEqual(t["render_buckets"], {"0": 1, "3": 1, "4": 1, "5": 1, "6": 1})
        self.assertEqual(t["multi_bucket_effects"], 2)
        self.assertEqual(t["multi_bucket_combinations"], {"0,3": 1, "4,5": 1})

        ninety = sps1_payload([(f"p{i}.sps", "fx_common", i % 4) for i in range(90)])
        self.assertEqual(ninety[:8], b"SPS1Z\x00\x00\x00")
        parsed90 = gamedata_sps1.parse_sps1(ninety)
        self.assertEqual(parsed90["header"]["low_count_byte_ascii"], "Z")
        self.assertEqual(parsed90["table"]["records"], 90)

    def test_rejects_malformed_or_trailing_inputs(self):
        good = sps1_payload([("a.sps", "a", 0)])
        with self.assertRaisesRegex(ValueError, "not an SPS1"):
            gamedata_sps1.parse_sps1(b"BAD1" + good[4:])
        with self.assertRaisesRegex(ValueError, "trailing"):
            gamedata_sps1.parse_sps1(good + b"\x00")
        with self.assertRaisesRegex(ValueError, "truncated"):
            gamedata_sps1.parse_sps1(good[:-1])
        # Missing trailing NUL in effect name
        bad_nul = bytearray(good)
        bad_nul[-2] = ord("x")
        with self.assertRaisesRegex(ValueError, "NUL"):
            gamedata_sps1.parse_sps1(bytes(bad_nul))
        # Non-printable byte in preset name
        bad_ctrl = sps1_payload([("a\x01.sps", "a", 0)])
        with self.assertRaisesRegex(ValueError, "non-printable"):
            gamedata_sps1.parse_sps1(bad_ctrl)

    def test_reads_bin_carve_dir_and_64k_continuation_streams(self):
        long_preset = "p_" + ("x" * 180) + ".sps"
        entries = [(f"{long_preset[:-4]}_{i}.sps", f"fx_{i % 5}", i % 7) for i in range(400)]
        blob = sps1_payload(entries)
        self.assertGreater(len(blob), 65536)

        self.add(b'<?xml version="1.0"?><RootA/>')
        self.add(blob, [blob[:65536], blob[65536:]])
        self.add(b"MHIU" + bytes(40))
        self.finish()

        raw_bin = os.path.join(self.root, "only_sps1.bin")
        with open(raw_bin, "wb") as f:
            f.write(blob)

        for target in (self.carve, raw_bin):
            loaded = gamedata_sps1.read_payloads(target)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["table"]["records"], 400)
            self.assertEqual(loaded[0]["size"], len(blob))
        self.assertEqual(gamedata_sps1.read_payloads(self.carve)[0]["chunks"], 2)

    def test_exact_64k_sps1_does_not_absorb_following_opaque_row(self):
        # 8B header + 327 * 200B + 128B == 65,536B exact, followed by an opaque binary row.
        entries_64k = [(f"p{i:03d}_" + ("x" * 180) + ".sps", "fx_std", i % 7) for i in range(327)]
        entries_64k.append(("p_end_" + ("y" * 107) + ".sps", "fx_std", 0))
        exact_64k = sps1_payload(entries_64k)
        self.assertEqual(len(exact_64k), 65536)

        self.add(exact_64k)
        self.add(b"\x01\x02\x03\x04" * 16)
        self.finish()
        loaded = gamedata_sps1.read_payloads(self.carve)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["size"], 65536)
        self.assertEqual(loaded[0]["chunks"], 1)
        self.assertEqual(loaded[0]["table"]["records"], 328)

        # Also verify an exact 131,072B two-chunk table stops after 2 chunks instead of absorbing a 3rd opaque row.
        self.rows.clear()
        entries_128k = [(f"p{i:03d}_" + ("x" * 180) + ".sps", "fx_std", i % 7) for i in range(655)]
        entries_128k.append(("p_end_" + ("z" * 43) + ".sps", "fx_std", 0))
        exact_128k = sps1_payload(entries_128k)
        self.assertEqual(len(exact_128k), 131072)
        self.add(exact_128k, [exact_128k[:65536], exact_128k[65536:]])
        self.add(b"\x05\x06\x07\x08" * 16)
        self.finish()
        loaded2 = gamedata_sps1.read_payloads(self.carve)
        self.assertEqual(len(loaded2), 1)
        self.assertEqual(loaded2[0]["size"], 131072)
        self.assertEqual(loaded2[0]["chunks"], 2)
        self.assertEqual(loaded2[0]["table"]["records"], 656)

    def test_cross_references_sgd5_and_matches_hashes_without_leaking_names(self):
        entries = [
            ("alpha_fx.sps", "alpha_fx", 0),
            ("alias_to_alpha.sps", "alpha_fx", 1),
            ("beta_only_alias.sps", "beta_fx", 4),
            ("gamma_missing.sps", "missing_fx", 2),
        ]
        blob = sps1_payload(entries)
        self.add(blob)
        self.finish()

        archive_bytes, layouts = synthetic_sgd5_archive(["alpha_fx", "beta_fx"], ["gamma_missing"])
        shaders_path = os.path.join(self.root, "synthetic_shaders.rpf")
        with open(shaders_path, "wb") as f:
            f.write(archive_bytes)

        with mock.patch.object(sgd5_effects, "DATABASES", layouts):
            payloads = gamedata_sps1.read_payloads(self.carve)
            xref = gamedata_sps1.cross_reference_sgd5(payloads, shaders_path)
            self.assertEqual(xref["sgd5_db0_effects"], 2)
            self.assertEqual(xref["sgd5_db1_effects"], 1)
            self.assertEqual(xref["records_resolved_db0"], 3)
            self.assertEqual(xref["records_resolved_db1"], 0)
            self.assertEqual(xref["records_unresolved"], 1)
            self.assertEqual(xref["distinct_effects_resolved_db0"], 2)
            self.assertEqual(xref["preset_stems_in_sgd5"], 2)   # alpha_fx, gamma_missing
            self.assertEqual(xref["sps1_only_preset_stems"], 2) # alias_to_alpha, beta_only_alias

        out_json = os.path.join(self.root, "reports", "sps1.json")
        # Query beta_only_alias via CLI? Use in-process main after patching DATABASES via test directly;
        # also run subprocess CLI without --shaders for --preset-hash + --json.
        beta_hash = f"{joaat('beta_only_alias'):08x}"
        proc = subprocess.run(
            [sys.executable, "-I", TOOL, self.carve, "--preset-hash", beta_hash, "--json", out_json],
            cwd=self.root, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        with open(out_json, "rb") as f:
            raw = f.read()
        self.assertIn(b"\n", raw)
        self.assertNotIn(b"\r\n", raw)
        # No raw preset or effect string appears anywhere in JSON or stdout.
        for secret in ("alpha_fx", "alias_to_alpha", "beta_only_alias", "beta_fx", "gamma_missing"):
            self.assertNotIn(secret, proc.stdout)
            self.assertNotIn(secret.encode("ascii"), raw)
        report = json.loads(raw.decode("utf-8"))
        self.assertEqual(len(report["hash_matches"]), 1)
        m = report["hash_matches"][0]
        self.assertEqual((m["record_index"], m["render_bucket"], m["aliased"]), (2, 4, True))
        self.assertEqual(m["preset_stem_hash"], f"{joaat('beta_only_alias'):08X}")
        self.assertEqual(m["effect_hash"], f"{joaat('beta_fx'):08X}")

    def test_missing_input_path_is_a_usage_error_naming_it(self):
        """BUILD-1046: a path that does not exist is exit 2 and one ERROR line, not a failed run."""
        missing = os.path.join(self.root, "no-such-input")
        self.assertFalse(os.path.exists(missing))
        for cmd in ([sys.executable, "-I", TOOL, missing],
                    [sys.executable, "-I", TOOL, self.carve, "--shaders", missing]):
            with self.subTest(cmd=cmd[3:]):
                p = subprocess.run(cmd, cwd=self.root, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace")
                self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
                self.assertNotIn("Traceback", p.stderr)
                lines = p.stderr.strip().splitlines()
                self.assertEqual(len(lines), 1, p.stderr)
                self.assertTrue(lines[0].startswith("ERROR:"), p.stderr)
                self.assertIn(missing, lines[0])

    def test_refuses_json_colliding_with_inputs_and_rejects_bad_archive(self):
        self.add(sps1_payload([("a.sps", "a", 0)]))
        self.finish()
        shaders_file = os.path.join(self.root, "shaders.rpf")
        with open(shaders_file, "wb") as f:
            f.write(b"dummy")
        out_dir = os.path.join(self.root, "existing_dir")
        os.makedirs(out_dir)

        for target, shaders, out in (
            (self.carve, None, os.path.join(self.carve, "index.json")),
            (self.carve, None, os.path.join(self.carve, f"{self.rows[0]['off']:012x}.bin")),
            (self.carve, shaders_file, shaders_file),
            (self.carve, None, out_dir),
        ):
            cmd = [sys.executable, "-I", TOOL, target, "--json", out]
            if shaders:
                cmd += ["--shaders", shaders]
            p = subprocess.run(cmd, cwd=self.root, capture_output=True, text=True, encoding="utf-8", errors="replace")
            self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
            self.assertIn("error:", p.stderr)
            self.assertNotIn("Traceback", p.stderr)

        bad_file = os.path.join(self.root, "not_rpf.bin")
        with open(bad_file, "wb") as f:
            f.write(b"NOPE1234")
        p = subprocess.run([sys.executable, "-I", TOOL, bad_file], cwd=self.root,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(p.returncode, 1)
        self.assertIn("neither an SPS1 payload nor an RPF9 archive", p.stderr)


if __name__ == "__main__":
    unittest.main()
