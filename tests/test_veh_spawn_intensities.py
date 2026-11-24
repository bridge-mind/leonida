"""Synthetic checks for tools/veh_spawn_intensities.py: the aggregate census of the vehicle
spawn-intensity table, band ranges, mud-band placement and rain grouping. Every entry name and
value in the fixtures is made up; the checker also asserts the CLI never prints key names, which
is game data.

    python -I tests/test_veh_spawn_intensities.py
"""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "veh_spawn_intensities.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import veh_spawn_intensities as vsi  # noqa: E402

# Entries shaped like the table: one 7-band ladder, one variant with mud in band 6, one 2-band.
A_SETS = [(-1.0, 0.0, 0.0, 0.0), (0.0, 0.1, 0.0, 0.0), (0.1, 0.2, 0.0, 0.0), (0.2, 0.4, 0.0, 0.0),
          (0.3, 0.6, 0.0, 0.0), (0.4, 0.8, 0.0, 0.0), (0.5, 1.0, 0.8, 1.0)]
B_SETS = A_SETS[:5] + [(0.45, 0.9, 0.9, 1.0), (0.5, 1.0, 0.8, 1.0)]
C_SETS = [(0.1, 0.3, 0.0, 0.0), (0.3, 0.9, 0.0, 0.0)]

DOC = """<?xml version="1.0" encoding="UTF-8"?>
<CGraphicsDefinitionsData>
  <VehSpawnIntensities>
    <Item key="spawn_alpha_test">
      <Name>spawn_alpha_test</Name>
      <IntensitySets>
        {a_sets}
      </IntensitySets>
      <DirtRainChance value="0.500000" /><DirtRainMin value="0.700000" /><DirtRainMax value="1.000000" />
    </Item>
    <Item key="spawn_beta_test">
      <Name>spawn_beta_test</Name>
      <IntensitySets>
        {b_sets}
      </IntensitySets>
      <DirtRainChance value="0.000000" /><DirtRainMin value="0.000000" /><DirtRainMax value="1.000000" />
    </Item>
    <Item key="spawn_gamma_test">
      <Name>spawn_gamma_test</Name>
      <IntensitySets>
        {c_sets}
      </IntensitySets>
      <DirtRainChance value="0.300000" /><DirtRainMin value="0.300000" /><DirtRainMax value="0.500000" />
    </Item>
  </VehSpawnIntensities>
</CGraphicsDefinitionsData>
""".format(
    a_sets="".join(f'<Item><DirtMin value="{a:g}" /><DirtMax value="{b:g}" />'
                   f'<MudMin value="{c:g}" /><MudMax value="{d:g}" /></Item>' for a, b, c, d in A_SETS),
    b_sets="".join(f'<Item><DirtMin value="{a:g}" /><DirtMax value="{b:g}" />'
                   f'<MudMin value="{c:g}" /><MudMax value="{d:g}" /></Item>' for a, b, c, d in B_SETS),
    c_sets="".join(f'<Item><DirtMin value="{a:g}" /><DirtMax value="{b:g}" />'
                   f'<MudMin value="{c:g}" /><MudMax value="{d:g}" /></Item>' for a, b, c, d in C_SETS),
)

WANT = {
    "bands_with_mud": [0, 0, 0, 0, 0, 1, 2],
    "entries": 3,
    "rain_group_sizes": [1, 1, 1],
    "rain_groups": 3,
    "set_counts": [2, 7],
    "signature_groups": [1, 1, 1],
    "signatures": 3,
    "per_set_ranges": [
        {"DirtMax": [0.0, 0.3], "DirtMin": [-1.0, 0.1], "MudMax": [0.0, 0.0], "MudMin": [0.0, 0.0]},
        {"DirtMax": [0.1, 0.9], "DirtMin": [0.0, 0.3], "MudMax": [0.0, 0.0], "MudMin": [0.0, 0.0]},
        {"DirtMax": [0.2, 0.2], "DirtMin": [0.1, 0.1], "MudMax": [0.0, 0.0], "MudMin": [0.0, 0.0]},
        {"DirtMax": [0.4, 0.4], "DirtMin": [0.2, 0.2], "MudMax": [0.0, 0.0], "MudMin": [0.0, 0.0]},
        {"DirtMax": [0.6, 0.6], "DirtMin": [0.3, 0.3], "MudMax": [0.0, 0.0], "MudMin": [0.0, 0.0]},
        {"DirtMax": [0.8, 0.9], "DirtMin": [0.4, 0.45], "MudMax": [0.0, 1.0], "MudMin": [0.0, 0.9]},
        {"DirtMax": [1.0, 1.0], "DirtMin": [0.5, 0.5], "MudMax": [1.0, 1.0], "MudMin": [0.8, 0.8]},
    ],
}


def write_doc(directory):
    path = os.path.join(directory, "graphics.xml")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(DOC)
    return path


class RecordsTest(unittest.TestCase):
    def test_records_and_aggregates(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(vsi.census(write_doc(tmp)), WANT)


class CliTest(unittest.TestCase):
    def run_tool(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            status = vsi.main(list(argv))
        return status, out.getvalue(), err.getvalue()

    def test_other_document_with_valid_table_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "other.xml")
            report = os.path.join(tmp, "report.json")
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(DOC.replace("CGraphicsDefinitionsData", "UnrelatedRoot"))
            status, out, err = self.run_tool(path, "--json", report)
            self.assertEqual(status, 2)
            self.assertEqual(out, "")
            self.assertEqual(err, "ERROR: expected CGraphicsDefinitionsData document root\n")
            self.assertFalse(os.path.exists(report))

    def test_stdout_aggregates_and_no_key_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, out, err = self.run_tool(write_doc(tmp))
            self.assertEqual(status, 0, err)
            self.assertEqual(json.loads(out), WANT)
            # Entry keys and Name values are the table's own text: the aggregate never carries them.
            for token in ("spawn_alpha_test", "spawn_beta_test", "spawn_gamma_test"):
                self.assertNotIn(token, out)
                self.assertNotIn(token, err)

    def test_json_output_matches_and_is_well_formed(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = os.path.join(tmp, "report.json")
            status, _, err = self.run_tool(write_doc(tmp), "--json", report)
            self.assertEqual(status, 0, err)
            with open(report, encoding="utf-8") as f:
                self.assertEqual(json.load(f), WANT)

    def test_json_over_the_input_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_doc(tmp)
            status, _, err = self.run_tool(path, "--json", path)
            self.assertEqual(status, 2)
            self.assertIn("--json must not overwrite the input", err)
            with open(path, encoding="utf-8") as f:
                self.assertIn("spawn_alpha_test", f.read())  # the input document is untouched

    def test_bad_input_is_one_error_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "broken.xml")
            with open(path, "w", encoding="utf-8") as f:
                f.write("<CGraphicsDefinitionsData><VehSpawnIntensities><Item/></VehSpawnIntensities>")
            status, _, err = self.run_tool(path)
            self.assertEqual(status, 2)
            self.assertTrue(err.startswith("ERROR:"))
            self.assertNotIn("Traceback", err)

    def test_a_non_finite_value_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_doc(tmp)
            with open(path, encoding="utf-8") as f:
                doc = f.read().replace('DirtMin value="-1"', 'DirtMin value="NaN"', 1)
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(doc)
            report = os.path.join(tmp, "report.json")
            status, out, err = self.run_tool(path, "--json", report)
            self.assertEqual(status, 2)
            self.assertTrue(err.startswith("ERROR:"), err)
            self.assertFalse(os.path.exists(report))
            with open(path, encoding="utf-8") as f:          # the input document is untouched
                self.assertIn('DirtMin value="NaN"', f.read())

    def test_carve_directory_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            blob = DOC.encode("utf-8")
            first, second = blob[:256], blob[256:]
            rows = []
            for i, chunk in enumerate((first, second)):
                off = 0x1000 + i * len(first)
                with open(os.path.join(tmp, f"{off:012x}.bin"), "wb") as f:
                    f.write(chunk)
                rows.append({"off": off, "stored": len(chunk), "size": len(chunk),
                             "pre": "00" * 16, "head": chunk[:16].hex()})
            with open(os.path.join(tmp, "index.json"), "w", encoding="utf-8") as f:
                json.dump({"files": rows}, f)
            status, out, err = self.run_tool(tmp)
            self.assertEqual(status, 0, err)
            self.assertEqual(json.loads(out), WANT)

    def test_carve_without_the_document_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = os.path.join(tmp, "0x1000.bin")
            with open(empty, "wb") as f:
                f.write(b"other plain bytes no xml")
            with open(os.path.join(tmp, "index.json"), "w", encoding="utf-8") as f:
                json.dump({"files": [{"off": 0x1000, "stored": 24, "size": 24,
                                      "pre": "00" * 16, "head": "other plain bytes ".encode().hex()[:32]}]}, f)
            status, _, err = self.run_tool(tmp)
            self.assertEqual(status, 2)
            self.assertIn("ERROR:", err)


if __name__ == "__main__":
    unittest.main()
