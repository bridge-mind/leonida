"""Synthetic export windows and keyed model observations."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import world_drawable_verify as verifier
import world_names


class VerifyTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def write(self, name, data):
        path = self.root / name
        path.write_text(json.dumps(data), encoding="utf-8", newline="\n")
        return path

    def test_join_counts_and_window_edges(self):
        candidates = []
        for name, offset in [("one", 10), ("two", 12), ("three", 13), ("four", 14), ("five", 20)]:
            candidates.append({"archetype": format(world_names.joaat(name), "x"),
                               "key": format(world_names.hash64(name), "x"), "offsets": [offset],
                               "status": "candidate", "placements": 3})
        # A 32-bit match alone cannot confirm the prediction.
        candidates[1]["archetype"] = candidates[0]["archetype"]
        candidates.append(dict(candidates[0], status="ambiguous", offsets=[10, 15]))
        predictions = self.write("lookup", {"archive": "source.rpf", "candidates": candidates})
        export = {"src": "source.rpf", "Scan": {"Start": 10, "Length": 10},
                  "Items": [{"Name": "one.#dr", "Offset": 10}, {"Name": "one", "Offset": 12},
                            {"Name": "three", "Offset": 18}, {"Name": "", "Offset": 14}]}
        first = self.write("first", export)
        second = self.write("second", dict(export, Items=export["Items"][:1]))
        result = verifier.verify(predictions, [first, second, first])
        self.assertEqual([r["status"] for r in result["rows"]],
                         ["confirmed", "conflicting_name", "conflicting_offset", "missing_evidence",
                          "outside_windows"])
        self.assertEqual(result["placements"], {s: 3 for s in result["identities"]})
        self.assertEqual(result["rows"][0]["duplicate_observations"], 1)
        self.assertEqual(result["rows"][0]["other_identity_offsets"], [12])
        full = self.write("full", dict(export, Scan={}))
        self.assertEqual(verifier.verify(predictions, [full])["identities"]["missing_evidence"], 2)

    def test_reject_unrelated_archive_and_bad_bounds(self):
        predictions = self.write("lookup", {"archive": "source.rpf", "candidates": []})
        for source, scan in [("elsewhere/source.rpf", {}), ("source.rpf", {"Start": 0}),
                             ("source.rpf", {"Start": True, "Length": 5}),
                             ("source.rpf", {"Start": 0, "Length": 2})]:
            manifest = self.write("manifest", {"src": source, "Scan": scan, "Items": [{"Offset": 2}]})
            with self.subTest(source=source, scan=scan), self.assertRaises(ValueError):
                verifier.verify(predictions, [manifest])


if __name__ == "__main__":
    unittest.main()
