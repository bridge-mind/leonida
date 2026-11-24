"""Synthetic coverage of the report join and its ambiguous/missing evidence."""
from itertools import product
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import world_drawable_lookup as lookup
import world_names


class DrawableLookupTest(unittest.TestCase):
    def test_only_export_candidates_with_consistent_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = ["one", "many", "conflict", "wrong_key", "unaligned", "cached", "missing_key"]
            definitions, streams, placements = [], [], []
            for i, name in enumerate(names):
                h = world_names.joaat(name)
                definitions.append({"name": h, "asset_name": h, "asset_type": 2})
                key = world_names.hash64(name)
                placements.append({"archetype": f"{h:x}", "key": f"{key:x}", "position": [0, 0, 0],
                                   "rotation": [0, 0, 0, 1], "lod_level": 6})
                streams.append({"offset": 4096 * i + 16, "names": [name + ".#dr"]})
            streams.append({"offset": 4096 * 8 + 16, "names": ["many.#dr"]})
            definitions.append(dict(definitions[2], asset_type=1))
            placements[3]["key"] = "0"
            streams[4]["offset"] += 1
            placements[-1].pop("key")
            for file, data in [("defs", {"archetypes": definitions}),
                               ("names", {"summary": {"archive": "test.rpf"}, "files": streams}),
                               ("manifest", {"Items": [{"Name": "cached"}]})]:
                (root / file).write_text(json.dumps(data), encoding="utf-8", newline="\n")
            (root / "entities").write_text("\n".join(map(json.dumps, placements)), encoding="utf-8", newline="\n")
            result = lookup.lookup(root / "entities", root / "defs", root / "names", [root / "manifest"])
            self.assertEqual(result["outcomes"], {"candidate": 1, "ambiguous": 1, "unresolved": 3,
                                                  "listed_in_manifest": 1, "missing_or_invalid_key": 1})
            self.assertEqual(sorted(r["offsets"] for r in result["candidates"]), [[16], [4112, 32784]])

    def test_scan_window_matches_exhaustive_intervals(self):
        for weights in product((0, 1, 3), repeat=4):
            rows = [{"status": "ambiguous", "offsets": [1, 4], "placements": 100}]
            for offset, weight in zip((1, 4, 5, 9), weights):
                if weight:
                    for n in (weight, 1):  # Two distinct identities at this offset.
                        rows.append({"status": "candidate", "offsets": [offset], "placements": n})
            report = {"archive": "one.rpf", "candidates": rows}
            for budget in range(1, 12):
                choices = []
                for start in range(10):
                    for length in range(1, budget + 1):
                        hits = [r for r in rows if r["status"] == "candidate"
                                and start <= r["offsets"][0] < start + length]
                        if hits:
                            first = min(r["offsets"][0] for r in hits)
                            last = max(r["offsets"][0] for r in hits)
                            choices.append((-sum(r["placements"] for r in hits), last, first, len(hits)))
                result = lookup.export_window(report, budget)
                if choices:
                    score, last, first, count = min(choices)
                    self.assertEqual(result, {"archive": "one.rpf", "start": first, "length": last-first+1,
                                              "predicted_placements": -score, "predicted_identities": count})
                else:
                    self.assertIsNone(result)
        for invalid in (0, -1, 1 << 63):
            with self.assertRaises(ValueError):
                lookup.export_window(report, invalid)


if __name__ == "__main__":
    unittest.main()
