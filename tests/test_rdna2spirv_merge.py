"""tools/rdna2spirv.py --merge: an incomplete shard set is refused, not published as a partial index.

    python -I tests/test_rdna2spirv_merge.py
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import rdna2spirv as R  # noqa: E402

TOOL = ROOT / "tools" / "rdna2spirv.py"


class MergeShardPartsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)

    def part(self, index, doc):
        (self.out / f"index.part{index}.json").write_text(json.dumps(doc), encoding="utf-8")

    def marker(self, index, count):
        (self.out / f"shard{index}.json").write_text(json.dumps({"index": index, "count": count}), encoding="utf-8")

    def plan(self, jobs):
        (self.out / "shards.json").write_text(json.dumps({"jobs": jobs}), encoding="utf-8")

    def merge_error(self):
        with self.assertRaises(R.ShardSetError) as cm:
            R.merge_shard_parts(str(self.out))
        return str(cm.exception)

    def test_complete_set_merges(self):
        self.part(0, {"a": {"r": 1}})
        self.part(1, {"b": {"r": 2}})
        self.marker(0, 2)
        self.marker(1, 2)
        self.assertEqual(R.merge_shard_parts(str(self.out)), {"a": {"r": 1}, "b": {"r": 2}})

    def test_a_complete_set_of_empty_parts_is_refused(self):
        # Every shard finished but translated nothing: index.json must not say the corpus is done.
        self.part(0, {})
        self.part(1, {})
        self.marker(0, 2)
        self.marker(1, 2)
        self.assertIn("holds no shaders", self.merge_error())

    def test_a_part_that_is_not_json_is_named(self):
        # A shard killed mid-write leaves a truncated part; the merge must say which part to delete, not traceback.
        (self.out / "index.part0.json").write_text('{"a": {', encoding="utf-8")
        self.marker(0, 1)
        message = self.merge_error()
        self.assertIn("cannot read", message)
        self.assertIn("index.part0.json", message)
        self.assertIn("delete that part", message)

    def test_missing_part_is_refused(self):
        self.part(0, {"a": {}})
        self.marker(0, 3)
        self.marker(1, 3)
        self.marker(2, 3)
        message = self.merge_error()
        self.assertIn("incomplete shard set", message)
        self.assertIn("1 of 3", message)
        self.assertIn("missing [1, 2]", message)

    def test_extra_part_is_refused(self):
        for i in range(3):
            self.part(i, {})
            self.marker(i, 2)
        message = self.merge_error()
        self.assertIn("incomplete shard set", message)
        self.assertIn("unexpected [2]", message)

    def test_parts_and_markers_must_match(self):
        self.part(0, {})
        self.part(1, {})
        self.marker(0, 2)
        self.assertIn("index parts and shard markers disagree", self.merge_error())

    def test_a_part_without_its_marker_is_refused(self):
        # A part left by an older rdna2spirv.py that did not write markers mixed with a newer set.
        for i in range(3):
            self.part(i, {})
        self.marker(1, 3)
        self.marker(2, 3)
        message = self.merge_error()
        self.assertIn("index parts and shard markers disagree", message)
        self.assertIn("parts [0, 1, 2]", message)
        self.assertIn("markers [1, 2]", message)

    def test_marker_counts_must_agree(self):
        self.part(0, {})
        self.part(1, {})
        self.marker(0, 2)
        self.marker(1, 3)
        self.assertIn("disagree on the shard count", self.merge_error())

    def test_plan_count_is_used_without_markers(self):
        self.part(0, {})
        self.plan(2)
        self.assertIn("incomplete shard set", self.merge_error())

    def test_legacy_parts_without_metadata_still_merge(self):
        self.part(0, {"a": {}})
        self.part(1, {"b": {}})
        self.assertEqual(sorted(R.merge_shard_parts(str(self.out))), ["a", "b"])

    def test_legacy_set_with_double_digit_indices_merges(self):
        for i in range(12):
            self.part(i, {str(i): {}})
        self.assertEqual(set(R.merge_shard_parts(str(self.out))), {str(i) for i in range(12)})

    def test_large_gaps_have_a_bounded_diagnostic(self):
        self.part(0, {"a": {}})
        self.part(10_000, {"b": {}})
        for expected in (None, 10_001):
            with self.subTest(expected=expected):
                if expected:
                    self.plan(expected)
                message = self.merge_error()
                self.assertIn("incomplete shard set", message)
                self.assertIn("missing [1, 2", message)
                self.assertIn("9983 more", message)
                self.assertLess(len(message), 300)

    def test_plan_and_markers_that_agree_merge(self):
        self.part(0, {"a": {}})
        self.part(1, {"b": {}})
        self.marker(0, 2)
        self.marker(1, 2)
        self.plan(2)
        self.assertEqual(sorted(R.merge_shard_parts(str(self.out))), ["a", "b"])

    def test_markers_must_match_the_plan_count(self):
        self.part(0, {})
        self.part(1, {})
        self.marker(0, 2)
        self.marker(1, 2)
        self.plan(3)
        self.assertIn("shard markers say 2 shards but the plan says 3", self.merge_error())

    def test_malformed_plan_fails_closed(self):
        self.part(0, {})
        self.part(1, {})
        self.marker(0, 2)
        self.marker(1, 2)
        (self.out / "shards.json").write_text("not json\n", encoding="utf-8")
        self.assertIn("cannot read the shard plan", self.merge_error())

    def test_plan_without_a_usable_job_count_fails_closed(self):
        self.part(0, {})
        self.plan(0)
        self.assertIn("cannot read the shard plan", self.merge_error())

    def test_a_gap_is_refused_when_the_count_is_unknown(self):
        self.part(0, {})
        self.part(2, {})
        message = self.merge_error()
        self.assertIn("not contiguous from 0", message)
        self.assertIn("missing [1]", message)

    def test_merge_with_no_parts_is_refused(self):
        self.assertIn("no index.part", self.merge_error())

    def test_a_corrupt_marker_is_refused(self):
        self.part(0, {})
        self.marker(0, 2)
        (self.out / "shard0.json").write_text("not json\n", encoding="utf-8")
        self.assertIn("cannot read the shard marker", self.merge_error())

    def test_a_marker_without_index_or_count_is_refused(self):
        self.part(0, {})
        (self.out / "shard0.json").write_text('{"count": 2}\n', encoding="utf-8")
        self.assertIn("is not a shard marker", self.merge_error())

    def test_unexpected_part_name_is_refused(self):
        (self.out / "index.partx.json").write_text("{}", encoding="utf-8")
        self.assertIn("not an index.part", self.merge_error())


class MergeCliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)

    def part(self, index, doc):
        (self.out / f"index.part{index}.json").write_text(json.dumps(doc), encoding="utf-8")

    def marker(self, index, count):
        (self.out / f"shard{index}.json").write_text(json.dumps({"index": index, "count": count}), encoding="utf-8")

    def run_merge(self):
        return subprocess.run([sys.executable, "-I", str(TOOL), str(self.out), "--out", str(self.out), "--merge"],
                              capture_output=True, text=True, timeout=120)

    def test_cli_refuses_a_complete_set_of_empty_parts(self):
        self.part(0, {})
        self.part(1, {})
        self.marker(0, 2)
        self.marker(1, 2)
        result = self.run_merge()
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("holds no shaders", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse((self.out / "index.json").exists())

    def test_cli_refuses_a_partial_set_and_writes_no_index(self):
        self.part(0, {"a": {}})
        self.marker(0, 2)
        self.marker(1, 2)
        result = self.run_merge()
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("ERROR: incomplete shard set", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse((self.out / "index.json").exists())

    def test_cli_removes_the_previous_index_when_the_merge_fails(self):
        # The old index describes the previous corpus; after a failed merge the parts no longer match it.
        (self.out / "index.json").write_text(json.dumps({"old": {}}), encoding="utf-8")
        self.part(0, {"a": {}})
        self.marker(0, 2)
        self.marker(1, 2)
        result = self.run_merge()
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("ERROR: incomplete shard set", result.stderr)
        self.assertFalse((self.out / "index.json").exists())

    def test_cli_merges_a_complete_set(self):
        self.part(0, {"a": {}})
        self.part(1, {"b": {}})
        self.marker(0, 2)
        self.marker(1, 2)
        result = self.run_merge()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("merged 2 entries", result.stdout)
        merged = json.loads((self.out / "index.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(merged), ["a", "b"])


    def test_cli_refuses_a_marker_count_that_disagrees_with_the_plan(self):
        self.part(0, {})
        self.part(1, {})
        self.marker(0, 2)
        self.marker(1, 2)
        (self.out / "shards.json").write_text(json.dumps({"jobs": 3}), encoding="utf-8")
        result = self.run_merge()
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("shard markers say 2 shards but the plan says 3", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse((self.out / "index.json").exists())

    def test_cli_refuses_a_part_that_is_not_json(self):
        (self.out / "index.part0.json").write_text('{"a": {', encoding="utf-8")
        self.marker(0, 1)
        result = self.run_merge()
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("ERROR: cannot read", result.stderr)
        self.assertIn("index.part0.json", result.stderr)
        self.assertIn("delete that part", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse((self.out / "index.json").exists())

    def test_cli_refuses_a_malformed_plan(self):
        self.part(0, {"a": {}})
        self.part(1, {"b": {}})
        self.marker(0, 2)
        self.marker(1, 2)
        (self.out / "shards.json").write_text("not json\n", encoding="utf-8")
        result = self.run_merge()
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("cannot read the shard plan", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse((self.out / "index.json").exists())

    def test_cli_refuses_a_corrupt_marker(self):
        self.part(0, {"a": {}})
        self.marker(0, 2)
        (self.out / "shard0.json").write_text("not json\n", encoding="utf-8")
        result = self.run_merge()
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("cannot read the shard marker", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse((self.out / "index.json").exists())


class ShardRoundTripTest(unittest.TestCase):
    """The writer and the merge through the real CLI, on an empty input directory (no .s and no llvm-mc needed)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.src = Path(self.tmp.name) / "src"
        self.out = Path(self.tmp.name) / "out"
        self.src.mkdir()
        self.out.mkdir()

    def run_tool(self, *args):
        return subprocess.run([sys.executable, "-I", str(TOOL), str(self.src), "--out", str(self.out), *args],
                              capture_output=True, text=True, timeout=120)

    def test_a_shard_writes_its_marker_and_part_and_merge_needs_the_whole_set(self):
        first = self.run_tool("--shard", "0/2")
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        self.assertTrue((self.out / "index.part0.json").is_file())
        marker = json.loads((self.out / "shard0.json").read_text(encoding="utf-8"))
        self.assertEqual(marker, {"index": 0, "count": 2})

        partial = self.run_tool("--merge")
        self.assertEqual(partial.returncode, 2, partial.stdout + partial.stderr)
        self.assertIn("incomplete shard set", partial.stderr)
        self.assertFalse((self.out / "index.json").exists())

        second = self.run_tool("--shard", "1/2")
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertTrue((self.out / "index.part1.json").is_file())
        self.assertTrue((self.out / "shard1.json").is_file())

        # Both shards finished, but over an empty source they hold nothing: the merge refuses to publish {}.
        full = self.run_tool("--merge")
        self.assertEqual(full.returncode, 2, full.stdout + full.stderr)
        self.assertIn("holds no shaders", full.stderr)
        self.assertFalse((self.out / "index.json").exists())


if __name__ == "__main__":
    unittest.main()
