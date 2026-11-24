"""tools/anim_skel_match.py: matching bone-table names to skeleton joints by joaat32 of the lowercased name, on
skeletons and names built here (no game files), and the exit code for a missing archive. Standard library only."""
import io
import os
import sys
import tempfile
import unittest
import zlib
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import anim_index  # noqa: E402
import anim_skel_match as sm  # noqa: E402


def skeleton(names, unnamed=(), spelled=None):
    """A skeleton whose joints are these names; `unnamed` joints keep their hash but lose the name string, and
    `spelled` maps a name to the string the joint stores instead."""
    spelled = spelled or {}
    return {"hashes": [anim_index.joaat32(n.lower()) for n in names],
            "names": [None if n in unnamed else spelled.get(n, n) for n in names]}


BODY = ["SKEL_ROOT", "SKEL_Pelvis", "SKEL_Spine0", "SKEL_Head", "SKEL_L_Hand", "SKEL_R_Hand", "PH_L_Hand"]


class Match(unittest.TestCase):
    def test_a_name_matches_by_joaat32_of_its_lowercased_spelling(self):
        report = sm.match({"s": ["SKEL_Head"]}, [skeleton(["skel_head"], spelled={"skel_head": "SKEL_Head"})])
        self.assertEqual(report["union"]["in_a_skeleton"], 1)
        self.assertEqual(report["union"]["same_spelling"], 1)

    def test_only_skeletons_holding_every_name_count(self):
        skels = [skeleton(BODY), skeleton(BODY), skeleton(BODY + ["SKEL_Tail"]), skeleton(BODY[:4])]
        report = sm.match({"hands": ["SKEL_L_Hand", "SKEL_R_Hand"], "head": ["SKEL_Head"]}, skels)
        self.assertEqual(report["distinct_skeletons"], 3)
        self.assertEqual(report["streams"]["hands"]["held_by_skeletons"],
                         {"distinct": 2, "copies": 3, "joint_counts": [7, 8]})
        self.assertEqual(report["streams"]["head"]["held_by_skeletons"]["copies"], 4)
        self.assertEqual(report["union"]["held_by_skeletons"]["joint_counts"], [7, 8])

    def test_unknown_names_unnamed_joints_and_other_spellings_are_counted_apart(self):
        skels = [skeleton(BODY, unnamed={"SKEL_Spine0"}, spelled={"SKEL_Pelvis": "skel_pelvis"})]
        report = sm.match({"s": ["SKEL_Pelvis", "SKEL_Spine0", "SKEL_Head", "SKEL_Nope"]}, skels)
        u = report["union"]
        self.assertEqual((u["names"], u["in_a_skeleton"], u["same_spelling"], u["joint_unnamed"]), (4, 3, 1, 1))
        self.assertEqual(report["not_in_any_skeleton"], 1)
        self.assertEqual(u["held_by_skeletons"]["distinct"], 0)

    def test_duplicate_names_in_a_stream_count_once(self):
        report = sm.match({"s": ["SKEL_Head", "SKEL_Head"]}, [skeleton(BODY)])
        self.assertEqual(report["streams"]["s"]["names"], 1)

    def test_report_prints_counts_only(self):
        report = sm.match({"s": BODY}, [skeleton(BODY)])
        with redirect_stdout(io.StringIO()) as out:
            sm.print_report(report)
        self.assertIn("7 of 7 names are joints", out.getvalue())
        self.assertNotIn("SKEL_", out.getvalue())


class CommandLine(unittest.TestCase):
    def test_a_missing_archive_is_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "anim.rpf")
            with redirect_stderr(io.StringIO()) as err:
                self.assertEqual(sm.main([missing, missing]), 2)
        self.assertIn("anim_skel_match:", err.getvalue())

    def test_json_must_not_replace_either_input_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            anim = Path(tmp) / "anim.rpf"
            packs = Path(tmp) / "packs.rpf"
            anim_data = zlib.compress(b"SKEL_Synthetic\x00")
            packs_data = b"\x00"
            args = [str(anim), str(packs), "--stream", "0", "--json"]
            for target in (anim, packs):
                anim.write_bytes(anim_data)
                packs.write_bytes(packs_data)
                with self.subTest(target=target.name), redirect_stderr(io.StringIO()) as err, \
                        redirect_stdout(io.StringIO()) as out:
                    self.assertEqual(sm.main([*args, str(target)]), 2)
                    self.assertIn("must not overwrite", err.getvalue())
                    self.assertEqual(out.getvalue(), "")
                    self.assertEqual(anim.read_bytes(), anim_data)
                    self.assertEqual(packs.read_bytes(), packs_data)
            report = Path(tmp) / "report.json"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(sm.main([*args, str(report)]), 0)
            self.assertTrue(report.is_file())

    def test_offsets_parse_as_hex_or_decimal(self):
        self.assertEqual(sm.parse_offset("0x101a32160"), 0x101A32160)
        self.assertEqual(sm.parse_offset("4096"), 4096)


if __name__ == "__main__":
    unittest.main()
