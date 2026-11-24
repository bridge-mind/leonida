"""scripts/merge_check.py on throwaway git repositories (no network, no game data).

    python -I tests/test_merge_check.py

Each case builds main and a pull request branch, merges it the way the merge rounds do ("Merge pull request #N"),
and checks what the script says about merging the same PR again: a rebased copy (the repeated-merge case that left
doubled paragraphs on main), a PR with a new commit after its merge, and a fresh PR.
"""
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import merge_check as mc  # noqa: E402

PARAGRAPH = ("The effect record is a fixed 48-byte header followed by its parameters, one 16-byte slot each.\n"
             "A slot whose type is zero ends the list early, before the count in the header says it should.\n"
             "Readers stop at that slot and ignore the bytes after it, which later versions use for padding.\n")


class MergeCheckTest(unittest.TestCase):
    def setUp(self):
        td = tempfile.TemporaryDirectory(prefix="leonida merge_check ")
        self.addCleanup(td.cleanup)
        self.repo = td.name
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        env.update({"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"})
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.git("init", "-q", "-b", "main")
        self.write("docs/notes.md", "# Notes\n\nAn introduction long enough to be counted as a line of its own here.\n")
        self.commit("start")

    def git(self, *args):
        r = subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                            "-c", "commit.gpgsign=false", "-C", self.repo, *args],
                           capture_output=True, encoding="utf-8")
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout.strip()

    def write(self, path, text):
        full = os.path.join(self.repo, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)

    def append(self, path, text):
        with open(os.path.join(self.repo, path), encoding="utf-8") as f:
            old = f.read()
        self.write(path, old + text)

    def commit(self, message):
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)
        return self.git("rev-parse", "HEAD")

    def pr_branch(self, name, base, text):
        self.git("checkout", "-q", "-b", name, base)
        self.append("docs/notes.md", text)
        sha = self.commit(f"{name}: add the effect record section")
        self.git("checkout", "-q", "main")
        return sha

    def merge(self, number, branch):
        self.git("merge", "-q", "--no-ff", "-m", f"Merge pull request #{number} from someone/{branch}", branch)

    def squash_merge(self, number, branch, subject=None):
        self.git("merge", "-q", "--squash", branch)
        title = subject or f"{branch}: add the effect record section"
        self.git("commit", "-q", "-m", f"{title} (#{number})")

    def run_check(self, *pairs):
        out = io.StringIO()
        with redirect_stdout(out):
            code = mc.main(["--repo", self.repo, "--main", "main", "--json", *pairs])
        return code, json.loads(out.getvalue())

    def run_text(self, *pairs):
        out = io.StringIO()
        with redirect_stdout(out):
            code = mc.main(["--repo", self.repo, "--main", "main", *pairs])
        return code, out.getvalue()

    def test_a_fresh_pr_is_clean(self):
        base = self.git("rev-parse", "main")
        self.pr_branch("fresh", base, "\n" + PARAGRAPH)
        code, [r] = self.run_check("12=fresh")
        self.assertEqual(code, 0)
        self.assertEqual(r["verdict"], "clean")
        self.assertEqual(r["merges"], [])
        self.assertEqual(r["commits"], {"total": 1, "on_main": 0})

    def test_a_rebased_copy_of_a_merged_pr_is_already_merged_and_would_duplicate(self):
        base = self.git("rev-parse", "main")
        self.pr_branch("first", base, "\n" + PARAGRAPH)
        self.merge(7, "first")
        # Unrelated work lands, then a rebased copy of #7 (same change, new commit) comes back for the next round.
        self.write("docs/other.md", "Another document that changes nothing in notes.md at all, but moves main.\n")
        self.commit("other work")
        self.git("checkout", "-q", "-b", "copy", base)
        self.write("docs/notes.md", "# Notes\n\nAn introduction long enough to be counted as a line of its own here.\n"
                   "\n" + PARAGRAPH)
        self.commit("first: add the effect record section (rebased)")  # a new commit with the same patch
        self.git("checkout", "-q", "main")
        code, [r] = self.run_check("7=copy")
        self.assertEqual(code, 1)
        self.assertEqual(r["verdict"], "already-merged")
        self.assertEqual(len(r["merges"]), 1)
        self.assertEqual(r["commits"], {"total": 1, "on_main": 1})

    def test_batch_keeps_prior_merges_separate(self):
        base = self.git("rev-parse", "main")
        self.pr_branch("landed", base, PARAGRAPH)
        self.merge(31, "landed")
        merged_sha = self.git("rev-parse", "--short=12", "main")
        self.pr_branch("next", "main", "A new line for the unmerged PR.\n")
        with mock.patch.object(mc, "git", wraps=mc.git) as commands:
            code, rows = self.run_check("32=next", "31=landed", "32=next")
        scans = [call for call in commands.call_args_list if call.args[1] == "log" and "--first-parent" in call.args]
        self.assertEqual(len(scans), 1)
        self.assertEqual(code, 1)
        self.assertEqual([(r["pr"], r["verdict"], r["merges"]) for r in rows],
                         [(32, "clean", []), (31, "already-merged", [merged_sha]), (32, "clean", [])])

    def test_text_already_on_main_is_reported_even_when_the_patch_differs(self):
        base = self.git("rev-parse", "main")
        self.pr_branch("first", base, "\n" + PARAGRAPH)
        self.merge(8, "first")
        # The copy puts the same paragraph at the top: a different patch, the same text twice after a merge.
        self.git("checkout", "-q", "-b", "moved", base)
        self.write("docs/notes.md", "# Notes\n\n" + PARAGRAPH +
                   "\nAn introduction long enough to be counted as a line of its own here.\n")
        self.commit("first: put the section first")
        self.git("checkout", "-q", "main")
        code, [r] = self.run_check("9=moved")
        self.assertEqual(code, 1)
        self.assertEqual(r["verdict"], "duplicates")
        self.assertEqual(r["merges"], [])  # another number: only the text check can see it
        self.assertEqual(r["duplicates"], {"docs/notes.md": 3})

    def test_patch_paths_and_added_header_like_lines_still_count_duplicates(self):
        paths = ("docs/quoted \u65e5\u672c\u8a9e.md", "docs/notes with spaces.md", "docs/header-like.md")
        header_like = "++ " + PARAGRAPH
        introduction = "A separator between copies keeps Git's added hunk easy to identify.\n"
        self.git("config", "core.quotePath", "true")
        for path in paths:
            self.write(path, PARAGRAPH + header_like + introduction)
        self.commit("Existing translated documents")
        self.git("checkout", "-q", "-b", "duplicated")
        for path in paths:
            self.append(path, "\n" + (header_like if path.endswith("header-like.md") else PARAGRAPH))
        self.commit("Duplicate docs")
        self.git("checkout", "-q", "main")

        code, [result] = self.run_check("18=duplicated")
        self.assertEqual((code, result["verdict"]), (1, "duplicates"))
        self.assertEqual(result["duplicates"], dict.fromkeys(paths, 3))

        self.git("config", "diff.noprefix", "true")
        self.git("config", "core.quotePath", "false")
        self.assertEqual(self.run_check("18=duplicated"), (code, [result]))

    def test_duplicate_blocks_use_git_line_boundaries(self):
        unicode_lines = "".join(f"Paragraph {i} begins with enough text to count as a line.\u2028"
                                "Its continuation is also longer than forty characters.\n" for i in range(3))
        bare_cr_lines = unicode_lines.replace("\u2028", "\r")
        paths = {"docs/notes.md": unicode_lines, "docs/cr.md": bare_cr_lines}
        for path, text in paths.items():
            self.write(path, text + "An unrelated line separates the two copies.\n")
        self.commit("Add paragraphs with embedded line separators")
        self.git("checkout", "-q", "-b", "duplicate-separators")
        for path, text in paths.items():
            # Read bytes here because normal text reads turn a bare CR into an LF.
            with open(os.path.join(self.repo, path), "ab") as f:
                f.write(text.encode("utf-8"))
        self.commit("Duplicate both paragraphs")
        self.git("checkout", "-q", "main")
        code, [result] = self.run_check("19=duplicate-separators")
        self.assertEqual((code, result["duplicates"]), (1, dict.fromkeys(paths, 3)))

    def test_a_block_the_pr_moves_is_not_a_duplicate(self):
        base = self.git("rev-parse", "main")
        self.pr_branch("first", base, "\n" + PARAGRAPH)
        self.merge(16, "first")
        # A later PR moves the paragraph above the introduction: one copy before, one copy after.
        self.git("checkout", "-q", "-b", "move", "main")
        self.write("docs/notes.md", "# Notes\n\n" + PARAGRAPH +
                   "\nAn introduction long enough to be counted as a line of its own here.\n")
        self.commit("move the section up")
        self.git("checkout", "-q", "main")
        code, [r] = self.run_check("17=move")
        self.assertEqual((code, r["verdict"], r["duplicates"]), (0, "clean", {}))

    def test_a_new_commit_after_the_merge_is_not_already_merged(self):
        base = self.git("rev-parse", "main")
        self.pr_branch("first", base, "\n" + PARAGRAPH)
        self.merge(10, "first")
        self.git("checkout", "-q", "first")
        self.append("docs/notes.md", "\nA follow-up sentence the author pushed after the first merge round ran.\n")
        self.commit("first: follow-up")
        self.git("checkout", "-q", "main")
        code, [r] = self.run_check("10=first")
        self.assertEqual(r["merges"] and r["verdict"], "clean")
        self.assertEqual(code, 0)
        self.assertEqual(r["commits"], {"total": 1, "on_main": 0})

    def test_one_or_two_repeated_lines_are_not_reported(self):
        base = self.git("rev-parse", "main")
        self.pr_branch("first", base, "\nself.assertEqual(result.returncode, 0, result.stdout + result.stderr)\n")
        self.merge(14, "first")
        self.git("checkout", "-q", "-b", "again", "main")
        self.append("docs/notes.md", "\nself.assertEqual(result.returncode, 0, result.stdout + result.stderr)\n")
        self.commit("the same assertion in another test")
        self.git("checkout", "-q", "main")
        code, [r] = self.run_check("15=again")
        self.assertEqual((code, r["duplicates"]), (0, {}))

    def test_short_lines_are_not_counted(self):
        base = self.git("rev-parse", "main")
        self.pr_branch("fence", base, "\n```\n---\n```\n")
        self.merge(11, "fence")
        self.git("checkout", "-q", "-b", "fence2", "main")
        self.append("docs/notes.md", "\n```\n---\n```\n")
        self.commit("another fence")
        self.git("checkout", "-q", "main")
        code, [r] = self.run_check("13=fence2")
        self.assertEqual(r["duplicates"], {})
        self.assertEqual(code, 0)

    def test_a_squash_merged_single_commit_pr_is_already_merged(self):
        base = self.git("rev-parse", "main")
        self.pr_branch("sq1", base, "\n" + PARAGRAPH)
        self.squash_merge(21, "sq1")
        self.write("docs/other.md", "Another document that changes nothing in notes.md at all, but moves main.\n")
        self.commit("other work")
        self.git("checkout", "-q", "-b", "sq1_copy", base)
        self.append("docs/notes.md", "\n" + PARAGRAPH)
        self.commit("sq1: add the effect record section (rebased)")
        self.git("checkout", "-q", "main")
        code, [r] = self.run_check("21=sq1_copy")
        self.assertEqual(code, 1)
        self.assertEqual(r["verdict"], "already-merged")
        self.assertEqual(len(r["merges"]), 1)
        self.assertEqual(r["commits"], {"total": 1, "on_main": 1})

    def test_a_multi_commit_squash_merged_pr_is_already_merged_despite_cherry_miss(self):
        base = self.git("rev-parse", "main")
        self.git("checkout", "-q", "-b", "sq2", base)
        lines = PARAGRAPH.splitlines(keepends=True)
        self.append("docs/notes.md", "\n" + lines[0])
        self.commit("sq2: first half")
        self.append("docs/notes.md", "".join(lines[1:]))
        self.commit("sq2: second half")
        self.git("checkout", "-q", "main")
        self.squash_merge(22, "sq2", "sq2: add the effect record section (#999)")
        # Unrelated change on main (separate file plus a non-conflicting edit at the top of notes.md).
        self.write("docs/other.md", "Another document that moves main without touching the squashed block.\n")
        self.write("docs/notes.md",
                   "# Notes (updated)\n\nAn introduction long enough to be counted as a line of its own here.\n"
                   "\n" + PARAGRAPH)
        self.commit("other work")
        code, [r] = self.run_check("22=sq2")
        self.assertEqual(code, 1)
        self.assertEqual(r["verdict"], "already-merged")
        self.assertEqual(len(r["merges"]), 1)
        self.assertEqual(r["commits"], {"total": 2, "on_main": 0})
        self.assertEqual(r["duplicates"], {})
        tcode, text = self.run_text("22=sq2")
        self.assertEqual(tcode, 1)
        self.assertIn("merged tree matches main", text)
        self.assertIn("merged before at", text)
        # Earlier issue number in the squash subject must not match as a prior merge.
        code_issue, [r_issue] = self.run_check("999=sq2")
        self.assertEqual(code_issue, 0)
        self.assertEqual(r_issue["merges"], [])

    def test_a_follow_up_commit_after_a_multi_commit_squash_merge_is_clean(self):
        base = self.git("rev-parse", "main")
        self.git("checkout", "-q", "-b", "sq3", base)
        lines = PARAGRAPH.splitlines(keepends=True)
        self.append("docs/notes.md", "\n" + lines[0])
        self.commit("sq3: first half")
        self.append("docs/notes.md", "".join(lines[1:]))
        self.commit("sq3: second half")
        self.git("checkout", "-q", "main")
        self.squash_merge(23, "sq3")
        self.git("checkout", "-q", "sq3")
        self.write("docs/followup.md", "A follow-up document pushed after the PR was squash-merged.\n")
        self.commit("sq3: follow-up")
        self.git("checkout", "-q", "main")
        code, [r] = self.run_check("23=sq3")
        self.assertEqual(code, 0)
        self.assertEqual(r["verdict"], "clean")
        self.assertEqual(len(r["merges"]), 1)
        self.assertEqual(r["commits"], {"total": 3, "on_main": 0})

    def test_a_revert_commit_is_not_counted_as_a_prior_merge(self):
        self.write("docs/other.md", "unrelated\n")
        self.commit('Revert "pr: something (#24)"')
        self.write("docs/other2.md", "unrelated2\n")
        self.commit("Revert pr: something (#24)")
        base = self.git("rev-parse", "main")
        self.pr_branch("fresh24", base, "\n" + PARAGRAPH)
        code, [r] = self.run_check("24=fresh24")
        self.assertEqual(code, 0)
        self.assertEqual(r["verdict"], "clean")
        self.assertEqual(r["merges"], [])

    def test_merged_pr_number_subject_parsing(self):
        self.assertEqual(mc._merged_pr_number("Merge pull request #7 from someone/first"), 7)
        self.assertEqual(mc._merged_pr_number("Merge pull request #7 from someone/fix (#99)"), 7)
        self.assertEqual(mc._merged_pr_number("docs: fix wording (#5515) (#5523)"), 5523)
        self.assertIsNone(mc._merged_pr_number('Revert "docs: fix wording (#5523)"'))
        self.assertIsNone(mc._merged_pr_number("Revert docs: fix wording (#5523)"))
        self.assertIsNone(mc._merged_pr_number("docs: see (#5523) for details"))
        self.assertIsNone(mc._merged_pr_number("docs: fix(#5523)"))

    def test_usage_errors(self):
        with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()), \
                mock.patch("sys.stderr", io.StringIO()):
            mc.main(["--repo", self.repo, "not-a-pair"])
        err = io.StringIO()
        with mock.patch("sys.stderr", err):
            self.assertEqual(mc.main(["--repo", self.repo, "--main", "main", "1=no-such-ref"]), 2)
        self.assertIn("merge_check:", err.getvalue())


if __name__ == "__main__":
    unittest.main()
