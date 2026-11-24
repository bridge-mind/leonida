"""The pull request template prompts for a `Depends on:` line and for whether the work was AI-assisted.

    python -I tests/test_docs_pr_template.py

The coordination issue builds its merge chains from `Depends on: #N` lines, and AGENTS.md asks a pull request to
say that it was AI-assisted; without a prompt in `.github/pull_request_template.md` those are the lines authors
leave out. This reads the template only, so rewording AGENTS.md or CONTRIBUTING.md cannot fail it. No game files.
"""
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(".github", "pull_request_template.md")
# A prompt is a line of its own that starts with the words an author fills in.
PROMPTS = ("Depends on:", "AI-assisted:")


def missing_prompts(template):
    """The prompts of PROMPTS that no line of `template` starts with."""
    return [p for p in PROMPTS if not re.search("^" + re.escape(p), template, re.M)]


class PullRequestTemplateTest(unittest.TestCase):
    def test_template_has_both_prompts(self):
        with open(os.path.join(ROOT, TEMPLATE), encoding="utf-8") as f:
            self.assertEqual(missing_prompts(f.read()), [])

    def test_a_template_without_a_prompt_is_reported(self):
        self.assertEqual(missing_prompts("### Tested\n"), ["Depends on:", "AI-assisted:"])
        # A mention inside another sentence is not a prompt: the line has to start with it.
        self.assertEqual(missing_prompts("no Depends on: here\nAI-assisted: no\n"), ["Depends on:"])


if __name__ == "__main__":
    unittest.main()
