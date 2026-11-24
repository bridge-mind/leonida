"""The synthetic compute shader metadata reaches the translator's workgroup declaration."""

from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime" / "test" / "shaders"))
sys.path.insert(0, str(ROOT / "tools"))

from compute_fixture import compute_header  # noqa: E402
from rdna2spirv import translate  # noqa: E402


class ComputeFixtureTests(unittest.TestCase):
    def test_group_tuple_whitespace_reaches_spirv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            disasm = root / "disasm"
            fssm = root / "fssm"
            disasm.mkdir()
            fssm.mkdir()
            for index, group in enumerate(("4,2,1", "4, 2, 1", "4 , 2,\t1")):
                with self.subTest(group=group):
                    shader = f"cs_group_{index}"
                    listing = disasm / f"{shader}.s"
                    listing.write_text(f"; group: {group}\n\ts_endpgm\n", encoding="utf-8", newline="\n")
                    (fssm / f"{shader}.fssm").write_bytes(compute_header(listing))
                    assembly = translate(str(listing))
                    local_size = next(line.strip() for line in assembly.splitlines()
                                      if line.lstrip().startswith("OpExecutionMode %main LocalSize "))
                    self.assertEqual(local_size, "OpExecutionMode %main LocalSize 4 2 1")


if __name__ == "__main__":
    unittest.main()
