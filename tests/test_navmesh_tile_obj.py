"""Read back OBJ primitives from a synthetic archive, independently of the writer."""
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_navmesh_decode import build_navmesh, make_rpf, nd

TOOL = HERE.parent / "tools" / "navmesh_tile_obj.py"


def read_obj(path):
    vertices = []
    groups = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        op, *args = line.split()
        if op == "v":
            vertices.append(tuple(map(float, args)))
        elif op == "g":
            name, = args
            assert name not in groups
            groups[name] = []
        elif op in ("p", "l"):
            refs = list(map(int, args))
            assert all(1 <= i <= len(vertices) for i in refs)
            groups[name].append((op, [vertices[i - 1] for i in refs]))
        elif op not in ("#", "o"):
            raise AssertionError(op)
    return groups


class ExportTest(unittest.TestCase):
    def test_archive_selection_and_record_round_trip(self):
        sizes = (3, 1, 0, 2, 4, 1)
        head, tail = build_navmesh(n_verts=3, poly_sizes=sizes, tiny_bm_cont=True)
        head = bytearray(head)
        (at, _), = nd.split_array(head, 0x70, 6, 3)
        struct.pack_into("<9H", head, at, 0, 0, 0, 65535, 0, 0, 0, 65535, 65535)
        a, b, c = (-768, 512, 0), (-640, 512, 0), (-768, 640, 16)
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "input.rpf"
            out = Path(directory) / "tile.obj"
            make_rpf(archive, [*build_navmesh(flags=5), head, tail, *build_navmesh(corrupt_vidx=True)])

            def run(tile, dest):
                return subprocess.run([sys.executable, "-I", TOOL, archive, "--tile", str(tile),
                                       "--out", dest], capture_output=True)

            result = run(1, out)
            self.assertEqual(result.returncode, 0, result.stderr)
            expected = [[("l", [a, b, c, a])], [("p", [a])], [], [("l", [b, c])],
                        [("l", [a, b, c, a, a])], [("p", [b])]]
            groups = read_obj(out)
            self.assertEqual(list(groups), [f"polygon_{i}" for i in range(len(sizes))])
            self.assertEqual(list(groups.values()), expected)
            original = out.read_bytes()
            for invalid in (-1, 0, 2, 3):
                self.assertEqual(run(invalid, out).returncode, 2)
                self.assertEqual(out.read_bytes(), original)
            self.assertEqual(run(1, archive).returncode, 2)


if __name__ == "__main__":
    unittest.main()
