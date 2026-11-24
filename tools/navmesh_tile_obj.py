"""Export one static navmesh tile's polygon boundaries to a private Wavefront OBJ.

--tile is the zero-based ordinal among complete navmesh resources in archive order,
including dynamic tiles. A dynamic or unreadable selection is an error. Groups
use zero-based polygon record numbers. Each record owns its OBJ vertices, so
coincident positions, repeated slots and winding survive without welding.
"""
import argparse
import os
import sys
import tempfile
from contextlib import closing
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from navmesh_nodes import navmesh_resources, tile_geometry
from report_paths import same_report_path


def write_obj(out, geometry, ordinal):
    out.write(f"o tile_{ordinal}\n")
    first = 1
    for record, (points, _, _) in enumerate(geometry["polygons"]):
        out.write(f"g polygon_{record}\n")
        if not points:
            out.write("# empty record\n")
            continue
        for x, y, z in points:
            out.write(f"v {x} {y} {z}\n")
        refs = list(range(first, first + len(points)))
        first += len(points)
        if len(points) >= 3:
            refs.append(refs[0])
        op = "p" if len(points) == 1 else "l"
        out.write(op + " " + " ".join(map(str, refs)) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("--tile", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True, help="local OBJ destination")
    args = parser.parse_args(argv)
    if args.tile < 0:
        parser.error("--tile must be non-negative")
    temp = None
    try:
        if same_report_path(args.archive, args.out):
            raise ValueError("--out aliases the archive")
        with closing(navmesh_resources(args.archive)) as resources:
            for ordinal, tile in enumerate(resources):
                if ordinal == args.tile:
                    geometry = tile_geometry(tile)
                    break
            else:
                raise ValueError("tile ordinal outside archive")
        if geometry is None:
            raise ValueError("selected tile is dynamic or unreadable")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=args.out.parent, delete=False) as out:
            temp = Path(out.name)
            write_obj(out, geometry, ordinal)
        os.replace(temp, args.out)
        temp = None
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
