"""DirectStorage BCPack bitstream inspection, decoding and profiling tool.

Provides CLI inspection, diagnostics, statistical frequency profiling and
decompression for Xbox Series BCPack streams (codecs BC1, BC3, BC4, BC5, BC6H, BC7).

Usage:
    python -I tools/bcpack_tool.py inspect <in.bin> --codec <1..7> --size <uncompressed_bytes> [--json]
    python -I tools/bcpack_tool.py decode <in.bin> <out.bin> --codec <1..7> --size <uncompressed_bytes> [--provisional]
    python -I tools/bcpack_tool.py profile <in.bin> --codec <1..7> --size <uncompressed_bytes>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bcpack
from report_paths import same_report_path


def inspect_stream(raw: bytes, codec: int, out_size: int) -> dict:
    """Analyze stream structure, compression ratio and per-lane statistics."""
    if codec not in bcpack.ALL_CODECS:
        raise ValueError(f"unsupported BCPack codec id: {codec}")

    block_size = bcpack.BLOCK_BYTES[codec]
    if out_size <= 0 or out_size % block_size != 0:
        raise ValueError(
            f"output size {out_size} is not a valid multiple of {bcpack.CODEC_NAMES[codec]} block size ({block_size})"
        )

    nblocks = out_size // block_size
    lanes, used = bcpack.decode_lanes(raw, codec, out_size)

    lane_widths = bcpack.LANE_WIDTHS[codec]
    lane_reports = []
    total_symbols = 0

    for idx, lane in enumerate(lanes):
        count = len(lane)
        total_symbols += count
        w = lane_widths[idx]
        zeros = lane.count(0)
        min_val = min(lane) if count else 0
        max_val = max(lane) if count else 0
        avg_val = (sum(lane) / count) if count else 0.0

        lane_reports.append({
            "lane": idx,
            "bit_width": w,
            "symbols": count,
            "zeros": zeros,
            "zero_percent": round(100.0 * zeros / count, 2) if count else 0.0,
            "min": min_val,
            "max": max_val,
            "avg": round(avg_val, 2),
        })

    trailing_bytes = len(raw) - used
    ratio = round(len(raw) / out_size, 4) if out_size else 0.0

    return {
        "codec": bcpack.CODEC_NAMES[codec],
        "codec_id": codec,
        "compressed_bytes": len(raw),
        "bytes_used": used,
        "trailing_zeros": trailing_bytes,
        "uncompressed_bytes": out_size,
        "blocks": nblocks,
        "compression_ratio": ratio,
        "total_symbols": total_symbols,
        "lanes": lane_reports,
    }


def profile_stream(raw: bytes, codec: int, out_size: int) -> dict:
    """Generate symbol frequency distribution profile across all lanes."""
    info = inspect_stream(raw, codec, out_size)
    lanes, _ = bcpack.decode_lanes(raw, codec, out_size)

    histograms = []
    for idx, lane in enumerate(lanes):
        counts: dict[int, int] = {}
        for sym in lane:
            counts[sym] = counts.get(sym, 0) + 1
        sorted_top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
        histograms.append({
            "lane": idx,
            "unique_symbols": len(counts),
            "top_symbols": [{"sym": s, "count": c} for s, c in sorted_top],
        })

    info["histograms"] = histograms
    return info


def decode_file(
    in_path: str,
    out_path: str,
    codec: int,
    out_size: int,
    allow_provisional: bool = False,
    history: str | None = None,
) -> int:
    """Decode compressed file and write decompressed blocks to disk."""
    if same_report_path(in_path, out_path):
        raise ValueError(
            f"output path '{out_path}' names the compressed input '{in_path}'; "
            f"refusing to overwrite it"
        )

    with open(in_path, "rb") as f:
        raw = f.read()

    try:
        data, prov = bcpack.decode_with_info(
            raw, codec, out_size, allow_provisional=allow_provisional, history=history
        )
    except bcpack.BcpackError:
        raise  # stream failures are already attributed to the stream, never to --history
    except ValueError as e:
        # json.JSONDecodeError from a --history file carries no filename; attribute it,
        # but do not double-prefix messages that already name the history path.
        if history is not None and os.path.basename(str(history)) not in str(e):
            raise ValueError(f"{history}: {e}") from e
        raise

    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(out_path, "wb") as f:
        f.write(data)

    return len(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="DirectStorage BCPack bitstream inspection, decoding and profiling tool."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # inspect
    p_insp = sub.add_parser("inspect", help="Inspect BCPack stream structure and lane statistics")
    p_insp.add_argument("file", help="Path to raw BCPack compressed file")
    p_insp.add_argument(
        "--codec",
        type=int,
        required=True,
        choices=sorted(bcpack.ALL_CODECS),
        help="BCPack codec ID (1=BC1, 3=BC3, 4=BC4, 5=BC5, 6=BC6H, 7=BC7)",
    )
    p_insp.add_argument(
        "--size",
        type=int,
        required=True,
        help="Expected uncompressed size in bytes",
    )
    p_insp.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output",
    )

    # profile
    p_prof = sub.add_parser("profile", help="Generate detailed symbol frequency distribution")
    p_prof.add_argument("file", help="Path to raw BCPack compressed file")
    p_prof.add_argument(
        "--codec",
        type=int,
        required=True,
        choices=sorted(bcpack.ALL_CODECS),
        help="BCPack codec ID",
    )
    p_prof.add_argument(
        "--size",
        type=int,
        required=True,
        help="Expected uncompressed size in bytes",
    )

    # decode
    p_dec = sub.add_parser("decode", help="Decompress BCPack file to raw BCn blocks")
    p_dec.add_argument("in_file", help="Path to input BCPack compressed file")
    p_dec.add_argument("out_file", help="Path to output decompressed file")
    p_dec.add_argument(
        "--codec",
        type=int,
        required=True,
        choices=sorted(bcpack.ALL_CODECS),
        help="BCPack codec ID",
    )
    p_dec.add_argument(
        "--size",
        type=int,
        required=True,
        help="Expected uncompressed size in bytes",
    )
    p_dec.add_argument(
        "--provisional",
        action="store_true",
        help="Allow provisional history and signatures",
    )
    p_dec.add_argument(
        "--history",
        type=str,
        default=None,
        help="Path to optional BC6H/BC7 history JSON or directory",
    )

    args = parser.parse_args(argv)

    # CLI-level guard: a missing/unreadable input or history file (OSError), a stream that fails
    # structural or rANS validation (bcpack.BcpackError, a ValueError), or a malformed --history
    # JSON (ValueError) must be one named line on stderr with exit status 2, not a traceback.
    # bcpack.py's library-level behavior is unchanged.
    src = args.in_file if args.command == "decode" else args.file
    try:
        return _run(args)
    except (ValueError, OSError) as e:
        print(f"ERROR: {src}: {e}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace) -> int:
    if args.command == "inspect":
        with open(args.file, "rb") as f:
            raw = f.read()
        info = inspect_stream(raw, args.codec, args.size)
        if args.json:
            print(json.dumps(info, indent=2))
        else:
            print(
                f"BCPack {info['codec']} stream: {info['compressed_bytes']} bytes -> "
                f"{info['uncompressed_bytes']} bytes ({info['blocks']} blocks, ratio {info['compression_ratio']:.3f})"
            )
            print(f"Lanes ({len(info['lanes'])} active):")
            for l in info['lanes']:
                print(
                    f"  lane {l['lane']:2d} (w={l['bit_width']:2d}b): {l['symbols']} syms, "
                    f"{l['zeros']} zeros ({l['zero_percent']}%), min {l['min']}, max {l['max']}, avg {l['avg']}"
                )
        return 0

    elif args.command == "profile":
        with open(args.file, "rb") as f:
            raw = f.read()
        prof = profile_stream(raw, args.codec, args.size)
        print(json.dumps(prof, indent=2))
        return 0

    elif args.command == "decode":
        written = decode_file(
            args.in_file,
            args.out_file,
            args.codec,
            args.size,
            allow_provisional=args.provisional,
            history=args.history,
        )
        print(f"Successfully decoded {written} bytes to {args.out_file}")
        return 0

    return 1

if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
