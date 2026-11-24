"""Convert one exported cue_regions.json to per-channel Audacity label tracks."""

import argparse
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory


def convert(sheet, output):
    data = json.loads(sheet.read_text(encoding="utf-8"))
    if (not isinstance(data, dict) or data.get("sample_rate") != 48000
            or not isinstance(data.get("regions"), list)):
        raise ValueError("expected a 48 kHz cue sheet with a regions list")

    tracks = {}
    for ordinal, row in enumerate(data["regions"], 1):
        if not isinstance(row, dict):
            raise ValueError(f"invalid cue at row {ordinal}")
        channel = row.get("file")
        cue = row.get("name_hash")
        start = row.get("start_sample")
        end = row.get("end_sample")
        if (not isinstance(channel, str) or not re.fullmatch(r"[0-9A-F]{8}\.ogg", channel)
                or not isinstance(cue, str) or not re.fullmatch(r"[0-9A-F]{8}", cue)
                or type(start) is not int or type(end) is not int or not 0 <= start <= end <= 0xffffffff):
            raise ValueError(f"invalid channel, cue hash or sample interval at row {ordinal}")
        tracks.setdefault(channel, []).append(
            f"{start / 48000:.9f}\t{end / 48000:.9f}\t{cue} #{ordinal}\n")

    # Publish a whole bank together, including an empty bank.
    if output.exists():
        raise ValueError("choose a new output directory for this bank")
    with TemporaryDirectory(dir=output.parent) as tmp:
        staged = Path(tmp) / "labels"
        staged.mkdir()
        for channel, lines in tracks.items():
            (staged / Path(channel).with_suffix(".txt")).write_text(
                "".join(lines), encoding="utf-8", newline="\n")
        staged.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sheet", type=Path, help="one bank's exported cue_regions.json")
    parser.add_argument("output", type=Path, help="new directory for this bank's label tracks")
    args = parser.parse_args()
    try:
        convert(args.sheet, args.output)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
