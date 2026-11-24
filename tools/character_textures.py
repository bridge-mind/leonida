"""Classify the texture bindings of decoded character ``model.json`` materials.

The five slot hashes below were measured on the retail package and named by the
case-sensitive joaat of the parameter name. They are the only slots this tool
resolves; anything else is reported as ``unknown 0x...`` rather than guessed:

    | slot         | role          | meaning                                  |
    | 0xC8E8C282   | diffusetex    | sRGB albedo (the only albedo slot)        |
    | 0xC6F46E79   | speculartex   | specular/gloss pack, not colour           |
    | 0x1E77BF4C   | bumptex       | tangent-space normal/detail normal        |
    | 0x739E3D0D   | speculartex2  | packed ORM/mask                           |
    | 0x8ADB8480   | materialid    | ID/palette map                            |

The garment shaders ``d5ed50f4``, ``ea8d09d0`` and ``39ddb9ca`` declare no
``0xC8E8C282`` slot at all. A material on one of those shaders with no albedo
binding is flagged ``garment-no-albedo``: a renderer must keep it clay instead
of treating ``speculartex`` as colour.

Input is one or more decoded model JSON files, or a directory searched
recursively for ``model.json``. The decoder's capitalised keys are read, with
lowercase equivalents accepted for hand-written files:

    {"Name": "player_head", "Materials": [
      {"Name": "head", "Shader": "a1b2c3d4", "Textures": [
        {"Param": "C8E8C282", "Name": "face_diffusetex"},
        {"Param": "1E77BF4C", "Name": "face_bumptex"}]}]}

A top-level ``Items``/``Models`` list is also accepted, so a model manifest can
be passed directly. When the model has no name, the ``File`` stem accepts either
slash convention regardless of the reporting host. The texture column keeps its
display name or hash. A separate ``asset_key`` column carries the 16-digit
resource identity from ``Id64`` or ``AssetKey``.

    python -I tools/character_textures.py <model.json|dir> [...] [--json FILE] [--csv FILE]

``--json`` writes the full per-material report and ``--csv`` one row per
binding; neither may overwrite an input. Standard library only.
"""
import argparse
import csv
import json
import ntpath
import os
import sys

# -I keeps the script directory off sys.path, so add it explicitly before importing the sibling module.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_paths import path_key, same_report_path  # noqa: E402

SLOT_ROLES = {
    "C8E8C282": "diffusetex",
    "C6F46E79": "speculartex",
    "1E77BF4C": "bumptex",
    "739E3D0D": "speculartex2",
    "8ADB8480": "materialid",
}
ALBEDO_SLOT = "C8E8C282"
GARMENT_SHADERS = ("d5ed50f4", "ea8d09d0", "39ddb9ca")

MODEL_LISTS = ("Items", "items", "Models", "models")
MATERIAL_LISTS = ("Materials", "materials")
TEXTURE_LISTS = ("Textures", "textures", "Bindings", "bindings", "TextureBindings", "texture_bindings")
MODEL_NAME_KEYS = ("Name", "name", "ModelName", "model_name", "Model", "model")
MODEL_FILE_KEYS = ("File", "file")
MATERIAL_NAME_KEYS = ("Name", "name", "MaterialName", "material_name")
SHADER_KEYS = ("Shader", "shader", "ShaderName", "shader_name", "ShaderHash", "shader_hash")
PARAM_KEYS = ("Param", "param", "Slot", "slot", "Parameter", "parameter", "Hash", "hash")
ASSET_KEY_FIELDS = ("Id64", "id64", "AssetKey", "asset_key")
TEXTURE_NAME_KEYS = ("Name", "name", "Texture", "texture", "Key", "key") + ASSET_KEY_FIELDS

CSV_FIELDS = ("source", "model", "material", "shader", "slot", "role", "texture", "flags", "asset_key")


def _first(mapping: dict, keys: tuple):
    """The first present, non-empty value for `keys`; 0 and False are values, not gaps."""
    for key in keys:
        if key not in mapping:
            continue
        value = mapping[key]
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _one_line(text: object) -> str:
    return " ".join(str(text).split())


def _hex_field(value: object, digits: int) -> str | None:
    """Normalise an unsigned field to its declared width, or None if malformed."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return f"{value:0{digits}X}" if 0 <= value < 16 ** digits else None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text[:2].lower() == "0x":
        text = text[2:]
    if len(text) == digits and all(ch in "0123456789abcdefABCDEF" for ch in text):
        return text.upper()
    return None


def _shader_id(value: object) -> str | None:
    """Shader hash as lowercase text for display and garment matching."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return f"{value:08x}" if 0 <= value <= 0xFFFFFFFF else str(value)
    text = str(value).strip()
    return text.lower() if text else None


def _texture_name(value: object) -> str:
    text = _one_line(value)
    return text or "-"


def model_name(model: dict, path: str) -> str:
    """Name from the model, else its File stem, else the containing folder for a plain model.json."""
    explicit = _first(model, MODEL_NAME_KEYS)
    if explicit is not None:
        return _one_line(explicit)
    file_name = _first(model, MODEL_FILE_KEYS)
    if isinstance(file_name, str) and file_name.strip():
        return _one_line(ntpath.splitext(ntpath.basename(file_name.strip()))[0])
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.lower() == "model":
        parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
        if parent:
            return parent
    return stem or "-"


def models_in(document: object) -> list[dict]:
    """The model objects in a decoded file, accepting a single model, a list, or a manifest."""
    if isinstance(document, list):
        return [item for item in document if isinstance(item, dict)]
    if not isinstance(document, dict):
        raise ValueError("top level is not a JSON object or list")
    for key in MODEL_LISTS:
        value = document.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    if any(isinstance(document.get(key), list) for key in MATERIAL_LISTS):
        return [document]
    raise ValueError("no Materials or Items list")


def texture_bindings(material: dict) -> list:
    """The material's binding list, also accepting a {slot: name} mapping."""
    for key in TEXTURE_LISTS:
        value = material.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            bindings = []
            for slot, entry in value.items():
                if isinstance(entry, dict):
                    binding = dict(entry)
                    binding.setdefault("Param", slot)
                else:
                    binding = {"Param": slot, "Name": entry}
                bindings.append(binding)
            return bindings
    return []


def classify_material(material: dict, index: int) -> dict:
    """Classify one decoded material."""
    name = _one_line(_first(material, MATERIAL_NAME_KEYS) or f"material[{index}]")
    shader = _shader_id(_first(material, SHADER_KEYS))
    bindings = []
    for reference in texture_bindings(material):
        if isinstance(reference, dict):
            slot = _hex_field(_first(reference, PARAM_KEYS), 8)
            texture = _texture_name(_first(reference, TEXTURE_NAME_KEYS))
            asset_key = _hex_field(_first(reference, ASSET_KEY_FIELDS), 16)
        else:
            slot, texture, asset_key = None, _texture_name(reference), None
        if slot is None:
            label, role = "-", "unknown"
        else:
            label = f"0x{slot}"
            role = SLOT_ROLES.get(slot, f"unknown {label}")
        bindings.append({"slot": label, "role": role, "texture": texture, "asset_key": asset_key})

    missing_albedo = not any(binding["slot"] == f"0x{ALBEDO_SLOT}" for binding in bindings)
    garment = shader is not None and (shader[2:] if shader.startswith("0x") else shader) in GARMENT_SHADERS
    flags = ["garment-no-albedo"] if garment and missing_albedo else []
    return {
        "name": name,
        "shader": shader or "-",
        "garment": garment,
        "missing_albedo": missing_albedo,
        "flags": flags,
        "bindings": bindings,
    }


def classify_file(path: str) -> dict:
    """Read one model file and classify every material in it."""
    try:
        with open(path, encoding="utf-8-sig") as f:
            document = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: {exc.msg} at line {exc.lineno} column {exc.colno}") from None
    return classify_document(path, document)


def classify_document(path: str, document: object) -> dict:
    models = models_in(document)
    if not models:
        raise ValueError(f"{path}: no models")
    report_models = []
    for model in models:
        materials = []
        material_list = []
        for key in MATERIAL_LISTS:
            if isinstance(model.get(key), list):
                material_list = model[key]
                break
        for index, material in enumerate(material_list):
            if not isinstance(material, dict):
                continue
            materials.append(classify_material(material, index))
        report_models.append({"name": model_name(model, path), "source": path, "materials": materials})
    return {"models": report_models}


def model_files(targets: list) -> list[str]:
    """Resolve explicit files and recurse into directories for model.json, de-duplicated."""
    def scan_failed(exc: OSError) -> None:
        raise exc

    found = []
    for target in targets:
        if os.path.isdir(target):
            matches = sorted(os.path.join(root, name)
                             for root, _, names in os.walk(target, onerror=scan_failed) for name in names
                             if name.lower() == "model.json")
            if not matches:
                raise ValueError(f"no model.json under {target}")
            found.extend(matches)
        elif os.path.isfile(target):
            found.append(target)
        else:
            raise FileNotFoundError(target)
    unique = []
    seen = set()
    for path in found:
        key = path_key(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def build_report(sources: list) -> dict:
    """Classify every source and add the summary and warning lines."""
    report = {"models": []}
    warnings = []
    material_count = 0
    binding_count = 0
    unknown_count = 0
    flagged = 0
    for path in sources:
        file_report = classify_file(path)
        for model in file_report["models"]:
            for material in model["materials"]:
                material_count += 1
                binding_count += len(material["bindings"])
                unknown_count += sum(1 for b in material["bindings"] if b["role"].startswith("unknown 0x"))
                if material["flags"]:
                    flagged += 1
                    warnings.append(
                        f"garment-no-albedo {model['name']} / {material['name']} ({material['shader']}): "
                        f"no 0x{ALBEDO_SLOT} binding; speculartex is not colour")
        report["models"].extend(file_report["models"])
    report["summary"] = {
        "models": len(report["models"]),
        "materials": material_count,
        "bindings": binding_count,
        "unknown_slots": unknown_count,
        "garment_without_albedo": flagged,
    }
    report["warnings"] = warnings
    return report


def report_rows(report: dict) -> list[dict]:
    """Flatten the report to one row per binding (or one placeholder row per empty material)."""
    rows = []
    for model in report["models"]:
        for material in model["materials"]:
            bindings = material["bindings"] or [{"slot": "-", "role": "no-bindings", "texture": "-"}]
            for binding in bindings:
                rows.append({
                    "source": model["source"],
                    "model": model["name"],
                    "material": material["name"],
                    "shader": material["shader"],
                    "slot": binding["slot"],
                    "role": binding["role"],
                    "texture": binding["texture"],
                    "flags": ";".join(material["flags"]),
                    "asset_key": binding.get("asset_key"),
                })
    return rows


def print_report(report: dict) -> None:
    print("model\tmaterial\tslot\trole\ttexture\tasset_key")
    for row in report_rows(report):
        print(*(_one_line(row[key]) for key in ("model", "material", "slot", "role", "texture")),
              row["asset_key"] or "-", sep="\t")
    for warning in report["warnings"]:
        print(f"FLAG {warning}")
    summary = report["summary"]
    print(f'{summary["models"]} models, {summary["materials"]} materials, '
          f'{summary["bindings"]} bindings, {summary["unknown_slots"]} unknown slots, '
          f'{summary["garment_without_albedo"]} garment materials without albedo')


def _make_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_csv(path: str, rows: list) -> None:
    _make_parent(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str, report: dict, rows: list) -> None:
    _make_parent(path)
    public = {"summary": report["summary"], "warnings": report["warnings"],
              "models": report["models"], "rows": rows}
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(public, f, indent=2)
        f.write("\n")


def check_report_path(parser: argparse.ArgumentParser, flag: str, path: str, sources: list) -> None:
    if os.path.isdir(path):
        parser.error(f"{flag} is a directory: {path}")
    for source in sources:
        if same_report_path(path, source):
            parser.error(f"{flag} would overwrite the input {source}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("targets", nargs="+", help="decoded model.json file(s), or a directory searched recursively")
    parser.add_argument("--json", metavar="FILE", help="write the full per-material report as JSON")
    parser.add_argument("--csv", metavar="FILE", help="write one row per texture binding as CSV")
    args = parser.parse_args()

    if args.json and args.csv and same_report_path(args.json, args.csv):
        parser.error("--json and --csv must use different paths")
    try:
        sources = model_files(args.targets)
    except FileNotFoundError as exc:
        print(f"not found: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    for flag, path in (("--json", args.json), ("--csv", args.csv)):
        if path:
            check_report_path(parser, flag, path, sources)

    try:
        report = build_report(sources)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    rows = report_rows(report)
    print_report(report)
    for path, writer, values in ((args.json, write_json, (report, rows)),
                                 (args.csv, write_csv, (rows,))):
        if path:
            try:
                writer(path, *values)
            except OSError as exc:
                print(f"ERROR: cannot write {path}: {exc.strerror or exc}", file=sys.stderr)
                return 2
            print("wrote", path)
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
