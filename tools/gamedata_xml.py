"""Catalog the XML documents recovered from a carved RPF9 archive.

RPF9 stores a logical XML document as one or more zlib streams. A carve from ``rpf9_carve.py`` has an
``index.json`` plus one inflated ``<offset>.bin`` per stream, so ``gamedata_index.py`` can group the streams
back into payloads. This tool parses those payloads and writes an aggregate catalog:

    python -I gamedata_xml.py <carve_dir> [--csv FILE] [--json FILE] [--check]
    python -I gamedata_xml.py <file.xml> [<file.xml> ...] [--csv FILE] [--json FILE] [--check]

For every file it records the root element, element count, maximum depth, attribute names, ``$(name)`` and
``platform:/`` references, and include-like structures. For every root it records the file count, category,
element-name histogram, attribute histogram and reference targets. ``--check`` validates well-formedness and
separates rootless fragments, tail-only documents and malformed documents.
Well-formed documents use the parser's expanded root name (including its XML namespace), just as
the element histogram does. A raw tag hint is only needed when recovering incomplete documents.

Only schema names, element/attribute names and reference tokens are written. Element text, attribute values,
``type``/``content`` values and entry paths are not. For a carve directory a row is identified by offset; an
individual file is identified by the basename that was passed. The output is a local work product: it must not
be committed or attached to an issue. Report paths are refused before the scan when they alias an input: a
nested XML in directory mode, or a carve ``index.json`` or stream. The test fixtures are synthetic and contain
no game data.
Usage: `python -I tools/gamedata_xml.py --help` lists the options.
"""
import argparse
import csv
import io
import json
import os
import re
import stat
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from xml.parsers.expat import errors

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import gamedata_index as gi  # noqa: E402
from report_paths import same_report_path  # noqa: E402

BOM = b"\xef\xbb\xbf"
MACRO_REF = re.compile(r"\$\(([A-Za-z_][\w.-]*)\)")
PLATFORM_REF = re.compile(r"platform:/([A-Za-z0-9_.-]+)?")
XI_INCLUDE = "{http://www.w3.org/2001/XInclude}include"
INCLUDE_ELEMENTS = ("fileList", "filename", "includedSets", "includedChangesets",
                    "changesetIncludeFlags", "loadPath")

# Category names and root membership follow the functional table in docs/gamedata.md. A category is a
# dispatch hint: the same root always belongs to one bucket, but a bucket is not a complete schema. Where a
# document straddles two labels, the table's bucket is kept so the file and root counts line up.
CATEGORY_ROOTS = {
    "time, weather, sky, water": """
        timecycle_modifier_data timecycle_keyframe_data CContentsOfWeatherXmlFile STS_Config CustomClouds
        CScatteringTransmittance__FogVolumeListData FogVolumeProfileList OceanSimPresets
        OceanWaveCalmingCurves EnvironmentalWaterDisturbManager ShoreWavePlacementData time_data
        RegionTintMap
    """,
    "vehicles, tyres, handling data": """
        CVehiclePoints CVehicleBoneOffsetsArray CVehiclePerformanceMaps CVehicleBaseRatingsAndEstimatesMap
        CVehicleComponentSetManager CVehicleDistantLightData CVehiclePointsMeta CTyreDataSet CParkingSplines
        CSnowTrackData CBreakableVehicleDoorLockManagerData CPaletteIndexedRecipeTintsArray
    """,
    "peds, schedules, appearance": """
        CPedAccuracyManager CPedComponentSetManager CPedFormationInfoManager CPedHeadbobSettingsManager
        CPedInventoryLoadOutManager CPedMoodConditionTreesManager CPedMoodSettingsManager
        CPedMotivationInfoManager CPedScheduleInfoManager CPedSpeedDataBlock CPedTimeWarpSettingsManager
        CPedVariationBodyTypeManager CPairedPedDataBlock CEnergyConfigInfos CIdlePreferenceManager
        CUpperBodyIdleConditionManager CUpperBodyIdleSettingsManager CHeadbobConditionManager
        CHealthConfigInfo sPedDamageModifiers sPedDistractionModifiers WalkAndTalkProfileData
        STPedMotionPredictionSetContainer STPedNavigationPredictiontPresetContainer CBaitingInfo
        CAnimalAnimationContexts CAnimalAttackBehaviours Relationships__GroupsData CMoodConditionManager
    """,
    "tuning, parameters, effects": """
        CTuningFile CCoverProcessorTuning GridAtlasInfo PatchConstants CClimbTuneMgr CoverShadowTunables
        LostTargetSearch__Tunables sCollisionEventScannerTuning hrsSimTuneMgr rage__fragTuneStruct
        DiffractionProfiles GlintsProfiles LightAnimationPresets CGraphicsDefinitionsData
    """,
    "mount / content manifests": """
        CDataFileMgr__ContentChangeSetDef CDataFileMgr__ContentDef
    """,
    "world, navigation, map": """
        GameShape__Data__LocationNameLoopkup PathingStyleData CTrainConfigs CHazardNodes
        MapState__Data__Templates aiNavIMapGroupData CLadderMetadataManager CLadderRailsMetadataManager
        CDoorInteractionDataList CEntityComponentLodManagerData CGarageInitDataList
        CSceneTerrainAnalyzerMgrDataInstance CScenicViewManager MaterialDataList TextureGridInfo
        Sprinklers__RootData CProceduralInfo CFluidVesselManager CObjectComponentSetManager
    """,
    "combat and AI behaviour": """
        CCombatInfoMgr CFiringPatternInfoManager CFleeDecisionMakerData Combat__GoalData__Store
        CSearchPointOfInterestRuleSet CTacticalAnalysisDistributionStore
        Modular__SM__AIBehaviour__LockControllerConfig Witnesses__Informations CTaskDataInfoManager
        CDrivingStyleCollection CLassoDataManager CAimUpperBodyDataBlock SSetupData
    """,
    "animation, poses, weapons": """
        WeaponAnimations__FullbodyControllerData WeaponAnimations__ConfigData CStandardPosesManager
        CAnimSceneAttachSlotSetManager StudioConfigurationData rage__fwRetargetCoordinatorConfig
    """,
    "damage": "CDamageDataManager",
    "audio, video, speech": """
        LandingPageVideos CTVPlaylistContainer naSpeechEventTypeBatchRequestParams rage__MediaDecoderSettings
    """,
    "missions, scenarios, respawn": "CDynamicScenarioManager CMissions Respawn__MetaData",
    "conversation": "CConversationList CConversationDictionary",
}
CATEGORY_BY_ROOT = {}
for _category, _roots in CATEGORY_ROOTS.items():
    for _root in _roots.split():
        CATEGORY_BY_ROOT[_root] = _category
UNCATALOGUED = "uncatalogued"


def category_for(root: str) -> str:
    return CATEGORY_BY_ROOT.get(root, UNCATALOGUED)


def scan_refs(text: str, report: dict) -> None:
    """Count symbolic macros and platform prefixes in one text or attribute value."""
    if not text:
        return
    if "$(" in text:
        for match in MACRO_REF.finditer(text):
            report["refs"]["$(" + match.group(1) + ")"] += 1
    if "platform:/" in text:
        for match in PLATFORM_REF.finditer(text):
            # Keep the bucket coarse: the first segment after a platform prefix can be a directory or a
            # file-like archive name, and only levels/other is needed to route a mount.
            report["platform"]["levels" if (match.group(1) or "").lower() == "levels" else "other"] += 1
            report["platform_total"] += 1


def scan_xml(data: bytes) -> dict:
    """Parse one document and return aggregate facts. Never keeps element text or values."""
    report = {
        "root": "",
        "well_formed": False,
        "mismatched_tag": False,
        "error": "",
        "element_count": 0,
        "max_depth": 0,
        "attributes": [],
        "comments": 0,
        "elements": Counter(),
        "attrs": Counter(),
        "refs": Counter(),
        "platform": Counter(),
        "platform_total": 0,
        "filename": 0,
        "parent_path": 0,
        "parent_file": 0,
        "xi_include": 0,
        "type_attrs": 0,
        "type_values": set(),
        "content_values": set(),
    }
    depth = 0
    ended = None
    try:
        parser = ET.iterparse(io.BytesIO(data), events=("start", "end", "comment"))
        for event, elem in parser:
            if event == "comment":
                report["comments"] += 1
                continue
            # A closing tag can be read before its tail. The next start or end event finalizes it.
            if ended is not None:
                scan_refs(ended.tail, report)
                ended.clear()
                ended = None
            if event == "start":
                depth += 1
                report["max_depth"] = max(report["max_depth"], depth)
                if not report["root"]:
                    report["root"] = elem.tag
                report["element_count"] += 1
                report["elements"][elem.tag] += 1
                for name, value in elem.attrib.items():
                    report["attrs"][name] += 1
                    if name == "type":
                        report["type_attrs"] += 1
                        report["type_values"].add(value)
                    elif name == "content":
                        report["content_values"].add(value)
                    elif name == "parentPath":
                        report["parent_path"] += 1
                    elif name == "parentFile":
                        report["parent_file"] += 1
                    scan_refs(value, report)
                if elem.tag == "filename":
                    report["filename"] += 1
                if elem.tag == XI_INCLUDE or elem.tag == "xi:include":
                    report["xi_include"] += 1
            else:
                scan_refs(elem.text, report)
                depth -= 1
                ended = elem
        report["well_formed"] = True
    except (ET.ParseError, LookupError, ValueError) as exc:
        # The parser also raises LookupError or ValueError for unsupported encoding declarations.
        report["error"] = str(exc)
        report["mismatched_tag"] = (isinstance(exc, ET.ParseError)
                                    and exc.code == errors.codes[errors.XML_ERROR_TAG_MISMATCH])
        # The parser stops at the error, so the comments after it would go uncounted (a truncated or tail-only
        # document): count them in the bytes instead, respecting the surrounding markup.
        report["comments"] = raw_comment_count(data)
    if ended is not None:
        scan_refs(ended.tail, report)
    report["attributes"] = sorted(report["attrs"])
    return report


COMMENT_MARKUP = re.compile(
    rb"(?P<comment><!--).*?(?:-->|\Z)"
    rb"|<!\[CDATA\[.*?(?:\]\]>|\Z)"
    rb"|<\?.*?(?:\?>|\Z)",
    re.S,
)


def raw_comment_count(data: bytes) -> int:
    """Count comment starts in malformed XML, consuming comments, CDATA and PIs through their terminator or EOF."""
    return sum(match.lastgroup == "comment" for match in COMMENT_MARKUP.finditer(data))


def looks_truncated(data: bytes, root: str) -> bool:
    """True when the bytes start inside a document instead of at its root element."""
    if not root:
        return False
    body = data[3:] if data.startswith(BOM) else data
    body = body.lstrip(b" \t\r\n")
    if body.startswith(b"<?xml"):
        return False
    return gi.xml_root(data) != root


def file_row(doc: dict, scan: dict) -> dict:
    # A truncated tail can begin with an inner element, so the index's closing-root hint wins.
    root = doc.get("root_hint") or scan["root"]
    header = doc.get("header")
    if header is None:
        header = gi.has_xml_decl(doc["data"])
    well_formed = scan["well_formed"]
    if well_formed:
        root = scan["root"]  # the parsed name includes its namespace; a recovery hint need not
        tail_only = fragment = False
    elif doc.get("from_carve"):
        tail_only = not header
        fragment = not root or not doc.get("complete", True)
    elif scan["mismatched_tag"]:
        # A cut inside a comment or CDATA changes the lexical meaning of the
        # retained tags. Accept that explanation only if restoring the opener
        # produces a complete document; a terminator alone is not evidence.
        repaired_root = ""
        data = doc["data"]
        closing_root = gi.tail_root(data)
        if not header and closing_root:
            body = data[3:] if data.startswith(BOM) else data
            for opener, terminator in ((b"<!--", b"-->"), (b"<![CDATA[", b"]]>")):
                if terminator not in body:
                    continue
                try:
                    ET.fromstring(b"<" + closing_root.encode("ascii") + b">" + opener + body)
                except (ET.ParseError, LookupError, ValueError):
                    continue
                repaired_root = closing_root
                break
        root = repaired_root or gi.xml_root(data) or scan["root"]
        tail_only, fragment = bool(repaired_root), False
    else:
        tail_only = looks_truncated(doc["data"], root)
        fragment = not root
    malformed = not well_formed and not tail_only and not fragment
    row = {
        "source": doc.get("name", ""),
        "offset": doc.get("offset"),
        "offset_hex": doc.get("offset_hex", ""),
        "bytes": len(doc["data"]),
        "chunks": doc.get("chunks", 1),
        "root": root,
        "category": category_for(root) if root else UNCATALOGUED,
        "header": bool(header),
        "complete": well_formed or doc.get("complete", False),
        "well_formed": well_formed,
        "fragment": fragment,
        "tail_only": tail_only and not fragment,
        "truncated": tail_only and not fragment and not scan["well_formed"],
        "malformed": malformed,
        "error": scan["error"],
        "element_count": scan["element_count"] if scan["well_formed"] else None,
        "max_depth": scan["max_depth"] if scan["well_formed"] else None,
        "comments": scan["comments"],
        "attributes": scan["attributes"],
        "refs": dict(sorted(scan["refs"].items())),
        "platform": dict(sorted(scan["platform"].items())),
        "platform_total": scan["platform_total"],
        "filename": scan["filename"],
        "parent_path": scan["parent_path"],
        "parent_file": scan["parent_file"],
        "xi_include": scan["xi_include"],
        "type_attrs": scan["type_attrs"],
        "type_values": len(scan["type_values"]),
        "content_values": len(scan["content_values"]),
    }
    row["_scan"] = scan
    return row


def new_root_stats(root: str) -> dict:
    return {
        "root": root,
        "category": category_for(root) if root else UNCATALOGUED,
        "files": 0,
        "well_formed": 0,
        "fragments": 0,
        "tail_only": 0,
        "truncated": 0,
        "malformed": 0,
        "bytes": 0,
        "elements": 0,
        "max_depth": 0,
        "element_names": Counter(),
        "attributes": Counter(),
        "refs": Counter(),
        "platform": Counter(),
        "platform_total": 0,
        "filename": 0,
        "parent_path": 0,
        "parent_file": 0,
        "xi_include": 0,
        "type_attrs": 0,
        "type_values": set(),
        "content_values": set(),
    }


def aggregate(rows: list) -> dict:
    roots = {}
    for row in rows:
        stats = roots.setdefault(row["root"], new_root_stats(row["root"]))
        stats["files"] += 1
        stats["bytes"] += row["bytes"]
        stats["fragments"] += int(row["fragment"])
        stats["tail_only"] += int(row["tail_only"])
        stats["truncated"] += int(row["truncated"])
        stats["malformed"] += int(row["malformed"])
        if not row["well_formed"]:
            continue
        scan = row["_scan"]
        stats["well_formed"] += 1
        stats["elements"] += scan["element_count"]
        stats["max_depth"] = max(stats["max_depth"], scan["max_depth"])
        stats["element_names"].update(scan["elements"])
        stats["attributes"].update(scan["attrs"])
        stats["refs"].update(scan["refs"])
        stats["platform"].update(scan["platform"])
        stats["platform_total"] += scan["platform_total"]
        stats["filename"] += scan["filename"]
        stats["parent_path"] += scan["parent_path"]
        stats["parent_file"] += scan["parent_file"]
        stats["xi_include"] += scan["xi_include"]
        stats["type_attrs"] += scan["type_attrs"]
        stats["type_values"].update(scan["type_values"])
        stats["content_values"].update(scan["content_values"])

    total_elements = Counter()
    total_attributes = Counter()
    total_refs = Counter()
    total_platform = Counter()
    all_type_values = set()
    all_content_values = set()
    by_category = defaultdict(lambda: {"files": 0, "roots": 0, "elements": 0, "bytes": 0})
    for root, stats in roots.items():
        total_elements.update(stats["element_names"])
        total_attributes.update(stats["attributes"])
        total_refs.update(stats["refs"])
        total_platform.update(stats["platform"])
        all_type_values.update(stats["type_values"])
        all_content_values.update(stats["content_values"])
        cat = by_category[stats["category"]]
        cat["files"] += stats["files"]
        cat["roots"] += 1
        cat["elements"] += stats["elements"]
        cat["bytes"] += stats["bytes"]

    for stats in roots.values():
        stats["element_names"] = dict(sorted(stats["element_names"].items(), key=lambda kv: (-kv[1], kv[0])))
        stats["attributes"] = dict(sorted(stats["attributes"].items(), key=lambda kv: (-kv[1], kv[0])))
        stats["refs"] = dict(sorted(stats["refs"].items()))
        stats["platform"] = dict(sorted(stats["platform"].items()))
        stats["type_values"] = len(stats["type_values"])
        stats["content_values"] = len(stats["content_values"])

    ordered_roots = dict(sorted(roots.items(), key=lambda kv: (-kv[1]["files"], kv[0])))
    return {
        "roots": ordered_roots,
        "elements": total_elements,
        "attributes": total_attributes,
        "refs": total_refs,
        "platform": total_platform,
        "type_values": len(all_type_values),
        "content_kinds": len(all_content_values),
        "categories": dict(sorted(by_category.items())),
    }


def summarize(rows: list, agg: dict, mode: str) -> dict:
    sizes = sorted(row["bytes"] for row in rows)
    header_sizes = sorted(row["bytes"] for row in rows if row["header"])
    return {
        "mode": mode,
        "documents": len(rows),
        "well_formed": sum(1 for row in rows if row["well_formed"]),
        "fragments": sum(1 for row in rows if row["fragment"]),
        "tail_only": sum(1 for row in rows if row["tail_only"]),
        "truncated": sum(1 for row in rows if row["truncated"]),
        "malformed": sum(1 for row in rows if row["malformed"]),
        "root_count": sum(1 for root in agg["roots"] if root),
        "unattributed": int("" in agg["roots"]),
        "catalog_roots": sum(1 for root in agg["roots"] if root in CATEGORY_BY_ROOT),
        "elements": sum(agg["elements"].values()),
        "element_names": len(agg["elements"]),
        "attributes": len(agg["attributes"]),
        "max_depth": max((stats["max_depth"] for stats in agg["roots"].values()), default=0),
        "comments": sum(row["comments"] for row in rows),
        "comments_header": sum(row["comments"] for row in rows if row["header"]),
        "header_documents": len(header_sizes),
        "type_values": agg["type_values"],
        "content_kinds": agg["content_kinds"],
        "bytes": sum(sizes),
        "min_bytes": sizes[0] if sizes else 0,
        "median_bytes": int(statistics.median(sizes)) if sizes else 0,
        "max_bytes": sizes[-1] if sizes else 0,
        "header_bytes": {
            "min": header_sizes[0] if header_sizes else 0,
            "median": int(statistics.median(header_sizes)) if header_sizes else 0,
            "max": header_sizes[-1] if header_sizes else 0,
        },
        "references": dict(sorted(agg["refs"].items())),
        "platform": dict(sorted(agg["platform"].items())),
        "platform_total": sum(agg["platform"].values()),
        "filename": sum(stats["filename"] for stats in agg["roots"].values()),
        "parent_path": sum(stats["parent_path"] for stats in agg["roots"].values()),
        "parent_file": sum(stats["parent_file"] for stats in agg["roots"].values()),
        "xi_include": sum(stats["xi_include"] for stats in agg["roots"].values()),
        "include_like": {name: agg["elements"].get(name, 0) for name in INCLUDE_ELEMENTS},
        "categories": agg["categories"],
        "catalog": {
            "roots": len(CATEGORY_BY_ROOT),
            "categories": len(CATEGORY_ROOTS),
        },
    }


def carve_documents(directory: str):
    """Yield the XML payloads of an rpf9_carve.py directory, rebuilt by gamedata_index."""
    source = gi.DirectorySource(directory)
    rows = gi.build_rows(source.rows, source)
    by_offset = {row["off"]: index for index, row in enumerate(source.rows)}
    for row in rows:
        if row["type"] != "xml":
            continue
        start = by_offset[row["offset"]]
        data = b"".join(source.read(source.rows[index]) for index in range(start, start + row["chunks"]))
        yield {
            "data": data,
            "from_carve": True,
            "offset": row["offset"],
            "offset_hex": row["offset_hex"],
            "chunks": row["chunks"],
            "header": row["header"],
            "complete": row["complete"],
            "root_hint": row["root"],
        }


def xml_files(directory: str):
    def scan_failed(exc: OSError) -> None:
        raise exc

    for base, _, names in os.walk(directory, onerror=scan_failed):
        for name in sorted(names):
            if name.lower().endswith(".xml"):
                yield os.path.join(base, name)


def input_files(targets: list):
    """Yield every file the scan will read, so a report path cannot overwrite one.

    Directory mode reads every nested ``*.xml``; carve mode reads ``index.json`` and the
    ``<offset>.bin``/``<offset>.bk2`` stream files next to it. A report path only has to match
    one of these, not the target argument itself.
    """
    for target in targets:
        if os.path.isdir(target):
            if os.path.isfile(os.path.join(target, "index.json")):
                yield os.path.join(target, "index.json")
                with os.scandir(target) as entries:
                    for entry in entries:
                        if entry.is_file() and entry.name.lower().endswith((".bin", ".bk2")):
                            yield entry.path
            else:
                yield from xml_files(target)
        elif os.path.isfile(target):
            yield target


def input_name_matches(name: str, target: str) -> bool:
    """Whether a report basename could be a name the scan reads from this directory target."""
    name = name.lower()
    if os.path.isdir(target):
        if os.path.isfile(os.path.join(target, "index.json")):
            return name == "index.json" or name.endswith((".bin", ".bk2"))
        return name.endswith(".xml")
    return False


def has_symlinked_input(target: str) -> bool:
    """Whether the scan can read a symlink to the report from this target."""
    if os.path.isdir(target):
        if os.path.isfile(os.path.join(target, "index.json")):
            with os.scandir(target) as entries:
                for entry in entries:
                    if entry.is_symlink():
                        leaf = entry.name.lower()
                        if leaf == "index.json" or leaf.endswith((".bin", ".bk2")):
                            return True
        else:
            for base, _, names in os.walk(target):
                for name in names:
                    if name.lower().endswith(".xml") and os.path.islink(os.path.join(base, name)):
                        return True
    return False


def report_needs_input_check(path: str, targets: list) -> bool:
    """Whether a report path has to be compared with every scanned input.

    A report that already exists as a regular file, is not a symlink and has one link is reachable
    only through its own name. When that name cannot be an input read from a directory target, and
    no scanned input is a symlink to it, the per-file ``same_report_path`` checks can be skipped.
    Missing reports, links, explicit file targets and input-like names are always compared.
    """
    try:
        st = os.lstat(path)
    except OSError:
        return True
    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
        return True
    name = os.path.basename(os.path.realpath(path))
    for target in targets:
        if os.path.isdir(target):
            if input_name_matches(name, target) or has_symlinked_input(target):
                return True
        elif os.path.isfile(target):
            return True      # an explicit file target can have any name
    return False


def document_mode(targets: list) -> str:
    for target in targets:
        if os.path.isdir(target) and os.path.isfile(os.path.join(target, "index.json")):
            return "carve"
    return "files"


def load_documents(targets: list):
    """Yield one document per XML payload. Directories with index.json use the carve path."""
    for target in targets:
        if os.path.isdir(target):
            if os.path.isfile(os.path.join(target, "index.json")):
                for doc in carve_documents(target):
                    yield doc
            else:
                for path in xml_files(target):
                    yield file_document(path)
        elif os.path.isfile(target):
            yield file_document(target)
        else:
            raise FileNotFoundError(target)


def file_document(path: str) -> dict:
    with open(path, "rb") as f:
        data = f.read()
    head, tail = gi.xml_root(data), gi.tail_root(data)
    root = head if (head and (gi.has_xml_decl(data) or not gi.xml_complete(head, data))) else (tail or head)
    return {
        "data": data,
        "from_carve": False,
        "name": os.path.basename(path),
        "root_hint": root,
    }


def _make_parent(path: str) -> None:
    """Create the folder an output path lives in: the scan is over by the time we write, so a missing
    folder used to end the run in a FileNotFoundError and lose the result."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_csv(path: str, rows: list) -> None:
    _make_parent(path)
    fields = ["source", "offset_hex", "bytes", "chunks", "root", "category", "header", "complete",
              "well_formed", "fragment", "tail_only", "truncated", "malformed", "element_count",
              "max_depth", "comments", "attributes", "refs", "platform", "platform_total", "filename",
              "parent_path", "parent_file", "xi_include", "type_attrs", "type_values", "content_values",
              "error"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["attributes"] = " ".join(row["attributes"])
            out["refs"] = ";".join(f"{key}={value}" for key, value in row["refs"].items())
            out["platform"] = ";".join(f"{key}={value}" for key, value in row["platform"].items())
            for flag in ("header", "complete", "well_formed", "fragment", "tail_only", "truncated",
                         "malformed"):
                out[flag] = str(row[flag]).lower()
            writer.writerow(out)


def plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def check_report(rows: list) -> bool:
    """Print the validation split. Return True when no complete document is malformed."""
    fragments = sum(1 for row in rows if row["fragment"])
    print(f"check: {sum(1 for row in rows if row['well_formed'])} well-formed, "
          f"{sum(1 for row in rows if row['truncated'])} truncated at the start, "
          f"{plural(fragments, 'rootless fragment')}, "
          f"{sum(1 for row in rows if row['malformed'])} malformed")
    for row in rows:
        if not (row["truncated"] or row["fragment"] or row["malformed"]):
            continue
        where = row["offset_hex"] or row["source"] or "?"
        if row["malformed"]:
            print(f"  malformed {row['root'] or '?'} @ {where}: {row['error']}")
        elif row["truncated"]:
            print(f"  truncated {row['root']} @ {where}: starts mid-document")
        else:
            print(f"  fragment @ {where}: no root element")
    return not any(row["malformed"] for row in rows)


def print_summary(summary: dict, agg: dict) -> None:
    docs = summary["documents"]
    print(f'{docs} XML payloads, {summary["bytes"]:,} bytes '
          f'(min {summary["min_bytes"]:,}, median {summary["median_bytes"]:,}, '
          f'max {summary["max_bytes"]:,})')
    print(f'{summary["well_formed"]} well-formed, {summary["tail_only"]} tail-only '
          f'({summary["truncated"]} truncated), {plural(summary["fragments"], "rootless fragment")}, '
          f'{summary["malformed"]} malformed')
    print(f'{summary["root_count"]} named roots (+{summary["unattributed"]} unattributed), '
          f'{summary["elements"]:,} elements, {summary["element_names"]:,} element names, '
          f'{summary["attributes"]} attributes, max depth {summary["max_depth"]}, '
          f'{summary["type_values"]} type values, {summary["content_kinds"]} content kinds')
    print(f'{summary["comments"]} comment starts '
          f'({summary["comments_header"]} in {summary["header_documents"]} declaration-bearing documents)')
    refs = ", ".join(f"{key} {value:,}" for key, value in summary["references"].items())
    platforms = ", ".join(f"{key} {value:,}" for key, value in summary["platform"].items())
    print(f"references: {summary['platform_total']:,} platform:/ ({platforms}); {refs}")
    print(f'include-like: {summary["filename"]:,} filename, {summary["parent_path"]} parentPath, '
          f'{summary["parent_file"]} parentFile, {summary["xi_include"]} xi:include')
    print("categories:")
    for name, stats in summary["categories"].items():
        print(f'  {plural(stats["files"], "file"):8} {plural(stats["roots"], "root"):8} '
              f'{stats["elements"]:7,} elements  {name}')
    print("roots:")
    for root, stats in list(agg["roots"].items())[:10]:
        if not root:
            continue
        print(f'  {plural(stats["files"], "file"):8} {stats["elements"]:7,} elements  '
              f'{root}  [{stats["category"]}]')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("targets", nargs="+", help="carve directory or XML file(s)")
    parser.add_argument("--csv", help="write one row per XML document to this CSV")
    parser.add_argument("--json", help="write the summary, per-root catalog and rows to this JSON")
    parser.add_argument("--check", action="store_true", help="validate well-formedness and report fragments")
    args = parser.parse_args()

    if args.csv and args.json and same_report_path(args.csv, args.json):
        parser.error("--csv and --json must use different paths")
    for flag, path in (("--csv", args.csv), ("--json", args.json)):
        if not path:
            continue
        if os.path.isdir(path):
            parser.error(f"{flag} is a directory: {path}")
        if report_needs_input_check(path, args.targets):
            for source in input_files(args.targets):
                if same_report_path(path, source):
                    parser.error(f"{flag} would overwrite the input {source}")

    rows = []
    mode = document_mode(args.targets)
    try:
        for doc in load_documents(args.targets):
            rows.append(file_row(doc, scan_xml(doc["data"])))
    except FileNotFoundError as exc:
        print(f"not found: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # Same one-line contract as gamedata_index.py: a bad carve index is an error, not a traceback.
        where = getattr(exc, "filename", None) or args.targets[0]
        print(f"ERROR: cannot read {where}: {getattr(exc, 'strerror', None) or exc!r}", file=sys.stderr)
        return 2

    agg = aggregate(rows)
    summary = summarize(rows, agg, mode)
    clean = True
    if args.check:
        clean = check_report(rows)
    print_summary(summary, agg)
    if args.csv:
        try:
            write_csv(args.csv, rows)
        except OSError as exc:
            print(f"ERROR: cannot write {args.csv}: {exc.strerror or exc}", file=sys.stderr)
            return 2
        print("wrote", args.csv)
    if args.json:
        public_rows = [{key: value for key, value in row.items() if key != "_scan"} for row in rows]
        try:
            _make_parent(args.json)
            with open(args.json, "w", encoding="utf-8", newline="\n") as f:
                json.dump({"summary": summary, "roots": agg["roots"], "rows": public_rows}, f, indent=1)
        except OSError as exc:
            print(f"ERROR: cannot write {args.json}: {exc.strerror or exc}", file=sys.stderr)
            return 2
        print("wrote", args.json)
    return 0 if clean else 1


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
