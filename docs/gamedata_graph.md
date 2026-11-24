# The `common.rpf` content-set graph

This note measures the mount/reference graph in the XML documents recovered from `common.rpf` in the
Xbox Series 1.0.2.1 package. It extends [gamedata.md](gamedata.md), which counts the documents and
reference tokens, and uses the same local carve. Only aggregate counts, root/category names and
reference-prefix names are recorded here. No archive, carved file, XML value, content-set id, entry
name or target path is committed.

`tools/gamedata_graph.py` rebuilds the documents with `gamedata_index.py` and
`gamedata_xml.load_documents`, then keeps the reference strings long enough to join them. It is
standard-library only and reads the same carve as the index and catalog. XML with an unsupported
encoding counts as an unparsed payload; it supplies no edges and does not stop the other documents.

## Graph model

| Node | Count | What it represents |
|---|---:|---|
| document | 267 | one XML payload, including the unparsed ones |
| content set | 30 | every `<id>`, plus every `<includedSets>` id |
| target | 3,364 | every distinct `<filename>`, `<loadPath>`, `parentPath` and `parentFile` value |
| **total** | **3,661** | |

| Edge | Count | Meaning |
|---|---:|---|
| `defines` | 28 | document to its top-level or nested `<contentSets>` id |
| `includes` | 14 | document to an `<includedSets>` id (12 resolve to a document) |
| `contains_file` | 3,315 | set (or document) to a `<filename>` target |
| `loads_changeset` | 28 | document to an `<includedChangesets>/<loadPath>` target |
| `parent_path` / `parent_file` | 40 / 3 | document to a `parentPath` / `parentFile` locator |
| **total** | **3,428** | |

The document/content-set part is 42 edges; the target part is 3,386 edges. An include becomes a
document-to-document edge only when its text exactly equals a set id. There is no fuzzy match and
no prefix normalization. Missing ids stay dangling content-set nodes, so a spelling difference is
visible as an unresolved include instead of becoming an invented document edge.

## Measured shape

| Measure | Value |
|---|---:|
| XML payloads | 267 |
| Complete documents | 266 |
| Parsed documents | 259 |
| Unparsed payloads | 8 (7 truncated, 1 rootless) |
| Named roots (+ one unattributed) | 118 (+1) |
| Manifest documents | 25 (23 `CDataFileMgr__ContentChangeSetDef`, 2 `CDataFileMgr__ContentDef`) |
| Content-set ids | 30 (28 defined, 2 dangling) |
| Duplicate set ids | 0 |
| Reference sites | 3,400 (3,386 target + 14 include) |
| Distinct target strings | 3,364 |
| Include sites / document edges | 14 / 12 |
| Cycles in the include graph | 0 |
| Include-graph roots (in-degree zero) | 13 |
| Reachable documents from those roots | 25 of 25 manifest documents |
| Maximum include depth | 1 |
| Reachable content sets | 30 |
| Reachable target sites | 3,343 |
| Reachable distinct targets | 3,343 |

The manifest layer is closed at the document level: all 12 resolved includes point at documents that
are already among the 25 manifests, so no non-manifest document is reachable by an id. The graph
therefore measures the mount walk, not a general include system. The 21 distinct targets outside the
reachable set are the `parentPath`/`parentFile` locators in the two non-manifest documents.

## Prefix rules

The four documented rules cover 3,312 of the 3,386 target sites. `common:/` also occurs but is not
one of those rules, so it is reported separately and is not rewritten to `$(common)/`.

| Reference form | Sites | Resolution in this catalog |
|---|---:|---|
| `$(common)/` | 587 | common archive namespace; the TOC name is missing |
| `$(target)/` | 23 | target data namespace; the TOC name is missing |
| `$(root)/` | 28 | changeset root; not a document in this archive |
| `platform:/levels/` | 2,346 | levels namespace; the TOC name is missing |
| other `platform:/` | 328 | other platform namespace; the TOC name is missing |
| `common:/` | 31 | unrecognized scheme alias; left unresolved |
| relative locator | 43 | `parentPath` / `parentFile`, not an archive path |
| **total** | **3,386** | |

`parentPath` values are XPath-like locators, not mount paths, and `parentFile` is a relative name.
The 43 locator sites are 21 distinct strings in 2 non-manifest documents. None of them is used to
invent a document edge.

### What the locators point at

The two kinds behave differently, and only one needs the table of contents:

| Locator | Sites | Distinct | Shape | Resolves |
|---|---:|---:|---|---|
| `parentPath` | 40 | 20 | always `../Item[@key='<key>']` | **40 of 40** to a sibling `Item` in the same list of the same document |
| `parentFile` | 3 | 1 | `<dir>/<name>.<ext>`, a relative file path | no: needs a file name, so the table of contents |

All 40 `parentPath` sites are in the one `CDoorInteractionDataList` document, on items of its
`DoorInteractionDataSets` list (56 items, 56 distinct `key`s). Read as an XPath step from the item, `..` is the
list and `Item[@key='K']` selects the sibling with that key; every `K` exists (case-insensitive). The result is
single inheritance inside the list: following parents gives depth 0 for 16 items, 1 for 35, 2 for 3, 3 for 1
and 4 for 1, with no cycles. So these are in-document references that a loader can resolve without any archive
path, by key lookup within the parent list. The 3 `parentFile` sites are all in `CEnergyConfigInfos` and name
the same other file.

## Unresolved references

| Bucket | Sites | Note |
|---|---:|---|
| Recognized archive namespace, no document identity | 3,312 | `$(common)` 587, `$(target)` 23, `$(root)` 28, `platform:/levels` 2,346, other `platform:/` 328 |
| Unrecognized scheme (`common:/`) | 31 | not one of the four prefix rules |
| Relative locator | 43 | 21 distinct `parentPath`/`parentFile` values |
| Include id with no exact definition | 2 | one differs only by the `common:/` vs `$(common)/` spelling; one has no exact set id |
| **total** | **3,388** | excludes the 12 resolved include edges |

The first bucket is not a failure by itself: the table of contents is encrypted, so an archive path
cannot be mapped to a carved document or to an entry name. It remains a target node. Only the two
include ids are unresolved content-set joins. Both are kept dangling rather than normalized.

## Runtime use

A mount resolver can follow the same shape: read a manifest, take its `<id>` and `<fileList>`,
resolve the four prefixes to archive namespaces, and follow `<includedSets>` only on an exact id. The
measured walk from the 13 in-degree-zero manifests reaches all 25 manifest documents and 30 set ids
in this archive, at depth 1. It does not reach any of the other 242 XML payloads, because those are
not mounted through content-set ids in this catalog.

This is not a runtime schema. The catalog knows 118 root names, 5,293 element names and 33 attribute
names, but not field types. It also cannot supply names for the archive targets: that needs a
documented TOC reader, which is outside this work.

`parentPath` inheritance is the one reference a loader can follow inside a document. The tool reports it
under `parent_path_inheritance`: a locator of the form `../Tag[@attr='value']` is resolved against the
siblings of the element that carries it, by exact string match. On this `common.rpf` all 40 sites take that
form and all 40 resolve (none ambiguous), in one document; the linked items sit at inheritance depth 1 (35),
2 (3), 3 (1) and 4 (1), with no cycles. These links stay outside the document graph and its edge counts.

## Limits

- The table of contents is encrypted, so target paths are namespaces and strings, not document ids.
  No path is joined to a carved document.
- Eight payloads are unparsed (seven truncated at the start, one rootless). Their references are
  unknown and contribute no edges.
- Content-set joining is exact string equality. `common:/`, `$(common)/`, case and slash variants
  are not normalized; the 31 `common:/` entries stay unresolved.
- `parentPath` and `parentFile` are locators, not mounts. They are counted, not followed by the tool.
  The 40 `parentPath` sites resolve inside their own document (see above); the 3 `parentFile` sites do not.
- Only `filename`, `loadPath`, `includedSets`, `parentPath` and `parentFile` are modeled. No
  XInclude occurs in the corpus, and no general include processor is implemented.
- The counts are for this version's `common.rpf`; another package can differ.

## Reproduce

```powershell
python -I tools/rpf9_carve.py common.rpf <work>\carve\common
python -I tools/gamedata_graph.py <work>\carve\common --json <work>\graph.json
python -I tests/test_gamedata_graph.py
```

The JSON is an aggregate work product: counts, root names, category names and prefix names. It
contains no target strings, content-set ids, offsets or per-file names, so it can be reviewed
without copying game data.
