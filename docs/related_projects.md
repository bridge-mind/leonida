# Related projects and prior art

Leonida is **GPL-3.0-or-later**. Some useful reverse-engineering work lives under other licences, and some of it
cannot be mixed with this repository. This page records what we may learn from each project and where the line is.

The rule is the same for every entry: **facts are fine; code is not.** A format layout, an ABI observation or a
measured behaviour can be reimplemented independently. The source that expressed it cannot be copied.

## ivall/xcl

<https://github.com/ivall/xcl>

An Xbox Series compatibility layer for Windows. It runs a title's x64 code natively and reimplements the console
system libraries, in the spirit of Wine. It recompiles RDNA2 shaders to HLSL/DXIL instead of SPIR-V. Its README
reports several retail titles as playable, and its design notes record measured facts relevant here: root-argument
placement in user SGPRs, thread-group lane layout on PC drivers and GFX10 texture tiling.

**Licence:** **GPL-2.0-only**, which is **incompatible with GPL-3.0-or-later**, as for americas-recomp below. **No
code, tables or comments may be copied.**

**What we use it for:** its measured facts only. Verify each one with Leonida's own census, a synthetic test or a
public specification before relying on it. This repository never contains game data, so a fact XCL measured on a retail package is checked here with synthetic inputs or a contributor's own copy.

## rpatty-dev/PercyStar

<https://github.com/rpatty-dev/PercyStar>

A source-available Rust explorer for RPF9/TTPF, with an RDNA2 disassembler, texture decoders and an `ADAT` Opus audio
decoder. Its format observations are relevant to the archive, texture and audio work here.

**Licence:** the "PercyStar Non-Commercial Attribution & Copyleft License (v1.2)". It is source-available, says
"All rights reserved", and adds non-commercial, attribution, naming and share-alike conditions. It is not
GPL-compatible, so **no PercyStar code, comments or tables may be copied into Leonida**.

**What we use it for:** reading about format facts, then implementing them independently. If a fact is useful,
record the observation in our own words, cite the project and verify it against the package or a public
specification. Do not paste even a small function "to save time".

## realdody/americas-recomp

<https://github.com/realdody/americas-recomp>

A **GPL-2.0-only** fork of [boykopovar/AnyPS5](https://github.com/boykopovar/AnyPS5), on its `xbox-series` branch. It
recompiles the same Xbox Series shaders and has run the package's DLLs against a PC D3D12 device behind an Xbox-ABI
proxy. Since 2026-11-22 its pull requests are limited to collaborators and its README points contributors here.

**Licence:** the repository is **GPL-2.0-only** (`LICENSE`), which is **incompatible with GPL-3.0-or-later**: GPLv2-only
code cannot be combined into a GPLv3 work, so none of it may be copied, and there is no "just this one file" exception.
Since 2026-11-22 its `REUSE.toml` on `xbox-series` (commit `f1c25d72`) lists **88 files under MIT**, each relicensed with
the agreement of everyone who wrote lines in it ([realdody/americas-recomp#179](https://github.com/realdody/americas-recomp/issues/179),
still open, so the list can change): 35 files at the top of its tools/xbox directory, 30 in tools/xbox/platform (for
example `xbox_dxil_lower.py`, `d3d12_xs.cpp`, `xmem.cpp`, `xbox_device_proxy.h`), one each under tools/xbox/msvc and
tools/xbox/platform/xgr, its 20 tools/tests/test_xbox_*.py tests and its docs/xbox/BVH_CONTAINER.md. Everything else,
including the recompiler in its core directory and the `xgameruntime` stand-in sources, stays GPL-2.0-only.

**Rule until the maintainer decides:** **no americas-recomp code is copied into Leonida**, MIT-listed or not. MIT is
compatible with GPL-3.0-or-later, so the listed files could be used the way `third_party/xtaskqueue` is (per file, only
paths listed in `REUSE.toml` at the commit used, the MIT notice kept); whether Leonida does that is the maintainer's
decision, not a contributor's.

**What we use it for:** the measured ABI facts only. The useful findings are summarised in
[d3d12_vulkan.md §7](d3d12_vulkan.md#7-prior-art-and-measured-xbox-abi): the Xbox device entry point, the slot
deviations, the descriptor-heap start-handle return convention, the root-signature flag word and the BVH builder's
compute-pipeline path. Those are observations to verify, not code to port. The document paraphrases them; no code or
long passages are copied.

## breadbbbb/rpf9-explorer

<https://github.com/breadbbbb/rpf9-explorer>

A Windows viewer for RPF archives, and the source of the Core library behind `tools/rpf9dump` (models, textures,
TTPF pools and AWC audio). It is a `third_party` **submodule**, not code copied into this repository.

**Licence:** the repository has **no licence file** (GitHub reports no licence), so all rights are reserved by
default. A submodule is an external dependency; it is not a grant to copy its code. Do not paste its functions,
tables or comments into Leonida. If a format fact learned from it is useful, verify it against the package and
write our own implementation. The model-recovery gap measured in [models.md](models.md) is in that submodule; the
fix should go upstream, not into a copy.

## meetox80/6stars

<https://github.com/meetox80/6stars> — a macOS Swift/Metal engine that renders the world out of RPF9,
with a Swift `RPFKit` format library and 44 Python research scripts. Independent work: it likewise
does not decode the TOC.

**Licence: none.** The repository has no `LICENSE` file (GitHub reports `license: null`); only the
vendored cimgui/imgui carry MIT. With no grant, default copyright applies and the project is
**all rights reserved** — stricter than PercyStar. **No 6stars code, comment, identifier or table may
be copied into Leonida.**

**What we use it for:** corroboration and a small number of hypotheses. As an independent reading of
the same formats it matches our `CEntityDef` member hashes (9/9), the PRD0 header, the navmesh tile
header, the meshlet group header and the PSIN/LSCH field layout. Its stream-role predicates and
navmesh surface bits are recorded as hypotheses and tested from our own package, never copied.
Where it disagrees with us (ETPF format and tile mode, the RDS5 codec field, `atArray` type `0x5C`,
PSO member hashing) our measurements are the validated ones.
<https://github.com/meetox80/6stars>

A macOS Swift/Metal engine that reads the world from RPF9, with its own archive index and format readers.

**Licence: none.** There is no `LICENSE` file and GitHub reports no licence, so all rights are reserved.
**No 6stars code, comments, identifiers or tables may be copied into Leonida.**

**What we use it for:** corroboration. Its RPF9 stream-boundary predicate and PRD0 member type codes were
re-measured against this package before entering [rpf.md](rpf.md) and [world.md](world.md); its navmesh
road/pavement bit labels did not reproduce and are not used. The navmesh polygon `+4` packing it reads is the one
our decoder already validates ([navmesh.md](navmesh.md)).

## p-portela/SCARAB

<https://github.com/p-portela/SCARAB>

A GPL-3.0 "Scarlett Compatibility And Runtime API Bridge" that forwards the console D3D12 runtime to PC D3D12 (and
to vkd3d-proton on Linux). It carries a measured catalogue of the console `d3d12_xs.dll` export surface, standard
and console-only IIDs, extension interfaces, command-list interception and presentation entry points.

**Licence:** GPL-3.0. That is compatible with this project's GPL-3.0-or-later for study, and adapted code can be
used with attribution and the original licence notice, unlike the two projects above. Even so, **facts first**:
verify a claim against SCARAB's code and, where possible, this package; write the implementation in our style; and
say in the pull request which file a borrowed fact came from. Do not copy a whole component just because the
licence allows it.

**What we use it for:** the measured console runtime surface (see `docs/d3d12_xs_surface.md` when that study
lands), the IID/extension tables, and the call-sequence observations that Leonida's own
[d3d12_vulkan.md §7](d3d12_vulkan.md#7-prior-art-and-measured-xbox-abi) still lists as unknown. SCARAB forwards to
D3D12; Leonida translates to Vulkan. The ABI facts carry over, the forwarding architecture does not.

## holdmysocks/SixNG

<https://github.com/holdmysocks/SixNG>

A Rust extractor for the same package: models, textures, map placements, collision, navmesh, the lane graph and
audio, with format notes in its module headers.

**Licence:** GPL-3.0, compatible with this project's GPL-3.0-or-later. As with SCARAB, **facts first**: nothing of
it is copied here.

**What we use it for:** two navmesh facts, read in `crates/sixng-core/src/navmesh.rs` at commit `0619601` and
re-measured on this package by `tools/navmesh_nodes.py`
([navmesh.md](navmesh.md#polygon-flags-under-the-lane-graph)): its label "road surface" for polygon `+0x18` bit
13, which it bases on a visual check, and the vertex formula `tile minimum + u16 / 65535 * extent`. The formula is
confirmed by the lane-graph heights. The label is consistent with the count, but not settled by it: nine in ten
lane nodes stand on polygons with both bit 8 and bit 13, and which bit or combination means road surface is open.

## microsoft/libHttpClient

<https://github.com/microsoft/libHttpClient>

Microsoft's HTTP/WebSocket library for GDK titles. Its `Source/Task` folder is the implementation of the GDK's
`XTaskQueue` and `XAsync`.

**Licence:** MIT, which is compatible with GPL-3.0-or-later.

**What we use:** the task queue's headers and Win32 sources, copied unmodified into `third_party/xtaskqueue` (commit
and file list in its README) with the licence file. `runtime/shims/xgameruntime.cpp` registers those functions as
`xgameruntime.dll` imports. The HTTP and WebSocket stack is not included. Unlike the projects above, this is code
use under a compatible licence, so the MIT notice must stay with the files.

## AMD machine-readable GPU ISA specification

<https://gpuopen.com/machine-readable-isa> (reader: <https://github.com/GPUOpen-Tools/isa_spec_manager>)

AMD's XML description of the RDNA and CDNA instruction sets: instructions, encodings, operands, data formats.

**Licence:** MIT (`<License>MIT</License>` in each XML file; `isa_spec_manager` is MIT too), compatible with
GPL-3.0-or-later.

**What we use it for:** `tools/isa_coverage.py` reads `amdgpu_isa_rdna2.xml` from a path the user supplies and counts
which instructions the translator has a rule for ([isa_coverage.md](isa_coverage.md)). The XML is not vendored (11 MB);
tests use a small fixture of our own in the same schema.

## Mesa (AMD register database and format tables) and LLVM's AMDGPU backend

<https://gitlab.freedesktop.org/mesa/mesa> (`src/amd/registers/*.json`, `src/amd/common/ac_descriptors.c`),
<https://github.com/llvm/llvm-project> (`llvm/lib/Target/AMDGPU/SIDefines.h`)

**Licence:** Mesa's AMD code is MIT, LLVM is Apache-2.0 with LLVM exceptions. Both are GPL-3.0-or-later compatible,
but nothing is copied.

**What we use:** field positions and enum values, written into this repository's own code. `tools/hw_oracle/tsharp.py`
builds the T# for the hardware oracle of `tools/image_fuzz.py` from Mesa's SQ_IMG_RSRC_WORD1..4 bit ranges
(`gfx10-rsrc.json`, `gfx11-rsrc.json`), the way `ac_build_gfx10_texture_descriptor` and `ac_set_mutable_tex_desc_fields`
fill them (linear images carry their pitch in WORD4), and the GFX10 and GFX11 format numbers (Mesa's `GFX10_FORMAT`,
and for GFX11 the order of LLVM's `UfmtGFX11::UnifiedFormat`). The GFX11 numbers were confirmed on a gfx1150 by the
hardware oracle itself (a wrong FORMAT changes the fetched words); the GFX10 numbers and layout are Mesa's and not yet
run on RDNA2 hardware.

## The line

| Fine | Not fine |
|---|---|
| A format layout, field offset or enum value | Copying the function that parses it |
| An ABI measurement (slot number, return register, flag word) | Copying the proxy, thunk or struct definitions |
| A behaviour observed by running a tool | Copying comments, identifiers or test code |
| A public specification or vendor documentation | A line-by-line port, a translated port, or a "cleaned up" copy |
| A table of facts you rebuilt from your own observations | A table copied from the source, even with renamed columns |

A table of constants is still expression. Rebuild it from your own measurements or a public specification. A
struct definition is still expression even if the field names are changed. If the only way to use a fact is to
copy the surrounding code, stop and find the fact another way.

When in doubt:

1. Write down the fact, not the code.
2. Cite the project and the section or file where you read it.
3. Verify the fact against the game package, a public specification or a test of your own.
4. Implement it independently, in this repository's style.

Never paste code from either project into a pull request, an issue, a commit message or a chat log that becomes
part of the repository history. If a contribution was written while looking at incompatible source, say so in the
pull request so the provenance can be checked.

Other dependencies keep their own licences; see the Credits and License sections of [README.md](../README.md).
