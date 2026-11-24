# Documentation

These notes record what has been measured in the local Xbox Series package and what is still open.
The headline counts are the measured numbers at the time of the note; the linked page has the
method, scope and caveats. No game data is included.

## Start here

| Document | What it covers |
|---|---|
| [design.md](design.md) | Overall project design, asset status, translator contract and open blockers. |
| [related_projects.md](related_projects.md) | Prior art we may read and the licence boundaries on it. |
| [rpf.md](rpf.md) | RPF9 archive layout, the carve tools and the unreadable encrypted TOC. |
| [testing.md](testing.md) | Running the GPU tests on your own GPU: setup per OS, wave64 emulation, running without a GPU on Mesa llvmpipe, and known problems. |
| [modding.md](modding.md) | The mod format: `mods/<folder>/mod.json`, checked and loaded in name order by `gta6rt`; a swap takes effect once the game's archetype lookup is hooked (needs the executable, RT-5). A mod names game models and holds no game data. |
| [progress_dashboard.md](progress_dashboard.md) | A local page of what works, what changed since it last passed and what blocks next: unit and runtime tests, the translator contract by confirmation method, measured imports, corpus counts and, with `--github`, CI and the D3D- items (`tools/progress_dashboard.py`). |

## Archives and assets

| Document | Measured content |
|---|---|
| [models.md](models.md) | 382,781 models (0 decode failures) at the `e293917` pin, including 159,511 drawable-dictionary children; 55,417 fragments, 89,887 bone-bound and 2,636,248,990 triangles over all ten archives (`levels.rpf` from a full export); the re-count table is in models.md. |
| [strand_grooms.md](strand_grooms.md) | GPU strand grooms (`smh`): hair, beards, brows and fur are groom LOD resources consumed by the `smh_*` compute passes, not triangle drawables (each is a one-triangle stub); the measured resource layout, the Class 0 strand scattering RNG and barycentric strand interpolation, decoded by `tools/smh_groom.py`. |
| [character_models.md](character_models.md) | `packs.rpf` character parts at `e293917`: 26,585 rows / 11,335 unique names, 2,865 fragments (all a flat 25-vertex / 32-triangle 5x5 bounds grid at z=0), Lucia's 22 named records and the decoded 13-part export, the `player_story000_00` drawable set, slot-name token counts, texture coverage (184/215), and what is not decoded (fragment children, ped variation, hair, bind pose). |
| [character_skeleton.md](character_skeleton.md) | Character `.#ft` fragments: the 597-name shared skeleton table (596 `joaat32` hashes), the 597-record instance tree with 580 measured parents, the model-side bone-hash mapping, and the measured vertex joint-index/skin-weight layouts (`tools/skin_streams.py`); the default bone transforms compose to the stored inverse bind matrices, and `tools/anim_glb.py` exports a skinned part with clips as an animated GLB. |
| [ped_catalogues.md](ped_catalogues.md) | Ped and wardrobe appearance catalogues (`0x8dcb4cf5`) in both framings: 2,716 catalogues and 632,778 presets (612,950 in resource-packed `0DRP`, 19,828 in one standalone `PSIN`), 345,780 presets with a parent (chain depth 3, 0 cycles), 896,115 `0x37e52277` states resolved, 0 parse errors (`tools/ped_catalogues.py`). |
| [strand_grooms.md](strand_grooms.md) | GPU strand grooms (`smh`): 5,751 assets / 20,046 LOD resources, 0 decode errors, 902,456,856 GPU buffer bytes and 112,491,621 strands; the `smh_gen_1` Class 0 RNG, the barycentric interpolation and split-chain rejoin, and a GLB export (`tools/smh_groom.py`). |
| [resource_names.md](resource_names.md) | RPF9 named-resource index: bounded-inflate lookup for `name.#dr`/`#ft`/`#dd` without the TOC. |
| [sgd5_render_states.md](sgd5_render_states.md) | The three effect-local render-state indices in an SGD5 pass, checked across all 28,798 passes. |
| [textures.md](textures.md) | 357,115 TTPF descriptors across 57 pools: `rpf9dump` (`e293917`) decodes 354,991 strict-clean (99.41%) directly including BCPack BC1/BC3/BC4/BC5; `tools/bcpack.py` decodes all six BCPack modes (BC1–BC7). |
| [textures_arrays.md](textures_arrays.md) | All 697 multi-slice TTPF descriptors resolved (127 mip-major `SW_64KB_D` 2D arrays, 567 streaming slice-major `SW_4KB_S` cubemaps, 3 `SW_4KB_S` 3D volumes) plus 357,115/357,115 chunk-page metadata validation. |
| [animation.md](animation.md) | 130,470 `anim.rpf` streams, 95.4% readable; 1,126 rig sets / 6,677 source-to-rig entries / 3,669 rigs. |
| [navmesh.md](navmesh.md) | 90,295 `navmeshes.rpf` streams (0 unclassified): 9,603 `0DRP` cover-point `0x7a` pages, 27,644 `0x00010015` navmesh tiles (`136×180` grid), 27,482 Band-3 sparse grids, 22 `PSIN`. |
| [bvh.md](bvh.md) | 565,596 `core_bvh4.rpf` streams; 271,627 BVH4 streams and 269,106 instances, prebuilt acceleration data. |
| [rbf0.md](rbf0.md) | 1,263 RBF0 resources, 475,461 elements; UI/Scaleform definitions, not a sectioned container; strict JSON/CSV/XML tree decoder. |
| [gamedata.md](gamedata.md) | `common.rpf`: 954 streams, 266 XML documents / 118 roots; 0 RBF0/PSIN; a streaming reader (`gamedata_reader.py`) converts 259 documents by root and `type`. |
| [gamedata_graph.md](gamedata_graph.md) | `common.rpf` mount graph: 25 manifest documents, 30 content-set ids, 14 include refs (12 resolve), 3,386 target refs, 0 cycles. |
| [weather.md](weather.md) | The weather document measured: 168 types (151 tagged / 17 untagged, a family split), exact-case `Mod*` vocabulary mismatches, axis-to-field Spearman (niceness tracks its field, `Rain` is a 7-of-25 data-set split), `TimeMult` and the inert `Honor*` values. |
| [world.md](world.md) | `levels.rpf` placement: 9,918 map files / 680,228 entities measured in 55% of the archive (lower bound; estimate 16k-39.6k / 1.03M-2.64M); `0x7a` resource layout. Building a placed world from entity rows (`world_build.py --placements`): shared geometry per chunk (`EXT_mesh_gpu_instancing`), bounded memory for the whole archive, and per-tier LOD chunk sets with distance bands (`--lod-sets`). |
| [world_census.md](world_census.md) | Full `levels.rpf` pass: 23,535 maps / 1,670,442 entities, 98.32% stream coverage, sharded and exact; the SixNG 30,785-map difference reconciled map by map (interiors and entity groups). |
| [world_names.md](world_names.md) | Name resolution: `archetypeName` and map `name`/`parent` are lower-cased joaat; 64-bit keys are MurmurHash3 x64 low 64. With `rpf9_names.py` names, 99.85% of the 1,670,435 entities a whole-archive `world_index` run reads resolve (64-bit verified); 5/23,535 map names, all five confirmed by the map 64-bit key. |
| [world_lights_paths.md](world_lights_paths.md) | Map lights (`CExtensionDefLightEffect`/`CLightAttrDef`, 280,105 lights, placed through map entities) and the vehicle lane graph (63,808 nodes, 145,518 links, 1,546 junctions, 369 signalised); identical to SixNG on the lane graph, lights reconciled resource by resource. |
| [page_directory.md](page_directory.md) | RPF9 `0DRP`: 0x60-byte header, `q38` 16-byte records; 15,768 pages / 60,880 records in `data.rpf`; targets and sizes, names still TOC-only. |
| [text.md](text.md) | 30,511 localized string tables (hash table, index, UTF-8 blob in `0DRP` resources) and the dialogue table: 138,104 lines joined to their subtitles in every language; language guessed per table. |
| [psin.md](psin.md) | PSIN metadata container: header, section walk, `LSCH`/`LENM` schema, and the instance data (block table, pointers, arrays, maps) read through it on 17,507 containers, 99.986% of block bytes reached; hash-typed members resolved to resource names. |
| [ped_catalogues.md](ped_catalogues.md) | Ped and wardrobe appearance catalogues: one schema rooted at struct `0x8dcb4cf5`, in `0DRP` v122 resources in `packs.rpf` and in the standalone `PSIN` player/wardrobe catalogue; both framings' 1-based block table and 64-bit pointer encoding (`block = ptr & 0xfffff`, `offset = ptr >> 20`). |
| [story.md](story.md) | `story000.rpf` 18.65 GiB mixed: 264 Bink 2 streams / 3.33 GB, 225,281 `RSC8` headers, 442 nested RPF9; `story000_script` PSIN, `shared_minigame` mixed content. |
| [restofworld.md](restofworld.md) | Six rest-of-world/script archives, 1.57 GB total: `core_restofworld` 1.21 GB (11 Bink 2, 37 models, 958 graphics pages), `story000_script` PSIN, small `appdata0`/licences. |
| [movies.md](movies.md) | Bink 2 streams and audio tracks; decode through the real loader and `Bink2Scarlett.dll`; stream names from the mount lists. |
| [pso_cache.md](pso_cache.md) | The PSO/shader cache: 0x80 header, 25,471 graphics and 2,693 compute records, two 8-byte index tables, meta chunk types 0x41/0x42/0x43 and 26,092 `s_endpgm`; `tools/psocache_extract.py` carves it, `tools/psocache_inspect.py --meta` decodes the chunks and `tools/psocache_usersig.py` maps the graphics user data to root parameters. |

## Audio

| Document | Measured content |
|---|---|
| [audio.md](audio.md) | AWC banks: Opus, PCM and XMA2; lossless Ogg Opus repack and WAV paths; stream names from dictionary hits with a salted control; names and subtitles written into the Ogg Opus tags. |
| [rsc8_audio.md](rsc8_audio.md) | 641,428 `RSC8` v8001 resources across 4 archives: 100% AWC tag `0x5c` facial/lip-sync `crClipDictionary` paired with 48 kHz mono Opus dialogue, 0 hidden audio containers; the undocumented `0x5a`/`0xd9` companion tags (1,185 `audio.rpf` streams) decoded structurally. |
| [awc_shorts.md](awc_shorts.md) | Export short streamed-music idents/stings by duration; stereo-aware, no re-encode. |
| [audio_config.md](audio_config.md) | The music/radio record table: 1,589 records, exact durations, AWC stream ids; no start offsets. AWC tag `0xBD`: 16-byte marker events on 938 of the 3,008 streamed-music banks, track markers in five long mixes. |
| [radio_stations.md](radio_stations.md) | Audio-config station blocks: 19 records, boundary u32, content-id lists; VCRR 80/80 mapping; no start offsets. |

## Runtime, GPU and validation

| Document | Measured content |
|---|---|
| [d3d12_vulkan.md](d3d12_vulkan.md) | Xbox D3D12 to Vulkan plan and status; device proxy, queues, command lists, barriers, copies. |
| [boot.md](boot.md) | The first real boot: the `gta6rt --trace`/`--trace-d3d12` runbook, the startup + D3D12 trace summary (`tools/boot_trace.py`), the triage order and the Q1–Q22 checklist. |
| [retail_boot_chain.md](retail_boot_chain.md) | How far `gta6rt --run` gets with the retail executable: the stops in order and what clears each, the physical-page export shapes, and device slots 26, 48 and 73. |
| [modding.md](modding.md) | The mod format: a `mods/<folder>/mod.json` next to `gta6rt.exe` or under `--mods DIR`, swapping a named (or `joaat`) archetype for another; load order, the start-up log, and the fact that a swap waits for the executable's archetype lookup address (RT-5). |
| [descriptor_lookup_apple.md](descriptor_lookup_apple.md) | M3 Max GPU timings for texture and sampler descriptor lookups at 1k, 10k and 100k live entries. |
| [descriptor_lookup_ada.md](descriptor_lookup_ada.md) | RTX 4070 SUPER (Ada, Windows) timings for the same lookups in VRAM, host-visible and ReBAR memory. |
| [descriptor_lookup_intel.md](descriptor_lookup_intel.md) | Intel UHD Graphics 770 (Xe-LP, Windows, driver 101.6129) timings for the same lookups at 1k–100k live entries; both memory placements within 0.6% of each other. |
| [d3d12_xs_surface.md](d3d12_xs_surface.md) | The 116-slot `d3d12_xs.dll` export surface; 83/83 standard GUIDs match the PC SDK. |
| [xdxr.md](xdxr.md) | `BATCHED_DESC2` (`0x50`) and `XDXR_STREAM_DESC` (`0x10`) from `XboxBvhBuilder.dll`'s reads; stream records' offsets and defaults; offline statistics not in the package. |
| [d3d12_command_objects.md](d3d12_command_objects.md) | Measured command queue/list/fence/`PresentX` surface; verified vs prior art vs open. |
| [xbox_d3d12_xs.md](xbox_d3d12_xs.md) | The `d3d12_xs` graphics ABI quoted from Microsoft's public GDK reference: root signatures, state objects, DXIL libraries, hit groups, shader cache sessions; which pages are missing or NDA-gated. |
| [runtime_imports.md](runtime_imports.md) | The 16 shipped modules' measured imports and shim/host/unimplemented split. |
| [xmem_exe_calls.md](xmem_exe_calls.md) | How the retail executable calls `xmem.dll`: call sites and constant arguments per import (`tools/import_call_sites.py`), the 23 `XMemVirtualAlloc` calls that all pass an attribute word, and the inferred shape of the physical-page family. |
| [audio_hal_exe_calls.md](audio_hal_exe_calls.md) | How the retail executable calls the 22 audio coprocessor, texture layout and DirectStorage imports that have no shim: call sites, constant arguments, and the four calls with a negative-result branch. |
| [xcl_compare.md](xcl_compare.md) | XCL's synthetic test title under `gta6rt --run` and under XCL (`tools/xcl_compare.py`): 5 of 50 checks pass here, both hangs are the documented manual-queue terminate deadlock, and each other difference mapped to an open item or a lead. |
| [gameinput.md](gameinput.md) | The GameInput shim on SDL3: `GameInputCreate` and the v0 `IGameInput`/`IGameInputDevice`/`IGameInputReading`/`IGameInputDispatcher` surface on any SDL gamepad, DualSense adaptive-trigger approximation and lightbar, force-feedback motors and trigger haptics through SDL's haptic API; 261 checks on virtual joysticks and a fake haptic backend; by hand on an Xbox Series X pad (Bluetooth) and a DualSense (USB). 261 checks on virtual joysticks and a fake haptic backend, not yet tried on a real pad. |
| [saves.md](saves.md) | Save containers: `JSAV` (index + JSON story save, byte-exact rebuild), `ASAV` header, `SLAYER1` packed copy (not decoded); `tools/saves.py` and import into the offline `XGameSave` folder; key hashes are case-sensitive joaat (`tools/save_keys.py`). |
| [pipeline_check.md](pipeline_check.md) | Driver-level corpus check: the full corpus, 26,503 of 26,503 modules and pipelines, 0 failures (RTX 4060 Ti); an earlier 10/12-shard run, 22,092 modules and 2,217 compute pipelines, also had 0 failures. Driver-level corpus check: 10/12 shards, 22,092 modules, 2,217 compute pipelines, 0 failures. |
| [multi_gpu_validation.md](multi_gpu_validation.md) | Running disjoint shards on eight identical RDNA1/RADV GPUs: PCI-BDF pinning, base controls, per-card logs, the V540 farm's aggregate results and its known driver limitations. |
| [shader_validation.md](shader_validation.md) | Differential CPU/GPU validation design and the current CPU interpreter prototype. |
| [isa_coverage.md](isa_coverage.md) | RDNA2 ISA coverage of the translator against AMD's machine-readable XML: which of its 1,153 instructions have a rule, and which matched names are not RDNA2. |
| [buffer_bounds.md](buffer_bounds.md) | V# `OOB_SELECT` bounds measured natively: RDNA2 (gfx1036) follows Mesa's GFX11 column (729/732 cases), not GFX10; raw mode also counts `soffset`. |
| [buffer_formats.md](buffer_formats.md) | Every GFX10 buffer data format a V# or MTBUF can name, loaded and stored, measured on a Radeon 880M (RDNA3.5, not RDNA2): 1.33 million validated elements, 0 mismatches; SCALED stores convert as NORM, float stores round toward zero. |
| [fssm_crosscheck.md](fssm_crosscheck.md) | FSSM/RDNA2 facts cross-checked against SCARAB; code starts at `code_off + 0x28`. |
| [fspx.md](fspx.md) | `FSPX`, the FidelityFX DLL's native compute programs (its shaders' `ACCC` part): directory as the DLL reads it, kind-0 dispatch state, the hash prologue, `gfx1030` for its `v_dot4` code; 13 of 29 translate with `tools/fspx_unpack.py`. |
| [first_light.md](first_light.md) | Draw an exported model's UVs or sample a supplied texture with `--dds`. The UV draw covered 78.1% on an RTX 2060 SUPER (4,213 vertices, 7,968 triangles). The first draw of a real game model with its own translated VS/PS on a GPU: 4,213 vertices / 7,968 triangles, 78.1% coverage on an RTX 2060 SUPER; `tools/first_light.py`, the `vk_harness` mesh mode, and the same draw through the D3D12 layer (`d3d12_first_light`). |
| [material_pass_replay.md](material_pass_replay.md) | How one authored mesh writes packed draw indices, how six compute programs read the layout, and which sibling passes read the surface material. |
| [embedded_shaders.md](embedded_shaders.md) | DXBC containers embedded in the package's DLLs: 52 in `XboxBvhBuilder.dll` (console `FSSM` via the `XDAT`/`XBCH` record; 51/52 pass `spirv-val` today) and 29 `ACCC`/`FSPX` in the FidelityFX upscaler; the scan/extract tool, the one rule still missing, and the build step and run-time lookup of the builder's programs (`<shaders>/bvh/index.json`). |
| [shader_library.md](shader_library.md) | The pipeline library in `shaders.rpf`'s zlib part, which the carve doesn't read: 25,471 graphics and 2,693 compute pipeline records (stage slots: PS, VS, MS) and 21,522 native programs, 21,509 of them pipeline builds of 6,522 carved shaders (named by their `HASH` digest); `tools/shader_library.py`. |
| [shader_inputs.md](shader_inputs.md) | Inputs that specific game shaders need to be right, as a first check for a black frame or a device-lost: the studio-shot lighting shader's 512-byte light records, eight-light loop, plane count at +0x4C (#1892 said +0x40), profile table and shadow-strength byte; the ACES LUT bakes' two tone-curve splines (reference order at +0x00, points-first at +0x70 behind the +0x118 switch), the +0x13C luminance mode and the quasi-log addressing (#1896); the sky LUT order and the +0xD0 dword that picks RGB9E5 or float outputs (#1897); the shared atmosphere constant buffer, its table slots (+0x220, +0xC60) and the layout with the two scale-height dwords no pass consumes (#3990). |
| [sandbox.md](sandbox.md) | Placed world props through one translated VS/PS pair with a real camera, TTPF albedo and the game's sampler: 442 of 446 draws textured; fly/walk/drive window, `tools/sandbox.py --check`. |
| [coder.md](coder.md) | The compressed mass in `story000_script.rpf`: entry framing (`align16(entry+8)` payload), the 32-value tag alphabet, deflate-shaped but proprietary, variable-length 7/8-bit LSB-first codes with no stored table, and the `n7 = 116..127` search that finishes it. |
| [mission_logic.md](mission_logic.md) | Mission state tables: 0x20 record layout (band counter, byte offsets), the 64-byte pool element with world transforms, case-folding joaat keys, 53.5 % of states named, 113 state-set clusters covering 82.4 % of files. |
| [gbuffer.md](gbuffer.md) | Deferred G-buffer targets (`PS_DeferredTextured` and all 2,989 `*Deferred*` pixel shaders, `PROC-146`): 5- and 6-target `OSG1`/`PSV0` signatures (`float32` + `exp ... compr` on all outputs), the 24-bit octahedral normal split across `mrt1.gba` (`2047.5` scale, `1/255` byte packing), sqrt-encoded `mrt2`, byte IDs in `mrt3.zw`, float motion in `mrt4`, and why binding `f16` targets mangles the normal bytes. |

## Other languages

Community quickstarts (stable onboarding text only; English docs remain authoritative): [i18n/README.md](i18n/README.md) (`fr`, `pt-BR`, `zh-CN`, `pl-PL`, `tr`, `ru`, `ja`, `es`, `it`, `de`, `ar`).
