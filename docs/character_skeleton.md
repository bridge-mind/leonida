# Character skeleton and skin format

Measured from the Xbox Series **1.0.2.1** package. This page covers the shared humanoid
skeleton stored in a character **fragment** (`.#ft`) and the model side that binds meshes to
it. It is separate from the world model exporter in [models.md](models.md): the world
drawables do not use this 597-name table.

No game data is reproduced here. Offsets, counts, field layouts and a few short name/hash
examples are enough to reproduce the measurements against a reader's own package.

The fragment carries the skeleton and a **25-vertex / 32-triangle bounds grid**, not a skin mesh. A
SixNG fragment-child walk finds no embedded drawables in any of the 2,865 `packs.rpf` character
fragments, while world fragments do have them; see
[character_models.md](character_models.md#what-is-not-decoded).

## The skeleton name table

Source resource: `ambient_femalebody_upper_nohead.#ft` in `packs.rpf`, stream offset
`0x45449F70` (stored 17,434 bytes, inflated 42,384 bytes).

This stream is the **name table**, not the skeleton fragment. `tools/anim_glb.py --skeleton` takes a
fragment offset; on 1.0.2.1 the sibling `ambient_femalebody_head.#ft` at `0x45417010` exports the
596-joint skeleton with a 5.13e-6 bind error, while passing `0x45449F70` fails with
`page directory: not a system pointer`.

| Range | Contents |
|---|---|
| `0x0008..0x7FF8` | 4,095-slot hash table, 8 bytes per slot: `u32 joaat32(name)`, `u32 boneIndex` |
| `0x8000` | page header; includes the `0x500280D0` pointer |
| `0x8088..0xA57E` | NUL-separated bone names (the first two entries repeat the fragment name) |
| `0xA57E` | `skel` marker |

The page contains **597 names and 596 hash entries**. Every matched hash equals
`joaat32(name)` over the lowercased ASCII name. The name table repeats the fragment name
(`ambient_femalebody_upper_nohead`), so 597 names share 596 hashes; the duplicate at index 596
is the entry without its own hash slot. Examples:

| Index | Name | `joaat32` |
|---:|---|---|
| 0 | `SKEL_ROOT` | `0x6686685C` |
| 1 | `SKEL_Pelvis` | `0x8133DE19` |
| 117 | `SKEL_L_Hand` | `0xA49BBD43` |
| 220 | `SKEL_Head` | `0x34C53E77` |
| 342 | `SKEL_R_Hand` | `0xB35ABA83` |

The table is a name/index dictionary, not a hierarchy. The hierarchy is in the next graphics
page.

## The skeleton instance table

The `.#ft` stream is followed by a 131,072-byte graphics page at `packs.rpf` offset
`0x4544F010`. Its `+0x08` pointer is `0x50027010`. The page-header pointer `0x500280D0` from
the `.#ft` page lands at `0x10C0` inside that graphics page. The page holds a **597-record
skeleton instance** starting at `0x150`, stride `0x50`:

| Record offset | Type | Meaning |
|---:|---|---|
| `+0x00` | `u32` | id/flags |
| `+0x04` | `u16` | next sibling |
| `+0x06` | `u16` | **parent index** (`0xFFFF` = root) |
| `+0x08` | `u32` | low 16 = bone index, high 16 = checksum |
| `+0x0C` | `u16` | bone index |
| `+0x0E` | `u16` | next/last-child link |
| `+0x10` | 4 × `f32` | reference rotation quaternion, `(x, y, z, w)` |
| `+0x20` | 3 × `f32` + pad | reference translation |
| `+0x30` | 3 × `f32` + pad | reference scale (all 1.0 in this page) |
| `+0x40` | `u32` + pad | pointer to another record |

Measured parent coverage:

- **580** non-root records have a valid measured parent.
- **16** records are roots: `SKEL_ROOT` (0) and the detached helper/attachment range
  572-595. `596` is the fragment-name record, not a deform bone.
- No cycles; every parent index is in range.

The parent field is enough to rebuild the real core hierarchy. Examples:

```
SKEL_ROOT(0)
  SKEL_Pelvis(1)
    SKEL_L_Thigh(2) -> SKEL_L_Calf(3) -> SKEL_L_Foot(4) -> SKEL_L_Toe0(7) ...
    SKEL_R_Thigh(38) -> SKEL_R_Calf(39) -> SKEL_R_Foot(40) -> SKEL_R_Toe0(43) ...
  SKEL_Spine_Root(106) -> SKEL_Spine0(107) ... SKEL_Spine6(113)
    SKEL_L_Clavicle(114) -> SKEL_L_UpperArm(115) -> SKEL_L_Forearm(116) -> SKEL_L_Hand(117) ...
    SKEL_Neck0(217) -> SKEL_Neck1(218) -> SKEL_Neck2(219) -> SKEL_Head(220)
    SKEL_R_Clavicle(339) -> SKEL_R_UpperArm(340) -> SKEL_R_Forearm(341) -> SKEL_R_Hand(342) ...
```

**Correction: the table above reads each record 0x40 bytes late.** The skeleton header that a
drawable points at with `+0x18` (the fragment's drawable for these humanoid fragments) gives the
record array, and in it a record starts with its transform:

| Header offset | Meaning |
|---:|---|
| `+0x28` | pointer to the 0x50-byte bone records |
| `+0x30` | pointer to one 64-byte **inverse bind matrix** per bone |
| `+0x40` | pointer to a `u16` parent index per bone (`0xFFFF` = root) |
| `+0x6C`, `+0x6E` | `u16` bone count, twice |

| Record offset | Type | Meaning |
|---:|---|---|
| `+0x00` | 4 × `f32` | default rotation quaternion `(x, y, z, w)`, local to the parent |
| `+0x10` | 3 × `f32` | default translation, local to the parent |
| `+0x1C` | `u32` | `joaat32` of the lowercased bone name |
| `+0x20` | 3 × `f32` | default scale |
| `+0x30` | pointer | bone name |
| `+0x46` | `u16` | parent index (the same value as the parent array) |
| `+0x48` | `u16` | the bone's own index |

So the table above's `+0x06` parent and `+0x08` index are `+0x46`/`+0x48` of this layout, and its
`+0x10`/`+0x20`/`+0x30` rotation, translation and scale are the **next** bone's. Measured on the
graphics page at `0x4544F010`: the parent at `+0x06` of each table row equals `+0x46` of the
record that starts 0x40 bytes earlier (597 of 597), that record's `+0x48` is its own index (596 of
596 bones), and a row's `+0x10` rotation is the next record's `+0x00` (596 of 596). Read from the
record start, the page's rotations and translations are those of the female skeleton below (596 of
596). The shifted pairing is why composing the table put the spine on the wrong side of the pelvis.

Read this way, the default pose is the bind pose: composing the default transforms down the
hierarchy (`world = parent_world * local`) reproduces every stored inverse bind matrix. The
largest difference between a joint's composed origin and the one its inverse bind matrix gives is
5.1e-6 m on the 596-bone female skeleton (`ambient_femalebody_head.#ft`) and 3.3e-6 m on the
560-bone male one (`ambient_malebody_head.#ft`). All 14,212 skeletons that read this way in
`packs.rpf` (13,900 drawable-dictionary children and 312 fragments) agree within 1e-3. The
matrices are row-vector (D3D) matrices; stored row by row they are glTF's column-major layout as
is. `tools/anim_glb.py` exports them as the glTF skin's `inverseBindMatrices` and checks them.

## Master skeletons and standing space

Master `crSkeletonData` tables (`root + 0x18`, bone count `u16` at `skel + 0x6c`, `0x50`-byte
`crBoneData` records at `skel + 0x28`, inverse bind matrices at `skel + 0x30`) live inside the 22
`ambient_*` ped fragment resources in `packs.rpf`. They fall into five master skeleton classes:

| Master skeleton fragment | Rig token | Named bones | `SKEL_*` | `BM_*` | `MH_*` | `FB_*` | `PH_*` | `OH_*` | `HH_*` | `RB_*` | `SEC_*` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ambient_femalebody` / `femalexxl` | `fr1` / `fo1` | **596** | 78 | 131 | 151 | 56 | 51 | 40 | 23 | 22 | 4 |
| `ambient_femalebb` | `fb1` | **574** | 78 | 131 | 134 | 54 | 50 | 40 | 23 | 22 | 4 |
| `ambient_malebody` / `malexxl` / `alien` | `mr1` / `mo1` | **560** | 78 | 131 | 129 | 46 | 51 | 40 | 23 | 22 | 4 |
| `ambient_malebb` | `mb1` | **552** | 78 | 130 | 126 | 44 | 49 | 40 | 23 | 22 | 4 |
| `ambient_kidbody` / `infantbody` | `mk1` / `mi1` | **454** | 78 | 104 | 124 | 44 | 15 | 36 | 0 | 22 | 4 |

Subsystem roles (counts shown for the 596-bone `ambient_femalebody` master rig):

- **`SKEL_*` (78):** Core animation tree — `SKEL_ROOT`, `SKEL_Pelvis`, a 7-segment spine
  (`SKEL_Spine_Root`, `SKEL_Spine0..6`), a 3-segment neck (`SKEL_Neck0..2`), `SKEL_Head`,
  clavicles, scapulae, limbs, 4-joint fingers (`Finger00..43`) and 3-part toes (`Toe0`, `Toe10`,
  `Toe20`).
- **`BM_*` (131):** Body-morph and anatomical definition drivers (`BM_L/R_Bicep`, `BM_L/R_Deltoid*`,
  `BM_L/R_Breast*`, `BM_Cleavage`, `BM_L/R_Butt`, `BM_Stomach*`, `BM_Spine0..5`).
- **`MH_*` (151 female / 129 male):** Muscle/crease helpers (`MH_L_KneeCrease`, `MH_L_PecUpper`,
  `MH_L_Butt`) and garment/jewelry anchors (`MH_BraStrap`, `MH_BeltLoop`, `MH_Necklace_*`,
  `MH_Choker_Spr_01`).
- **`FB_*` (56 female / 46 male):** Oral, jaw and mask rig — `FB_Root`, `FB_Jaw`, 24 individual
  tooth bones, an 11-bone articulated tongue, mouth bags, plus female mask and earring anchors.
- **`PH_*` (51) and `HH_*` (23):** Prop, weapon, ground-contact and holster attachment bones.
- **`OH_*` (40) and `RB_*` (22):** Orientation/recoil helpers and limb twist/roll bones.
- **`SEC_*` (4):** Dedicated secondary-motion breast physics bones (`SEC_L/R_Full_Breast`,
  `SEC_L/R_Mid_Breast`); buttocks and belly have zero `SEC_*` / `SPR_*` bones.
- **Ear and specialised helpers (40):** `EarRig_*` (10), `EarJaw_*` (2), `TGT_*` (2), `XH_*` (5),
  `EO_*` (4), `IK_*` (4), `Tuck_L/R_Tgt` (2), `CH_*`/`WH_*` (4), `UV_L/R_Scapular` (2), plus watch,
  ponytail, capsule-offset and camera-attach helpers. The 36-bone female-over-male delta is 39
  female clothing/jewelry/mask/scapular helpers minus 3 male-only bones (`MH_L/R_Toe`, `PH_HAT_LOC`).

**Standing-space coordinate convention:** In bind space, bipedal feet sit near `Z = -1.0 m` below
`SKEL_ROOT` at the origin, whereas modular drawable bounds (`root + 0x30..0x4f`) are authored in
standing space with **`SKEL_ROOT = (0, 0, +1.0 m)`** in RAGE Z-up (`(0, +1.0 m, 0)` in glTF Y-up)
so feet rest on the ground plane. Translating the skeleton root node (or static mesh vertices) by
`+1.0 m` vertically aligns all modular ped parts without per-slot offset tables; see
`tools/ped_assemble.py` and [character_models.md](character_models.md#standing-space-coordinate-convention).

## Model-side bone mapping

Each character drawable's `grmModel` header points at a **bone-hash array** (at `+0x28`, as
rpf9-explorer's `RageSkeleton.ReadModelPalette` reads it; an earlier version of this table said `+0x20`,
which in 3,000 sampled part models is never a hash list that resolves):

| `grmModel` offset | Meaning |
|---:|---|
| `+0x08` | geometry pointer array |
| `+0x10` | geometry count (also duplicated at +0x12 and +0x36) |
| `+0x14` | local bone count |
| `+0x18` | bounds pointer |
| `+0x28` | **bone-hash array pointer** |

The array is `local bone count × u32`, sorted ascending. Each hash is matched against the
597-entry `joaat32` table to obtain the global bone index. Measured coverage:

| Part | Local bones | Resolved to the 597 table |
|---|---:|---:|
| head | 51 | 51 |
| body | 28 | 28 |
| hands | 198/199 | 199 |
| feet | 35 | 35 |
| undies | 44 / 18 | 43 / 18 |
| brabikini | 75 / 33 | 55 / 33 |
| jacket | 107 | 0 (hash-only) |
| crop tee | 137 | 80 |

Unresolved hashes are genuinely absent from the ambient table; they should be emitted as
`bone_0x...` fallbacks, never given invented names.

## Vertex skin streams (ASSET-12)

Measured with `tools/skin_streams.py` over every drawable resource that carves out of `packs.rpf`,
`data.rpf` and `levels.rpf` of the 1.0.2.1 package: the tool walks the same drawable → LOD →
`grmModel` → geometry chain as the readers above, joins each resource's pages (system pages first,
then graphics pages, one zlib stream per page) and reads the FVF semantic table, whose per-semantic
entries are a `u32` stream offset at `fvf + 4*i`, a `u8` element size at `fvf + 0xD0 + i` and a
`u8` format at `fvf + 0x104 + i` (vertex count = `u32` at `fvf + 0x138` >> 10). A size-0 slot is
a placeholder with no stream. The candidate scan only starts a resource at a 4096-aligned offset
with 16 preceding zero bytes — rpf9-explorer's recovery convention — and does not recurse into
nested archives, so the counts are a floor for the package.

| Layout | Weights (semantic 16/17) | Indices (semantic 20/21) | Meshes | Vertices |
|---|---|---|---:|---:|
| 4 influences, 8-bit | 16: `R8G8B8A8_UNORM` | 20: `R8G8B8A8_UINT` | 38,721 | 40,631,019 |
| 8 influences, 8-bit | 16, 17: `R8G8B8A8_UNORM` | 20, 21: `R8G8B8A8_UINT` | 23,860 | 47,611,935 |
| 4 influences, packed | 16: `R8G8B8A8_UNORM` | 20: `R32G32B32A32_UINT` | 18,247 | 39,468,907 |
| 8 influences, packed | 16, 17: `R8G8B8A8_UNORM` | 20, 21: `R32G32B32A32_UINT` | 16,111 | 9,512,230 |
| rigid, 8-bit | none | 20: `R8_UINT` | 5,364 | 3,606,677 |
| rigid, 16-bit | none | 20: `R16_UINT` | — | — |

Total: 102,303 skin meshes / 140,830,768 vertices. The meshes split by archive: `packs.rpf`
45,854 meshes / 60,142,591 vertices, `data.rpf` 3,222 / 3,854,484 (all 4-influence 8-bit),
`levels.rpf` 53,227 / 76,833,693. No 16-bit index form appears in the top-level streams of these
three archives; the 16-bit forms were measured (over nested-archive recovery) in the independent
measurement linked below. The format names are the DXGI equivalents of the measured format bytes —
28 for the weight slots, 30 for the 8-bit index slots, 3 for the packed index slots, 62 for the
rigid 8-bit index slot — anchored on the decoded bytes, not on a name table: the 1.0.2.1 FVF
format byte is not desktop DXGI numbering (desktop `R32G32B32A32_UINT` is 3 but desktop
`R8G8B8A8_UNORM` is 24, and here it is 28; `R16_UINT` is 56 in desktop DXGI and 57 here, as
breadbbbb/rpf9-explorer#10 reads it).

Rules the bytes support, validated per mesh against the model's own bone count (`grmModel + 0x14`)
on a deterministic sample (96 meshes per layout per archive; the census itself covers every mesh):

- **The index is into the model's own bone list.** Every component with a non-zero weight has an
  index below the local bone count — zero violations in every sampled mesh; a zero-weight
  component may carry any index, so the check masks by weight.
- **The weights are the UNORM bytes and sum to 255** over all present weight slots. Every sampled
  vertex of every layout in `data.rpf` and `levels.rpf`, and of the 8-influence and 8-bit
  4-influence layouts in `packs.rpf`, sums to exactly 255. The packed 4-influence meshes in
  `packs.rpf` round: of 50,648 sampled vertices, 31,884 sum to exactly 255, 11,604 to 247–254,
  and 7,160 fall below 247 (as low as 164).
- **Packed `R32G32B32A32_UINT`: the bone index is bits 0–11 of each u32.** Bits 9–11 were never
  set in the sampled meshes; bit 8 appears only where the model has more than 256 bones. Bits
  12–31 hold a separate, dense, unidentified field (smooth per-vertex values in the high bits —
  not a weight: the weights are already complete in 16/17).
- **Rigid meshes carry one index per vertex and no weight slot.**
- The vertex buffer head carries one native buffer descriptor word per used semantic; every skin
  slot's stream offset (bits 7–28 of the word) matched its descriptor. The word's size code
  (bits 3–6) equals the element bytes for the 4- and 8-byte slots, reads 1 for the 16-byte packed
  index slot and 0 for the 1-byte rigid slot, so only the offset is checked.

This corrects the earlier reading of this section (semantic 20 as packed `(u16 bone, u16 weight)`
pairs, semantic 21 as weights): semantic 21 is always index-typed, and the weights are the UNORM
bytes of semantics 16/17. The earlier "16 bytes that do not form `(bone, weight)` pairs" note was
the packed index layout above. [An independent measurement of the same layouts, over rpf9-explorer
recovery with the bone-palette pull requests applied, was posted in
#634](https://github.com/ItsNotRexed/Leonida/issues/634#issuecomment-7074504326); the counts there
are higher because that recovery also walks nested archives.

## Animated glTF export

`tools/anim_glb.py` puts the three readers together: the skeleton above, the part meshes' skin
streams, and the clip dictionaries of `anim.rpf` (`tools/anim_clips.py`, see
[animation.md](animation.md#keyframe-decoding)). It writes one GLB with a glTF skin and one glTF
animation per `.clip`:

```powershell
python -I tools/anim_glb.py <package>\packs.rpf <work>\walk.glb --skeleton <fragment offset> `
    --part <part offset>[:<child>] --part ... `
    --anim <package>\anim.rpf --dictionary <clip dictionary offset> [--clip <name> ...]
python -I tools/anim_glb.py <package>\packs.rpf --coverage --skeleton <fragment offset> `
    --anim <package>\anim.rpf --jobs 8 --report <work>\coverage.json
```

On 1.0.2.1, `--skeleton 0x45417010` names the `ambient_femalebody_head.#ft` fragment;
`0x45449F70` in the section above is the name-table stream, not a fragment.

`--clip` matches the full clip label or the dictionary hash. When the hash is the leaf-name hash,
one value selects every path with that leaf: in dictionary `0xfc91b010`, `0xe61dc582` is
`joaat32("move")` and selects all six `/move` clips. Pass the full label (for example
`gameplay@locomotion@female_generic@normal@standing@walk@slope@up/move`) to export one clip.

Offsets are zlib stream offsets (as `tools/skin_streams.py` and `rpf9dump` report them). The
output is a standard glTF 2.0 file; Blender and the three.js editor play its animations. What
it does and why:

- **Joints** are the skeleton's bones with their default transforms as the rest pose, and the
  stored inverse bind matrices. The export refuses a skeleton whose default transforms disagree
  with them by 1e-3 or more.
- **Part meshes** bind through their model's bone list: vertex bone index → `grmModel + 0x28` hash
  → the skeleton bone with that `joaat32`. Weights are the UNORM bytes divided by their sum; up to
  eight influences become `JOINTS_0/1` and `WEIGHTS_0/1`. three.js reads only the first set, and
  eight matter here: on the arms part below, 3,746 of 5,821 vertices have more than four
  influences, with up to 0.37 of the weight beyond the largest four. `--max-influences 4` keeps
  the largest four, renormalised, for such viewers. A part with a *weighted* bone that is not in
  the skeleton is refused with the missing hashes, never bound to a guessed bone.
- **Clips:** `bonerotation`, `bonetranslation` and `bonescale` tracks drive the joint whose hash is
  the track's bone hash. Bone hash 0 drives a `mover` node above the skeleton (root motion;
  `--no-root-motion` leaves it out). Other track types are counted, not exported. A clip entry
  plays its animation's `[start, end]` window at `rate`, so animation time `t` is clip time
  `(t − start) / rate`. `(end − start) / rate` equals the clip's `+0x78` duration for all 785,174
  entries of all 61,505 dictionaries, to within 2.2e-4 relative (see [animation.md](animation.md#clip-dictionaries-are-rage-resources)).
- The whole skeleton sits under one node that turns RAGE Z-up into glTF Y-up. Skinned mesh nodes
  sit at the scene root, because glTF ignores a skinned mesh node's own transform.

Measured on the 1.0.2.1 copy:

| Check | Result |
|---|---|
| Skinned part drawables in `packs.rpf` (dictionary children with a bone list, no own skeleton) that read: positions, clustered triangles, skin streams | 9,571 of 9,571 (9,999 meshes: 9,091 8-bit, 908 packed indices) |
| … whose weighted bones all bind to the female skeleton | 6,359 (6,322 also bind to the male one) |
| … that use bones outside both skeletons (mostly clothing with its own cloth bones) | 3,212 |
| Clip bone tracks (rotation, translation, scale; mover left out) whose bone hash is a joint of the female skeleton | 9,167,307 of 10,137,825 (90.4%), in 52,752 dictionaries |
| … of the male skeleton | 9,164,859 (90.4%) |
| Dictionaries with at least 90% of their bones in the female skeleton | 44,346 |
| Clip rotation convention: joints whose default rotation is far from identity (rotation w below 0.9 in magnitude) and whose clip rotation is closer to the default than to its conjugate | 112 of 125, in four female locomotion and interaction dictionaries |

Clip translations match the default bone lengths except on helper bones that are meant to move
(`IK_*`, `PH_*` ground references and placements, `SKEL_ROOT`), so clip values are local
transforms in the same convention as the skeleton's defaults.

An export of a female head, arms and legs part with a locomotion walk dictionary (three clips)
was checked in Node with three r160's `GLTFLoader` and `AnimationMixer`:

- At rest, skinned vertices equal the stored ones turned to Y-up, within 4.1e-7.
- Over nine samples of every clip, joint matrices are finite and bone lengths of joints without
  translation tracks change by at most 3.9e-7.
- The figure is 1.72 m tall. The feet alternate front and back by about ±0.65 m, four steps per
  3.7 s clip, and the last frame equals the first.

The same body with 40 randomly chosen female clip dictionaries (240 clips) exports without error.
The Khronos glTF Validator reports no errors, warnings or infos on any of the 40 files, and the
three.js checks above pass on all 240 clips.

No textures are exported. A part binds to the one skeleton given, so clothing whose cloth bones
are not in it does not bind (see the table).
## Skinned GLB export

`tools/skinned_glb.py` writes one skinned glTF 2.0 binary from a decoded ped skeleton
`model.json` and one or more decoded drawable directories. It reads the drawable's palette
(`grmModel + 0x20`), maps each palette hash through the drawable's own skeleton/string table
and then the ped table, and reads the 8-influence streams from the graphics buffer. The
section offsets are also in `drawable.buffers[vertex_buffer].mesh_blob.sections` as
`[semantic, offset, format]`; the 23-word mesh-block header encodes
`offset = (word >> 8) * 2`, `format = word & 0xFF`.

| Component | Slot | Stream | Offset | Format | Bytes |
|---|---|---:|---:|---:|---:|
| head | 8 | `JOINTS_0`, 4 x u8 | 215,720 | `0x25` | 35,932 |
| head | 9 | `JOINTS_1`, 4 x u8 | 251,652 | `0x25` | 35,932 |
| head | 6 | `WEIGHTS_0`, 4 x u8 UNORM | 143,856 | `0x26` | 35,932 |
| head | 7 | `WEIGHTS_1`, 4 x u8 UNORM | 179,788 | `0x26` | 35,932 |
| jacket | 8 | `JOINTS_0` | 275,032 | `0x25` | 39,272 |
| jacket | 9 | `JOINTS_1` | 314,304 | `0x25` | 39,272 |
| jacket | 6 | `WEIGHTS_0` | 196,488 | `0x26` | 39,272 |
| jacket | 7 | `WEIGHTS_1` | 235,760 | `0x26` | 39,272 |
| *packed layout* | 8, 9 | `JOINTS_0/1`, 4 x u32 | section entry | `0x8C` | 16 x vertices |

Joint slots may also use the packed `R32G32B32A32_UINT` layout, format `0x8C`: 16 bytes
per vertex, four little-endian u32 words whose low 12 bits are the joint index (bits
12–31 are a separate unidentified field). The exporter converts it to the same 4 x u8 row
as `0x25` and rejects an index above 255. Only `0x8C` is accepted: no `0x8B` joint layout
has been measured, so `0x8B` is rejected rather than guessed.

The output has one skin. Its joints are the full ped skeleton plus any drawable-local helper
joints named by the palette; a palette hash that resolves nowhere is emitted as
`bone_0x........`, never given an invented name. Every vertex keeps all eight influences as
`JOINTS_0`/`JOINTS_1` and `WEIGHTS_0`/`WEIGHTS_1`. The source u8 weights sum to 255 and are
normalised to 1.0. Bind and inverse-bind matrices use the same Z-up to Y-up conversion as
positions, `(x, y, z) -> (x, z, -y)`.

```bash
python -I tools/skinned_glb.py out.glb \
    --skeleton <ped>/model.json \
    --component <head>/model.json \
    --component <jacket>/model.json
```

The measured head/jacket export uses 596 ped joints plus 107 garment helper joints (703
total). This is a static-export skip: only the components named on the command line are
exported, and no body or other unrequested component is assembled. No game data or generated
GLB is stored in the repository; `tests/test_skinned_glb.py` builds its own skeleton, skin
buffer and GLB.

`python -I tools/glb_inspect.py out.glb` reads the written file back with no glTF library and
prints its mesh, skin (joint count) and animation counts, the POSITION bounding box and any
field the exporter dropped; [models.md](models.md#skinned-glb-export) lists what the rest of
the report carries.

`python -I tools/glb_skinning.py out.glb` checks the skin data itself: each skin's joint and
`inverseBindMatrices` counts, the accessor types and finite values, which primitives carry
skinning data and which are static, `JOINTS_n`/`WEIGHTS_n` pairing, every vertex's weight sum
as FLOAT or normalized `UNSIGNED_BYTE`/`UNSIGNED_SHORT`, the joint-index range and the POSITION
bounds computed from the accessor. A malformed container, a file with no skin, or a skin whose
`inverseBindMatrices` count differs from its joint count exits nonzero; `--json report.json`
writes the same report as LF-only JSON. `python -I tests/test_glb_skinning.py` runs its 17
synthetic tests and needs no game files.

## Texture parameter split

The character material blocks carry several 32-bit parameter hashes before each 64-bit asset
key. Across the measured character materials, the texture content separates cleanly:

| Parameter | Common formats | Measured role |
|---|---|---|
| `0xC8E8C282` | BC1/BC3/BC7 sRGB, some BC1/BC7 UNORM | **sRGB albedo** when present |
| `0xC6F46E79` | BC7 UNORM, BC1 UNORM | `speculartex`: specular/gloss (packed), not a normal map |
| `0x1E77BF4C` | BC5 UNORM, some BC1/BC7 | `bumptex`: tangent-space normal/detail normal |
| `0x739E3D0D` | BC7 UNORM, some BC1 | `speculartex2`: ORM/mask (packed) |
| `0x8ADB8480` | BC4 UNORM | `materialid`: ID/palette map |

This is a measured corpus split, not a claim that the hashes are globally unique semantics.
The practical rule for exporters: use `0xC8E8C282` as albedo, and **do not fall back to the
first texture** when it is absent. A normal map painted as a base colour is the usual symptom
of that fallback.

To apply this rule, pass `--base-color-param C8E8C282` to `tools/world_build.py` (or
`base_color_param="C8E8C282"` to `build_world`): only references with that parameter hash are
used as base colour, and materials without a resolvable match stay untextured clay. Without the
option the exporter keeps its existing first-resolvable-reference behaviour, because these hash
roles are measured only on the character corpus.

## The Mandy search

Mandy is a KeyLento persistent/cutscene character schedule, not a unique drawable. `common.rpf`
contains the schedule entry `KeyLento/Perschars/Peds/Unique/Mandy` (pack
`PSP_KeyLento_perschars_peds`) in the `CPedScheduleInfoManager` XML. A read-only search of the
1.0.2.1 package found no `mandy`/`cs_mandy`/`ig_mandy`/`mp_mandy` drawable or fragment:

| Container | Search | Mandy drawable |
|---|---|---|
| `packs.rpf` | 144,364 streams / 9.74 GB; 26,585 decoded models at `e293917` (24,034 at `917e9b9`); 12,338 name records | none |
| `data.rpf` | 28,509 streams / 1.41 GB | none (localization/UI text only) |
| `common.rpf` | 954 streams / 50 MB | none (schedule and ped-appearance XML) |
| `levels.rpf` | 2,051 page-aligned nested archives, every zlib stream's first 4 MiB | none (an iFruit billboard and a house photo prop are the only name hits) |
| `story000.rpf` | 0-4 GiB and the 14 GiB region | none (animation clips `*_mandy.clip`, no drawable) |
| `story000_script.rpf`, `script.rpf`, `shared_minigame000.rpf`, rest-of-world, `audio.rpf`, `anim.rpf`, `movies.rpf`, `textures.rpf`/peds TTPF | full raw scans | none |

The name controls resolve `cs_kenton`, `u_f_o_rhonda`, `cs_monalisa` and `cs_lucia`; Mandy does
not. The delivered `mandy/` preview export is therefore a clearly labelled `cs_monalisa` proxy,
not a verified Mandy model.

Two further checks refine that negative. A missing `mandy` name is the common case, not an
anomaly: of the 13 `KeyLento/Perschars/Peds/Unique` entries, two resolve cleanly by name
(`kenton` -> `cs_kenton`, `rhonda` -> `u_f_o_rhonda`), one plausibly maps (`Bert` ->
`hc_cs_herbert_watson`), and the other ten, including `Mandy`, have no name hit. Also, the
package manifest lists 108 files and its `pack/` directory contains only `story000`,
`core_restofworld`, `core_restofworld001`, `shared_minigame000` and their `_bvh4`/`_licenses`
companions; there is no `psp_*` archive, so the `PSP_KeyLento_perschars_peds` pack named by the
schedule is not present in this build. Mandy is therefore not disproven: her appearance is either
a generic ped assembled at runtime or in a pack this package does not include.

Caveats: `levels.rpf` was prefix-bounded at 4 MiB per inflated stream, raw RSC8 bodies and
Bink/opaque regions were not decoded, and the encrypted RPF9 table of contents prevents a full
path enumeration (27 top-level entries remain). The evidence supports a shared ped-appearance
actor assembled from existing components, not a unique `mandy` resource.
The mission binding is now measured. `COM_KBOS_NEIGHB` is the internal id of the "Pipework"
mission: the replay table carries that `<MissionId>`, and the mission text names "Mandy's
house", "Follow Mandy" and "Bert died". Five PSIN state tables in
`pack/story000/story000_script.rpf` (`03fe4000`, `03fe8000`, `03feb000`, `03fee000`,
`03ff0000`) contain both characters as entity link hashes: `0x03A885E6` (`Mandy`) and
`0xE34ACBC8` (`Bert`); the schedule pack hash is `0x865F0DA9`
(`PSP_KeyLento_perschars_peds`).

The model binding is also decoded. `PERSCHARS_INFO` (`data.rpf@0x1AA5F000`, record 3499) maps
`Name = 0x03A885E6` (`mandy`) to `ModelSetName = 0x03A885E6`; `CPedModelInfo__InitData`
(`packs.rpf@0x11D1FF000`, record 1219) maps that link to `cs_mandy` (`0xD7C0CDFF`). `cs_mandy`
is a live model id — a `story000.rpf` cutscene actor/voice table uses it — but there is **no
named `cs_mandy` drawable**. The `cs_mandy` ID above is the 32-bit `ModelName` at row offset
`+0x0C` (schema type `0x4D`). The adjacent 64-bit `FragName` at `+0x10` (type `0x4F`)
`0xC9888B65009B33B7` is the shared female cutscene base, carried by **904** female rows:
`player_story000_00` (Lucia's model id `0xE472B687`), `cs_monalisa` (`0x9CB9BCC6`) and
`cs_mandy` (`0xD7C0CDFF`), plus 901 others. No ped-model row carries `joaat("cs_lucia")`
(`0x95F1E309`); that is a name prefix, not a model id. A second base
`0x0A704516C676E805` is shared by `cs_shanese` (`0x3570D92E`) and 11 unnamed female rows, and
8 female models have wholly model-specific fragments. None resolves to a name, and neither
table contains an adult-token, `mp_f_*`, `nude`, `topless`, `stripper`, `prostitute`, `dancer`,
`escort`, `hooker`, `swim`, `lingerie` or `club` female model. The realistic full-body in the
shared base is `cs_lucia_legacypedcs`
(874 v), so Mandy uses the same body as Lucia. Her nine-entry variation table has 18 u32 hashes;
16 are unique to her. A deeper search — 373,484 shipped names / 725,869 normalised variants plus
about 25 million generated strings — gives **0/18** exact matches. The sibling nine-entry table
resolves 9 of 18 values to `joaat("outfit_01")` … `joaat("outfit_10")` (frequency corroborates
it), so the table stores build-time design labels, not shipped asset names; the 18 component
values are not recoverable by name enumeration. `common.rpf` holds `CPedVariationBodyTypeManager`
(77,422 bytes, 67 keys) and `CPedComponentSetManager`; neither maps a variation index to a
drawable plus texture. The first manager's keys are literal model names, and `joaat32` of all
67 gives 0 hits against the 18. The `3cba3fe6` float (0.942 / 0.971 for Lucia and Monalisa) is a
per-character body/height scale, not a name; there is no component/outfit-name manager in
`common.rpf`, and `outfit_NN` exists only as a hash in the packs table.

The entities linked to Mandy carry three different per-state archetype hashes
(`0x18BB2550`, `0x73E83886`, `0x57DDFFC2`), which fits her outfit changing between states, not
three models. None of those hashes resolves through the package name dictionary, any drawable
or prop name pool, the `packs.rpf` PSOs, or a `0x7a` meta hash scan, so they are state/archetype
ids, not model names. The archetype-to-component-set lookup remains unidentified.

**Component variation (ASSET-279).** At `packs.rpf@0x11D1FF000`, the first 16 inflated 64 KiB
pages cover the instance blocks and row schema and end inside LSNS. The 16-byte root has one
array of 2,714 model rows, each 384 bytes. The row schema has no array or map, and both struct
pointers (row offsets `+0x118` and `+0x178`) are null in every row. Only record 1219 carries its `ModelName` in this table. Its `FragName`
occurs in 906 rows, including the 904 female rows counted above.
This layout provides no inline array or live struct pointer for either nine-entry table.

At `data.rpf@0x1B076000`, the PSIN walker reached 25 structures across all seven blocks
(756/756 instance bytes). One owner embeds both 12-byte settings at owner offsets `+0x0C`
and `+0x18`. Each holds a `textureIndex` and two floats. Four separate root maps contain
seven entries between them. The two settings are inline fields, and their subsystem is unknown.
Two XML roots in `common.rpf` describe scale and body-shape profiles (`0x226000`) and entity
components (`0x228000`). A byte comparison against the 1,576,148-byte character metadata prefix
at `data.rpf@0x1AA5F000` found none of the settings candidate's nine distinct nonzero typed
hashes at their original little-endian widths. That prefix did not validate as a complete PSIN.
None of these checks establishes a variation selector. An external lookup keyed by a model-row
field remains possible; confirming it requires tracing a component variation to both its
drawable and its texture.

`Mandy_FirstTingle`, `Mandy_ComingUp` and `Drug_Mandy_*` in `common.rpf` are MDMA drug effects,
not the character. They are a red herring in any `mandy` name search.

Caveats: the RPF9 table of contents is still not decoded; names come from carving and recovery
(318,266 unique `levels.rpf` names, 24,034 `packs.rpf` names), and raw RSC8 bodies and
Bink/opaque regions were not decoded. The evidence supports a shared ped-appearance actor: a
`cs_mandy` model id, a body fragment shared with `cs_lucia` and `cs_monalisa`, and a variation
table, not a unique `mandy` drawable.

## Reproduction

Steps 1-4 were measured with local tools that are not part of this repository; a reader can
reproduce them with a standard zlib/RPF9 reader:

1. Inflate the `.#ft` **name-table** stream at `0x45449F70`; parse the 4,095-slot hash table and the
   NUL-separated names; verify `joaat32` lowercased.
2. Inflate the following 131,072-byte graphics page at `0x4544F010`; parse the records at
   `0x110 + 0x50*i`: rotation at +0x00, translation at +0x10, hash at +0x1C, parent at +0x46
   (see the correction above). `tools/anim_glb.py` reads the same records through the
   skeleton header at drawable `+0x18`.
3. For a character `#dr` system page, read `grmModel+0x28`, then the `u32` bone hashes.
4. Map each hash through step 1; unresolved hashes are hash-only fallbacks.
5. Run `python -I tools/skin_streams.py <your copy's packs.rpf> [...] --json out.json` for the
   skin-stream census and validation; `python -I tests/test_skin_streams.py` runs the same walk
   and the same checks on synthetic resources built by the test.
6. Run `python -I tools/anim_glb.py ... --coverage` (above) for the clip-track counts, and
   `python -I tests/test_anim_glb.py` for the export on synthetic skeleton, part and clip data.

No archive, model, texture or audio file is stored in the repository or in any test fixture.
