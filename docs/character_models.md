# Character models and ped parts

Measured from the pinned `e293917` export of `packs.rpf` (`tools/rpf9dump models`): **26,585
manifest rows**, **11,335 unique names**, **23,720 `.#dr` drawable rows** and **2,865 `.#ft`
fragment rows**. One row is one LOD or variant, so these are row counts, not character counts.

No game data is reproduced here.

## A character is a drawable plus a fragment

- A `.#dr` drawable carries the geometry, materials and texture references.
- A `.#ft` fragment carries the shared 597-name humanoid skeleton and the character's bounds. The
  measured skeleton is in [character_skeleton.md](character_skeleton.md).

The current exporter writes every fragment as a **flat 25-vertex / 32-triangle bounds grid** (a 5x5
grid at z=0, `Min`/`Max` z = 0) with `Embedded: null` (2,865 of 2,865). Fragment child drawables
are not walked, so a part that exists only as a fragment child is not in the OBJ/GLB output even
though its parent fragment is.

## Lucia records

22 unique names contain `lucia` (12 contain `cs_lucia`). They are parts, not one assembled model.
The decoded export used by the previews contains 13 parts:

| Part | Source drawable | Vertices | Triangles | Diffuse key (`0xC8E8C282` only) |
|---|---|---:|---:|---|
| head (hi) | `player_story000_00_head_fr1_lat_000_m.#dr` | 8,983 | 13,320 | `0x8EDB5664DA38C47E` (head-skin preset) |
| head L0 | same | 4,638 | 4,940 | same preset |
| head L2 | same | 599 | 514 | same preset |
| eyes | `…_head_fr1_lat_000_m_eyes.#dr` | 434 | 816 | `0x429453B7A2EB9980` (sRGB + alpha) |
| eyecap | `…_head_fr1_lat_000_m_eyecap.#dr` | 214 | 304 | `0x5CE55BAA340248BD` (sRGB + alpha) |
| hands | `hand_fr1_000_nude_cull.#dr` | 10,440 | 13,534 | no exact hit for `0x4D89F2311DBBB7A8` in the 57 scanned TTPF pools (see [Hands and feet](#hands-and-feet)) |
| feet | `feet_fr1_000_full.#dr` | 2,309 | 4,082 | no exact hit for `0xA54D9F8B51CA6034` in the 57 scanned TTPF pools (see [Hands and feet](#hands-and-feet)) |
| body (legacy) | `cs_lucia_legacypedcs_fr1_000_fullbody_ws00_full.#dr` | 874 | 908 | `0x996FDBA2DB30CC12` |
| legacy head | `cs_lucia_legacyheadcs_fr1_001_regular_ws00_full.#dr` | 592 | 1,028 | `0x707771F769AA3542` |
| jacket (biker) | `cs_lucia_jacketbiker_fr1_000_regular_c_hl_long_full.#dr` | 9,818 | 10,837 | none — `speculartex` only (`0xF5073AA039BB3CED`) |
| crop tee | `player_story000_00_tshirtcrop_fr1_001_slim_cc_un_long_full.#dr` | 4,932 | 6,756 | none — `speculartex` only (`0x46E8FA9436E9D43C`) |
| bikini top | `brabikini_fr1_012_regular_ws00_full.#dr` | 6,547 | 5,697 | `0xDF3615ADFB1F4787` (sRGB + alpha) |
| undies | `undiesgstring_fr1_002_slim_hw_full.#dr` | 961 | 913 | `0xCFD339832A9C724A` (sRGB + alpha) |

### Higher-detail variants

The rows above are the preview build, not the densest rows in the archive. LOD0 counts come from
the manifests; "distinct" is unique vertex positions decoded from a GLB, which removes duplicate
cull seams:

| Component | Denser variant | LOD0 V / T | Distinct |
|---|---|---:|---:|
| jacket | `player_story000_00_jacketbiker_fr1_000_regular_o_hl_long_full` | 11,093 / 12,023 | 9,597 |
| crop tee | `player_story000_00_tshirtcrop_fr1_001_slim_cc_un_long_cull_00` | 12,254 / 15,437 | 4,558 |
| jeans | `player_story000_00_pantsjeans_fr1_002_regular_hw_mid_full` | 4,917 / 5,550 | 2,826 |
| boots | `cs_roxy_bootsankle_fr1_000_mid_hh_full` | 7,764 / 7,960 | - |

The library maximum for jeans is `pantsjeans_fr1_019_slim_mw_long_full` at 23,880 V / 25,230 T;
that is a single 0.41 x 0.26 x 0.95 m mesh, but the raw count includes seams, so validate it
visually. `_cull_00` variants can have more raw vertices and less surface than their `_full`
sibling; compare distinct positions, not the manifest total.

Two caveats that matter for any render:

- **`body` is the shared player/cutscene body.** The only `lucia`-named full-body record in
  `packs.rpf` is `cs_lucia_legacypedcs_fr1_000_fullbody_ws00_full`. Its LOD0 is **874 vertices /
  908 triangles** (468 distinct positions); the 1,748 / 1,816 figure is LOD0 + LOD1 counted
  twice, and the 640-vertex row is LOD2. Its `FragName` is byte-identical to `cs_monalisa` and
  `cs_mandy`, so it is the shared player/cutscene body, not a unique high-detail mesh. Do not
  present it as a high-detail hero body.
- **The head drawable has no material texture references**, but its face comes from a **head-skin
  preset** keyed by the head name's `joaat32` (see Textures).

`player_story000_00` (Lucia's slot prefix) has 104 unique names: **63 drawables** plus **41
`char_bipedbounds` fragments**. The largest drawables:

| Drawable | Vertices | Triangles |
|---|---:|---:|
| `…_hoodiezip_fr1_001_regular_o_hd_long_cull_00.#dr` | 20,311 | 14,256 |
| `…_sweatshirtcrewneck_fr1_000_loose_cc_un_long_cull_00.#dr` | 19,217 | 13,600 |
| `…_hoodiezip_fr1_001_regular_h_hd_long_cull_00.#dr` | 19,141 | 14,106 |
| `…_shirtdress_fr1_000_slim_c_oc_un_long_cull_00.#dr` | 14,341 | 16,146 |
| `…_tshirtcrop_fr1_001_slim_cc_un_long_cull_00.#dr` | 12,254 | 15,437 |
| `…_jacketbiker_fr1_000_regular_o_hl_long_full.#dr` | 11,093 | 12,023 |

## Slot tokens

These are counts of unique names containing the token. They are a naming heuristic, not the game's
component table.

| Token | Unique names | Note |
|---|---:|---|
| `player_story` | 196 | Lucia (`000_00`) and Jason (`000_01`) |
| `head` | 3,741 | heads and head attachments across all characters |
| `hair` | 278 | includes accessories and non-human fur |
| `belt` | 44 | accessories |
| `pants` | 459 | `pantsjeans`, `pantscargo`, `pantsleggings`, `pantstrack`, … |
| `jeans` | 157 | `pantsjeans` / `shortsjeans` |
| `jacket` | 327 | `jacketbiker`, `jacketdenim`, `jacketblazer`, `jackettrack`, … |
| `uppr` | 17 | 14 are `slod_uppr_*`; 3 are `char_bipedbounds_female_armor…_uppr_l/r.#ft` |
| `lowr` | 21 | 17 are `slod_lowr_*`; 4 are `char_bipedbounds_female_armor…_lowr_l/r.#ft` |
| `jbib` | 0 | the GTA V slot code is not a name token here |
| `hands` | 8 | the singular `hand_` appears in 35 names |
| `feet` | 26 | bare feet plus creature variants |
| `bikini` | 45 | `brabikini_*` and related |
| `undies` | 102 | `undiesgstring`, `undiesbriefs`, `undiesboxers`, … |

## Textures

The character export requested 215 texture keys and matched **184**. The only sRGB albedo slot is
`0xC8E8C282` (`diffusetex`); `0xC6F46E79` is `speculartex` (specular/gloss pack), `0x1E77BF4C` is
`bumptex` (the tangent-space normal), `0x739E3D0D` is `speculartex2` (packed ORM), and
`0x8ADB8480` is `materialid` (an ID/palette map). The jacket's and the crop tee's material rows
reference one (`0x6CD70ED1054EF726`, `0xF0722B975F329141`, both `BC4_UNORM` textures in the
`ttdb_peds` pools); the bikini top's and the undies' rows have no such slot. Whether a shader
samples it is not measured. The garment
shaders (`d5ed50f4`, `ea8d09d0`, `39ddb9ca`) do not declare a `0xC8E8C282` slot at all: none of
their 7,948 material rows in the export has one. The jacket and the crop tee are on them, so they
have no albedo. A part without one keeps a flat colour; it must not
be painted with `speculartex` or an ORM map. The garments' `_COLOR0` vertex data is baked shading
(alpha 255 on every vertex), not a tint, and the shader
declares no tint or palette slot. No authored base colour exists to recover for those two.

The bikini top and the undies are not on the garment shaders, and they do have an albedo. All
three records of each drawable bind `0xC8E8C282` to the key in the table: the bikini top on
shaders `84af5c61` and `81153bcc` (`76c45dd8` and `428b44d6` in its smallest record), the undies
on `a9dfb1e7` (`428b44d6` in its smallest record). Every material row on those five shaders in
the export binds the slot (1,372 of 1,372). Both keys are 512x256 `BC3_UNORM_SRGB` textures with
7 mips in the `ttdb_peds` pools: an sRGB format with an alpha channel, as the eyes' and eyecap's
`BC7_UNORM_SRGB` keys are. Their `speculartex` keys are
other textures (`0x93CB433D30220BA6` and `0x5D237CCCD91F8E4D`, 256x128 `BC7_UNORM`), where the
jacket's and the crop tee's table keys are `BC7_UNORM` `speculartex` sheets themselves.

The hands and feet bind albedo keys with no exact hit in the scanned TTPF catalogs (see [Hands and feet](#hands-and-feet)).
The head drawable has no material texture reference, but its face is selected by
a **head-skin preset** keyed by the head name's `joaat32`:

| Head | Face albedo | Face normal |
|---|---|---|
| `player_story000_00_head_fr1_lat_000_m` | `0x8EDB5664DA38C47E` | `0x953715153D75447C` |
| `cs_monalisa_head_fr1_afr_000_m_full` | `0x53A2896F4E5A6BA6` | `0xE82C7F6E02C8B701` |

The preset library covers 897 heads; `cs_lucia_head_fr1_lat_000_y_full` is not in it.

### Hands and feet

Measured against the catalogs of all 57 TTPF pools (357,115 textures). The hands and the feet bind
the four maps on the same two shaders (`210FBC6A`, `8D69EDAB`); for the two examples below,
**only the normal-map keys have exact catalog hits**. "None" means no matching full 64-bit
`AssetKey` in these scanned catalogs, not proof that no runtime source exists:

| Part | `diffusetex` | `speculartex` | `bumptex` (normal) | `speculartex2` (ORM) |
|---|---|---|---|---|
| hands (`hand_fr1_000_*`) | `0x4D89F2311DBBB7A8` — none | `0xA805B9A446CCE950` — none | `0x9D337DABCA636906` — `ttdb_peds_2_3`, 512×1024 BC5 | `0x09D2C98C5B40D16C` — none |
| feet (`feet_fr1_000_*`) | `0xA54D9F8B51CA6034` — none | `0x752EA8A7A086558F` — none | `0x5091E86DE6196BC3` — `ttdb_peds_3_3`, 512×1024 BC7 | `0xD5C8D2A180326DE7` — none |

Each material has ten slots; the other six bind `0x5D7A3A87CC19967D` on both shaders, with no
exact pool hit. The [independent indexed-record check](https://github.com/ItsNotRexed/Leonida/pull/3256#pullrequestreview-6014627459)
found this same reference in all 176 material
rows, including the 12 positive-control rows whose four named maps all resolve. An engine-supplied
default is a hypothesis: the repeated unresolved reference alone does not establish its runtime use.

**Shared static material references.** The hands and feet have bindings in common with body skin
pieces in `pack/core_restofworld/core_restofworld.rpf`; these references do not establish runtime
sheet selection:

- `bodynipplesgen_fr1_000_regular_ws00_full`, the female upper body, binds **the same four keys**
  as the hands, on the same two shaders.
- `bodygenitalsgen_f{r,b,o}1_008_regular_ws00_full`, the female lower body (and the `fr1`
  `_alt_00`–`_02` variants), binds the feet's
  `speculartex`, `bumptex` and `speculartex2`, and its own `diffusetex` `0x5B6829CFEC0F41A1`, also in
  no pool.
- The male lower body, `bodygenitalsgen_mr1_000_*`, has all four maps in the pools
  (`diffusetex` `0xBCE4D91DA23689C7`).

The unresolved female-body albedo, specular and ORM references above also have no exact hit in
these pools. The recorded search found none of these keys in another resource of `data.rpf`,
`common.rpf`, `script.rpf`, `packs.rpf`,
`anim.rpf`, `core_restofworld.rpf` or `core_restofworld001.rpf` (`story000.rpf` and
`story000_restofworld.rpf` recover no entries with this reader). These are static lookup results;
the source of skin colour and the runtime's sheet selection remain open.

Across 162 indexed records (176 material rows, 56 distinct `hand_`/`feet_` drawable names), all
176 rows carry the four named references. Of the 56 names in `packs.rpf`, 52 have no exact
diffuse-key hit in the scanned pools; the four exceptions with all four maps resolved are
`hand_fsk1_000_nude_cull`, `feet_fsk1_000_nude_cull` and the first-person hands
`hand_mb1_000_nude_cull_fp` and `hand_mo1_000_nude_cull_fp`. The male hands and feet
(`hand_mr1_000_nude_cull`, `feet_mr1_000_nude_cull`) also have exact hits only for their normal
maps, and male and female feet bind the same `diffusetex` key. For previews, use the resolved normal
map and a flat skin tone; this is a fallback, not a reconstruction of the runtime's skin colouring.
Do not borrow another part's albedo.

## Materials

A binding audit of 120 exported scenes (232 material rows) found 35 rows with a real `C8E8C282`
albedo and 174 rows with `C6F46E79` but no `C8E8C282`, all on garment shaders. The four rows that
produced teal previews were the specular sheet used as the base colour, not a normal map. Those
parts have **no albedo**, so a preview must render them as clay rather than invent a colour.

- **The belt is not a separate part.** It is 172 connected components / 2,229 vertices inside the
  jeans mesh, with no node, material or texture key; it cannot be separated without editing the
  geometry.
- **UVs above 1 are wraps, not atlas pages.** The blazer reaches U=2.0014 (1,628 vertices); after
  subtracting 1, 90.9% of those vertices are within 1e-4 of a page-0 vertex. The jeans reach
  U=-0.9936, a -1 mirror.
- **The body atlas is a shared collage.** `996FDBA2DB30CC12` is 256x256 and contains several ped
  variants (faces, a tropical shirt, blue jeans, dark cloth). Only 43.7% of the sheet is covered
  by the body's islands, so bilinear sampling at an island edge can read a neighbouring variant.
  The blue in an earlier preview came from the authored `baseColorFactor` `(0.200, 0.270, 0.400)`,
  not from this atlas.

## Bodies

A geometry-first census — every `packs.rpf` drawable and fragment, all 50,802 `levels.rpf`
fragments, 178,073 of 184,778 `levels.rpf` drawables (96.4%; the 6,705 above 3 MB were not
decoded), all 4,195 `story000.rpf` drawables, and the `anim.rpf` rig dictionaries — found no
Lucia or Mandy body. `story000.rpf` is mission map cells, not ped actors: its full read finds
4,195 drawables, 177,811 meshes, **0 skeletons, 0 fragments and 0 body-like names**, 28.5 M
vertices of unrigged 64 m cells. `anim.rpf`'s `LOD0_Body_Shapes_Female` is a rig program with no
mesh.

A raw RSC8 scan of **862,555** resources across all 13 indexed RPF9 archives (every `rageobj`
entry, unnamed pages and the >3 MiB band included; 25.6 GB compressed / 56.3 GB inflated) found
2,631 ped-skeleton resources: 342 animal bounds, 2,212 `char_bipedbounds_*` fragments,
4 mannequins and 73 anim clips. **0** combined a ped skeleton with the skin/flesh shader, and
0 of the 908 resources that use the skin/flesh shader contained a single `SKEL_*` bone name.
The unnamed-resource route therefore does not add a body.

The only **nude** continuous human-skeleton meshes are a **generic female mannequin** in
`levels.rpf`: `gmt_mannequin_proxy_f_01a` (15,488 vertices / 27,725 triangles, 95 bones,
1.72 m, A-pose) and `gmt_mannequin_f_01b` (the same body posed mid-stride). It has no Lucia or
Mandy name or texture link, and its rig is a clothing form, not the 596-bone ped skeleton. 78 of
95 bone names are shared, and all 71 weighted bones match; its pivots are a rigid 180° yaw plus a
31.7 mm drop, not a mirror, so no L/R weight swap is needed. A 39.59° upper-arm correction lands
the hand on the ped pivot. It has no skin albedo — only a flat tint swatch atlas — so it is an
untextured base proxy for a reconstruction, not Lucia's body.

A separate bare female mesh exists only as a **scene monster** in `packs.rpf`:
`sc_fast_demon_bodyuppergen_fr1_000_m_full` (10,048 v / 13,124 t) and
`sc_fast_demon_bodylowergen_fr1_000_m_full` (3,717 v / 4,082 t), shader `0x210FBC6A`, with no
garment parts. It is a monster asset, not a player or ped body. The `bodyupper`/`bodylower`
families confirm the negative: all 18 rows are human male (`cs_farmm`, `cs_mister_mangrove`,
`u_m_m_farm_monster_man`) or monsters (`sc_*`), none female. The `hc_stripper_delights`,
`hc_stripper_joh` and `hc_stripper_nine1nine` characters ship heads and eyes only, with no body.

The player and cutscene body is a separate shared **clothed** fragment,
`cs_lucia_legacypedcs_fr1_000_fullbody_ws00_full` (874 v), used by `cs_lucia`, `cs_monalisa`
and `cs_mandy` (their `FragName` is identical). It is the body the game uses; it is low-poly and
clothed, so it is not a high-detail nude body. A flat-skin render of the 874-v body with its
normal map disconnected still shows the modelled top, shorts and ankle boots, so the clothing
silhouette is geometry, not a painted-on atlas.

## Modular ped construction pipeline and body assembly

RAGE 9 constructs human peds from modular streamed slots rather than monolithic rigged meshes.
Spawning a character resolves a five-stage data graph across `packs.rpf`, `data.rpf`, `common.rpf`,
`anim.rpf` and `data/ttdb/peds/*.ttpf`:

1. **Identity and archetype table.** `packs.rpf` archetype PSO (`00011d1ff000.pso`, root
   `0xdd77771e`, 2,714 records, 384 B/record) holds each archetype's `joaat32` hash (`+0x08`) and
   voice/short name (`+0x174`).
2. **Outfit slot resolution.** `packs.rpf` outfit-slot PSO (`00011cfbf000.pso`, root `0x890aca98`,
   2,713 archetypes -> 33,621 outfit hashes) maps each archetype to valid outfit slot hashes with a
   1:1 link to compiled `0DRP` appearance catalogues.
3. **Compiled appearance catalogues (`0DRP`) and variation pools.** 2,709 resource-packed `0DRP`
   catalogues (`0x8dcb4cf5`, v122), the wardrobe catalogue (`00011ccb2000.pso`, 19,827 presets) and
   80 `0x1edee47e` variation PSOs resolve 88 B presets (`0xf795e8ba`) and 184 B wearable states
   (`0x37e52277`) to `murmur64` drawable, albedo, normal and `cull_drawable` (`0x46e1a048`) keys.
4. **Modular geometry, skeleton, deformation and strand grooms.** Three streamed geometry LOD tiers
   per slot, master `crSkeletonData` in `ambient_*` fragments (454–596 named bones; see
   [character_skeleton.md](character_skeleton.md#master-skeletons-and-standing-space)),
   `char_bipedbounds_*` collision rigs, 6,756 quantized blend-shape delta buffers, 449 ML Deformer
   weight models and 5,751 `smh` GPU strand grooms (20,046 LOD resources; hair drawables in
   `packs.rpf` are 3-vertex / 1-triangle stubs).
5. **GPU compute skinning, morphing and deferred/RT shading.** `ped_morphable_model_cs` ->
   `ped_standard_head_cs` (blendshapes, sticky lips, per-meshlet stretch/wrinkles) ->
   `ped_skinning_cs` -> `ped_skin_processing` / `ped_standard_skin_tattoo_bake` -> `bvh_skinning`
   -> `meshlet_culling` (`CS_BatchCullPreSkinnedMeshlets`) -> deferred G-buffer, GPU subsurface
   scattering and ray-tracing resolve.

### Standing-space coordinate convention

All human modular drawables share a single standing skeleton space anchored at
**`SKEL_ROOT = (0, 0, +1.0 m)`** in RAGE Z-up (`(0, +1.0 m, 0)` in glTF Y-up). Vertex buffers
store root-relative bind positions (feet sit near `Z = -1.0 m`), while drawable bounding boxes
(`root + 0x30..0x4f`) store standing-space bounds. Translating skinned parts by `(0, 0, +1.0 m)`
aligns `head`, `eyes`, `eyecap`, `teeth`, `hand_*_nude_cull`, `feet_*_nude_cull`,
`bodynipplesgen_*`, `bodygenitalsgen_*` and garments to sub-millimetre precision without per-slot
offset tables. Disabled slots use `<12 cm` at-root stub meshes at `(0, 0, -1.0 m)`.

### Modular body slots, `_nude_cull` islands and watertight stitching

Despite their legacy `hand_` and `feet_` prefixes, the canonical base nude body drawables cover the
entire torso and pelvis:

- **Upper body (`hand_<rig>_000_nude_cull`):** torso, chest, shoulders, arms and hands (first-person
  arms ship separately as `hand_<rig>_000_nude_cull_fp`).
- **Lower body (`feet_<rig>_000_nude_cull`):** pelvis, hips, buttocks, legs and feet.
- **Female nipples (`bodynipplesgen_fr1_000_regular_ws00_full`):** separate `fr1` chest patch
  sharing the `hand_fr1` `512×1024` UV layout (male `mr1` torsos bake nipples directly into
  `hand_mr1_000_nude_cull`).
- **Adult genitals (`bodygenitalsgen_fr1_008_*` and `bodygenitalsgen_mr1_000_*`):** modular `fr1`
  and `mr1` pelvic patches with four geometry variants (`base`, `_alt_00`, `_alt_01`, `_alt_02`;
  female depth `6.83–7.58 cm`, male resting `8.9–9.0 cm` and extended `13.4–13.5 cm`).

RAGE 9 uses a two-level culling architecture to prevent skin poke-through without runtime mesh
rebuilds:

1. **Catalogue swaps (`0x46e1a048` = `"cull_drawable"`):** 385 unique drawables carry `_cull`
   suffixes — 351 `_cull_00` garments (348/351 paired directly with a `_full` sibling) plus 28
   `_nude_cull` / `_nude_cull_fp` base skin meshes across the 12 rigs. Wearable states in `0DRP`
   catalogues map `0x46e1a048` to `0x3d780b74` records (`25,292` resolving to `*_cull*` drawables).
2. **Physical `_nude_cull` cull islands and coplanar overlap bands:** Base nude drawables are
   physically split into hundreds of disconnected island patches along garment cuts:

| Drawable | Physical cull islands | Tier 0 V / T | Measured coplanar overlap |
|---|---:|---:|---|
| `hand_fr1_000_nude_cull` | **449** | 10,440 / 13,534 | **258** waist tris (`Z = 0.8981..1.0367 m`, 13.86 cm) + **26** nipple tris |
| `hand_mr1_000_nude_cull` | **290** | 9,914 / 13,534 | **258** waist tris (`Z = 0.9346..1.0739 m`, 13.93 cm) |
| `feet_fr1_000_nude_cull` | **257** | 4,103 / 4,462 | **64** female genital tris behind `bodygenitalsgen_fr1_008_*` |
| `feet_mr1_000_nude_cull` | **264** | 4,107 / 4,462 | shares the 258-tri waist band (`<1.5 mm` apart) |

At runtime, an outfit occlusion mask suppresses the overlapping waist strip on one mesh plus any
islands hidden under garments or modular chest/pelvic patches. Rendering both `hand_*_nude_cull`
and `feet_*_nude_cull` offline without culling causes Z-fighting on those 258 waist triangles (and
on the 26 nipple and 64 female genital backing triangles). Culling the coplanar overlapping
triangles stitches the body with **0 boundary-edge holes** (watertight on both `fr1` and `mr1`).
`tools/ped_assemble.py` implements standing-space alignment, canonical vertex-key plus
coplanar-centroid/normal overlap culling and multi-part glTF 2.0 export, verified on synthetic
overlapping body meshes in `tests/test_ped_assemble.py`:

```powershell
python -I tools/ped_assemble.py assemble <parts...> -o <out.glb>
python -I tools/ped_assemble.py cull-overlap --target <upper> --ref <lower> -o <out.glb>
python -I tools/ped_assemble.py inspect-seams <parts...>
```

### 15-channel meshlet vertex layout

Character geometry uses GPU meshlets (16-byte headers, up to 128 local vertices per meshlet,
32-bit packed triangles `7b a | 7b b | 7b c | 11b meshlet_id` plus `uint16` vertex remap). The
23-slot vertex buffer header (`0x00..0x5b`) carries 15 active streams on modern character heads:

| Slot | Format | Stride | Purpose |
|---:|---|---:|---|
| `0` | `0x42` (`0x4042`) | 8 B | Quantized 16-bit XYZ position (`base + q * step`) |
| `2` | `0x21` | 4 B | Packed octahedral normal + 15-bit tangent frame |
| `18` / `19` | `0x27` | 4 B | Secondary tangent / rest-pose octahedral normal (`19` heads, `18` body skin) |
| `6`, `7` | `0x26` | 4 + 4 B | Skinning blend weights (up to 8 influences per vertex) |
| `8`, `9` | `0x8c` or `0x25` | 16 + 16 B or 4 + 4 B | Bone indices: extended 32 B/vert (`0x8c`, hero heads) vs 8 B/vert (`0x25`, body skin / ambient) |
| `1`, `12`, `13` | `0x26` | 4 + 4 + 4 B | Three `RGBA8` vertex color / AO / blend-mask / skin-curvature channels |
| `3`, `4`, `14`, `15` | `0x28` / `0x42` | 4–8 B each | Four UV channels (`UV0` albedo, `UV1` tattoo/suntan/damage, `UV2` wrinkle/micro-detail, `UV3` morphable-model) |
| `17` | `0x84` | 16 B | Per-vertex basis / blend-shape / ML-deformer stream |

The ped importer and exporter preserve authored `TEXCOORD_1` and higher sets alongside
`TEXCOORD_0`, including normalized integer input accessors. Ped JSON stores these extra
sets in `uvs_extra`, an object whose keys are the positive set indices (`"1"`, `"2"`,
and so on); `uvs` remains set 0. Overlap culling and vertex compaction keep each set
aligned with the surviving vertices. This preserves vertex data without expanding
the material importer's supported texture-coordinate bindings.

### 3-tier LOD budgets and thresholds

Modular ped drawables in `packs.rpf` stream three separate dictionary tiers per item name plus a
fourth `slod_human` crowd proxy. Body skin, eyes, teeth, genitals, nipples and garments encode
`rmcDrawable` `m_LodDist[4]` (`root + 0x70..0x7f`) as **`15.0 m` (LOD 0 -> 1), `30.0 m` (LOD 1 -> 2),
`60.0 m` (LOD 2 -> 3/SLOD), `9998.0 m`**; heads and `slod_human` store `(9998.0, 9998.0, 9998.0,
9998.0)` because head tier selection is driven dynamically by the ped component LOD manager.

| Component | Tier 0 (High) | Tier 1 (Med) | Tier 2 (Low) | Notes |
|---|---:|---:|---:|---|
| Human head (canonical) | 8,983 v / 13,320 t | 2,958 v / 3,362 t | 599 v / 514 t | Shared topology across 1,600+ modern heads |
| Eyes (`*_eyes_full`) | 434 v / 816 t | 230 v / 398–408 t | 50–55 v / 80 t | Dual eyeball + cornea |
| Teeth (`teethmodular_*`) | 3,628 v / 5,582 t | 1,359 v / 2,144 t | 373 v / 556 t | Upper/lower arches + gums + tongue |
| Upper body nude (`hand_fr1_000_nude_cull`) | 10,440 v / 13,534 t | 5,821 v / 5,168 t | 2,365 v / 1,392 t | `hand_mr1`: 9,914 v / 13,534 t |
| Lower body nude (`feet_fr1_000_nude_cull`) | 4,103 v / 4,462 t | 2,794 v / 2,112 t | 1,134 v / 606 t | `feet_mr1`: 4,107 v / 4,462 t |
| First-person arms (`hand_*_nude_cull_fp`) | 8,127 v / 12,342 t | 4,337–4,360 v / 6,000 t | 932–934 v / 1,266 t | Dedicated close-up FP arms |
| Genitals — female (`bodygenitalsgen_fr1_008_*`) | 267 v / 356 t | 132–139 v / 153–170 t | 35 v / 38 t | Identical across `base`, `alt_00..02` |
| Genitals — male (`bodygenitalsgen_mr1_000_*`) | 246 v / 352 t | 113–115 v / 176 t | 22 v / 35–36 t | Resting + extended `alt_00..02` |
| Nipples — female (`bodynipplesgen_fr1_000_*`) | 106 v / 182 t | 90 v / 118 t | 3 v / 1 t | Collapses to 1-tri stub at Tier 2 |
| SLOD far crowd proxy (`slod_human`) | 392 v / 338 t | — | — | Whole distant character (-99.1%) |

Assembled nude adult budget: ~**27,300–28,000 v / 38,000–38,250 t** at Tier 0 -> ~**13,400 t** at
Tier 1 (-65%) -> ~**3,200 t** at Tier 2 (-92%) -> **338 t** at SLOD (-99.1%). Clothed characters
typically span ~**35,000–55,000 t** at Tier 0 excluding GPU hair strands.

## What is not decoded

- **Fragment children.** A raw child-array walk (`tools/fragment_children.py`) covers every fragment
  of `packs.rpf` and `levels.rpf`. The fragment header's u16 at `+0x30` is the number of drawables;
  the records are 0x110 bytes apart, starting at the main drawable that `+0x18` points at, so
  records 1 and up are the child drawables. All 2,865 `packs.rpf` fragments have a single drawable
  and the flat 25-vertex / 32-triangle bounds grid, with **no child arrays**. The four
  `gmt_mannequin*` fragments have no child arrays either; `ambient_femalebody` is a 596-bone
  skeleton plus the same grid, with no skin mesh. The path is alive on world fragments
  (`bus2_frag_hi`: 14 model structs, 11 children, a 650 KB `model_parts.glb`), so the negative is
  a property of the character files; those figures come from a different walk and were not
  re-derived by this one. In `levels.rpf` the walk finds **50,802 fragments** (the exporter's
  manifest counts 50,798): 50,068 have a single drawable and **734 have a child array**, 733 of
  one child and 1 of two, **735 child drawables** in all. Every declared child record passes the
  drawable-body test, and a scan of all other 16-byte offsets of those fragments' system pages
  finds 0 further drawables. 10,097 of the 50,802 have a null `+0x10`, `+0x20` or `+0x28`, and 6
  have a main drawable that fails the embedded-drawable header test. For comparison `data.rpf` has
  1,758 fragments, all single-drawable. The other header counts are unrelated to children: the
  `+0x28` array of 0x50-byte part records (the u16 at `+0x34`) has 153,028 records over
  `levels.rpf` (13,871 fragments with more than one) and 27,778 over `packs.rpf` (2,605). The gap
  is closed: every fragment in both archives was walked, and the only ones with child drawables
  are 734 of the `levels.rpf` ones.
- **A ped-variation table.** `common.rpf` contains `CPedVariationBodyTypeManager`
  (`0x226000`) and `CPedComponentSetManager` (`0x228000`), but neither maps a component variation
  index to a drawable plus texture. The first is height and body-shape presets
  (`SCALE_PLAYER_STORY000_00` is 1.62 m); the second lists gameplay components. Slot selection
  from names is a heuristic, not the game's rule.
- **Lucia's hair.** The `packs.rpf` manifest has three `player_story000_00` hair rows
  (`hairbob`, `hairpixie`, `hairponytail`), each **3 vertices / 1 triangle**; `hairbuzzcut`
  appears only in male `_mr1` rows. All 54 non-SLOD rows of those four styles are 3 v / 1 t.
  Across `packs.rpf`, 219 of the 267 measurable hair rows are <=3 v / 1 t. The 48 larger rows
  are 29 accessories (`hairaccsband`, `hairaccsclip`, `hairaccessoriescuff`; up to 7,251 v), 10
  SLOD proxies (71-147 v), 3 fused `cs_wyman_head_mr1_cau_000_o_hair_alt0` rows (up to 13,529 v),
  3 `hairbun` and 1 `hairlob` unit-box stubs (24 v / 12 t), and 2 blowfly
  `cri_fly_blow_m_00_hair_00` rows (1,436 / 2,040 v); no stand-alone full-resolution female hair
  mesh exists. `levels.rpf` and `story000.rpf` have zero character hair rows. A static Lucia
  export is bald by design; a render needs a clearly labelled placeholder, not a missing decode.
- **`levels.rpf`.** The full archive was name-enumerated: 1,116,073 streams, 98.32% inflated,
  235,814 named streams, 318,266 unique names. It contains no Lucia, Mandy, `player_story` or
  `ambient_female` drawable; the Lucia and Mandy hits are props (photos, frames, luggage, a
  curtain). The only human-skeleton meshes in `levels.rpf` are the generic mannequin pair in
  **Bodies**.
- **Bind pose and skinning in static exports.** The static preview and `rpf9dump` GLBs do not apply
  the 8-influence bone-palette skinning (preview GLBs carry 0 skins, 0 `JOINTS_0` and 0 `WEIGHTS_0`;
  `rpf9dump` emits identity inverse-bind matrices). The real inverse bind matrices are at skeleton
  header `+0x30` (drawable `+0x18` points at the header) and agree with the default bone transforms;
  `tools/anim_glb.py`, `tools/skinned_glb.py` and `tools/ped_assemble.py` export glTF skins with
  those matrices (see [character_skeleton.md](character_skeleton.md#animated-gltf-export)).

## Reproduce

```powershell
rpf9dump models <packs.rpf> <outdir>
python -I tools/fragment_children.py <package>\levels.rpf --jobs 12 --json <work>\levels_fragments.json
```

The second command prints the fragment and child-drawable totals quoted under "Fragment children"; it
reads the archive in place (about 22 minutes for `levels.rpf` with 12 workers over a WSL mount of
the dump) and writes nothing else.

The material rows are in the same manifest, one per shader of a record, each with the texture
references by slot:

```python
import json

items = json.load(open(r"<outdir>/packs/models/manifest.json", encoding="utf-8"))["Items"]
for item in items:
    if item["Name"] == "brabikini_fr1_012_regular_ws00_full.#dr":
        for material in item["Materials"]:
            print(material["Shader"], {t["Param"]: t["Id64"] for t in material["Textures"]})
```

A key's size and format come from the pool that holds it: the `AssetKey`, `Width`, `Height`,
`DxgiFormat` and `Mips` fields of the manifest `rpf9dump ttpf <pool.ttpf> <outdir>` writes.

The [Hands and feet](#hands-and-feet) tables also need `core_restofworld.rpf` and the 57 pool
catalogs (`AssetKey` in each pool's `manifest.json`).

`rpf9dump ttpf` writes a pool's catalog to `<output root>\<pool basename>\manifest.json`, and pool
basenames repeat: the 57 pools in [`game/xbox-1.0.2.1.json`](../game/xbox-1.0.2.1.json) have 51
distinct basenames (`ttdb_maps` and `ttdb_terrain_image_based` twice, `ttdb_ui` and `ttdb_vfx` three
times). With one output root for all of them, later pools overwrite earlier catalogs. Give each pool
its package-relative parent folder under `<outdir>\ttpf` as its output root:

```powershell
$game = (Resolve-Path "<game root>").Path.TrimEnd("\")
rpf9dump models "$game\pack\core_restofworld\core_restofworld.rpf" <outdir>
Get-Item "$game\data\ttdb", "$game\pack\*\data\ttdb" | Get-ChildItem -Recurse -File -Filter *.ttpf | ForEach-Object {
    rpf9dump ttpf $_.FullName "<outdir>\ttpf\$($_.DirectoryName.Substring($game.Length + 1))"
}
```

Each catalog records its source pool in `src`. The lookup finds catalogs at any depth under
`<outdir>\ttpf` and stops unless their sources are exactly the 57 pools of the game root, so a
catalog that was overwritten, exported in another layout or taken from another folder is caught
before an absent key is read as a lookup miss:

```python
import json
import os
from pathlib import Path

maps = {"C8E8C282": "diffusetex", "C6F46E79": "speculartex", "1E77BF4C": "bumptex", "739E3D0D": "speculartex2"}
game = Path(r"<game root>")
pool_root = Path(r"<outdir>/ttpf")


def source(path):
    return os.path.normcase(os.path.abspath(path))


pools = {source(p) for pattern in ("data/ttdb/**/*.ttpf", "pack/*/data/ttdb/**/*.ttpf") for p in game.glob(pattern)}
sources, pooled = set(), set()
for m in pool_root.rglob("manifest.json"):
    catalog = json.loads(m.read_text(encoding="utf-8"))
    sources.add(source(catalog.get("src", m)))
    pooled.update(t["AssetKey"] for t in catalog.get("Textures", []))
if len(pools) != 57 or sources != pools:
    raise ValueError(f"{len(pools)} pools under {game} (expected 57); {len(pools - sources)} have no catalog "
                     f"under {pool_root}, {len(sources - pools)} catalogs come from other sources")
items = [i for m in (r"<outdir>/packs/models/manifest.json", r"<outdir>/core_restofworld/models/manifest.json")
         for i in json.load(open(m, encoding="utf-8"))["Items"]]
for name in ("hand_fr1_000_nude_cull.#dr", "feet_fr1_000_full.#dr", "bodynipplesgen_fr1_000_regular_ws00_full.#dr",
             "bodygenitalsgen_fr1_008_regular_ws00_full.#dr", "bodygenitalsgen_mr1_000_regular_ws00_full.#dr"):
    keys = {maps[t["Param"]]: t["Id64"] for i in items if i["Name"] == name
            for mat in i["Materials"] for t in mat["Textures"] if t["Param"] in maps}
    print(name, {k: (v, v in pooled) for k, v in keys.items()})
```

The fragment and skinning fields are in the same manifest; the skeleton measurements are in
[character_skeleton.md](character_skeleton.md), and the world model export is in
[models.md](models.md).
