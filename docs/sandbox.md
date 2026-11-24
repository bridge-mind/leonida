# Real-shader sandbox: placed props through the game's own translated shaders

Follows [first_light.md](first_light.md) §5: a real camera through a vertex shader's constants, a TTPF albedo
through the bindless set with the game's sampler, and many placed world props instead of one model. Measured on
2026-11-22 from one retail Xbox Series install (1.0.2.1), on an NVIDIA GeForce RTX 4070 Ti SUPER (driver 621.24).
Nothing below reproduces game data; every number is a count.

## 1. What runs

`tools/sandbox.py` packs the props around a point (`tools/sandbox_scene.py`), picks one translated VS/PS pair from
the work folder and runs `vk_sandbox` (`runtime/test/vk_sandbox.cpp`), which draws every prop with that pair: vertex buffers at
locations 0–2 (position, normal, UV), a D32 depth buffer, one indexed draw per mesh and one descriptor table per
draw. `--window` opens a window to fly, walk or drive through the scene; `--check` renders one frame headless and
prints only counts.

The shaders are not stand-ins: the VS transforms each vertex with the world rows and view-projection it loads from
the heap and lights it with its own `max(0.25, n·l)`; the PS samples the albedo through the T# it loads and the
S# literal in its code, and multiplies by the VS colour. Lighting beyond that, shadows and the G-buffer passes are
not part of this pair.

## 2. How the pair is chosen

The material → program table is in the executable (#2519), so the pair is chosen by what the shaders read, from
`<work>/shaders/spirv/index.json` and the listings in `<work>/shaders/dxbc/disasm`:

| | Vertex shader | Pixel shader |
|---|---|---|
| inputs | POSITION (xyz), NORMAL (xyz), TEXCOORD (xy) | SV_Position, TEXCOORD, COLOR |
| user data | user SGPR 8 only (`ptr32_lo`) | user SGPR 8 (`ptr32_lo`) |
| bindings | none | 0, 1, 2 (images, samplers, hash table) |
| listing | three `s_load_dwordx4` from `s[8:9]` at 0x0, 0x40, 0x60; a 4-dword buffer load at 0x30 and a 16-dword one at 0x160 | one `s_load_dwordx8` from `s[8:9]` + 0x20; one `image_sample`; `exp mrt0` |

On the 1.0.2.1 corpus 3 VS and 231 PS match the signatures, and 2 VS and 4 PS also match the listing patterns;
they share one table layout. `tests/test_sandbox_pick.py` checks the picker on hand-written listings.

## 3. The table the pair reads

User SGPR 8 is a 32-bit pointer (the shaders force the high dword to 0) to one table in the heap:

| Offset | Descriptor | Contents (byte offsets in the buffer) |
|---|---|---|
| 0x00 | V# | world rows 0–2 at 0x00, 0x10, 0x20 (xyz each), translation at 0x30; world = x·row0 + y·row1 + z·row2 + row3 |
| 0x20 | T# | the albedo image (8 dwords) |
| 0x40 | V# | light direction at 0xa0; view-projection at 0x160, clip = wx·r0 + wy·r1 + wz·r2 + r3 (D3D clip, y up, z 0..1) |
| 0x60 | V# | colour at 0x00 (rgb × the lighting term, alpha as is) |

`vk_sandbox` writes raw V#s (stride 0, NUM_RECORDS = the buffer's bytes, OOB_SELECT 3), registers each texture's
T# words and the PS's S# literal in the descriptor hash table, builds the sampler from those S# words
(`gfx10::decode`), and draws through a negative-height viewport so the D3D-convention matrices stay unchanged.

## 4. Reproduce it

```sh
python -I tools/world_index.py ... --entities-out <work>/entities.jsonl     # placements (docs/world.md)
rpf9dump models "<package>/levels.rpf" <work>/assets
rpf9dump ttpf "<package>/data/ttdb/maps/ttdb_maps_<n>_7.ttpf" <work>/ttpf_maps          # n = 0..7
rpf9dump ttpf "<package>/data/ttdb/terrain/ttdb_terrain_<n>_7.ttpf" <work>/ttpf_terrain  # n = 0..7
cmake --build runtime/build --target vk_sandbox
python -I tools/sandbox.py --work <work> --entities <work>/entities.jsonl --models <work>/assets \
    --textures <work>/ttpf_maps --textures <work>/ttpf_terrain [--center X Y] --window
```

Window keys: WASD move, right mouse button held to look, Q/E down/up (fly), Shift faster, 1 fly, 2 walk (eye 1.7 m
over the highest scene triangle under the camera), 3 drive (eye 1.2 m, accelerates and steers), P screenshot,
Esc quit. `--center` is in RAGE world metres; without it the densest 50 m cell of resolved props is used.

## 5. Measured

| | |
|---|---|
| map pools (8) | 133,697 textures exported, 0 failed, 7,030 skipped |
| terrain pools (8) | 46,115 exported, 0 failed, 26,903 skipped (`rpf9dump`'s limits in [textures.md](textures.md)) |
| test scene (one `levels.rpf` window) | 600 entities within 120 m, 150 placed (402 have no exported model in that window), 446 draws, 178 meshes, 345,782 triangles |
| albedo | the material's DiffuseTex reference (`Param` = `joaat("DiffuseTex")` = `C8E8C282`): 422 of 446 draws textured with 93 textures from the map and terrain pools (69 with the map pools alone). Two-layer blend materials have no DiffuseTex: 11 references each, two albedos, two normal maps, packed maps and a blend mask; the first layer's albedo (`Param` `346394EA`, name unknown) brings it to 442 of 446 draws and 107 textures |
| `--check` (640×360) | 114,503 pixels covered, 4,736 distinct colours (147 with the textures left out) |
| window | about 90 frames per second with a `vkQueueWaitIdle` per frame; a 60-frame windowed run ends on the same pixels as the headless render |
| walk / drive | eye at 1.70 m / 1.20 m over the roof the camera starts above |

## 6. Limits and next

* One pair, one forward pass. The world's own material shaders pull vertices (`SV_VertexID` only) and write the
  G-buffer; drawing those needs their vertex fetch tables and the lighting passes that read the G-buffer.
* The scene stores mip 0 box-scaled to 256 px (`--tex-size`), so the sampler's LOD range is clamped to 0.
* Ground following uses the scene's own triangles; there is no collision against walls.
* Draws are not culled or sorted, and every frame waits for the GPU; a large `--max-props` gets slow (each prop is
  one draw per mesh: 150 props gave 446 draws).
