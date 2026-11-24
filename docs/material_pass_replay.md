# Following a mesh from its material to a pixel shader

A stored effect can offer several ways to draw the same mesh. On one Xbox 1.0.2.1 drawable, the first geometry selects material slot 0. Its effect has 45 techniques. One vertex/pixel pair writes a packed integer target. Two other techniques each pair a mesh shader with a pixel shader that declares textures. The first of those pixel shaders directly reads the drawable's stored material constants. The active technique for a particular frame remains unknown.

## Identify the draw

To locate the example in a local 1.0.2.1 copy, filter the decoded `data.rpf` drawables to three meshes and three material slots. The first mesh has 1,656 vertices and 809 triangles. Its slot declares five textures and 46 numeric members. Select stored technique 2, pass 0, then confirm the pixel stage declares one constant buffer, one structured SRV and one four-component unsigned output. These checks distinguish the input used here from the UV example in [first_light.md](first_light.md).

Follow the geometry's material-slot index to the effect, then follow the technique and pass selectors to the original program records. Use their archive offsets to distinguish programs and verify the native shader hashes before replay. On the selected vertex shader's packed-index branch, the mesh's original buffers reproduced 2,427 triangle corners in order and 1,656 object-space positions bit for bit. Swapping the first and third corner changed 1,618 indices. Changing the position-axis field eliminated every exact position match. The replay ends before the object-to-clip transform. [Effect lookup and geometry check](https://github.com/ItsNotRexed/Leonida/pull/2519#issuecomment-7085667975) · [Effect and pass reader](https://github.com/ItsNotRexed/Leonida/pull/3036)

## Follow the integer target through its reader

For this pixel shader the first dword of `b501, space0` supplies a row index. The paired vertex shader uses that index to write three more incoming dwords into one 12-byte structured record. The pixel shader also reads a word at byte offset 20 in a separate structured SRV. Three other shader programs use that field to address records 48 bytes apart in a second table.

Before its compressed export, the pixel shader forms the second component as `((word & 0xffff) << 16) | ((row + 1) & 0xffff)`. Six compute programs load a four-component image under the same resource key and interpret that layout. Each tests the second component for zero. Each subtracts one from the low half to index the 12-byte table and uses the high half to index the other table. The different stages identify the tables by matching resource keys even where the register numbers differ. Connecting this particular draw and dispatch still requires the actual image binding and execution order.

To check the handoff, vary just the row and read back the **whole second component**. The following synthetic inputs were supplied to the selected translated pixel shader. A synthetic vertex shader supplied constant interpolants. Both pixel shaders used the same descriptor contents and `R32G32B32A32_UINT` target. [Tested translator revision](https://github.com/ItsNotRexed/Leonida/commit/4f0c41fb3241870230483c02918f99e901c3a5fe)

| Buffer word | Row | Before the half conversion | Selected shader readback | Direct 32-bit control |
|---:|---:|---:|---:|---:|
| `0x3f80` | 6 | `0x3f800007` | `0x00003c00` | `0x3f800007` |
| `0x3f80` | 7 | `0x3f800008` | `0x00003c00` | `0x3f800008` |

Two further cases varied the buffer word and crossed a half-precision quantization boundary. All four pixels of each of eight 2×2 draws matched their expected values on an NVIDIA GeForce RTX 5070, driver 621.01, Intel Core Ultra 7 265K, Windows 11 build 26200.9974. A typed output alone does not preserve these two indices through the selected translation. The Xbox frame's export configuration and render-target view remain open.

## Find the surface inputs

Compare pixel-stage declarations across the same material's passes. The packed pass has no texture or sampler declaration. The textured sibling declares a 272-byte material constant buffer. Of its 46 authored members, 45 match the program by hash and byte offset. For the last member, the effect's parameter record supplies both the authored and compiled name hashes at the same offset. A different material slot gave only 11 direct matches. The textured shader reads the stored values and uses the first texture handle for a 2-D RGB sample. The drawable supplies four texture references across five declarations. The live handle table, sampler, and the descriptor used for the empty declaration still need independent verification.

The empty texture declaration has a narrower consequence on the first textured pass. Two stored material values satisfy a uniform branch condition. For an active wave reaching that branch, the taken path skips all four 2-D array sampling sites. Changing either value independently made those sites reachable in a control-flow check. The shader still loads the handle word. This result concerns that material and pass.

Both tested color techniques' mesh and pixel programs translated and passed Vulkan 1.2 SPIR-V validation, including the translator's 32-lane emulation of the mesh stages' 64-lane waves. The stage-interface checker also found compatible mesh outputs and pixel inputs for each original pair in both modes. Reproducing final appearance still requires their draw constants, descriptors, transforms, and the frame's pass selection.
