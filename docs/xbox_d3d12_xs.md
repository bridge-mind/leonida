# The `d3d12_xs` graphics ABI, from Microsoft's public GDK reference

Status: reference, sourced. This page transcribes, for the items the D3D12-to-Vulkan layer proxies, the exact C
signatures and constraints Microsoft publishes for the Xbox GDK graphics API, and names what the public reference does
not cover. It is not a design document and not a reimplementation guide; [d3d12_vulkan.md](d3d12_vulkan.md) holds the
plan and [d3d12_xs_surface.md](d3d12_xs_surface.md) holds the measured export surface.

Source: <https://docs.xbox.com>. The site serves Markdown per page (append `.md`) behind plain-text indexes. The path
taken was <https://docs.xbox.com/llms.txt> → <https://docs.xbox.com/_llms/en/apis/gdk-apis/graphics-api-reference.txt>
→ the `d3d12` sub-index → the page URLs listed in each section below. All pages were read on 2026-11-23.

How to read the quotes: **Documented** means the text block is copied from the page as published, including its
inconsistencies. Where a published block contradicts itself or cannot be the real ABI, the item is marked **Doc
quirk** and the reason is stated; nothing is silently corrected. **Not documented in the public GDK reference** means
the page does not exist (404) or the index does not list it, and no value is guessed.

## 1. What the reference covers

The graphics index has two units: `d3d12` (520 pages) and `d3d12_xs` (7 pages). The console-specific pages are thin:
six enums (`D3D_FEATURE_LEVEL`, `D3D_NAME`, `D3D_SHADER_INPUT_TYPE`, `D3D12_MESH_SHADER_TIER`,
`D3D12_STANDARD_MULTISAMPLE_QUALITY_LEVELS`, `D3D12_WAVE_MMA_TIER`), one function (`D3D10CreateBlob`) and one struct
(`DXGI_ADAPTER_DESC2`) ([index](https://docs.xbox.com/_llms/en/apis/gdk-apis/graphics-api-reference.txt)).

Everything about root signatures, state objects, DXIL libraries, hit groups and shader caches lives in the `d3d12`
unit and is labelled "(XBOX Series consoles)" in its title, with `Header: d3d12_xs.h` and `Library: d3d12_xs.lib` in
its Requirements block. Those labels are the only machine-readable difference from the PC pages; the signatures are
the PC shapes.

The `d3d12_xs/interfaces/...` paths that several `See Also` links point at (`d3d12_xs/interfaces/ID3D12StateObject/...`,
`d3d12_xs/interfaces/ID3D12PipelineState/id3d12pipelinestate_xs.md`, `d3d12_xs/interfaces/ID3D12CommandQueue/...`) all
return 404, and the `_llms` index does not list them. **The console interface pages are not documented in the public
GDK reference.**

## 2. Root signatures

### 2.1 `D3D12_LOCAL_ROOT_SIGNATURE`

```cpp
typedef struct D3D12_LOCAL_ROOT_SIGNATURE
{
    ID3D12RootSignature  pLocalRootSignature;
} D3D12_LOCAL_ROOT_SIGNATURE
```

`pLocalRootSignature` is "the root signature that will function as a local root signature. A state object holds a
reference to this signature." The subobject is optional; the global plus local signatures attached to a shader
function must cover every resource binding the shader declares, with no overlap. If one function in a call graph is
associated with a local root signature, every other function in that graph must use the same one or none, and the
shader entry (the graph root) must be associated with it, because all code reachable from an entry runs from a single
shader record carrying one set of local root arguments.

Requirements: `Header: d3d12_xs.h or d3d12_x.h`, `Library: d3d12_xs.lib or d3d12_x.lib`, both console families.
Source: <https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_local_root_signature_public.md>

**Doc quirk:** the block declares `ID3D12RootSignature pLocalRootSignature;` but the member table types it
`ID3D12RootSignature *` and the prose says the state object "holds a reference". The published block has no pointer
star and no trailing semicolon; that is a transcription of the page, not the header.

### 2.2 `D3D12_GLOBAL_ROOT_SIGNATURE`

```cpp
typedef struct D3D12_GLOBAL_ROOT_SIGNATURE
{
    ID3D12RootSignature  pGlobalRootSignature;
} D3D12_GLOBAL_ROOT_SIGNATURE
```

Same no-overlap rule, stated per global signature. The page adds the console constraint that matters to a replay
layer: every shader a particular
[`DispatchRays`](https://docs.xbox.com/reference/graphics/d3d12/interfaces/id3d12graphicscommandlist4/methods/id3d12graphicscommandlist4_dispatchrays_public.md)
reaches from a command list must carry the same global root signature as the compute root signature bound on that
command list, while one large state object may hold several global root signatures for different shader subsets.
Source: <https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_global_root_signature_public.md>

### 2.3 The graphics/compute root signature, shared with the PC shape

```cpp
typedef struct D3D12_ROOT_SIGNATURE_DESC
{
    UINT NumParameters;
    const D3D12_ROOT_PARAMETER  pParameters;
    UINT NumStaticSamplers;
    const D3D12_STATIC_SAMPLER_DESC  pStaticSamplers;
    D3D12_ROOT_SIGNATURE_FLAGS Flags;
} D3D12_ROOT_SIGNATURE_DESC
```

`NumParameters` is "the number of slots in the root signature" and the length of `pParameters`. The page states that
"there is one graphics root signature, and one compute root signature", and that this struct is consumed by
`D3D12SerializeRootSignature` and returned by `ID3D12RootSignatureDeserializer::GetRootSignatureDesc`.
Source: <https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_root_signature_desc_public.md>

```cpp
typedef struct D3D12_ROOT_PARAMETER
{
    D3D12_ROOT_PARAMETER_TYPE ParameterType;
    D3D12_ROOT_DESCRIPTOR_TABLE DescriptorTable;
    D3D12_ROOT_CONSTANTS Constants;
    D3D12_ROOT_DESCRIPTOR Descriptor;
    D3D12_SHADER_VISIBILITY ShaderVisibility;
} D3D12_ROOT_PARAMETER
```

`ParameterType` selects the union member: `DescriptorTable` (ranges contiguous in a heap), `Constants` (inline
constants that appear to the shader as one constant buffer), `Descriptor` (inline CBV/SRV/UAV). `ShaderVisibility`
limits which shaders may read the slot.
Source: <https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_root_parameter_public.md>

```cpp
typedef enum D3D12_ROOT_PARAMETER_TYPE
{
    D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE = 0,
    D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS = 0,
    D3D12_ROOT_PARAMETER_TYPE_CBV = 0,
    D3D12_ROOT_PARAMETER_TYPE_SRV = 0,
    D3D12_ROOT_PARAMETER_TYPE_UAV = 0
} D3D12_ROOT_PARAMETER_TYPE
```

**Doc quirk, and the one that matters most to a traceless implementation:** every constant is published as `0`, so
the page cannot be used to recover the numeric values. The prose still distinguishes the five slot kinds
(descriptor table, 32-bit constants, CBV, SRV, UAV). Take the values from the header the title is compiled against,
not from this page.
Source: <https://docs.xbox.com/reference/graphics/d3d12/enums/d3d12_root_parameter_type_public.md>

```cpp
typedef enum D3D_ROOT_SIGNATURE_VERSION
{
    D3D_ROOT_SIGNATURE_VERSION_1 = 1,
    D3D_ROOT_SIGNATURE_VERSION_1_0 = 1,
    D3D_ROOT_SIGNATURE_VERSION_1_1 = 2
} D3D_ROOT_SIGNATURE_VERSION
```

1.1 is the version that adds the `D3D12_ROOT_SIGNATURE_DESC1` / `D3D12_ROOT_PARAMETER1` / `D3D12_DESCRIPTOR_RANGE1`
family; the page links Microsoft's "Root Signature Version 1.1" article but does not restate the differences.
Source: <https://docs.xbox.com/reference/graphics/d3d12/enums/d3d_root_signature_version_public.md>

```cpp
HRESULT CreateRootSignature(
    UINT nodeMask,
    const void  pBlobWithRootSignature,
    SIZE_T blobLengthInBytes,
    const IID & riid,
    void  ppvRootSignature
)
```

`nodeMask` is zero for single-GPU operation. The blob is the serialized signature produced by
`D3D12SerializeRootSignature`; `E_INVALIDARG` means the blob is invalid. Header `d3d12_xs.h or d3d12_x.h`.
Source: <https://docs.xbox.com/reference/graphics/d3d12/interfaces/id3d12device/methods/id3d12device_createrootsignature_public.md>

`ID3D12RootSignature` itself documents only that "currently, there is one graphics and one compute root signature per
app" and that command lists link to the resources the shaders require.
Source: <https://docs.xbox.com/reference/graphics/d3d12/interfaces/id3d12rootsignature/id3d12rootsignature_public.md>

### 2.4 How the console model differs from the PC model

Three differences are documented:

- **A second axis: global versus local.** On the console a root signature can be attached to a state object as a
  subobject, and the pair (global, local) is what a shader function binds. The PC root signature has no such split;
  it is the signature bound on the command list. (§2.1, §2.2.)
- **State objects carry root signatures by reference.** `D3D12_LOCAL_ROOT_SIGNATURE` and
  `D3D12_GLOBAL_ROOT_SIGNATURE` hold an `ID3D12RootSignature`, so a state object outlives and ref-counts the
  signature it was built from.
- **The graphics/compute signature stays the PC object.** `D3D12_ROOT_SIGNATURE_DESC`, `CreateRootSignature` and the
  `SetGraphicsRootSignature`/`SetComputeRootSignature` pair are documented with the same shapes and the same
  `d3d12_xs.h` header, so a console title serializes and creates signatures exactly as a PC title does before the
  state-object path enters the picture.

**User-data slots: not documented in the public GDK reference.** No page in the graphics index contains the term, no
`D3D12_ROOT_PARAMETER_TYPE_USER_DATA`-style constant appears in the published enum, and the `D3D12_SHADER_VISIBILITY`
and `D3D12_ROOT_PARAMETER` pages describe only the three inline kinds plus descriptor tables. What the reference does
document is the inline-constant mechanism (`D3D12_ROOT_CONSTANTS`, exposed through
`SetGraphicsRoot32BitConstant(s)` and `SetComputeRoot32BitConstant(s)`), so any console notion of a fixed user-data
window per stage is outside this reference. The working model in this repository is measured, not documented:
parameters are placed inline in user SGPRs from `s4` in root order while they fit whole, and the first that does not
fit plus everything after it go into a packed array reached through a 40-bit pointer at `s[0:1]`
([d3d12_vulkan.md §8.2 item 3](d3d12_vulkan.md#82-unknown--open-questions)).

**`D3D12_ROOT_SIGNATURE_FLAGS`: not documented in the public GDK reference.** Both
`d3d12/enums/d3d12_root_signature_flags_public.md` and `d3d12_xs/enums/d3d12_root_signature_flags_xs.md` return 404,
although `D3D12_ROOT_SIGNATURE_DESC::Flags` is typed by it and the PC page links the Xbox name. The console flag
values, including the high-byte bits this repository measures as `0x03000000`
([d3d12_vulkan.md §8.2 item 19](d3d12_vulkan.md#82-unknown--open-questions)), have no public page.

## 3. State objects, DXIL libraries and hit groups

### 3.1 `D3D12_STATE_SUBOBJECT_TYPE`

```cpp
typedef enum D3D12_STATE_SUBOBJECT_TYPE
{
    D3D12_STATE_SUBOBJECT_TYPE_STATE_OBJECT_CONFIG = 0,
    D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE = 1,
    D3D12_STATE_SUBOBJECT_TYPE_LOCAL_ROOT_SIGNATURE = 2,
    D3D12_STATE_SUBOBJECT_TYPE_NODE_MASK = 3,
    D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY = 5,
    D3D12_STATE_SUBOBJECT_TYPE_EXISTING_COLLECTION = 6,
    D3D12_STATE_SUBOBJECT_TYPE_SUBOBJECT_TO_EXPORTS_ASSOCIATION = 7,
    D3D12_STATE_SUBOBJECT_TYPE_DXIL_SUBOBJECT_TO_EXPORTS_ASSOCIATION = 8,
    D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG = 9,
    D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG = 10,
    D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP = 11,
    D3D12_STATE_SUBOBJECT_TYPE_MAX_VALID = 0
} D3D12_STATE_SUBOBJECT_TYPE
```

The console-relevant constraint in the constant table: a `NODE_MASK` subobject carrying node `0` is mishandled by
some runtime versions (it should mean node 1), so the page says to write an explicit `1` or omit the subobject.
Requirements say `Header: d3d12_x.h`, `Library: d3d12_x.lib`, `Supported Platforms: XBOX One family`, which conflicts
with the other state-object pages that say `d3d12_xs`; treat the header line as a copy of the Xbox One page.

**Doc quirk:** `D3D12_STATE_SUBOBJECT_TYPE_MAX_VALID` is published as `0`, which collides with
`STATE_OBJECT_CONFIG`. `4` is unused in the list; the page does not say why and does not name a type 4.
Source: <https://docs.xbox.com/reference/graphics/d3d12/enums/d3d12_state_subobject_type_public.md>

### 3.2 `D3D12_DXIL_LIBRARY_DESC`

```cpp
typedef struct D3D12_DXIL_LIBRARY_DESC
{
    D3D12_SHADER_BYTECODE DXILLibrary;
    UINT NumExports;
    D3D12_EXPORT_DESC  pExports;
} D3D12_DXIL_LIBRARY_DESC
```

Constraints the page states:

- `DXILLibrary` "must have been compiled with library target 6.3 or higher".
- The same library may appear several times in one state object or collection, or across collections, "as long as the
  names exported each time don't conflict in a given state object".
- `NumExports` is the length of `pExports`; `0` means "everything gets exported from the library".
- `pExports` is optional. Requirements: `Header: d3d12_xs.h`, `Library: d3d12_xs.lib`, `XBOX Series consoles`.

Source: <https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_dxil_library_desc_public.md>

```cpp
typedef struct D3D12_SHADER_BYTECODE
{
    const void  pShaderBytecode;
    SIZE_T BytecodeLength;
} D3D12_SHADER_BYTECODE
```

The bytecode pointer is opaque to the API and may be a whole compiled blob as loaded from disk. This is the type the
library, the compute PSO and the graphics PSO all carry their programs in.
Source: <https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_shader_bytecode_public.md>

### 3.3 `D3D12_EXPORT_DESC` and `D3D12_EXPORT_FLAGS`

```cpp
typedef struct D3D12_EXPORT_DESC
{
    LPCWSTR Name;
    LPCWSTR ExportToRename;
    D3D12_EXPORT_FLAGS Flags;
} D3D12_EXPORT_DESC
```

`Name` is the export name; for an overloaded function a compiler-modified name (retrievable from HLSL reflection)
disambiguates. With a non-null `ExportToRename`, `Name` is the new name and must be the unmodified one, while
`ExportToRename` may be modified or unmodified; one internal name may be exported several times under different
renames.
Source: <https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_export_desc_public.md>

```cpp
typedef enum D3D12_EXPORT_FLAGS
{
    D3D12_EXPORT_FLAG_NONE = 0
} D3D12_EXPORT_FLAGS
```

The page states plainly that "no export flags are defined in the current release", so a layer can reject every
non-zero flag value it sees on the console build without losing documented behaviour.
Source: <https://docs.xbox.com/reference/graphics/d3d12/enums/d3d12_export_flags_public.md>

### 3.4 `D3D12_HIT_GROUP_DESC` and `D3D12_HIT_GROUP_TYPE`

```cpp
typedef struct D3D12_HIT_GROUP_DESC
{
    LPCWSTR HitGroupExport;
    D3D12_HIT_GROUP_TYPE Type;
    LPCWSTR AnyHitShaderImport;
    LPCWSTR ClosestHitShaderImport;
    LPCWSTR IntersectionShaderImport;
} D3D12_HIT_GROUP_DESC
```

`AnyHitShaderImport` and `ClosestHitShaderImport` are optional and legal for every hit-group type.
`IntersectionShaderImport` is optional and legal only for a procedural-primitive hit group.
Source: <https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_hit_group_desc_public.md>

```cpp
typedef enum D3D12_HIT_GROUP_TYPE
{
    D3D12_HIT_GROUP_TYPE_TRIANGLES = 0,
    D3D12_HIT_GROUP_TYPE_PROCEDURAL_PRIMITIVE = 1
} D3D12_HIT_GROUP_TYPE
```

A triangle hit group "can't contain an intersection shader"; a procedural-primitive hit group "must contain" one.
Source: <https://docs.xbox.com/reference/graphics/d3d12/enums/d3d12_hit_group_type_public.md>

### 3.5 The two raytracing configuration subobjects

```cpp
typedef struct D3D12_RAYTRACING_SHADER_CONFIG
{
    UINT MaxPayloadSizeInBytes;
    UINT MaxAttributeSizeInBytes;
} D3D12_RAYTRACING_SHADER_CONFIG
```

`MaxPayloadSizeInBytes` is the payload budget in scalars of 4 bytes; `MaxAttributeSizeInBytes` is the attribute
budget, capped at `D3D12_RAYTRACING_MAX_ATTRIBUTE_SIZE_IN_BYTES`. A pipeline needs one such configuration; if several
are present, for example one per collection so each compiles early, they must all match when combined.
Source: <https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_raytracing_shader_config_public.md>

```cpp
typedef struct D3D12_RAYTRACING_PIPELINE_CONFIG
{
    UINT MaxTraceRecursionDepth;
} D3D12_RAYTRACING_PIPELINE_CONFIG
```

The recursion limit "must be in the range of 0 to 31"; a `TraceRay` at the limit puts the device into removed state.
Duplicating this subobject per collection buys no early compilation, unlike the shader config above.
Source: <https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_raytracing_pipeline_config_public.md>

### 3.6 The state-object container

```cpp
typedef struct D3D12_STATE_OBJECT_DESC
{
    D3D12_STATE_OBJECT_TYPE Type;
    UINT NumSubobjects;
    const D3D12_STATE_SUBOBJECT  pSubobjects;
} D3D12_STATE_OBJECT_DESC
```

Passed to `ID3D12Device5::CreateStateObject`; `NumSubobjects` is the length of `pSubobjects`.
Source: <https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_state_object_desc_public.md>

```cpp
typedef struct D3D12_STATE_SUBOBJECT
{
    D3D12_STATE_SUBOBJECT_TYPE Type;
    const void  pDesc;
} D3D12_STATE_SUBOBJECT
```

`Type` selects the meaning and layout of `pDesc`. The related-topics block of this page links
`d3d12_xs/enums/d3d12_state_subobject_type_xs.md`, which 404s; the live page is the `d3d12` one in §3.1.
Source: <https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_state_subobject_public.md>

```cpp
typedef enum D3D12_STATE_OBJECT_TYPE
{
    D3D12_STATE_OBJECT_TYPE_COLLECTION = 0,
    D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE = 3
} D3D12_STATE_OBJECT_TYPE
```

Two values only: a collection, and a raytracing pipeline. Value 3 with no documented 1 or 2 is what the page
publishes; the enum is shared with the PC shape, where those numbers historically held graphics and compute state
objects. A collection is the unit `D3D12_EXISTING_COLLECTION_DESC` re-exports.
Source: <https://docs.xbox.com/reference/graphics/d3d12/enums/d3d12_state_object_type_public.md>

```cpp
typedef struct D3D12_EXISTING_COLLECTION_DESC
{
    ID3D12StateObject  pExistingCollection;
    UINT NumExports;
    D3D12_EXPORT_DESC  pExports;
} D3D12_EXISTING_COLLECTION_DESC
```

The enclosing object ref-counts the collection; `NumExports` 0 exports the whole collection.
Source: <https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_existing_collection_desc_public.md>

```cpp
typedef struct D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION
{
    const D3D12_STATE_SUBOBJECT  pSubobjectToAssociate;
    UINT NumExports;
    LPCWSTR  pExports;
} D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION
```

This is how a local root signature, a shader config or another subobject is bound to named exports. `NumExports == 0`
declares a default association, which is also what omitting the subobject means. With cross-linkage opted into
through `D3D12_STATE_OBJECT_CONFIG`, an associated export need not exist yet in the current state object and is
resolved later, at RTPSO creation.
Source: <https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_subobject_to_exports_association_public.md>

```cpp
typedef struct D3D12_DXIL_SUBOBJECT_TO_EXPORTS_ASSOCIATION
{
    LPCWSTR SubobjectToAssociate;
    UINT NumExports;
    LPCWSTR  pExports;
} D3D12_DXIL_SUBOBJECT_TO_EXPORTS_ASSOCIATION
```

The whole page is one sentence: "this subobject is unsupported in the current release." It is still a valid
`D3D12_STATE_SUBOBJECT_TYPE` value (8), so a layer sees the type and must reject it rather than guess semantics.
Source: <https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_dxil_subobject_to_exports_association_public.md>

### 3.7 Creating and querying a state object

```cpp
HRESULT CreateStateObject(
    const D3D12_STATE_OBJECT_DESC  pDesc,
    const IID & riid,
    void  ppStateObject
)
```

On `ID3D12Device5`. `riid` is `__uuidof(ID3D12StateObject)`. Documented failures: `E_INVALIDARG` for a bad input,
`E_OUTOFMEMORY` for the handle. Header `d3d12_xs.h`, `Library: d3d12_xs.lib`, `XBOX Series consoles`.
Source: <https://docs.xbox.com/reference/graphics/d3d12/interfaces/id3d12device5/methods/id3d12device5_createstateobject_public.md>

```cpp
const GetShaderIdentifier(
    LPCWSTR pExportName
)
```

On `ID3D12StateObjectProperties`. Returns a pointer to the shader identifier, valid while the state object lives, of
`D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES`; the page recommends copying and caching it. The identifier is global to the
shader, so the same shader in another state object with the same associations yields the same bytes, and the return
is `nullptr` when the shader is not fully resolved.

**Doc quirk:** the block omits the return type and the parameter star (`const GetShaderIdentifier(`), and the return
value is described only as "Type: const". The prose is consistent with `const void* GetShaderIdentifier(LPCWSTR)`.
Source: <https://docs.xbox.com/reference/graphics/d3d12/interfaces/id3d12stateobjectproperties/methods/id3d12stateobjectproperties_getshaderidentifier_public.md>

`ID3D12StateObjectProperties` also documents `GetShaderStackSize`, `GetPipelineStackSize` and `SetPipelineStackSize`
for raytracing stack management
([index](https://docs.xbox.com/_llms/en/apis/gdk-apis/graphics-api-reference/d3d12/interfaces.txt)).

### 3.8 Ray-tracing library shapes

The public reference documents the building blocks above and stops there. What is missing, and therefore what a layer
cannot take from this source:

- **No console RTPSO page.** `D3D12_RAYTRACING_PIPELINE_CONFIG`, `..._SHADER_CONFIG`, hit groups and the DXIL library
  are described with PC semantics plus the console global/local root signature rules; no page describes an
  Xbox-specific pipeline layout, shader-table format or offline pipeline object.
- **The conceptual XDXR pages are NDA-gated.** `.../graphics/newfeatures/raytracing/index.md` is public and lists
  cards for BVH builders, BVH compression, k-DOP and offline RTPSO, but fetching `rt-offlinertpso.md` or `rt-xdxr.md`
  returns Microsoft's "Access restricted" page: the content requires an Xbox developer program sign-in
  ([raytracing index](https://docs.xbox.com/build/core-features/graphics/newfeatures/raytracing/index.md)).
- **Shader tables have no page.** `D3D12_DISPATCH_RAYS_DESC` exists in the index and `GetShaderIdentifier` describes a
  shader record as a blob the app fills in GPU memory, but no page in the graphics reference specifies the
  record layout beyond the identifier size constant.

## 4. Shader cache sessions

### 4.1 Creation

```cpp
HRESULT CreateShaderCacheSession(
    const D3D12_SHADER_CACHE_SESSION_DESC  pDesc,
    const IID & riid,
    void  ppvSession
)
```

On `ID3D12Device9`. Returns `DXGI_ERROR_ALREADY_EXISTS` when "you tried to create a cache with an existing
identifier", which the description below qualifies to the case where sessions are still open. `riid` receives an
`ID3D12ShaderCacheSession`. Header `d3d12_xs.h`, `Library: d3d12_xs.lib`, `XBOX Series consoles`.
Source: <https://docs.xbox.com/reference/graphics/d3d12/interfaces/id3d12device9/methods/id3d12device9_createshadercachesession_public.md>

```cpp
typedef struct D3D12_SHADER_CACHE_SESSION_DESC
{
    GUID Identifier;
    D3D12_SHADER_CACHE_MODE Mode;
    D3D12_SHADER_CACHE_FLAGS Flags;
    UINT MaximumInMemoryCacheSizeBytes;
    UINT MaximumInMemoryCacheEntries;
    UINT MaximumValueFileSizeBytes;
    UINT64 Version;
} D3D12_SHADER_CACHE_SESSION_DESC
```

The documented semantics of each field:

- `Identifier`: caches with different identifiers are stored side by side; caches with the same identifier are shared
  by all sessions in the process. Creating a disk cache with an existing identifier opens that cache unless `Version`
  differs, in which case it is wiped and re-created if no other session holds it; if others do, creation fails with
  `DXGI_ERROR_ALREADY_EXISTS`.
- `Mode`: memory or disk (§4.2).
- `Flags`: modifier bits (§4.3).
- `MaximumInMemoryCacheSizeBytes`: for a memory cache the whole budget; for a disk cache the size of the staging area
  that holds stored and found entries until newer ones evict them. **Default 1 KB.**
- `MaximumInMemoryCacheEntries`: how many entries fit in that memory. **Default 128.**
- `MaximumValueFileSizeBytes`: disk-cache maximum file size. **Default 128 MB.**
- `Version`: mismatch wipes and re-creates the cache, which is the intended way to invalidate after an app update.

Source: <https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_shader_cache_session_desc_public.md>

### 4.2 `D3D12_SHADER_CACHE_MODE`

```cpp
typedef enum D3D12_SHADER_CACHE_MODE
{
    D3D12_SHADER_CACHE_MODE_MEMORY = 0,
    D3D12_SHADER_CACHE_MODE_DISK = 0
} D3D12_SHADER_CACHE_MODE
```

`MEMORY`: no backing file, stores are discarded with the session. `DISK`: files persist across runs until cleared
(§4.5).

**Doc quirk:** both values are published as `0`, so the page cannot be used to tell the two modes apart numerically.
It is a transcription error in the page; a layer should accept the host runtime's value rather than assume 0 means
disk or memory.
Source: <https://docs.xbox.com/reference/graphics/d3d12/enums/d3d12_shader_cache_mode_public.md>

### 4.3 `D3D12_SHADER_CACHE_FLAGS`

```cpp
typedef enum D3D12_SHADER_CACHE_FLAGS
{
    D3D12_SHADER_CACHE_FLAG_NONE = 0,
    D3D12_SHADER_CACHE_FLAG_DRIVER_VERSIONED = 1,
    D3D12_SHADER_CACHE_FLAG_USE_WORKING_DIR = 2
} D3D12_SHADER_CACHE_FLAGS
```

`DRIVER_VERSIONED` makes the cache implicitly versioned per driver, stored side by side per adapter on multi-GPU
systems, with `Version` as an extra constraint. `USE_WORKING_DIR` moves the cache from temporary storage (subject to
disk cleanup) into the current working directory, and is "not valid for UWP apps".
Source: <https://docs.xbox.com/reference/graphics/d3d12/enums/d3d12_shader_cache_flags_public.md>

### 4.4 The session object

`ID3D12ShaderCacheSession` "represents a shader cache session", is listed under `ID3D12DeviceChild` in its See Also,
and declares four methods ([interface list](https://docs.xbox.com/_llms/en/apis/gdk-apis/graphics-api-reference/d3d12/interfaces.txt)):

```cpp
HRESULT StoreValue(
    const void  pKey,
    UINT KeySize,
    const void  pValue,
    UINT ValueSize
)
```

Adds an entry. `DXGI_ERROR_ALREADY_EXISTS` for a duplicate key, `DXGI_ERROR_CACHE_HASH_COLLISION` when the hash
matches but the key does not, `DXGI_ERROR_CACHE_FULL` when the insertion would exceed the maximum size.
Source: <https://docs.xbox.com/reference/graphics/d3d12/interfaces/id3d12shadercachesession/methods/id3d12shadercachesession_storevalue_public.md>

```cpp
HRESULT FindValue(
    const void  pKey,
    UINT KeySize,
    void  pValue,
    UINT  pValueSize
)
```

Exact-key lookup, called twice: once for the size, once for the data. Errors are
`DXGI_ERROR_CACHE_HASH_COLLISION` and `DXGI_ERROR_NOT_FOUND`. The page notes that in-memory temporary storage makes
the two-call pattern cheap.
Source: <https://docs.xbox.com/reference/graphics/d3d12/interfaces/id3d12shadercachesession/methods/id3d12shadercachesession_findvalue_public.md>

```cpp
D3D12_SHADER_CACHE_SESSION_DESC GetDesc(
)
```

Returns the description the session was created with.
Source: <https://docs.xbox.com/reference/graphics/d3d12/interfaces/id3d12shadercachesession/methods/id3d12shadercachesession_getdesc_public.md>

`SetDeleteOnDestroy`: when all sessions for a cache are destroyed, the cache is cleared. The Remarks list the five
documented ways a disk cache can be cleared, and the on-disk file extensions (`*.idx`, `*.val`, `*.lock`) that only
`USE_WORKING_DIR` caches may be deleted by hand.
Source: <https://docs.xbox.com/reference/graphics/d3d12/interfaces/id3d12shadercachesession/methods/id3d12shadercachesession_setdeleteondestroy_public.md>

### 4.5 `ID3D12Device9::ShaderCacheControl`

```cpp
HRESULT ShaderCacheControl(
    D3D12_SHADER_CACHE_KIND_FLAGS Kinds,
    D3D12_SHADER_CACHE_CONTROL_FLAGS Control
)
```

"May be used only in developer mode." `DISABLE` and `ENABLE` cannot be combined, and at least one flag is required.
Any of the named caches may or may not exist.
Source: <https://docs.xbox.com/reference/graphics/d3d12/interfaces/id3d12device9/methods/id3d12device9_shadercachecontrol_public.md>

```cpp
typedef enum D3D12_SHADER_CACHE_KIND_FLAGS
{
    D3D12_SHADER_CACHE_KIND_FLAG_IMPLICIT_D3D_CACHE_FOR_DRIVER = 1,
    D3D12_SHADER_CACHE_KIND_FLAG_IMPLICIT_D3D_CONVERSIONS = 2,
    D3D12_SHADER_CACHE_KIND_FLAG_IMPLICIT_DRIVER_MANAGED = 4,
    D3D12_SHADER_CACHE_KIND_FLAG_APPLICATION_MANAGED = 8
} D3D12_SHADER_CACHE_KIND_FLAGS
```

`APPLICATION_MANAGED` covers every session created by `CreateShaderCacheSession`, and a `CLEAR` on it also clears
on-disk caches created without `USE_WORKING_DIR`.
Source: <https://docs.xbox.com/reference/graphics/d3d12/enums/d3d12_shader_cache_kind_flags_public.md>

```cpp
typedef enum D3D12_SHADER_CACHE_CONTROL_FLAGS
{
    D3D12_SHADER_CACHE_CONTROL_FLAG_DISABLE = 1,
    D3D12_SHADER_CACHE_CONTROL_FLAG_ENABLE = 2,
    D3D12_SHADER_CACHE_CONTROL_FLAG_CLEAR = 4
} D3D12_SHADER_CACHE_CONTROL_FLAGS
```

`DISABLE` also makes an attempt to use or create a cache return `DXGI_ERROR_NOT_CURRENTLY_AVAILABLE`.
Source: <https://docs.xbox.com/reference/graphics/d3d12/enums/d3d12_shader_cache_control_flags_public.md>

### 4.6 Support flags and the pipeline-library gate

```cpp
typedef enum D3D12_SHADER_CACHE_SUPPORT_FLAGS
{
    D3D12_SHADER_CACHE_SUPPORT_NONE = 0,
    D3D12_SHADER_CACHE_SUPPORT_SINGLE_PSO = 1,
    D3D12_SHADER_CACHE_SUPPORT_LIBRARY = 2,
    D3D12_SHADER_CACHE_SUPPORT_AUTOMATIC_INPROC_CACHE = 4,
    D3D12_SHADER_CACHE_SUPPORT_AUTOMATIC_DISK_CACHE = 8
} D3D12_SHADER_CACHE_SUPPORT_FLAGS
```

`SINGLE_PSO` (the `CachedPSO` member of the PSO descriptions) "is always supported"; `LIBRARY` is what gates
`ID3D12PipelineLibrary`.
Source: <https://docs.xbox.com/reference/graphics/d3d12/enums/d3d12_shader_cache_support_flags_public.md>

`D3D12_FEATURE_DATA_SHADER_CACHE` is a single `SupportFlags` field, answered by `CheckFeatureSupport` under
`D3D12_FEATURE_SHADER_CACHE`.
Source: <https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_feature_data_shader_cache_public.md>

### 4.7 Documented limits

| Limit | Value | Source |
|---|---|---|
| In-memory staging size, default | 1 KB | [session desc](https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_shader_cache_session_desc_public.md) |
| In-memory entries, default | 128 | [session desc](https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_shader_cache_session_desc_public.md) |
| Disk value file, default | 128 MB | [session desc](https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_shader_cache_session_desc_public.md) |
| Key/value sizes | no documented cap; failures surface as `DXGI_ERROR_CACHE_FULL` | [StoreValue](https://docs.xbox.com/reference/graphics/d3d12/interfaces/id3d12shadercachesession/methods/id3d12shadercachesession_storevalue_public.md) |
| Console-specific cache sizes or memory budget | not documented in the public GDK reference | — |

## 5. Device, queue and pipeline creation entry points

### 5.1 Device

**`D3D12CreateDevice` is not documented in the public GDK graphics reference.**
`d3d12/functions/d3d12createdevice_public.md` returns 404 and the functions index lists only six free functions:
`D3D12CreateRootSignatureDeserializer`, `D3D12CreateVersionedRootSignatureDeserializer`,
`D3D12EnableExperimentalFeatures`, `D3D12GetDebugInterface`, `D3D12GetInterface`, `D3D12SerializeRootSignature`
([functions list](https://docs.xbox.com/_llms/en/apis/gdk-apis/graphics-api-reference/d3d12.txt)). The console entry
point this project calls `D3D12XboxCreateDevice` (exported by `d3d12_xs.dll` as `D3D12ConsCreateDevice`, see
[d3d12_xs_surface.md §2.1](d3d12_xs_surface.md)) has no page under either name.

What is documented next to it is the interface selector:

```cpp
HRESULT D3D12GetInterface(
    const IID & rclsid,
    const IID & riid,
    void  ppvDebug
)
```

"Selects an SDK version at runtime when the system is in Windows Developer Mode." The page defines four CLSIDs
(`CLSID_D3D12Debug`, `CLSID_D3D12Tools`, `CLSID_D3D12DeviceRemovedExtendedData`, `CLSID_D3D12SDKConfiguration`) with
their GUID words and maps each to an interface. It creates no device.
Source: <https://docs.xbox.com/reference/graphics/d3d12/functions/d3d12getinterface_public.md>

### 5.2 Command queue

```cpp
HRESULT CreateCommandQueue(
    const D3D12_COMMAND_QUEUE_DESC  pDesc,
    const IID & riid,
    void  ppCommandQueue
)
```

The desc:

```cpp
typedef struct D3D12_COMMAND_QUEUE_DESC
{
    D3D12_COMMAND_LIST_TYPE Type;
    INT Priority;
    D3D12_COMMAND_QUEUE_FLAGS Flags;
    UINT NodeMask;
} D3D12_COMMAND_QUEUE_DESC
```

`Priority` is a `D3D12_COMMAND_QUEUE_PRIORITY` constant, `NodeMask` must set exactly one bit, and `NodeMask` 0 means
the single-GPU case. The same struct is returned by `ID3D12CommandQueue::GetDesc`. `ID3D12Device9::CreateCommandQueue1`
adds a creator ID and is documented separately.
Source: <https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_command_queue_desc_public.md>,
<https://docs.xbox.com/reference/graphics/d3d12/interfaces/id3d12device/methods/id3d12device_createcommandqueue_public.md>

### 5.3 Compute and graphics PSOs

```cpp
HRESULT CreateComputePipelineState(
    const D3D12_COMPUTE_PIPELINE_STATE_DESC  pDesc,
    const IID & riid,
    void  ppPipelineState
)
```

```cpp
typedef struct D3D12_COMPUTE_PIPELINE_STATE_DESC
{
    ID3D12RootSignature  pRootSignature;
    D3D12_SHADER_BYTECODE CS;
    UINT NodeMask;
    D3D12_CACHED_PIPELINE_STATE CachedPSO;
    D3D12_PIPELINE_STATE_FLAGS Flags;
} D3D12_COMPUTE_PIPELINE_STATE_DESC
```

`CachedPSO` may be null/zero. `Flags` is typed by `D3D12_PIPELINE_STATE_FLAGS`, whose page is one of the 404s
(`d3d12_xs/enums/d3d12_pipeline_state_flags_xs.md`). The returned object is immutable and "contains no methods".
Source: <https://docs.xbox.com/reference/graphics/d3d12/interfaces/id3d12device/methods/id3d12device_createcomputepipelinestate_public.md>,
<https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_compute_pipeline_state_desc_public.md>

```cpp
HRESULT CreateGraphicsPipelineState(
    const D3D12_GRAPHICS_PIPELINE_STATE_DESC  pDesc,
    const IID & riid,
    void  ppPipelineState
)
```

Same return contract (`E_OUTOFMEMORY` and the standard D3D12 codes) and the same immutable result. The graphics desc
page is linked as `d3d12_x/structs/d3d12_graphics_pipeline_state_desc.md`, which is not part of the public `d3d12`
index, so the field list is **not documented in the public GDK reference**.
Source: <https://docs.xbox.com/reference/graphics/d3d12/interfaces/id3d12device/methods/id3d12device_creategraphicspipelinestate_public.md>

**The stream-desc PSO has no page.** `D3D12_PIPELINE_STATE_STREAM_DESC` is documented and its Remarks name
`ID3D12Device2::CreatePipelineState` as the consumer, but that method's page is absent: the
`ID3D12Device2` entry in the interfaces index carries no methods, and
`.../id3d12device2/methods/id3d12device2_createpipelinestate_public.md` returns 404. The struct itself is specified in
full, including the alignment rule:

```cpp
typedef struct D3D12_PIPELINE_STATE_STREAM_DESC
{
    SIZE_T SizeInBytes;
    void  pPipelineStateSubobjectStream;
} D3D12_PIPELINE_STATE_STREAM_DESC
```

The stream is "an alternating set of `D3D12_PIPELINE_STATE_SUBOBJECT_TYPE`" and its subobject, each pair aligned to
the natural word alignment (`alignas(void*)` or a union of the whole subobject with `void*`, not of the enum alone,
which misaligns types such as `D3D12_PRIMITIVE_TOPOLOGY_TYPE`). The pipeline kind (COMPUTE, GRAPHICS, MESH) is
inferred from which of VS/CS/MS is non-null; none or more than one fails creation. Irrelevant subobjects are ignored,
missing ones take defaults, duplicate or derived-duplicate subobjects fail validation.
Source: <https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_pipeline_state_stream_desc_public.md>

**`D3D12_PIPELINE_STATE_SUBOBJECT_TYPE` is not documented in the public GDK reference**; its page is linked as
`d3d12_x/enums/d3d12_pipeline_state_subobject_type.md` and is not in the index.

### 5.4 Pipeline libraries

```cpp
HRESULT CreatePipelineLibrary(
    const void  pLibraryBlob,
    SIZE_T BlobLength,
    const IID & riid,
    void  ppPipelineLibrary
)
```

On `ID3D12Device1`. Gate it on `CheckFeatureSupport(D3D12_FEATURE_SHADER_CACHE)` returning
`D3D12_SHADER_CACHE_SUPPORT_LIBRARY`, otherwise the call always returns `DXGI_ERROR_NOT_SUPPORTED`. A non-empty blob
is validated and parsed but not copied, so the caller must keep it alive for the library's lifetime; an empty blob
starts an empty library. Documented failures: `E_INVALIDARG` (corrupt or unrecognised blob),
`D3D12_ERROR_DRIVER_VERSION_MISMATCH` (data from an old driver or runtime), `D3D12_ERROR_ADAPTER_NOT_FOUND` (data
from other hardware). Passing `nullptr` for the output validates the blob and returns `S_FALSE`.
Source: <https://docs.xbox.com/reference/graphics/d3d12/interfaces/id3d12device1/methods/id3d12device1_createpipelinelibrary_public.md>

```cpp
HRESULT StorePipeline(
    LPCWSTR pName,
    ID3D12PipelineState  pPipeline
)
```

Adds a PSO under a unique name; overwriting is unsupported, and `E_INVALIDARG` means the name exists.
Source: <https://docs.xbox.com/reference/graphics/d3d12/interfaces/id3d12pipelinelibrary/methods/id3d12pipelinelibrary_storepipeline_public.md>

### 5.5 Present

**Presentation is not documented in the public GDK graphics reference.** The graphics API index contains only
`d3d12` and `d3d12_xs`, and the only DXGI item in either is `DXGI_ADAPTER_DESC2` under `d3d12_xs/structs`; there is no
`IDXGISwapChain`, `DXGI_PRESENT` or `PresentX` page. The console's present entry point (`PresentX` in this
repository's usage, [d3d12_command_objects.md](d3d12_command_objects.md)) and its frame-event handshake have no
public page under any URL form tried, including `d3d12/functions/d3d12createdevice_public.md`'s neighbours.

## 6. What this means for Leonida

Mapping the documented surface onto what the layer does today. Each row names the code that would change; "gap"
means the code is absent, not that the API is undocumented.

| Area | Documented contract | State in this repository | Consequence |
|---|---|---|---|
| Root signature blobs | `D3D12_ROOT_SIGNATURE_DESC` 1.0/1.1, `D3D_ROOT_SIGNATURE_VERSION`, `CreateRootSignature` (§2.3) | `runtime/shims/d3d12_xs_rootsig.cpp` rewrites the measured `0x03000000` high byte and downgrades 1.2 to 1.1 before forwarding to the host serializer | Correct as a forwarding shim; the console flag semantics stay unknown because the flags enum is not published (§2.4) |
| Console root parameters to `s0..s31` | Only `D3D12_ROOT_PARAMETER`'s three inline kinds are published; no user-data slot type (§2.4) | Measured model: inline from `s4`, spill array via a 40-bit pointer at `s[0:1]` ([d3d12_vulkan.md §8.2](d3d12_vulkan.md#82-unknown--open-questions)) | The layer cannot cite a document for the mapping; the shader-side check stays the only test |
| State objects and RTPSOs | `CreateStateObject`, subobject enum, DXIL library, hit groups, shader/pipeline configs (§3) | No `CreateStateObject`, `DispatchRays` or `GetShaderIdentifier` in `runtime/shims/d3d12_vulkan_device.cpp` | Full DXR stays unimplemented; `RaytracingTier` is answered `NOT_SUPPORTED` and that answer is what the game reacts to |
| DXIL library exports | Library target 6.3+, `NumExports == 0` means export all, `D3D12_EXPORT_FLAG_NONE` is the only flag (§3.2, §3.3) | Not reached: no state-object path | When a state object path is added, the export-set and rename rules are the part a translator must preserve to keep `GetShaderIdentifier` stable |
| Graphics PSOs | `CreateGraphicsPipelineState` with `D3D12_GRAPHICS_PIPELINE_STATE_DESC`; the stream desc's alignment and kind-inference rules (§5.3) | Desc-form creation exists in `d3d12_vulkan_device.cpp`; the stream-desc `CreatePipelineState` is not implemented there (only `LoadPipeline` consumes a stream desc) | A title that builds PSOs through `ID3D12Device2` needs that path; the documented alignment rule is the spec for it |
| Compute PSO on the console slot | `CreateComputePipelineState` (§5.3) | `d3d12_xs_device.cpp` proxies PC slots 10/11 plus the measured console compute-PSO slot 62 | The console-only entry point has three documented equivalents (compute, graphics, state object); the slot's arity is measured, not documented |
| Draws and dispatch | `DrawInstanced`, `DrawIndexedInstanced`, `Dispatch`, root-argument setters (interfaces index) | Present and GPU-tested ([first_light.md](first_light.md)) | Nothing to change from the documentation side |
| Shader cache | Session desc with its three documented defaults, `StoreValue`/`FindValue` two-call pattern, `ShaderCacheControl` in developer mode, library gate (§4) | Implemented with `ID3D12Device9::CreateShaderCacheSession`, the session object and `ShaderCacheControl` (D3D-34, `runtime/test/d3d12_shader_cache_test.cpp`) | The defaults (1 KB / 128 entries / 128 MB) are the numbers the layer must reproduce for a host-parity answer |
| Present | No page (§5.5) | `runtime/shims/xbox_present.cpp` plus a DXGI swapchain fallback | The layer's presenter is built on measurement; there is no document to check it against |
| Device creation | No `D3D12CreateDevice` page; `D3D12GetInterface` only (§5.1) | `D3D12XboxCreateDevice` with an unmeasured 24-word params block | Params and flags stay open; nothing in the public reference narrows them |

Two consequences follow for how this documentation is used. First, the parts a replay layer needs most precisely
(the enum values) are the parts this reference publishes worst: three enums in this page print colliding or all-zero
constants, and two enum pages do not exist. Values must come from the header the layer is compiled against, with the
prose here used only for meaning. Second, the console-specific behaviour the layer actually depends on, the root
signature to user-SGPR mapping and the present path, is exactly what Microsoft keeps out of the public reference, so
the measured notes in [d3d12_vulkan.md](d3d12_vulkan.md) and [d3d12_command_objects.md](d3d12_command_objects.md)
remain the operative source for those two.

## 7. Pages used

| Page | URL |
|---|---|
| Graphics API reference | <https://docs.xbox.com/_llms/en/apis/gdk-apis/graphics-api-reference.txt> |
| `d3d12` index | <https://docs.xbox.com/_llms/en/apis/gdk-apis/graphics-api-reference/d3d12.txt> |
| `d3d12` interfaces index | <https://docs.xbox.com/_llms/en/apis/gdk-apis/graphics-api-reference/d3d12/interfaces.txt> |
| `D3D12_LOCAL_ROOT_SIGNATURE` | <https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_local_root_signature_public.md> |
| `D3D12_GLOBAL_ROOT_SIGNATURE` | <https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_global_root_signature_public.md> |
| `D3D12_ROOT_SIGNATURE_DESC` | <https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_root_signature_desc_public.md> |
| `D3D12_ROOT_PARAMETER` | <https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_root_parameter_public.md> |
| `D3D12_ROOT_PARAMETER_TYPE` | <https://docs.xbox.com/reference/graphics/d3d12/enums/d3d12_root_parameter_type_public.md> |
| `D3D_ROOT_SIGNATURE_VERSION` | <https://docs.xbox.com/reference/graphics/d3d12/enums/d3d_root_signature_version_public.md> |
| `CreateRootSignature` | <https://docs.xbox.com/reference/graphics/d3d12/interfaces/id3d12device/methods/id3d12device_createrootsignature_public.md> |
| `ID3D12RootSignature` | <https://docs.xbox.com/reference/graphics/d3d12/interfaces/id3d12rootsignature/id3d12rootsignature_public.md> |
| `D3D12_STATE_SUBOBJECT_TYPE` | <https://docs.xbox.com/reference/graphics/d3d12/enums/d3d12_state_subobject_type_public.md> |
| `D3D12_DXIL_LIBRARY_DESC` | <https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_dxil_library_desc_public.md> |
| `D3D12_SHADER_BYTECODE` | <https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_shader_bytecode_public.md> |
| `D3D12_EXPORT_DESC` | <https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_export_desc_public.md> |
| `D3D12_EXPORT_FLAGS` | <https://docs.xbox.com/reference/graphics/d3d12/enums/d3d12_export_flags_public.md> |
| `D3D12_HIT_GROUP_DESC` | <https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_hit_group_desc_public.md> |
| `D3D12_HIT_GROUP_TYPE` | <https://docs.xbox.com/reference/graphics/d3d12/enums/d3d12_hit_group_type_public.md> |
| `D3D12_RAYTRACING_SHADER_CONFIG` | <https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_raytracing_shader_config_public.md> |
| `D3D12_RAYTRACING_PIPELINE_CONFIG` | <https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_raytracing_pipeline_config_public.md> |
| `D3D12_STATE_OBJECT_DESC` | <https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_state_object_desc_public.md> |
| `D3D12_STATE_SUBOBJECT` | <https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_state_subobject_public.md> |
| `D3D12_STATE_OBJECT_TYPE` | <https://docs.xbox.com/reference/graphics/d3d12/enums/d3d12_state_object_type_public.md> |
| `D3D12_EXISTING_COLLECTION_DESC` | <https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_existing_collection_desc_public.md> |
| `D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION` | <https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_subobject_to_exports_association_public.md> |
| `D3D12_DXIL_SUBOBJECT_TO_EXPORTS_ASSOCIATION` | <https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_dxil_subobject_to_exports_association_public.md> |
| `CreateStateObject` | <https://docs.xbox.com/reference/graphics/d3d12/interfaces/id3d12device5/methods/id3d12device5_createstateobject_public.md> |
| `GetShaderIdentifier` | <https://docs.xbox.com/reference/graphics/d3d12/interfaces/id3d12stateobjectproperties/methods/id3d12stateobjectproperties_getshaderidentifier_public.md> |
| `CreateShaderCacheSession` | <https://docs.xbox.com/reference/graphics/d3d12/interfaces/id3d12device9/methods/id3d12device9_createshadercachesession_public.md> |
| `ShaderCacheControl` | <https://docs.xbox.com/reference/graphics/d3d12/interfaces/id3d12device9/methods/id3d12device9_shadercachecontrol_public.md> |
| `D3D12_SHADER_CACHE_SESSION_DESC` | <https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_shader_cache_session_desc_public.md> |
| `D3D12_SHADER_CACHE_MODE` | <https://docs.xbox.com/reference/graphics/d3d12/enums/d3d12_shader_cache_mode_public.md> |
| `D3D12_SHADER_CACHE_FLAGS` | <https://docs.xbox.com/reference/graphics/d3d12/enums/d3d12_shader_cache_flags_public.md> |
| `D3D12_SHADER_CACHE_KIND_FLAGS` | <https://docs.xbox.com/reference/graphics/d3d12/enums/d3d12_shader_cache_kind_flags_public.md> |
| `D3D12_SHADER_CACHE_CONTROL_FLAGS` | <https://docs.xbox.com/reference/graphics/d3d12/enums/d3d12_shader_cache_control_flags_public.md> |
| `D3D12_SHADER_CACHE_SUPPORT_FLAGS` | <https://docs.xbox.com/reference/graphics/d3d12/enums/d3d12_shader_cache_support_flags_public.md> |
| `D3D12_FEATURE_DATA_SHADER_CACHE` | <https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_feature_data_shader_cache_public.md> |
| `ID3D12ShaderCacheSession` methods | <https://docs.xbox.com/reference/graphics/d3d12/interfaces/id3d12shadercachesession/methods/id3d12shadercachesession_storevalue_public.md>, `..._findvalue_public.md`, `..._getdesc_public.md`, `..._setdeleteondestroy_public.md` |
| `D3D12GetInterface` | <https://docs.xbox.com/reference/graphics/d3d12/functions/d3d12getinterface_public.md> |
| `CreateCommandQueue` | <https://docs.xbox.com/reference/graphics/d3d12/interfaces/id3d12device/methods/id3d12device_createcommandqueue_public.md> |
| `D3D12_COMMAND_QUEUE_DESC` | <https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_command_queue_desc_public.md> |
| `CreateComputePipelineState` | <https://docs.xbox.com/reference/graphics/d3d12/interfaces/id3d12device/methods/id3d12device_createcomputepipelinestate_public.md> |
| `D3D12_COMPUTE_PIPELINE_STATE_DESC` | <https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_compute_pipeline_state_desc_public.md> |
| `CreateGraphicsPipelineState` | <https://docs.xbox.com/reference/graphics/d3d12/interfaces/id3d12device/methods/id3d12device_creategraphicspipelinestate_public.md> |
| `D3D12_PIPELINE_STATE_STREAM_DESC` | <https://docs.xbox.com/reference/graphics/d3d12/structs/d3d12_pipeline_state_stream_desc_public.md> |
| `CreatePipelineLibrary` | <https://docs.xbox.com/reference/graphics/d3d12/interfaces/id3d12device1/methods/id3d12device1_createpipelinelibrary_public.md> |
| `StorePipeline` | <https://docs.xbox.com/reference/graphics/d3d12/interfaces/id3d12pipelinelibrary/methods/id3d12pipelinelibrary_storepipeline_public.md> |
| Raytracing concept index | <https://docs.xbox.com/build/core-features/graphics/newfeatures/raytracing/index.md> |

Pages that were looked for and do not exist publicly (404): `d3d12/functions/d3d12createdevice_public.md`,
`d3d12/enums/d3d12_root_signature_flags_public.md`, `d3d12_xs/enums/d3d12_root_signature_flags_xs.md`,
`d3d12/interfaces/id3d12device2/methods/id3d12device2_createpipelinestate_public.md`,
`d3d12_xs/enums/d3d12_pipeline_state_flags_xs.md`, every `d3d12_xs/interfaces/...` page, and the NDA-gated
`rt-xdxr.md` / `rt-offlinertpso.md`.
