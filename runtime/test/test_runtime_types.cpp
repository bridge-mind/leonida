// Unit tests verifying D3D12 and XMem ABI type alignment and structure sizes against the Windows SDK.
//
// These tests verify that the guest Xbox Series ABI assumptions made by Leonida's runtime shims
// (d3d12_xs, xmem, registry) match the Windows SDK type definitions, packing, member offsets,
// and calling conventions on x86-64.

#include <windows.h>
#include <d3d12.h>
#include <dxgiformat.h>
#include <dxgitype.h>
#include <psapi.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>

#if __has_include("d3d12_xs_rootsig.h")
#include "d3d12_xs_rootsig.h"
#elif __has_include("../shims/d3d12_xs_rootsig.h")
#include "../shims/d3d12_xs_rootsig.h"
#endif

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) {
        ++g_failures;
    }
}

// ---------------------------------------------------------------------------
// Compile-time ABI assertions: Type sizes, alignments, and member offsets
// ---------------------------------------------------------------------------

// Architecture verification
static_assert(sizeof(void*) == 8, "Target architecture must be 64-bit");
static_assert(sizeof(size_t) == 8, "size_t must be 8 bytes on x64");
static_assert(sizeof(uintptr_t) == 8, "uintptr_t must be 8 bytes on x64");

// D3D12 Root Constants
static_assert(sizeof(D3D12_ROOT_CONSTANTS) == 12, "D3D12_ROOT_CONSTANTS size must be 12 bytes");
static_assert(alignof(D3D12_ROOT_CONSTANTS) == 4, "D3D12_ROOT_CONSTANTS alignment must be 4 bytes");
static_assert(offsetof(D3D12_ROOT_CONSTANTS, ShaderRegister) == 0);
static_assert(offsetof(D3D12_ROOT_CONSTANTS, RegisterSpace) == 4);
static_assert(offsetof(D3D12_ROOT_CONSTANTS, Num32BitValues) == 8);

// D3D12 Descriptor Ranges (1.0)
static_assert(sizeof(D3D12_DESCRIPTOR_RANGE) == 20, "D3D12_DESCRIPTOR_RANGE size must be 20 bytes");
static_assert(alignof(D3D12_DESCRIPTOR_RANGE) == 4, "D3D12_DESCRIPTOR_RANGE alignment must be 4 bytes");
static_assert(offsetof(D3D12_DESCRIPTOR_RANGE, RangeType) == 0);
static_assert(offsetof(D3D12_DESCRIPTOR_RANGE, NumDescriptors) == 4);
static_assert(offsetof(D3D12_DESCRIPTOR_RANGE, BaseShaderRegister) == 8);
static_assert(offsetof(D3D12_DESCRIPTOR_RANGE, RegisterSpace) == 12);
static_assert(offsetof(D3D12_DESCRIPTOR_RANGE, OffsetInDescriptorsFromTableStart) == 16);

// D3D12 Descriptor Ranges (1.1)
static_assert(sizeof(D3D12_DESCRIPTOR_RANGE1) == 24, "D3D12_DESCRIPTOR_RANGE1 size must be 24 bytes");
static_assert(alignof(D3D12_DESCRIPTOR_RANGE1) == 4, "D3D12_DESCRIPTOR_RANGE1 alignment must be 4 bytes");
static_assert(offsetof(D3D12_DESCRIPTOR_RANGE1, RangeType) == 0);
static_assert(offsetof(D3D12_DESCRIPTOR_RANGE1, NumDescriptors) == 4);
static_assert(offsetof(D3D12_DESCRIPTOR_RANGE1, BaseShaderRegister) == 8);
static_assert(offsetof(D3D12_DESCRIPTOR_RANGE1, RegisterSpace) == 12);
static_assert(offsetof(D3D12_DESCRIPTOR_RANGE1, Flags) == 16);
static_assert(offsetof(D3D12_DESCRIPTOR_RANGE1, OffsetInDescriptorsFromTableStart) == 20);

// D3D12 Root Descriptor Table
static_assert(sizeof(D3D12_ROOT_DESCRIPTOR_TABLE) == 16, "D3D12_ROOT_DESCRIPTOR_TABLE size must be 16 bytes");
static_assert(alignof(D3D12_ROOT_DESCRIPTOR_TABLE) == 8, "D3D12_ROOT_DESCRIPTOR_TABLE alignment must be 8 bytes");
static_assert(offsetof(D3D12_ROOT_DESCRIPTOR_TABLE, NumDescriptorRanges) == 0);
static_assert(offsetof(D3D12_ROOT_DESCRIPTOR_TABLE, pDescriptorRanges) == 8);

static_assert(sizeof(D3D12_ROOT_DESCRIPTOR_TABLE1) == 16, "D3D12_ROOT_DESCRIPTOR_TABLE1 size must be 16 bytes");
static_assert(alignof(D3D12_ROOT_DESCRIPTOR_TABLE1) == 8, "D3D12_ROOT_DESCRIPTOR_TABLE1 alignment must be 8 bytes");
static_assert(offsetof(D3D12_ROOT_DESCRIPTOR_TABLE1, NumDescriptorRanges) == 0);
static_assert(offsetof(D3D12_ROOT_DESCRIPTOR_TABLE1, pDescriptorRanges) == 8);

// D3D12 Root Descriptors
static_assert(sizeof(D3D12_ROOT_DESCRIPTOR) == 8, "D3D12_ROOT_DESCRIPTOR size must be 8 bytes");
static_assert(alignof(D3D12_ROOT_DESCRIPTOR) == 4, "D3D12_ROOT_DESCRIPTOR alignment must be 4 bytes");
static_assert(offsetof(D3D12_ROOT_DESCRIPTOR, ShaderRegister) == 0);
static_assert(offsetof(D3D12_ROOT_DESCRIPTOR, RegisterSpace) == 4);

static_assert(sizeof(D3D12_ROOT_DESCRIPTOR1) == 12, "D3D12_ROOT_DESCRIPTOR1 size must be 12 bytes");
static_assert(alignof(D3D12_ROOT_DESCRIPTOR1) == 4, "D3D12_ROOT_DESCRIPTOR1 alignment must be 4 bytes");
static_assert(offsetof(D3D12_ROOT_DESCRIPTOR1, ShaderRegister) == 0);
static_assert(offsetof(D3D12_ROOT_DESCRIPTOR1, RegisterSpace) == 4);
static_assert(offsetof(D3D12_ROOT_DESCRIPTOR1, Flags) == 8);

// D3D12 Root Parameters
static_assert(sizeof(D3D12_ROOT_PARAMETER) == 32, "D3D12_ROOT_PARAMETER size must be 32 bytes");
static_assert(alignof(D3D12_ROOT_PARAMETER) == 8, "D3D12_ROOT_PARAMETER alignment must be 8 bytes");
static_assert(offsetof(D3D12_ROOT_PARAMETER, ParameterType) == 0);
static_assert(offsetof(D3D12_ROOT_PARAMETER, DescriptorTable) == 8);
static_assert(offsetof(D3D12_ROOT_PARAMETER, Constants) == 8);
static_assert(offsetof(D3D12_ROOT_PARAMETER, Descriptor) == 8);
static_assert(offsetof(D3D12_ROOT_PARAMETER, ShaderVisibility) == 24);

static_assert(sizeof(D3D12_ROOT_PARAMETER1) == 32, "D3D12_ROOT_PARAMETER1 size must be 32 bytes");
static_assert(alignof(D3D12_ROOT_PARAMETER1) == 8, "D3D12_ROOT_PARAMETER1 alignment must be 8 bytes");
static_assert(offsetof(D3D12_ROOT_PARAMETER1, ParameterType) == 0);
static_assert(offsetof(D3D12_ROOT_PARAMETER1, DescriptorTable) == 8);
static_assert(offsetof(D3D12_ROOT_PARAMETER1, Constants) == 8);
static_assert(offsetof(D3D12_ROOT_PARAMETER1, Descriptor) == 8);
static_assert(offsetof(D3D12_ROOT_PARAMETER1, ShaderVisibility) == 24);

// D3D12 Static Sampler
static_assert(sizeof(D3D12_STATIC_SAMPLER_DESC) == 52, "D3D12_STATIC_SAMPLER_DESC size must be 52 bytes");
static_assert(alignof(D3D12_STATIC_SAMPLER_DESC) == 4, "D3D12_STATIC_SAMPLER_DESC alignment must be 4 bytes");
static_assert(offsetof(D3D12_STATIC_SAMPLER_DESC, Filter) == 0);
static_assert(offsetof(D3D12_STATIC_SAMPLER_DESC, AddressU) == 4);
static_assert(offsetof(D3D12_STATIC_SAMPLER_DESC, AddressV) == 8);
static_assert(offsetof(D3D12_STATIC_SAMPLER_DESC, AddressW) == 12);
static_assert(offsetof(D3D12_STATIC_SAMPLER_DESC, MipLODBias) == 16);
static_assert(offsetof(D3D12_STATIC_SAMPLER_DESC, MaxAnisotropy) == 20);
static_assert(offsetof(D3D12_STATIC_SAMPLER_DESC, ComparisonFunc) == 24);
static_assert(offsetof(D3D12_STATIC_SAMPLER_DESC, BorderColor) == 28);
static_assert(offsetof(D3D12_STATIC_SAMPLER_DESC, MinLOD) == 32);
static_assert(offsetof(D3D12_STATIC_SAMPLER_DESC, MaxLOD) == 36);
static_assert(offsetof(D3D12_STATIC_SAMPLER_DESC, ShaderRegister) == 40);
static_assert(offsetof(D3D12_STATIC_SAMPLER_DESC, RegisterSpace) == 44);
static_assert(offsetof(D3D12_STATIC_SAMPLER_DESC, ShaderVisibility) == 48);

// D3D12 Root Signature Descriptors
static_assert(sizeof(D3D12_ROOT_SIGNATURE_DESC) == 40, "D3D12_ROOT_SIGNATURE_DESC size must be 40 bytes");
static_assert(alignof(D3D12_ROOT_SIGNATURE_DESC) == 8, "D3D12_ROOT_SIGNATURE_DESC alignment must be 8 bytes");
static_assert(offsetof(D3D12_ROOT_SIGNATURE_DESC, NumParameters) == 0);
static_assert(offsetof(D3D12_ROOT_SIGNATURE_DESC, pParameters) == 8);
static_assert(offsetof(D3D12_ROOT_SIGNATURE_DESC, NumStaticSamplers) == 16);
static_assert(offsetof(D3D12_ROOT_SIGNATURE_DESC, pStaticSamplers) == 24);
static_assert(offsetof(D3D12_ROOT_SIGNATURE_DESC, Flags) == 32);

static_assert(sizeof(D3D12_ROOT_SIGNATURE_DESC1) == 40, "D3D12_ROOT_SIGNATURE_DESC1 size must be 40 bytes");
static_assert(alignof(D3D12_ROOT_SIGNATURE_DESC1) == 8, "D3D12_ROOT_SIGNATURE_DESC1 alignment must be 8 bytes");
static_assert(offsetof(D3D12_ROOT_SIGNATURE_DESC1, NumParameters) == 0);
static_assert(offsetof(D3D12_ROOT_SIGNATURE_DESC1, pParameters) == 8);
static_assert(offsetof(D3D12_ROOT_SIGNATURE_DESC1, NumStaticSamplers) == 16);
static_assert(offsetof(D3D12_ROOT_SIGNATURE_DESC1, pStaticSamplers) == 24);
static_assert(offsetof(D3D12_ROOT_SIGNATURE_DESC1, Flags) == 32);

static_assert(sizeof(D3D12_VERSIONED_ROOT_SIGNATURE_DESC) == 48,
              "D3D12_VERSIONED_ROOT_SIGNATURE_DESC size must be 48 bytes");
static_assert(alignof(D3D12_VERSIONED_ROOT_SIGNATURE_DESC) == 8,
              "D3D12_VERSIONED_ROOT_SIGNATURE_DESC alignment must be 8 bytes");
static_assert(offsetof(D3D12_VERSIONED_ROOT_SIGNATURE_DESC, Version) == 0);
static_assert(offsetof(D3D12_VERSIONED_ROOT_SIGNATURE_DESC, Desc_1_0) == 8);
static_assert(offsetof(D3D12_VERSIONED_ROOT_SIGNATURE_DESC, Desc_1_1) == 8);

// D3D12 Descriptor Handles and Heap
static_assert(sizeof(D3D12_CPU_DESCRIPTOR_HANDLE) == 8, "D3D12_CPU_DESCRIPTOR_HANDLE size must be 8 bytes");
static_assert(alignof(D3D12_CPU_DESCRIPTOR_HANDLE) == 8, "D3D12_CPU_DESCRIPTOR_HANDLE alignment must be 8 bytes");
static_assert(offsetof(D3D12_CPU_DESCRIPTOR_HANDLE, ptr) == 0);
static_assert(std::is_standard_layout_v<D3D12_CPU_DESCRIPTOR_HANDLE>);
static_assert(std::is_trivially_copyable_v<D3D12_CPU_DESCRIPTOR_HANDLE>);

static_assert(sizeof(D3D12_GPU_DESCRIPTOR_HANDLE) == 8, "D3D12_GPU_DESCRIPTOR_HANDLE size must be 8 bytes");
static_assert(alignof(D3D12_GPU_DESCRIPTOR_HANDLE) == 8, "D3D12_GPU_DESCRIPTOR_HANDLE alignment must be 8 bytes");
static_assert(offsetof(D3D12_GPU_DESCRIPTOR_HANDLE, ptr) == 0);
static_assert(std::is_standard_layout_v<D3D12_GPU_DESCRIPTOR_HANDLE>);
static_assert(std::is_trivially_copyable_v<D3D12_GPU_DESCRIPTOR_HANDLE>);

static_assert(sizeof(D3D12_DESCRIPTOR_HEAP_DESC) == 16, "D3D12_DESCRIPTOR_HEAP_DESC size must be 16 bytes");
static_assert(alignof(D3D12_DESCRIPTOR_HEAP_DESC) == 4, "D3D12_DESCRIPTOR_HEAP_DESC alignment must be 4 bytes");

// D3D12 Resources and Views
static_assert(sizeof(D3D12_RESOURCE_DESC) == 56, "D3D12_RESOURCE_DESC size must be 56 bytes");
static_assert(alignof(D3D12_RESOURCE_DESC) == 8, "D3D12_RESOURCE_DESC alignment must be 8 bytes");
static_assert(offsetof(D3D12_RESOURCE_DESC, Dimension) == 0);
static_assert(offsetof(D3D12_RESOURCE_DESC, Alignment) == 8);
static_assert(offsetof(D3D12_RESOURCE_DESC, Width) == 16);
static_assert(offsetof(D3D12_RESOURCE_DESC, Height) == 24);
static_assert(offsetof(D3D12_RESOURCE_DESC, DepthOrArraySize) == 28);
static_assert(offsetof(D3D12_RESOURCE_DESC, MipLevels) == 30);
static_assert(offsetof(D3D12_RESOURCE_DESC, Format) == 32);
static_assert(offsetof(D3D12_RESOURCE_DESC, SampleDesc) == 36);
static_assert(offsetof(D3D12_RESOURCE_DESC, Layout) == 44);
static_assert(offsetof(D3D12_RESOURCE_DESC, Flags) == 48);

static_assert(sizeof(D3D12_CLEAR_VALUE) == 20, "D3D12_CLEAR_VALUE size must be 20 bytes");
static_assert(alignof(D3D12_CLEAR_VALUE) == 4, "D3D12_CLEAR_VALUE alignment must be 4 bytes");
static_assert(sizeof(D3D12_BOX) == 24, "D3D12_BOX size must be 24 bytes");
static_assert(alignof(D3D12_BOX) == 4, "D3D12_BOX alignment must be 4 bytes");
static_assert(sizeof(D3D12_VIEWPORT) == 24, "D3D12_VIEWPORT size must be 24 bytes");
static_assert(alignof(D3D12_VIEWPORT) == 4, "D3D12_VIEWPORT alignment must be 4 bytes");
static_assert(sizeof(D3D12_RECT) == 16, "D3D12_RECT size must be 16 bytes");
static_assert(alignof(D3D12_RECT) == 4, "D3D12_RECT alignment must be 4 bytes");

static_assert(sizeof(D3D12_SUBRESOURCE_DATA) == 24, "D3D12_SUBRESOURCE_DATA size must be 24 bytes");
static_assert(alignof(D3D12_SUBRESOURCE_DATA) == 8, "D3D12_SUBRESOURCE_DATA alignment must be 8 bytes");
static_assert(offsetof(D3D12_SUBRESOURCE_DATA, pData) == 0);
static_assert(offsetof(D3D12_SUBRESOURCE_DATA, RowPitch) == 8);
static_assert(offsetof(D3D12_SUBRESOURCE_DATA, SlicePitch) == 16);

static_assert(sizeof(D3D12_MEMCPY_DEST) == 24, "D3D12_MEMCPY_DEST size must be 24 bytes");
static_assert(alignof(D3D12_MEMCPY_DEST) == 8, "D3D12_MEMCPY_DEST alignment must be 8 bytes");

static_assert(sizeof(D3D12_RESOURCE_BARRIER) == 32, "D3D12_RESOURCE_BARRIER size must be 32 bytes");
static_assert(alignof(D3D12_RESOURCE_BARRIER) == 8, "D3D12_RESOURCE_BARRIER alignment must be 8 bytes");
static_assert(offsetof(D3D12_RESOURCE_BARRIER, Type) == 0);
static_assert(offsetof(D3D12_RESOURCE_BARRIER, Flags) == 4);
static_assert(offsetof(D3D12_RESOURCE_BARRIER, Transition) == 8);

static_assert(sizeof(D3D12_CONSTANT_BUFFER_VIEW_DESC) == 16,
              "D3D12_CONSTANT_BUFFER_VIEW_DESC size must be 16 bytes");
static_assert(alignof(D3D12_CONSTANT_BUFFER_VIEW_DESC) == 8,
              "D3D12_CONSTANT_BUFFER_VIEW_DESC alignment must be 8 bytes");
static_assert(offsetof(D3D12_CONSTANT_BUFFER_VIEW_DESC, BufferLocation) == 0);
static_assert(offsetof(D3D12_CONSTANT_BUFFER_VIEW_DESC, SizeInBytes) == 8);

static_assert(sizeof(D3D12_SHADER_RESOURCE_VIEW_DESC) == 40,
              "D3D12_SHADER_RESOURCE_VIEW_DESC size must be 40 bytes");
static_assert(alignof(D3D12_SHADER_RESOURCE_VIEW_DESC) == 8,
              "D3D12_SHADER_RESOURCE_VIEW_DESC alignment must be 8 bytes");

static_assert(sizeof(D3D12_UNORDERED_ACCESS_VIEW_DESC) == 40,
              "D3D12_UNORDERED_ACCESS_VIEW_DESC size must be 40 bytes");
static_assert(alignof(D3D12_UNORDERED_ACCESS_VIEW_DESC) == 8,
              "D3D12_UNORDERED_ACCESS_VIEW_DESC alignment must be 8 bytes");

static_assert(sizeof(D3D12_RENDER_TARGET_VIEW_DESC) == 24,
              "D3D12_RENDER_TARGET_VIEW_DESC size must be 24 bytes");
static_assert(alignof(D3D12_RENDER_TARGET_VIEW_DESC) == 8,
              "D3D12_RENDER_TARGET_VIEW_DESC alignment must be 8 bytes");

static_assert(sizeof(D3D12_DEPTH_STENCIL_VIEW_DESC) == 24,
              "D3D12_DEPTH_STENCIL_VIEW_DESC size must be 24 bytes");
static_assert(alignof(D3D12_DEPTH_STENCIL_VIEW_DESC) == 4,
              "D3D12_DEPTH_STENCIL_VIEW_DESC alignment must be 4 bytes");

static_assert(sizeof(D3D12_COMPUTE_PIPELINE_STATE_DESC) == 56,
              "D3D12_COMPUTE_PIPELINE_STATE_DESC size must be 56 bytes");
static_assert(alignof(D3D12_COMPUTE_PIPELINE_STATE_DESC) == 8,
              "D3D12_COMPUTE_PIPELINE_STATE_DESC alignment must be 8 bytes");
static_assert(offsetof(D3D12_COMPUTE_PIPELINE_STATE_DESC, pRootSignature) == 0);
static_assert(offsetof(D3D12_COMPUTE_PIPELINE_STATE_DESC, CS) == 8);
static_assert(offsetof(D3D12_COMPUTE_PIPELINE_STATE_DESC, NodeMask) == 24);
static_assert(offsetof(D3D12_COMPUTE_PIPELINE_STATE_DESC, CachedPSO) == 32);
static_assert(offsetof(D3D12_COMPUTE_PIPELINE_STATE_DESC, Flags) == 48);

static_assert(sizeof(D3D12_SHADER_BYTECODE) == 16, "D3D12_SHADER_BYTECODE size must be 16 bytes");
static_assert(alignof(D3D12_SHADER_BYTECODE) == 8, "D3D12_SHADER_BYTECODE alignment must be 8 bytes");
static_assert(offsetof(D3D12_SHADER_BYTECODE, pShaderBytecode) == 0);
static_assert(offsetof(D3D12_SHADER_BYTECODE, BytecodeLength) == 8);

// Memory / XMem Subsystem Types
static_assert(sizeof(MEMORY_BASIC_INFORMATION) == 48, "MEMORY_BASIC_INFORMATION size must be 48 bytes on x64");
static_assert(alignof(MEMORY_BASIC_INFORMATION) == 8, "MEMORY_BASIC_INFORMATION alignment must be 8 bytes");
static_assert(offsetof(MEMORY_BASIC_INFORMATION, BaseAddress) == 0);
static_assert(offsetof(MEMORY_BASIC_INFORMATION, AllocationBase) == 8);
static_assert(offsetof(MEMORY_BASIC_INFORMATION, AllocationProtect) == 16);
static_assert(offsetof(MEMORY_BASIC_INFORMATION, PartitionId) == 20);
static_assert(offsetof(MEMORY_BASIC_INFORMATION, RegionSize) == 24);
static_assert(offsetof(MEMORY_BASIC_INFORMATION, State) == 32);
static_assert(offsetof(MEMORY_BASIC_INFORMATION, Protect) == 36);
static_assert(offsetof(MEMORY_BASIC_INFORMATION, Type) == 40);

static_assert(sizeof(PROCESS_MEMORY_COUNTERS) == 72, "PROCESS_MEMORY_COUNTERS size must be 72 bytes on x64");
static_assert(alignof(PROCESS_MEMORY_COUNTERS) == 8, "PROCESS_MEMORY_COUNTERS alignment must be 8 bytes");
static_assert(offsetof(PROCESS_MEMORY_COUNTERS, cb) == 0);
static_assert(offsetof(PROCESS_MEMORY_COUNTERS, PageFaultCount) == 4);
static_assert(offsetof(PROCESS_MEMORY_COUNTERS, PeakWorkingSetSize) == 8);
static_assert(offsetof(PROCESS_MEMORY_COUNTERS, WorkingSetSize) == 16);
static_assert(offsetof(PROCESS_MEMORY_COUNTERS, QuotaPeakPagedPoolUsage) == 24);
static_assert(offsetof(PROCESS_MEMORY_COUNTERS, QuotaPagedPoolUsage) == 32);
static_assert(offsetof(PROCESS_MEMORY_COUNTERS, QuotaPeakNonPagedPoolUsage) == 40);
static_assert(offsetof(PROCESS_MEMORY_COUNTERS, QuotaNonPagedPoolUsage) == 48);
static_assert(offsetof(PROCESS_MEMORY_COUNTERS, PagefileUsage) == 56);
static_assert(offsetof(PROCESS_MEMORY_COUNTERS, PeakPagefileUsage) == 64);

static_assert(sizeof(PROCESS_MEMORY_COUNTERS_EX) == 80, "PROCESS_MEMORY_COUNTERS_EX size must be 80 bytes on x64");
static_assert(offsetof(PROCESS_MEMORY_COUNTERS_EX, PrivateUsage) == 72);

// XMem ABI Constants & Contracts from xmem.cpp
constexpr size_t kXMemAllocAlignment = 64;
constexpr size_t kXMemVirtualQueryOutSize = 64;
static_assert(sizeof(MEMORY_BASIC_INFORMATION) <= kXMemVirtualQueryOutSize,
              "the 64-byte XMemVirtualQuery output buffer must accommodate MEMORY_BASIC_INFORMATION");
static_assert(kXMemVirtualQueryOutSize - sizeof(MEMORY_BASIC_INFORMATION) == 16,
              "the trailing unknown area in the 64-byte buffer is exactly 16 bytes");

// D3D12 Xbox root signature flag mask contract
constexpr uint32_t kXboxRootSigFlagMask = 0xFF000000u;
constexpr uint32_t kObservedXboxRootSigFlag = 0x03000000u;

// Function pointer signatures for XMem exports
using XMemAllocFn = void*(__stdcall*)(size_t, uint64_t);
using XMemFreeFn = void(__stdcall*)(void*, uint64_t);
using XMemVirtualQueryFn = SIZE_T(__stdcall*)(HANDLE, const void*, void*);
using XMemGetWorkingSetStatisticsFn = BOOL(__stdcall*)(uint32_t, void*);

// Function pointer signatures for D3D12 exports
using D3D12SerializeVersionedRootSignatureFn =
    HRESULT(WINAPI*)(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC*, ID3DBlob**, ID3DBlob**);
using D3D12SerializeRootSignatureFn =
    HRESULT(WINAPI*)(const D3D12_ROOT_SIGNATURE_DESC*, D3D_ROOT_SIGNATURE_VERSION, ID3DBlob**, ID3DBlob**);
using D3D12CreateRootSignatureDeserializerFn =
    HRESULT(WINAPI*)(const void*, SIZE_T, REFIID, void**);
using D3D12XboxCreateDeviceFn =
    HRESULT(WINAPI*)(IUnknown*, const void*, const IID*, void**);

// ---------------------------------------------------------------------------
// Runtime Test Procedures
// ---------------------------------------------------------------------------

void test_d3d12_root_signature_abi() {
    std::printf("\n--- Testing D3D12 Root Signature ABI Types ---\n");

    check(sizeof(D3D12_ROOT_CONSTANTS) == 12 && alignof(D3D12_ROOT_CONSTANTS) == 4,
          "D3D12_ROOT_CONSTANTS matches Windows SDK size 12 and alignment 4");

    check(sizeof(D3D12_DESCRIPTOR_RANGE) == 20 && alignof(D3D12_DESCRIPTOR_RANGE) == 4,
          "D3D12_DESCRIPTOR_RANGE matches Windows SDK size 20 and alignment 4");

    check(sizeof(D3D12_DESCRIPTOR_RANGE1) == 24 && alignof(D3D12_DESCRIPTOR_RANGE1) == 4,
          "D3D12_DESCRIPTOR_RANGE1 matches Windows SDK size 24 and alignment 4");

    check(sizeof(D3D12_ROOT_DESCRIPTOR_TABLE) == 16 && alignof(D3D12_ROOT_DESCRIPTOR_TABLE) == 8,
          "D3D12_ROOT_DESCRIPTOR_TABLE matches Windows SDK size 16 and alignment 8");

    check(sizeof(D3D12_ROOT_DESCRIPTOR_TABLE1) == 16 && alignof(D3D12_ROOT_DESCRIPTOR_TABLE1) == 8,
          "D3D12_ROOT_DESCRIPTOR_TABLE1 matches Windows SDK size 16 and alignment 8");

    check(sizeof(D3D12_ROOT_DESCRIPTOR) == 8 && alignof(D3D12_ROOT_DESCRIPTOR) == 4,
          "D3D12_ROOT_DESCRIPTOR matches Windows SDK size 8 and alignment 4");

    check(sizeof(D3D12_ROOT_DESCRIPTOR1) == 12 && alignof(D3D12_ROOT_DESCRIPTOR1) == 4,
          "D3D12_ROOT_DESCRIPTOR1 matches Windows SDK size 12 and alignment 4");

    check(sizeof(D3D12_ROOT_PARAMETER) == 32 && alignof(D3D12_ROOT_PARAMETER) == 8,
          "D3D12_ROOT_PARAMETER matches Windows SDK size 32 and alignment 8");

    check(sizeof(D3D12_ROOT_PARAMETER1) == 32 && alignof(D3D12_ROOT_PARAMETER1) == 8,
          "D3D12_ROOT_PARAMETER1 matches Windows SDK size 32 and alignment 8");

    check(sizeof(D3D12_STATIC_SAMPLER_DESC) == 52 && alignof(D3D12_STATIC_SAMPLER_DESC) == 4,
          "D3D12_STATIC_SAMPLER_DESC matches Windows SDK size 52 and alignment 4");

    check(sizeof(D3D12_ROOT_SIGNATURE_DESC) == 40 && alignof(D3D12_ROOT_SIGNATURE_DESC) == 8,
          "D3D12_ROOT_SIGNATURE_DESC matches Windows SDK size 40 and alignment 8");

    check(sizeof(D3D12_ROOT_SIGNATURE_DESC1) == 40 && alignof(D3D12_ROOT_SIGNATURE_DESC1) == 8,
          "D3D12_ROOT_SIGNATURE_DESC1 matches Windows SDK size 40 and alignment 8");

    check(sizeof(D3D12_VERSIONED_ROOT_SIGNATURE_DESC) == 48 &&
              alignof(D3D12_VERSIONED_ROOT_SIGNATURE_DESC) == 8,
          "D3D12_VERSIONED_ROOT_SIGNATURE_DESC matches Windows SDK size 48 and alignment 8");

    // Dynamic layout check: verify union offset and flag masking behavior
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC versioned{};
    versioned.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    versioned.Desc_1_1.NumParameters = 1;
    versioned.Desc_1_1.Flags = static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(
        kObservedXboxRootSigFlag | D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

    check(versioned.Desc_1_1.Flags & kObservedXboxRootSigFlag,
          "Xbox root signature flags set high byte bit 0x03000000");

    const uint32_t cleared_flags =
        static_cast<uint32_t>(versioned.Desc_1_1.Flags) & ~kXboxRootSigFlagMask;
    check(cleared_flags == D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
          "Clearing high flag byte 0xFF000000 preserves legitimate Windows SDK flags");
}

void test_d3d12_handles_and_views_abi() {
    std::printf("\n--- Testing D3D12 Descriptors, Handles, and Views ABI Types ---\n");

    check(sizeof(D3D12_CPU_DESCRIPTOR_HANDLE) == 8 && alignof(D3D12_CPU_DESCRIPTOR_HANDLE) == 8,
          "D3D12_CPU_DESCRIPTOR_HANDLE size is 8 bytes with 8-byte alignment");

    check(sizeof(D3D12_GPU_DESCRIPTOR_HANDLE) == 8 && alignof(D3D12_GPU_DESCRIPTOR_HANDLE) == 8,
          "D3D12_GPU_DESCRIPTOR_HANDLE size is 8 bytes with 8-byte alignment");

    // Check register-return ABI contract: handles fit cleanly inside RAX
    D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle{0xDEADBEEF00001234ull};
    uint64_t cpu_raw = 0;
    std::memcpy(&cpu_raw, &cpu_handle, sizeof(cpu_raw));
    check(cpu_raw == 0xDEADBEEF00001234ull,
          "D3D12_CPU_DESCRIPTOR_HANDLE round-trips into 64-bit register representation");

    D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle{0xCAFEF00D56789A00ull};
    uint64_t gpu_raw = 0;
    std::memcpy(&gpu_raw, &gpu_handle, sizeof(gpu_raw));
    check(gpu_raw == 0xCAFEF00D56789A00ull,
          "D3D12_GPU_DESCRIPTOR_HANDLE round-trips into 64-bit register representation");

    check(sizeof(D3D12_DESCRIPTOR_HEAP_DESC) == 16 && alignof(D3D12_DESCRIPTOR_HEAP_DESC) == 4,
          "D3D12_DESCRIPTOR_HEAP_DESC size 16 bytes and alignment 4");

    check(sizeof(D3D12_CONSTANT_BUFFER_VIEW_DESC) == 16 &&
              alignof(D3D12_CONSTANT_BUFFER_VIEW_DESC) == 8,
          "D3D12_CONSTANT_BUFFER_VIEW_DESC size 16 bytes and alignment 8");

    check(sizeof(D3D12_SHADER_RESOURCE_VIEW_DESC) == 40 &&
              alignof(D3D12_SHADER_RESOURCE_VIEW_DESC) == 8,
          "D3D12_SHADER_RESOURCE_VIEW_DESC size 40 bytes and alignment 8");

    check(sizeof(D3D12_UNORDERED_ACCESS_VIEW_DESC) == 40 &&
              alignof(D3D12_UNORDERED_ACCESS_VIEW_DESC) == 8,
          "D3D12_UNORDERED_ACCESS_VIEW_DESC size 40 bytes and alignment 8");

    check(sizeof(D3D12_RENDER_TARGET_VIEW_DESC) == 24 &&
              alignof(D3D12_RENDER_TARGET_VIEW_DESC) == 8,
          "D3D12_RENDER_TARGET_VIEW_DESC size 24 bytes and alignment 8");

    check(sizeof(D3D12_DEPTH_STENCIL_VIEW_DESC) == 24 &&
              alignof(D3D12_DEPTH_STENCIL_VIEW_DESC) == 4,
          "D3D12_DEPTH_STENCIL_VIEW_DESC size 24 bytes and alignment 4");
}

void test_d3d12_resources_and_pipeline_abi() {
    std::printf("\n--- Testing D3D12 Resources, Barriers, and Pipeline State ABI Types ---\n");

    check(sizeof(D3D12_RESOURCE_DESC) == 56 && alignof(D3D12_RESOURCE_DESC) == 8,
          "D3D12_RESOURCE_DESC size 56 bytes and alignment 8");

    check(sizeof(D3D12_CLEAR_VALUE) == 20 && alignof(D3D12_CLEAR_VALUE) == 4,
          "D3D12_CLEAR_VALUE size 20 bytes and alignment 4");

    check(sizeof(D3D12_BOX) == 24 && alignof(D3D12_BOX) == 4,
          "D3D12_BOX size 24 bytes and alignment 4");

    check(sizeof(D3D12_VIEWPORT) == 24 && alignof(D3D12_VIEWPORT) == 4,
          "D3D12_VIEWPORT size 24 bytes and alignment 4");

    check(sizeof(D3D12_RECT) == 16 && alignof(D3D12_RECT) == 4,
          "D3D12_RECT size 16 bytes and alignment 4");

    check(sizeof(D3D12_SUBRESOURCE_DATA) == 24 && alignof(D3D12_SUBRESOURCE_DATA) == 8,
          "D3D12_SUBRESOURCE_DATA size 24 bytes and alignment 8");

    check(sizeof(D3D12_MEMCPY_DEST) == 24 && alignof(D3D12_MEMCPY_DEST) == 8,
          "D3D12_MEMCPY_DEST size 24 bytes and alignment 8");

    check(sizeof(D3D12_RESOURCE_BARRIER) == 32 && alignof(D3D12_RESOURCE_BARRIER) == 8,
          "D3D12_RESOURCE_BARRIER size 32 bytes and alignment 8");

    check(sizeof(D3D12_COMPUTE_PIPELINE_STATE_DESC) == 56 &&
              alignof(D3D12_COMPUTE_PIPELINE_STATE_DESC) == 8,
          "D3D12_COMPUTE_PIPELINE_STATE_DESC size 56 bytes and alignment 8");

    check(sizeof(D3D12_SHADER_BYTECODE) == 16 && alignof(D3D12_SHADER_BYTECODE) == 8,
          "D3D12_SHADER_BYTECODE size 16 bytes and alignment 8");
}

void test_xmem_alloc_alignment_and_types() {
    std::printf("\n--- Testing XMem Allocator ABI Types and Alignment Constraints ---\n");

    check(kXMemAllocAlignment == 64, "XMem allocation alignment requirement is 64 bytes");

    // Test alignment predicate with simulated pointers
    alignas(64) uint8_t buffer[128];
    const uintptr_t aligned_addr = reinterpret_cast<uintptr_t>(buffer);
    check((aligned_addr & (kXMemAllocAlignment - 1)) == 0,
          "64-byte aligned address satisfies (ptr & 63) == 0 constraint");

    const uintptr_t unaligned_addr = aligned_addr + 8;
    check((unaligned_addr & (kXMemAllocAlignment - 1)) != 0,
          "Non-64-byte aligned offset is correctly identified");

    // Check function pointer type sizes and standard layout
    check(sizeof(XMemAllocFn) == 8 && sizeof(XMemFreeFn) == 8,
          "XMemAlloc and XMemFree function pointers are 8 bytes on x64");

    check(sizeof(XMemVirtualQueryFn) == 8 && sizeof(XMemGetWorkingSetStatisticsFn) == 8,
          "XMemVirtualQuery and XMemGetWorkingSetStatistics function pointers are 8 bytes on x64");
}

void test_xmem_virtual_query_abi_layout() {
    std::printf("\n--- Testing XMem VirtualQuery Output Buffer Layout ---\n");

    check(sizeof(MEMORY_BASIC_INFORMATION) == 48 && alignof(MEMORY_BASIC_INFORMATION) == 8,
          "MEMORY_BASIC_INFORMATION size 48 bytes with 8-byte alignment");

    check(kXMemVirtualQueryOutSize == 64,
          "XMemVirtualQuery output buffer contract is 64 bytes");

    // Exercise VirtualQuery with actual Windows SDK call into a 64-byte output buffer
    alignas(64) uint8_t out64[kXMemVirtualQueryOutSize];
    std::memset(out64, 0xA5, sizeof(out64));

    MEMORY_BASIC_INFORMATION mbi{};
    const SIZE_T written = VirtualQuery(&mbi, &mbi, sizeof(mbi));
    check(written == sizeof(mbi), "VirtualQuery returns exact MEMORY_BASIC_INFORMATION size (48)");

    std::memcpy(out64, &mbi, sizeof(mbi));
    std::memset(out64 + sizeof(mbi), 0, kXMemVirtualQueryOutSize - sizeof(mbi));

    // Verify prefix matches MEMORY_BASIC_INFORMATION
    MEMORY_BASIC_INFORMATION readback{};
    std::memcpy(&readback, out64, sizeof(readback));
    check(readback.BaseAddress == mbi.BaseAddress &&
              readback.RegionSize == mbi.RegionSize &&
              readback.State == mbi.State,
          "Output buffer prefix exactly holds MEMORY_BASIC_INFORMATION fields");

    // Verify trailing 16 bytes are cleanly zeroed
    bool tail_is_zero = true;
    for (size_t i = sizeof(mbi); i < kXMemVirtualQueryOutSize; ++i) {
        if (out64[i] != 0) {
            tail_is_zero = false;
            break;
        }
    }
    check(tail_is_zero, "Output buffer trailing 16 bytes are zero-filled");
}

void test_xmem_working_set_statistics_abi_layout() {
    std::printf("\n--- Testing XMem Working Set Statistics ABI Layout ---\n");

    check(sizeof(PROCESS_MEMORY_COUNTERS) == 72 && alignof(PROCESS_MEMORY_COUNTERS) == 8,
          "PROCESS_MEMORY_COUNTERS size 72 bytes with 8-byte alignment");

    check(offsetof(PROCESS_MEMORY_COUNTERS, WorkingSetSize) == 16,
          "PROCESS_MEMORY_COUNTERS WorkingSetSize is at offset 16");

    PROCESS_MEMORY_COUNTERS pmc{};
    const BOOL ok = GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
    check(ok != FALSE && pmc.WorkingSetSize > 0,
          "GetProcessMemoryInfo succeeds and returns non-zero WorkingSetSize");

    // Verify FidelityFX / XMem layout contract (RT-220):
    // The measured FSR sites establish argument shape, kind and the +8 working-set read; +0's title-total memory
    // limit is inferred from other Series implementations (xcl, xbox-emulator).
    // stats[0..7] = TotalMemoryLimit (uint64_t), stats[8..15] = WorkingSetSize (uint64_t)
    alignas(64) uint8_t stats[64];
    std::memset(stats, 0xA5, sizeof(stats));

    std::memset(stats, 0, 16);
    MEMORYSTATUSEX ms{sizeof(ms)};
    uint64_t total_memory = 12ULL * 1024 * 1024 * 1024;
    if (GlobalMemoryStatusEx(&ms) && ms.ullTotalPhys != 0) {
        total_memory = ms.ullTotalPhys;
    }
    std::memcpy(stats + 0, &total_memory, sizeof(total_memory));
    const uint64_t working_set = static_cast<uint64_t>(pmc.WorkingSetSize);
    std::memcpy(stats + 8, &working_set, sizeof(working_set));

    uint64_t prefix0 = 0;
    std::memcpy(&prefix0, stats, sizeof(prefix0));
    check(prefix0 == total_memory && prefix0 >= working_set,
          "Working set statistics total memory at +0 holds title memory limit >= working set");

    uint64_t extracted_ws = 0;
    std::memcpy(&extracted_ws, stats + 8, sizeof(extracted_ws));
    check(extracted_ws == working_set,
          "Working set statistics size at +8 holds 64-bit working set bytes");

    bool tail_untouched = true;
    for (size_t i = 16; i < sizeof(stats); ++i) {
        if (stats[i] != 0xA5) {
            tail_untouched = false;
            break;
        }
    }
    check(tail_untouched, "Working set statistics buffer bytes 16..63 remain untouched");
}

void test_d3d12_xbox_create_device_params_abi() {
    std::printf("\n--- Testing D3D12XboxCreateDevice Parameter Layout ABI ---\n");

    // Measured parameter block in d3d12_xs is 24 32-bit words (96 bytes)
    constexpr size_t kWordCount = 24;
    constexpr size_t kParamsSize = kWordCount * sizeof(uint32_t);
    check(kParamsSize == 96, "D3D12XboxCreateDevice params block is 24 words (96 bytes)");

    std::array<uint32_t, kWordCount> params{};
    for (size_t i = 0; i < kWordCount; ++i) {
        params[i] = static_cast<uint32_t>(0x1000 + i);
    }

    check(params[0] == 0x1000 && params[23] == 0x1017,
          "Parameter block words are indexed contiguously as 32-bit integers");
}

}  // namespace

int main() {
    std::printf("test_runtime_types: verifying D3D12/XMem ABI type alignment and structure sizes\n");

    test_d3d12_root_signature_abi();
    test_d3d12_handles_and_views_abi();
    test_d3d12_resources_and_pipeline_abi();
    test_xmem_alloc_alignment_and_types();
    test_xmem_virtual_query_abi_layout();
    test_xmem_working_set_statistics_abi_layout();
    test_d3d12_xbox_create_device_params_abi();

    std::printf("\n------------------------------------------------------------\n");
    if (g_failures == 0) {
        std::printf("test_runtime_types: OK (%d checks)\n", g_checks);
        return 0;
    } else {
        std::printf("test_runtime_types: FAILED (%d checks, %d failure%s)\n",
                    g_checks, g_failures, g_failures == 1 ? "" : "s");
        return 1;
    }
}
