// Xbox XDXR build descriptions as XboxBvhBuilder.dll (package 1.0.2.1) reads them. docs/xdxr.md gives the evidence
// for every field. Names come from the DLL's exported symbols or the public D3D12 structs they extend; the stream
// record fields have no names in the DLL, so they are named by offset. Plain types only: this builds without the
// Windows SDK, and xdxr_descs_test checks it against d3d12.h where that is available.
#pragma once

#include <cstddef>
#include <cstdint>

namespace leonida::xdxr {

// D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS (public layout).
struct BuildInputs {
    uint32_t type;           // D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE: 0 top level, 1 bottom level
    uint32_t flags;          // D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS
    uint32_t num_descs;
    uint32_t descs_layout;   // D3D12_ELEMENTS_LAYOUT
    uint64_t descs;          // InstanceDescs (GPU VA) or pGeometryDescs / ppGeometryDescs
};

// D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC (public layout).
struct PostbuildInfoDesc {
    uint64_t dest_buffer;
    uint32_t info_type;
    uint32_t padding;
};

// D3D12XBOX_XDXR_STREAM_DESC: SizeInBytes of 16-byte StreamRecords at pStream.
struct StreamDesc {
    uint64_t size_in_bytes;
    const void* stream;
};

// One record of the stream; its type decides how the 12 bytes after it are read.
struct StreamRecord {
    uint32_t type;
    uint8_t payload[12];
};

struct StreamRecord0 {       // type 0
    uint32_t type;
    float value_04;          // default -1.0
    float value_08;          // negative: default_record0_value_08(flags)
    uint8_t bytes_0c[4];     // default 0
};

struct StreamRecord1 {       // type 1
    uint32_t type;
    uint8_t unread_04[9];
    uint8_t level_0d;        // bottom level only; 0 keeps the default, otherwise min(level, 15)
    uint8_t flag_0e;         // top level only
    uint8_t unread_0f;
};

struct StreamRecord2 {       // type 2
    uint32_t type;
    uint32_t unread_04;
    float count_08;          // truncated to an integer; -1 means the default (0.2)
    float value_0c;          // 0.0 means the default (0.75)
};

// D3D12XBOX_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_BATCHED_DESC2. The first 0x30 bytes are the public
// D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC.
struct BatchedDesc2 {
    uint64_t dest_acceleration_structure_data;   // 256-byte aligned
    BuildInputs inputs;
    uint64_t source_acceleration_structure_data;
    uint64_t scratch_acceleration_structure_data;
    uint32_t num_postbuild_info_descs;
    uint32_t padding;                            // the single-build wrapper writes 0; never read
    const PostbuildInfoDesc* postbuild_info_descs;
    StreamDesc stream;
};

static_assert(sizeof(BuildInputs) == 0x18);
static_assert(sizeof(PostbuildInfoDesc) == 0x10 && offsetof(PostbuildInfoDesc, info_type) == 0x08);
static_assert(sizeof(StreamDesc) == 0x10 && offsetof(StreamDesc, stream) == 0x08);
static_assert(sizeof(StreamRecord) == 0x10 && sizeof(StreamRecord0) == 0x10 && sizeof(StreamRecord1) == 0x10 &&
              sizeof(StreamRecord2) == 0x10);
static_assert(offsetof(StreamRecord0, value_04) == 0x04 && offsetof(StreamRecord0, value_08) == 0x08 &&
              offsetof(StreamRecord0, bytes_0c) == 0x0c);
static_assert(offsetof(StreamRecord1, level_0d) == 0x0d && offsetof(StreamRecord1, flag_0e) == 0x0e);
static_assert(offsetof(StreamRecord2, count_08) == 0x08 && offsetof(StreamRecord2, value_0c) == 0x0c);
static_assert(sizeof(BatchedDesc2) == 0x50);
static_assert(offsetof(BatchedDesc2, inputs) == 0x08);
static_assert(offsetof(BatchedDesc2, source_acceleration_structure_data) == 0x20);
static_assert(offsetof(BatchedDesc2, scratch_acceleration_structure_data) == 0x28);
static_assert(offsetof(BatchedDesc2, num_postbuild_info_descs) == 0x30);
static_assert(offsetof(BatchedDesc2, postbuild_info_descs) == 0x38);
static_assert(offsetof(BatchedDesc2, stream) == 0x40);

constexpr uint32_t kBuildFlagPreferFastTrace = 0x04;
constexpr uint32_t kBuildFlagMinimizeMemory = 0x10;

// D3D12XBOX_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_OFFLINE_STATISTICS: the DLL rejects it for every build
// it does (offline builds only).
constexpr uint32_t kPostbuildInfoOfflineStatistics = 0x10;

// The record a setting is read from, the way the DLL finds it: the first record of `type` from the start, stepping
// over types 0, 1 and 2, and stopping at any other type. Null when there is none.
inline const StreamRecord* find_stream_record(const StreamDesc& desc, uint32_t type) {
    if (desc.size_in_bytes == 0 || desc.stream == nullptr)
        return nullptr;
    const auto* record = static_cast<const StreamRecord*>(desc.stream);
    const auto* end = reinterpret_cast<const StreamRecord*>(static_cast<const uint8_t*>(desc.stream) +
                                                            desc.size_in_bytes);
    for (; record < end; ++record) {
        if (record->type == type)
            return record;
        if (record->type > 2)
            return nullptr;
    }
    return nullptr;
}

// Type 0's value at 0x08 when it is negative or absent, from Inputs.Flags.
inline float default_record0_value_08(uint32_t flags) {
    const bool fast_trace = (flags & kBuildFlagPreferFastTrace) != 0;
    const bool minimize_memory = (flags & kBuildFlagMinimizeMemory) != 0;
    if (fast_trace)
        return minimize_memory ? 1.005f : 0.0f;
    return minimize_memory ? 1.01f : 1.005f;
}

}  // namespace leonida::xdxr
