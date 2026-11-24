// src/xdxr_descs.h: the XDXR build descriptions measured in XboxBvhBuilder.dll (docs/xdxr.md). The layouts are
// static_asserts in the header; this checks the stream lookup and the flag defaults against the DLL's measured
// behaviour, and, where the Windows SDK is available, that the public part matches d3d12.h. No GPU, no game files.
#include "../src/xdxr_descs.h"

#include <cstdio>
#include <cstring>

#if defined(_WIN32) && __has_include(<d3d12.h>)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <d3d12.h>
// Older d3d12.h headers (MinGW's, for one) have no DXR types.
#ifdef D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT
#define XDXR_HAVE_D3D12 1
#endif
#endif

#define CHECK(condition) do {                                    \
    if (!(condition)) {                                          \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                \
    }                                                            \
} while (0)

using namespace leonida::xdxr;

namespace {

StreamDesc stream_of(const StreamRecord* records, size_t count) {
    return StreamDesc{count * sizeof(StreamRecord), records};
}

int stream_lookup() {
    StreamRecord records[4] = {};

    // No stream, a zero size or a null pointer: every setting keeps its default.
    CHECK(find_stream_record(StreamDesc{}, 0) == nullptr);
    CHECK(find_stream_record(StreamDesc{0, records}, 0) == nullptr);
    CHECK(find_stream_record(StreamDesc{sizeof(records), nullptr}, 0) == nullptr);

    // Types 0, 1 and 2 are stepped over until the wanted one.
    records[0].type = 2;
    records[1].type = 0;
    records[2].type = 1;
    const StreamDesc three = stream_of(records, 3);
    CHECK(find_stream_record(three, 0) == &records[1]);
    CHECK(find_stream_record(three, 1) == &records[2]);
    CHECK(find_stream_record(three, 2) == &records[0]);

    // The first record of a type wins.
    records[3].type = 0;
    CHECK(find_stream_record(stream_of(records, 4), 0) == &records[1]);

    // Any other type ends the walk, so records after it are not seen.
    records[0].type = 3;
    CHECK(find_stream_record(three, 0) == nullptr);
    CHECK(find_stream_record(three, 2) == nullptr);
    // ...but a match before it is still found.
    records[0].type = 0;
    records[1].type = 7;
    CHECK(find_stream_record(three, 0) == &records[0]);
    CHECK(find_stream_record(three, 1) == nullptr);

    // Nothing past SizeInBytes is read.
    records[0].type = 1;
    records[1].type = 0;
    CHECK(find_stream_record(stream_of(records, 1), 0) == nullptr);
    CHECK(find_stream_record(stream_of(records, 2), 0) == &records[1]);
    return 0;
}

int flag_defaults() {
    CHECK(default_record0_value_08(0) == 1.005f);
    CHECK(default_record0_value_08(kBuildFlagPreferFastTrace) == 0.0f);
    CHECK(default_record0_value_08(kBuildFlagMinimizeMemory) == 1.01f);
    CHECK(default_record0_value_08(kBuildFlagPreferFastTrace | kBuildFlagMinimizeMemory) == 1.005f);
    // Other bits (ALLOW_UPDATE, ALLOW_COMPACTION, PREFER_FAST_BUILD, PERFORM_UPDATE) don't change it.
    CHECK(default_record0_value_08(0x01 | 0x02 | 0x08 | 0x20) == 1.005f);
    return 0;
}

int record_views() {
    // The typed views alias the generic record: the type word is shared and the payload starts at 4.
    StreamRecord record = {};
    record.type = 1;
    record.payload[0x0d - 4] = 9;
    record.payload[0x0e - 4] = 1;
    StreamRecord1 one;
    std::memcpy(&one, &record, sizeof(one));
    CHECK(one.type == 1 && one.level_0d == 9 && one.flag_0e == 1);
    return 0;
}

#ifdef XDXR_HAVE_D3D12
// The public part of BATCHED_DESC2 is the public build desc, as the DLL's standard-build export copies it.
static_assert(sizeof(D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS) == sizeof(BuildInputs));
static_assert(offsetof(D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS, Flags) == offsetof(BuildInputs, flags));
static_assert(offsetof(D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS, NumDescs) ==
              offsetof(BuildInputs, num_descs));
static_assert(offsetof(D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS, DescsLayout) ==
              offsetof(BuildInputs, descs_layout));
static_assert(offsetof(D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS, InstanceDescs) ==
              offsetof(BuildInputs, descs));
static_assert(sizeof(D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC) == offsetof(BatchedDesc2, num_postbuild_info_descs));
static_assert(offsetof(D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC, Inputs) == offsetof(BatchedDesc2, inputs));
static_assert(offsetof(D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC, SourceAccelerationStructureData) ==
              offsetof(BatchedDesc2, source_acceleration_structure_data));
static_assert(offsetof(D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC, ScratchAccelerationStructureData) ==
              offsetof(BatchedDesc2, scratch_acceleration_structure_data));
static_assert(sizeof(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC) == sizeof(PostbuildInfoDesc));
static_assert(offsetof(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC, InfoType) ==
              offsetof(PostbuildInfoDesc, info_type));
// The alignment the DLL enforces on DestAccelerationStructureData.
static_assert(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT == 256);
static_assert(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE == kBuildFlagPreferFastTrace);
static_assert(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_MINIMIZE_MEMORY == kBuildFlagMinimizeMemory);

int public_desc_copy() {
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC desc = {};
    desc.DestAccelerationStructureData = 0x100000;
    desc.Inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    desc.Inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_MINIMIZE_MEMORY;
    desc.Inputs.NumDescs = 3;
    desc.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS;
    desc.Inputs.InstanceDescs = 0x2000;
    desc.SourceAccelerationStructureData = 0x300000;
    desc.ScratchAccelerationStructureData = 0x400000;
    BatchedDesc2 batched = {};
    std::memcpy(&batched, &desc, sizeof(desc));
    CHECK(batched.dest_acceleration_structure_data == 0x100000);
    CHECK(batched.inputs.type == 1 && batched.inputs.flags == kBuildFlagMinimizeMemory);
    CHECK(batched.inputs.num_descs == 3 && batched.inputs.descs_layout == 1 && batched.inputs.descs == 0x2000);
    CHECK(batched.source_acceleration_structure_data == 0x300000);
    CHECK(batched.scratch_acceleration_structure_data == 0x400000);
    CHECK(batched.stream.size_in_bytes == 0 && batched.stream.stream == nullptr);
    return 0;
}
#endif

}  // namespace

int main() {
    if (stream_lookup() || flag_defaults() || record_views())
        return 1;
#ifdef XDXR_HAVE_D3D12
    if (public_desc_copy())
        return 1;
    std::printf("xdxr_descs: ok (checked against d3d12.h)\n");
#else
    std::printf("xdxr_descs: ok (no DXR d3d12.h: public layout not compared)\n");
#endif
    return 0;
}
