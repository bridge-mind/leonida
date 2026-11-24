// BuildRaytracingAccelerationStructure on the standalone Vulkan device writes XDXR BVH4 records (docs/d3d12_vulkan.md
// 4.8, src/xdxr_build.h), the layout the game's translated traversal shaders read. A BLAS of two geometries (an
// unindexed triangle soup, and an R16-indexed grid with a 16-byte stride and a 3x4 transform) and a TLAS of three
// instances are built through the D3D12 API, sized by GetRaytracingAccelerationStructurePrebuildInfo, copied to
// readback buffers and written to <out dir>; the BLAS is then updated in place (PERFORM_UPDATE) after its soup moves,
// and written again as d3d12_blas_update; then its postbuild COMPACTED_SIZE and a TLAS rebuild's CURRENT_SIZE are
// checked against the prebuild sizes, and the BLAS is compacted (CopyRaytracingAccelerationStructure COMPACT) into a
// buffer of exactly that size, written as d3d12_blas_compacted as <case>.bin + <case>.json at their console GPU virtual addresses.
// Last, the BLAS again with MINIMIZE_MEMORY (BOX16 nodes: a smaller prebuild size and postbuild size), written as
// d3d12_blas_box16, then updated in place with the same flags (d3d12_blas_box16_update); and once more without
// ALLOW_UPDATE, so that its triangles are paired into TRI0/TRI1 leaves (d3d12_blas_paired; the ALLOW_UPDATE builds
// must have no pairs, since a refit rewrites the leaves one triangle each).
// `python -I tools/xdxr_bvh.py check-build <out dir>` validates them and traces rays against brute force (ctest:
// d3d12_xdxr_rt + d3d12_xdxr_rt_check). Exit 77 without a Vulkan device, or with LEONIDA_RT_AS=khr. No game files.
#include "../shims/d3d12_com.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

template<class T> struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
    T** put() { return &p; }
};

int fail(const char* what, HRESULT hr = S_OK) {
    std::fprintf(stderr, "FAIL: %s (hr 0x%08lx)\n", what, static_cast<unsigned long>(hr));
    return 1;
}

HRESULT buffer(ID3D12Device* device, D3D12_HEAP_TYPE heap, UINT64 size, D3D12_RESOURCE_STATES state,
               D3D12_RESOURCE_FLAGS flags, ID3D12Resource** out) {
    D3D12_HEAP_PROPERTIES props{};
    props.Type = heap;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size ? size : 1;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    return device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(out));
}

std::string num(double x) {
    char s[64];
    std::snprintf(s, sizeof s, "%.9g", x);
    return s;
}

bool write_file(const std::string& path, const void* data, std::size_t bytes) {
    std::ofstream f(path, std::ios::binary);
    f.write(static_cast<const char*>(data), std::streamsize(bytes));
    return bool(f);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: d3d12_xdxr_rt_test <out dir>\n");
        return 2;
    }
    const std::string dir = argv[1];
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        const auto ext = entry.path().extension();
        if (ext == ".bin" || ext == ".json") std::filesystem::remove(entry.path(), ec);
    }

    // LEONIDA_RT_AS=khr builds #743's VkAccelerationStructureKHR instead, whose bytes are the driver's own.
    char rt_as[8] = {};
    std::size_t rt_as_length = 0;
    if (getenv_s(&rt_as_length, rt_as, "LEONIDA_RT_AS") == 0 && rt_as_length > 1 && std::strcmp(rt_as, "khr") == 0) {
        std::puts("SKIP: LEONIDA_RT_AS=khr");
        return 77;
    }
    Ref<ID3D12Device5> device;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(device.put())))) {
        std::puts("SKIP: no Vulkan D3D12 device");
        return 77;
    }

    // ---- geometry: a triangle soup, and an R16-indexed grid (stride 16) with a transform ----
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> spread(-4.0f, 4.0f), jitter(-0.6f, 0.6f);
    const UINT soup_tris = 300;
    std::vector<float> soup;
    for (UINT t = 0; t < soup_tris; ++t) {
        const float c[3] = {spread(rng), spread(rng), spread(rng)};
        for (int k = 0; k < 9; ++k) soup.push_back(c[k % 3] + jitter(rng));
    }
    const UINT n = 8;
    std::vector<float> grid;
    for (UINT j = 0; j <= n; ++j)
        for (UINT i = 0; i <= n; ++i) {
            grid.push_back(float(i) - n * 0.5f);
            grid.push_back(float(j) - n * 0.5f);
            grid.push_back(0.25f * float((i * 7 + j * 3) % 5));
            grid.push_back(9.0f);                                    // padding: the stride is 16
        }
    std::vector<std::uint16_t> indices;
    for (UINT j = 0; j < n; ++j)
        for (UINT i = 0; i < n; ++i) {
            const std::uint16_t a = std::uint16_t(j * (n + 1) + i);
            const std::uint16_t q[6] = {a, std::uint16_t(a + 1), std::uint16_t(a + n + 2), a, std::uint16_t(a + n + 2),
                                        std::uint16_t(a + n + 1)};
            indices.insert(indices.end(), q, q + 6);
        }
    const float transform[12] = {0, 2, 0, 1, 0, 0, 1, -2, 2, 0, 0, 3};   // exact in float

    const UINT64 soup_bytes = soup.size() * 4, grid_bytes = grid.size() * 4, index_bytes = indices.size() * 2;
    const UINT64 upload_size = 256 + soup_bytes + grid_bytes + index_bytes + 64 + 3 * 64;
    Ref<ID3D12Resource> upload;
    HRESULT hr = buffer(device.p, D3D12_HEAP_TYPE_UPLOAD, upload_size, D3D12_RESOURCE_STATE_GENERIC_READ,
                        D3D12_RESOURCE_FLAG_NONE, upload.put());
    if (FAILED(hr)) return fail("upload buffer", hr);
    char* mapped = nullptr;
    if (FAILED(upload->Map(0, nullptr, reinterpret_cast<void**>(&mapped)))) return fail("Map upload");
    UINT64 at = 0;
    auto place = [&](const void* data, UINT64 bytes, UINT64 align) {
        at = (at + align - 1) / align * align;
        std::memcpy(mapped + at, data, std::size_t(bytes));
        const UINT64 va = upload->GetGPUVirtualAddress() + at;
        at += bytes;
        return va;
    };
    const UINT64 soup_va = place(soup.data(), soup_bytes, 16);
    const UINT64 grid_va = place(grid.data(), grid_bytes, 16);
    const UINT64 index_va = place(indices.data(), index_bytes, 16);
    const UINT64 transform_va = place(transform, sizeof transform, 16);

    D3D12_RAYTRACING_GEOMETRY_DESC geometries[2]{};
    geometries[0].Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
    geometries[0].Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
    geometries[0].Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
    geometries[0].Triangles.VertexCount = soup_tris * 3;
    geometries[0].Triangles.VertexBuffer = {soup_va, 12};
    geometries[1].Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
    geometries[1].Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
    geometries[1].Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
    geometries[1].Triangles.VertexCount = (n + 1) * (n + 1);
    geometries[1].Triangles.VertexBuffer = {grid_va, 16};
    geometries[1].Triangles.IndexFormat = DXGI_FORMAT_R16_UINT;
    geometries[1].Triangles.IndexCount = UINT(indices.size());
    geometries[1].Triangles.IndexBuffer = index_va;
    geometries[1].Triangles.Transform3x4 = transform_va;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS blas_inputs{};
    blas_inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    blas_inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;   // updated below
    blas_inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    blas_inputs.NumDescs = 2;
    blas_inputs.pGeometryDescs = geometries;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO blas_info{};
    device->GetRaytracingAccelerationStructurePrebuildInfo(&blas_inputs, &blas_info);
    if (!blas_info.ResultDataMaxSizeInBytes) {
        std::puts("SKIP: the device does not build XDXR acceleration structures (no bufferDeviceAddress)");
        return 77;
    }

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlas_inputs{};
    tlas_inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    tlas_inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    tlas_inputs.NumDescs = 3;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO tlas_info{};
    device->GetRaytracingAccelerationStructurePrebuildInfo(&tlas_inputs, &tlas_info);
    if (!tlas_info.ResultDataMaxSizeInBytes) return fail("TLAS prebuild info is empty");

    const D3D12_RESOURCE_FLAGS uav = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    const D3D12_RESOURCE_STATES as_state = D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
    Ref<ID3D12Resource> blas, tlas, scratch, blas_rb, tlas_rb;
    const UINT64 scratch_size = blas_info.ScratchDataSizeInBytes > tlas_info.ScratchDataSizeInBytes
                                    ? blas_info.ScratchDataSizeInBytes : tlas_info.ScratchDataSizeInBytes;
    if (FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, blas_info.ResultDataMaxSizeInBytes, as_state, uav, blas.put())) ||
        FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, tlas_info.ResultDataMaxSizeInBytes, as_state, uav, tlas.put())) ||
        FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, scratch_size, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, uav,
                           scratch.put())) ||
        FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_READBACK, blas_info.ResultDataMaxSizeInBytes,
                           D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, blas_rb.put())) ||
        FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_READBACK, tlas_info.ResultDataMaxSizeInBytes,
                           D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, tlas_rb.put())))
        return fail("acceleration-structure buffers", hr);
    if (blas->GetGPUVirtualAddress() % 256 || tlas->GetGPUVirtualAddress() % 256)
        return fail("acceleration structures must be 256-byte aligned");

    // ---- instances: identity, rotated + translated, and masked out ----
    const float c = std::cos(0.7f), s = std::sin(0.7f);
    const float xforms[3][12] = {{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0},
                                 {c, -s, 0, 9, s, c, 0, -1, 0, 0, 1, 0.5f},
                                 {2, 0, 0, -9, 0, 0.5f, 0, 2, 0, 0, 1.5f, 1}};
    const UINT masks[3] = {0xff, 0x01, 0x00};
    D3D12_RAYTRACING_INSTANCE_DESC instances[3]{};
    for (int i = 0; i < 3; ++i) {
        std::memcpy(instances[i].Transform, xforms[i], sizeof xforms[i]);
        instances[i].InstanceID = 20 + i;
        instances[i].InstanceMask = masks[i];
        instances[i].AccelerationStructure = blas->GetGPUVirtualAddress();
    }
    const UINT64 instances_va = place(instances, sizeof instances, 16);
    upload->Unmap(0, nullptr);
    tlas_inputs.InstanceDescs = instances_va;

    // ---- record: BLAS, barrier, TLAS, barrier, copies to readback ----
    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList4> list;
    Ref<ID3D12Fence> fence;
    if (FAILED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(queue.put()))) ||
        FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(allocator.put()))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                         IID_PPV_ARGS(list.put()))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.put()))))
        return fail("queue, allocator, list or fence");
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build{};
    build.Inputs = blas_inputs;
    build.DestAccelerationStructureData = blas->GetGPUVirtualAddress();
    build.ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress();
    list->BuildRaytracingAccelerationStructure(&build, 0, nullptr);
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = nullptr;                         // every UAV: the BLAS and the scratch
    list->ResourceBarrier(1, &barrier);
    build.Inputs = tlas_inputs;
    build.DestAccelerationStructureData = tlas->GetGPUVirtualAddress();
    list->BuildRaytracingAccelerationStructure(&build, 0, nullptr);
    list->ResourceBarrier(1, &barrier);
    // The layer keeps buffers in one memory, so a copy out of an acceleration structure reads its bytes as they are
    // (D3D12 itself only allows CopyRaytracingAccelerationStructure; this is a test of the layer's records).
    list->CopyBufferRegion(blas_rb.p, 0, blas.p, 0, blas_info.ResultDataMaxSizeInBytes);
    list->CopyBufferRegion(tlas_rb.p, 0, tlas.p, 0, tlas_info.ResultDataMaxSizeInBytes);
    if (FAILED(hr = list->Close())) return fail("Close", hr);
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    if (FAILED(queue->Signal(fence.p, 1))) return fail("Signal");
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);   // reused for the update below
    const bool done = event && SUCCEEDED(fence->SetEventOnCompletion(1, event)) &&
                      WaitForSingleObject(event, 10000) == WAIT_OBJECT_0;
    if (event) CloseHandle(event);
    if (!done) return fail("the builds did not complete");
    if (FAILED(device->GetDeviceRemovedReason())) return fail("device removed", device->GetDeviceRemovedReason());

    // ---- dump ----
    auto dump = [&](ID3D12Resource* rb, UINT64 size, const std::string& name, const std::string& json) {
        void* p = nullptr;
        if (FAILED(rb->Map(0, nullptr, &p))) return false;
        std::uint32_t units = 0;
        std::memcpy(&units, static_cast<char*>(p) + 4, 4);    // header totalSizeIn64BNodes
        const UINT64 used = (UINT64(units) + 1) * 64;   // the field does not count the 64-byte header
        const bool ok = used && used <= size && write_file(dir + "/" + name + ".bin", p, std::size_t(used)) &&
                        write_file(dir + "/" + name + ".json", json.data(), json.size());
        rb->Unmap(0, nullptr);
        return ok;
    };
    // The BLAS's triangles as the reference sees them, for the soup's current positions.
    auto triangles_json = [&](const std::vector<float>& soup_now) {
        std::string tris = "[";
        bool first = true;
        auto add_tri = [&](const float v[3][3], int geometry, UINT primitive) {
            tris += first ? "\n  [" : ",\n  [";
            first = false;
            for (int k = 0; k < 3; ++k)
                tris += (k ? ", [" : "[") + num(v[k][0]) + ", " + num(v[k][1]) + ", " + num(v[k][2]) + "]";
            tris += ", " + std::to_string(geometry) + ", " + std::to_string(primitive) + "]";
        };
        for (UINT t = 0; t < soup_tris; ++t) {
            float v[3][3];
            for (int k = 0; k < 9; ++k) v[k / 3][k % 3] = soup_now[9 * t + k];
            add_tri(v, 0, t);
        }
        for (UINT t = 0; t < indices.size() / 3; ++t) {
            float v[3][3];
            for (int k = 0; k < 3; ++k) {
                const float* p = &grid[4 * indices[3 * t + k]];
                for (int r = 0; r < 3; ++r)
                    v[k][r] = transform[4 * r] * p[0] + transform[4 * r + 1] * p[1] + transform[4 * r + 2] * p[2] +
                              transform[4 * r + 3];
            }
            add_tri(v, 1, t);
        }
        return tris + "]";
    };
    const std::string tris = triangles_json(soup);
    // ALLOW_UPDATE: never paired, since a refit rewrites the leaves one triangle each ("pairs": false)
    if (!dump(blas_rb.p, blas_info.ResultDataMaxSizeInBytes, "d3d12_blas",
              "{\"kind\": \"blas\", \"pairs\": false, \"box16\": false, \"address\": " + std::to_string(blas->GetGPUVirtualAddress()) +
                  ", \"triangles\": " + tris + "}\n"))
        return fail("BLAS readback");
    std::string inst = "[";
    for (int i = 0; i < 3; ++i) {
        inst += i ? ", {\"transform\": [" : "{\"transform\": [";
        for (int k = 0; k < 12; ++k) inst += (k ? ", " : "") + num(xforms[i][k]);
        inst += "], \"mask\": " + std::to_string(masks[i]) + ", \"blas\": 0}";
    }
    inst += "]";
    if (!dump(tlas_rb.p, tlas_info.ResultDataMaxSizeInBytes, "d3d12_tlas",
              "{\"kind\": \"tlas\", \"address\": " + std::to_string(tlas->GetGPUVirtualAddress()) +
                  ", \"bias\": 0, \"blas\": [\"d3d12_blas\"], \"instances\": " + inst + "}\n"))
        return fail("TLAS readback");

    // ---- PERFORM_UPDATE in place: the soup moves (every vertex), the grid stays; the BLAS is refitted ----
    std::vector<float> moved = soup;
    std::uniform_real_distribution<float> nudge(-1.0f, 1.0f);
    for (std::size_t v = 0; v < moved.size(); v += 3) {
        moved[v] = moved[v] * 1.3f + nudge(rng);
        moved[v + 1] = moved[v + 1] * 0.7f + nudge(rng);
        moved[v + 2] += nudge(rng);
    }
    if (FAILED(upload->Map(0, nullptr, reinterpret_cast<void**>(&mapped)))) return fail("Map upload for the update");
    std::memcpy(mapped + (soup_va - upload->GetGPUVirtualAddress()), moved.data(), std::size_t(soup_bytes));
    upload->Unmap(0, nullptr);
    if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.p, nullptr))) return fail("Reset for the update");
    build.Inputs = blas_inputs;
    build.Inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
    build.DestAccelerationStructureData = blas->GetGPUVirtualAddress();
    build.SourceAccelerationStructureData = blas->GetGPUVirtualAddress();
    list->BuildRaytracingAccelerationStructure(&build, 0, nullptr);
    list->ResourceBarrier(1, &barrier);
    list->CopyBufferRegion(blas_rb.p, 0, blas.p, 0, blas_info.ResultDataMaxSizeInBytes);
    if (FAILED(hr = list->Close())) return fail("Close (update)", hr);
    queue->ExecuteCommandLists(1, lists);
    if (FAILED(queue->Signal(fence.p, 2))) return fail("Signal (update)");
    event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    const bool updated = event && SUCCEEDED(fence->SetEventOnCompletion(2, event)) &&
                         WaitForSingleObject(event, 10000) == WAIT_OBJECT_0;
    if (event) CloseHandle(event);
    if (!updated) return fail("the update did not complete");
    if (!dump(blas_rb.p, blas_info.ResultDataMaxSizeInBytes, "d3d12_blas_update",
              "{\"kind\": \"blas\", \"pairs\": false, \"address\": " + std::to_string(blas->GetGPUVirtualAddress()) +
                  ", \"triangles\": " + triangles_json(moved) + "}\n"))
        return fail("BLAS readback after the update");

    // ---- postbuild info and compaction: EmitRaytracingAccelerationStructurePostbuildInfo for both structures, a
    // TLAS rebuild with a CURRENT_SIZE postbuild desc, then COMPACT the BLAS into a buffer of exactly its size ----
    auto submit = [&](UINT64 value) {
        if (FAILED(list->Close())) return false;
        queue->ExecuteCommandLists(1, lists);
        if (FAILED(queue->Signal(fence.p, value))) return false;
        HANDLE e = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        const bool ok = e && SUCCEEDED(fence->SetEventOnCompletion(value, e)) &&
                        WaitForSingleObject(e, 10000) == WAIT_OBJECT_0;
        if (e) CloseHandle(e);
        return ok;
    };
    Ref<ID3D12Resource> info, info_rb;
    if (FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, 24, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, uav, info.put())) ||
        FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_READBACK, 24, D3D12_RESOURCE_STATE_COPY_DEST,
                           D3D12_RESOURCE_FLAG_NONE, info_rb.put())))
        return fail("postbuild buffers", hr);
    if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.p, nullptr))) return fail("Reset for postbuild");
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC compacted_desc{
        info->GetGPUVirtualAddress(), D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE};
    const D3D12_GPU_VIRTUAL_ADDRESS both[2] = {blas->GetGPUVirtualAddress(), tlas->GetGPUVirtualAddress()};
    list->EmitRaytracingAccelerationStructurePostbuildInfo(&compacted_desc, 2, both);
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC current_desc{
        info->GetGPUVirtualAddress() + 16, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_CURRENT_SIZE};
    build = {};
    build.Inputs = tlas_inputs;
    build.DestAccelerationStructureData = tlas->GetGPUVirtualAddress();
    build.ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress();
    list->BuildRaytracingAccelerationStructure(&build, 1, &current_desc);
    list->ResourceBarrier(1, &barrier);
    list->CopyBufferRegion(info_rb.p, 0, info.p, 0, 24);
    if (!submit(3)) return fail("postbuild submission");
    UINT64 sizes[3] = {};
    {
        void* p = nullptr;
        if (FAILED(info_rb->Map(0, nullptr, &p))) return fail("Map postbuild");
        std::memcpy(sizes, p, sizeof sizes);
        info_rb->Unmap(0, nullptr);
    }
    // A build's size is the nodes it wrote, within the prebuild size (room for the most a tree of that many
    // elements can have), and it is already as small as it gets: COMPACTED_SIZE and CURRENT_SIZE are that size. The
    // same TLAS built twice has the same tree.
    auto fits = [](UINT64 size, UINT64 max) { return size && size % 64 == 0 && size <= max; };
    if (!fits(sizes[0], blas_info.ResultDataMaxSizeInBytes) || !fits(sizes[1], tlas_info.ResultDataMaxSizeInBytes) ||
        sizes[2] != sizes[1]) {
        std::fprintf(stderr, "postbuild sizes %llu %llu %llu, prebuild %llu %llu\n",
                     static_cast<unsigned long long>(sizes[0]), static_cast<unsigned long long>(sizes[1]),
                     static_cast<unsigned long long>(sizes[2]),
                     static_cast<unsigned long long>(blas_info.ResultDataMaxSizeInBytes),
                     static_cast<unsigned long long>(tlas_info.ResultDataMaxSizeInBytes));
        return fail("postbuild sizes");
    }
    Ref<ID3D12Resource> compacted, compacted_rb;
    if (FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, sizes[0], as_state, uav, compacted.put())) ||
        FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_READBACK, sizes[0], D3D12_RESOURCE_STATE_COPY_DEST,
                           D3D12_RESOURCE_FLAG_NONE, compacted_rb.put())))
        return fail("compacted buffers", hr);
    if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.p, nullptr))) return fail("Reset for compaction");
    list->CopyRaytracingAccelerationStructure(compacted->GetGPUVirtualAddress(), blas->GetGPUVirtualAddress(),
                                              D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT);
    list->ResourceBarrier(1, &barrier);
    list->CopyBufferRegion(compacted_rb.p, 0, compacted.p, 0, sizes[0]);
    if (!submit(4)) return fail("compaction submission");
    if (!dump(compacted_rb.p, sizes[0], "d3d12_blas_compacted",
              "{\"kind\": \"blas\", \"address\": " + std::to_string(compacted->GetGPUVirtualAddress()) +
                  ", \"triangles\": " + triangles_json(moved) + "}\n"))
        return fail("compacted BLAS readback");

    // ---- MINIMIZE_MEMORY: the same BLAS with BOX16 nodes (half boxes, 64 bytes a node instead of 128). Prebuild
    // info must already count the smaller nodes, and an update with the same flags must refit them as BOX16 ----
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS box16_inputs = blas_inputs;
    box16_inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_MINIMIZE_MEMORY;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO box16_info{};
    device->GetRaytracingAccelerationStructurePrebuildInfo(&box16_inputs, &box16_info);
    if (!box16_info.ResultDataMaxSizeInBytes || box16_info.ResultDataMaxSizeInBytes >= blas_info.ResultDataMaxSizeInBytes)
        return fail("MINIMIZE_MEMORY prebuild size is not below the default one");
    Ref<ID3D12Resource> box16_blas, box16_rb;
    if (FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, box16_info.ResultDataMaxSizeInBytes, as_state, uav,
                           box16_blas.put())) ||
        FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_READBACK, box16_info.ResultDataMaxSizeInBytes,
                           D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, box16_rb.put())))
        return fail("MINIMIZE_MEMORY buffers", hr);
    if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.p, nullptr))) return fail("Reset for MINIMIZE_MEMORY");
    build = {};
    build.Inputs = box16_inputs;
    build.DestAccelerationStructureData = box16_blas->GetGPUVirtualAddress();
    build.ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress();
    list->BuildRaytracingAccelerationStructure(&build, 1, &compacted_desc);   // its COMPACTED_SIZE, at info + 0
    list->ResourceBarrier(1, &barrier);
    list->CopyBufferRegion(box16_rb.p, 0, box16_blas.p, 0, box16_info.ResultDataMaxSizeInBytes);
    list->CopyBufferRegion(info_rb.p, 0, info.p, 0, 8);
    if (!submit(5)) return fail("MINIMIZE_MEMORY submission");
    UINT64 box16_size = 0;
    {
        void* p = nullptr;
        if (FAILED(info_rb->Map(0, nullptr, &p))) return fail("Map postbuild (MINIMIZE_MEMORY)");
        std::memcpy(&box16_size, p, 8);
        info_rb->Unmap(0, nullptr);
    }
    // the nodes the build wrote and the spare unit, within the prebuild size and below the BOX32 build's own size
    if (!box16_size || box16_size % 64 || box16_size > box16_info.ResultDataMaxSizeInBytes || box16_size >= sizes[0])
        return fail("MINIMIZE_MEMORY postbuild size");
    if (!dump(box16_rb.p, box16_info.ResultDataMaxSizeInBytes, "d3d12_blas_box16",
              "{\"kind\": \"blas\", \"pairs\": false, \"box16\": true, \"address\": " +
                  std::to_string(box16_blas->GetGPUVirtualAddress()) +
                  ", \"triangles\": " + triangles_json(moved) + "}\n"))
        return fail("MINIMIZE_MEMORY BLAS readback");
    // the soup moves back to where it started, and the BOX16 BLAS is refitted in place
    if (FAILED(upload->Map(0, nullptr, reinterpret_cast<void**>(&mapped)))) return fail("Map upload (MINIMIZE_MEMORY)");
    std::memcpy(mapped + (soup_va - upload->GetGPUVirtualAddress()), soup.data(), std::size_t(soup_bytes));
    upload->Unmap(0, nullptr);
    if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.p, nullptr))) return fail("Reset for the BOX16 update");
    build.Inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
    build.SourceAccelerationStructureData = box16_blas->GetGPUVirtualAddress();
    list->BuildRaytracingAccelerationStructure(&build, 0, nullptr);
    list->ResourceBarrier(1, &barrier);
    list->CopyBufferRegion(box16_rb.p, 0, box16_blas.p, 0, box16_info.ResultDataMaxSizeInBytes);
    if (!submit(6)) return fail("BOX16 update submission");
    if (!dump(box16_rb.p, box16_info.ResultDataMaxSizeInBytes, "d3d12_blas_box16_update",
              "{\"kind\": \"blas\", \"pairs\": false, \"box16\": true, \"address\": " +
                  std::to_string(box16_blas->GetGPUVirtualAddress()) +
                  ", \"triangles\": " + tris + "}\n"))
        return fail("BOX16 BLAS readback after the update");
    std::printf("d3d12_xdxr_rt: MINIMIZE_MEMORY BLAS %llu bytes (BOX16)\n", static_cast<unsigned long long>(box16_size));

    // ---- a BLAS that will not be updated (no ALLOW_UPDATE): sibling triangles that share an edge are paired into
    // TRI0/TRI1 leaves. Pairing changes neither the layout nor the size ----
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS static_inputs = blas_inputs;
    static_inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO static_info{};
    device->GetRaytracingAccelerationStructurePrebuildInfo(&static_inputs, &static_info);
    if (static_info.ResultDataMaxSizeInBytes != blas_info.ResultDataMaxSizeInBytes)
        return fail("a BLAS without ALLOW_UPDATE has another prebuild size");
    Ref<ID3D12Resource> static_blas, static_rb;
    if (FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, static_info.ResultDataMaxSizeInBytes, as_state, uav,
                           static_blas.put())) ||
        FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_READBACK, static_info.ResultDataMaxSizeInBytes,
                           D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, static_rb.put())))
        return fail("static BLAS buffers", hr);
    if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.p, nullptr))) return fail("Reset for the static BLAS");
    build = {};
    build.Inputs = static_inputs;
    build.DestAccelerationStructureData = static_blas->GetGPUVirtualAddress();
    build.ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress();
    list->BuildRaytracingAccelerationStructure(&build, 0, nullptr);
    list->ResourceBarrier(1, &barrier);
    list->CopyBufferRegion(static_rb.p, 0, static_blas.p, 0, static_info.ResultDataMaxSizeInBytes);
    if (!submit(7)) return fail("static BLAS submission");
    if (!dump(static_rb.p, static_info.ResultDataMaxSizeInBytes, "d3d12_blas_paired",
              "{\"kind\": \"blas\", \"pairs\": true, \"address\": " +
                  std::to_string(static_blas->GetGPUVirtualAddress()) + ", \"triangles\": " + tris + "}\n"))
        return fail("static BLAS readback");
    std::printf("d3d12_xdxr_rt: BLAS %llu bytes (prebuild %llu), TLAS (prebuild %llu), written to %s\n",
                static_cast<unsigned long long>(blas_info.ResultDataMaxSizeInBytes),
                static_cast<unsigned long long>(blas_info.ResultDataMaxSizeInBytes),
                static_cast<unsigned long long>(tlas_info.ResultDataMaxSizeInBytes), dir.c_str());
    return 0;
}
