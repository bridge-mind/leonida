// DXR acceleration-structure prebuild and command-list recording validation parity test.
// Compares valid prebuild/build/postbuild/copy acceptance against native Windows D3D12
// (WARP and native hardware adapter when DXR-capable) and verifies Leonida's strict
// validation and crash-safety rules with zero Vulkan validation errors.
// Synthetic resources only; no game content.
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <utility>

namespace {
template<class T> struct Ref {
    T* p = nullptr;
    Ref() = default;
    Ref(Ref&& o) noexcept : p(o.p) { o.p = nullptr; }
    Ref& operator=(Ref&& o) noexcept {
        if (this != &o) {
            if (p) p->Release();
            p = o.p;
            o.p = nullptr;
        }
        return *this;
    }
    Ref(const Ref&) = delete;
    Ref& operator=(const Ref&) = delete;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
    T** put() { return &p; }
};

bool make_buffer(ID3D12Device* dev, UINT64 size, D3D12_HEAP_TYPE heap_type,
                 D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state,
                 Ref<ID3D12Resource>& out) {
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = heap_type;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = size;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = flags;
    return SUCCEEDED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, state, nullptr,
                                                  IID_PPV_ARGS(out.put())));
}

bool make_list(ID3D12Device5* dev, Ref<ID3D12CommandAllocator>& alloc, Ref<ID3D12GraphicsCommandList4>& cl) {
    return SUCCEEDED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(alloc.put()))) &&
           SUCCEEDED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr,
                                            IID_PPV_ARGS(cl.put())));
}

bool execute_and_wait(ID3D12Device5* dev, ID3D12CommandQueue* queue, ID3D12GraphicsCommandList4* cl) {
    if (FAILED(cl->Close())) return false;
    ID3D12CommandList* lists[] = {cl};
    queue->ExecuteCommandLists(1, lists);
    Ref<ID3D12Fence> fence;
    if (FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.put()))) ||
        FAILED(queue->Signal(fence.p, 1)))
        return false;
    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!ev) return false;
    const bool ok = SUCCEEDED(fence->SetEventOnCompletion(1, ev)) &&
                    WaitForSingleObject(ev, 10000) == WAIT_OBJECT_0;
    CloseHandle(ev);
    return ok;
}

bool is_zero_prebuild(const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO& info) {
    return info.ResultDataMaxSizeInBytes == 0 &&
           info.ScratchDataSizeInBytes == 0 &&
           info.UpdateScratchDataSizeInBytes == 0;
}

// Runs valid prebuild, valid GPU BLAS+TLAS build, and valid recording cases on any DXR-capable device.
// When check_strict_validation is true (Leonida), also asserts all invalid inputs are rejected cleanly.
int run_suite(ID3D12Device* base_dev, const char* name, bool check_strict_validation) {
    Ref<ID3D12Device5> dev;
    if (FAILED(base_dev->QueryInterface(IID_PPV_ARGS(dev.put())))) {
        std::printf("%s: no ID3D12Device5, skipping\n", name);
        return 77;
    }
    D3D12_FEATURE_DATA_D3D12_OPTIONS5 o5{};
    if (FAILED(dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &o5, sizeof(o5)))) {
        std::printf("%s: OPTIONS5 query failed, skipping\n", name);
        return 77;
    }
    if (!check_strict_validation && o5.RaytracingTier == D3D12_RAYTRACING_TIER_NOT_SUPPORTED) {
        std::printf("%s: raytracing not supported (tier=%d), skipping\n", name, static_cast<int>(o5.RaytracingTier));
        return 77;
    }

    Ref<ID3D12Resource> vb, inst_buf, scratch, blas_buf, tlas_buf, src_as, post_buf;
    if (!make_buffer(dev.p, 4096, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                     D3D12_RESOURCE_STATE_GENERIC_READ, vb) ||
        !make_buffer(dev.p, 4096, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                     D3D12_RESOURCE_STATE_GENERIC_READ, inst_buf) ||
        !make_buffer(dev.p, 262144, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                     D3D12_RESOURCE_STATE_COMMON, scratch) ||
        !make_buffer(dev.p, 262144, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                     D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, blas_buf) ||
        !make_buffer(dev.p, 262144, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                     D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, tlas_buf) ||
        !make_buffer(dev.p, 262144, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                     D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, src_as) ||
        !make_buffer(dev.p, 4096, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                     D3D12_RESOURCE_STATE_COMMON, post_buf)) {
        std::fprintf(stderr, "%s: failed to allocate synthetic buffers\n", name);
        return 1;
    }

    // Upload a single valid triangle and a single valid TLAS instance.
    {
        float* verts = nullptr;
        D3D12_RANGE no_read{0, 0};
        if (FAILED(vb->Map(0, &no_read, reinterpret_cast<void**>(&verts))) || !verts) return 1;
        const float tri_verts[9] = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
        std::memcpy(verts, tri_verts, sizeof(tri_verts));
        D3D12_RANGE written{0, sizeof(tri_verts)};
        vb->Unmap(0, &written);

        D3D12_RAYTRACING_INSTANCE_DESC* inst = nullptr;
        if (FAILED(inst_buf->Map(0, &no_read, reinterpret_cast<void**>(&inst))) || !inst) return 1;
        std::memset(inst, 0, sizeof(*inst));
        inst->Transform[0][0] = 1.0f;
        inst->Transform[1][1] = 1.0f;
        inst->Transform[2][2] = 1.0f;
        inst->InstanceMask = 0xFF;
        inst->AccelerationStructure = blas_buf->GetGPUVirtualAddress();
        D3D12_RANGE inst_written{0, sizeof(*inst)};
        inst_buf->Unmap(0, &inst_written);
    }

    const D3D12_GPU_VIRTUAL_ADDRESS vb_va = vb->GetGPUVirtualAddress();
    const D3D12_GPU_VIRTUAL_ADDRESS inst_va = inst_buf->GetGPUVirtualAddress();
    const D3D12_GPU_VIRTUAL_ADDRESS scr_va = scratch->GetGPUVirtualAddress();
    const D3D12_GPU_VIRTUAL_ADDRESS blas_va = blas_buf->GetGPUVirtualAddress();
    const D3D12_GPU_VIRTUAL_ADDRESS tlas_va = tlas_buf->GetGPUVirtualAddress();
    const D3D12_GPU_VIRTUAL_ADDRESS src_va = src_as->GetGPUVirtualAddress();
    const D3D12_GPU_VIRTUAL_ADDRESS post_va = post_buf->GetGPUVirtualAddress();

    D3D12_RAYTRACING_GEOMETRY_DESC tri{};
    tri.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
    tri.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
    tri.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
    tri.Triangles.VertexCount = 3;
    tri.Triangles.VertexBuffer.StartAddress = vb_va;
    tri.Triangles.VertexBuffer.StrideInBytes = 12;
    const D3D12_RAYTRACING_GEOMETRY_DESC* tri_ptr = &tri;
    const D3D12_RAYTRACING_GEOMETRY_DESC* null_tri_ptr = nullptr;

    // 1. Valid PrebuildInfo parity: BLAS (ARRAY & ARRAY_OF_POINTERS) + TLAS (ARRAY & ARRAY_OF_POINTERS).
    {
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS in{};
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info{};

        in.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        in.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        in.NumDescs = 1;
        in.pGeometryDescs = &tri;
        dev->GetRaytracingAccelerationStructurePrebuildInfo(&in, &info);
        if (info.ResultDataMaxSizeInBytes == 0 || info.ScratchDataSizeInBytes == 0) {
            if (check_strict_validation && o5.RaytracingTier == D3D12_RAYTRACING_TIER_NOT_SUPPORTED) {
                std::printf("%s: neither XDXR nor KHR acceleration structures supported, skipping\n", name);
                return 77;
            }
            std::fprintf(stderr, "%s: valid BLAS ARRAY prebuild returned zero\n", name);
            return 1;
        }
        const auto blas_array_res = info.ResultDataMaxSizeInBytes;

        info = {};
        in.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS;
        in.ppGeometryDescs = &tri_ptr;
        dev->GetRaytracingAccelerationStructurePrebuildInfo(&in, &info);
        if (info.ResultDataMaxSizeInBytes != blas_array_res || info.ScratchDataSizeInBytes == 0) {
            std::fprintf(stderr, "%s: valid BLAS ARRAY_OF_POINTERS prebuild mismatch\n", name);
            return 1;
        }

        info = {};
        in = {};
        in.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
        in.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        in.NumDescs = 1;
        in.InstanceDescs = inst_va;
        dev->GetRaytracingAccelerationStructurePrebuildInfo(&in, &info);
        if (info.ResultDataMaxSizeInBytes == 0 || info.ScratchDataSizeInBytes == 0) {
            std::fprintf(stderr, "%s: valid TLAS ARRAY prebuild returned zero\n", name);
            return 1;
        }

        info = {};
        in.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS;
        dev->GetRaytracingAccelerationStructurePrebuildInfo(&in, &info);
        if (info.ResultDataMaxSizeInBytes == 0 || info.ScratchDataSizeInBytes == 0) {
            std::fprintf(stderr, "%s: valid TLAS ARRAY_OF_POINTERS prebuild returned zero\n", name);
            return 1;
        }
    }

    // 2. Valid GPU BLAS + TLAS build sanity check (executed on queue, zero Vulkan validation errors).
    {
        Ref<ID3D12CommandQueue> queue;
        D3D12_COMMAND_QUEUE_DESC qdesc{};
        qdesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        if (FAILED(dev->CreateCommandQueue(&qdesc, IID_PPV_ARGS(queue.put())))) return 1;

        Ref<ID3D12CommandAllocator> alloc;
        Ref<ID3D12GraphicsCommandList4> cl;
        if (!make_list(dev.p, alloc, cl)) return 1;

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC blas_desc{};
        blas_desc.DestAccelerationStructureData = blas_va;
        blas_desc.ScratchAccelerationStructureData = scr_va;
        blas_desc.Inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        blas_desc.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        blas_desc.Inputs.NumDescs = 1;
        blas_desc.Inputs.pGeometryDescs = &tri;
        cl->BuildRaytracingAccelerationStructure(&blas_desc, 0, nullptr);

        D3D12_RESOURCE_BARRIER uav{};
        uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        uav.UAV.pResource = blas_buf.p;
        cl->ResourceBarrier(1, &uav);

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC tlas_desc{};
        tlas_desc.DestAccelerationStructureData = tlas_va;
        tlas_desc.ScratchAccelerationStructureData = scr_va;
        tlas_desc.Inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
        tlas_desc.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        tlas_desc.Inputs.NumDescs = 1;
        tlas_desc.Inputs.InstanceDescs = inst_va;
        cl->BuildRaytracingAccelerationStructure(&tlas_desc, 0, nullptr);

        if (!execute_and_wait(dev.p, queue.p, cl.p)) {
            std::fprintf(stderr, "%s: valid BLAS+TLAS GPU execution failed\n", name);
            return 1;
        }
    }

    auto valid_blas_desc = [&]() {
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC d{};
        d.DestAccelerationStructureData = blas_va;
        d.ScratchAccelerationStructureData = scr_va;
        d.Inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        d.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        d.Inputs.NumDescs = 1;
        d.Inputs.pGeometryDescs = &tri;
        return d;
    };

    auto valid_tlas_desc = [&]() {
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC d{};
        d.DestAccelerationStructureData = tlas_va;
        d.ScratchAccelerationStructureData = scr_va;
        d.Inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
        d.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        d.Inputs.NumDescs = 1;
        d.Inputs.InstanceDescs = inst_va;
        return d;
    };

    // 3. Valid recording parity across Build (with postbuild, with PERFORM_UPDATE, without PERFORM_UPDATE),
    //    EmitPostbuild (InfoType 0..3), and Copy (mode 0..4).
    {
        auto expect_ok = [&](const char* label, auto&& record) -> bool {
            Ref<ID3D12CommandAllocator> alloc;
            Ref<ID3D12GraphicsCommandList4> cl;
            if (!make_list(dev.p, alloc, cl)) return false;
            record(cl.p);
            const HRESULT hr = cl->Close();
            if (FAILED(hr)) {
                std::fprintf(stderr, "%s: valid case '%s' failed Close (0x%08X)\n",
                             name, label, static_cast<unsigned>(hr));
                return false;
            }
            return true;
        };

        if (!expect_ok("Build BLAS + postbuild CURRENT_SIZE", [&](ID3D12GraphicsCommandList4* cl) {
                auto d = valid_blas_desc();
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC pb{
                    post_va, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_CURRENT_SIZE};
                cl->BuildRaytracingAccelerationStructure(&d, 1, &pb);
            })) return 1;

        if (!expect_ok("Build PERFORM_UPDATE aligned src", [&](ID3D12GraphicsCommandList4* cl) {
                auto d = valid_blas_desc();
                d.Inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE |
                                 D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
                d.SourceAccelerationStructureData = src_va;
                cl->BuildRaytracingAccelerationStructure(&d, 0, nullptr);
            })) return 1;

        if (!expect_ok("Build non-update ignores src", [&](ID3D12GraphicsCommandList4* cl) {
                auto d = valid_blas_desc();
                d.SourceAccelerationStructureData = src_va + 128;
                cl->BuildRaytracingAccelerationStructure(&d, 0, nullptr);
            })) return 1;

        const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_TYPE info_types[] = {
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE,
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_TOOLS_VISUALIZATION,
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_SERIALIZATION,
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_CURRENT_SIZE,
        };
        for (const auto t : info_types) {
            if (!expect_ok("EmitPostbuild valid InfoType", [&](ID3D12GraphicsCommandList4* cl) {
                    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC pb{post_va, t};
                    cl->EmitRaytracingAccelerationStructurePostbuildInfo(&pb, 1, &blas_va);
                })) return 1;
        }

        const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE modes[] = {
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_CLONE,
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT,
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_VISUALIZATION_DECODE_FOR_TOOLS,
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_SERIALIZE,
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_DESERIALIZE,
        };
        for (const auto m : modes) {
            if (!expect_ok("CopyAS valid mode", [&](ID3D12GraphicsCommandList4* cl) {
                    cl->CopyRaytracingAccelerationStructure(blas_va, src_va, m);
                })) return 1;
        }
    }

    if (!check_strict_validation) {
        std::printf("%s (tier=%d): valid prebuild, GPU BLAS+TLAS build, and recording parity OK\n",
                    name, static_cast<int>(o5.RaytracingTier));
        return 0;
    }

    // 4. Leonida strict validation & crash-safety checks.
    // 4a. GetRaytracingAccelerationStructurePrebuildInfo invalid inputs -> zeroed out without crashing.
    {
        auto expect_zero_prebuild = [&](const char* label,
                                        const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS* in) -> bool {
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info{0xDEAD, 0xDEAD, 0xDEAD};
            dev->GetRaytracingAccelerationStructurePrebuildInfo(in, &info);
            if (!is_zero_prebuild(info)) {
                std::fprintf(stderr, "%s: Prebuild '%s' did not return zeroed info (%llu, %llu, %llu)\n",
                             name, label, info.ResultDataMaxSizeInBytes, info.ScratchDataSizeInBytes,
                             info.UpdateScratchDataSizeInBytes);
                return false;
            }
            return true;
        };

        // Null output pointer must not crash.
        {
            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS in{};
            in.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
            in.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
            dev->GetRaytracingAccelerationStructurePrebuildInfo(&in, nullptr);
        }

        if (!expect_zero_prebuild("desc=nullptr", nullptr)) return 1;

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS in{};
        in.Type = static_cast<D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE>(99);
        in.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        in.NumDescs = 1;
        in.pGeometryDescs = &tri;
        if (!expect_zero_prebuild("unknown Type=99", &in)) return 1;

        in.Type = static_cast<D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE>(-1);
        if (!expect_zero_prebuild("unknown Type=-1", &in)) return 1;

        in.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        in.DescsLayout = static_cast<D3D12_ELEMENTS_LAYOUT>(99);
        in.NumDescs = 1;
        in.pGeometryDescs = &tri;
        if (!expect_zero_prebuild("BLAS invalid DescsLayout=99 Num=1", &in)) return 1;

        in.NumDescs = 0;
        in.pGeometryDescs = nullptr;
        if (!expect_zero_prebuild("BLAS invalid DescsLayout=99 Num=0", &in)) return 1;

        in.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
        in.DescsLayout = static_cast<D3D12_ELEMENTS_LAYOUT>(99);
        in.NumDescs = 1;
        in.InstanceDescs = inst_va;
        if (!expect_zero_prebuild("TLAS invalid DescsLayout=99 Num=1", &in)) return 1;

        in.NumDescs = 0;
        in.InstanceDescs = 0;
        if (!expect_zero_prebuild("TLAS invalid DescsLayout=99 Num=0", &in)) return 1;

        in.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        in.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        in.NumDescs = 1;
        in.pGeometryDescs = nullptr;
        if (!expect_zero_prebuild("BLAS ARRAY Num=1 pGeometryDescs=null", &in)) return 1;

        in.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS;
        in.NumDescs = 1;
        in.ppGeometryDescs = nullptr;
        if (!expect_zero_prebuild("BLAS PTRS Num=1 ppGeometryDescs=null", &in)) return 1;

        in.ppGeometryDescs = &null_tri_ptr;
        if (!expect_zero_prebuild("BLAS PTRS Num=1 ppGeometryDescs[0]=null", &in)) return 1;
    }

    // 4b. Command-list recording invalid inputs -> Close() must fail.
    auto expect_fail = [&](const char* label, auto&& record) -> bool {
        Ref<ID3D12CommandAllocator> alloc;
        Ref<ID3D12GraphicsCommandList4> cl;
        if (!make_list(dev.p, alloc, cl)) return false;
        record(cl.p);
        const HRESULT hr = cl->Close();
        if (SUCCEEDED(hr)) {
            std::fprintf(stderr, "%s: invalid case '%s' unexpectedly passed Close()\n", name, label);
            return false;
        }
        return true;
    };

    // BuildRaytracingAccelerationStructure rejections
    if (!expect_fail("Build desc=null", [](ID3D12GraphicsCommandList4* cl) {
            cl->BuildRaytracingAccelerationStructure(nullptr, 0, nullptr);
        })) return 1;

    if (!expect_fail("Build postbuild_count=1 postbuild=null", [&](ID3D12GraphicsCommandList4* cl) {
            auto d = valid_blas_desc();
            cl->BuildRaytracingAccelerationStructure(&d, 1, nullptr);
        })) return 1;

    if (!expect_fail("Build postbuild InfoType=4", [&](ID3D12GraphicsCommandList4* cl) {
            auto d = valid_blas_desc();
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC pb{
                post_va, static_cast<D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_TYPE>(4)};
            cl->BuildRaytracingAccelerationStructure(&d, 1, &pb);
        })) return 1;

    if (!expect_fail("Build postbuild DestBuffer=0", [&](ID3D12GraphicsCommandList4* cl) {
            auto d = valid_blas_desc();
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC pb{
                0, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_CURRENT_SIZE};
            cl->BuildRaytracingAccelerationStructure(&d, 1, &pb);
        })) return 1;

    if (!expect_fail("Build postbuild DestBuffer+4", [&](ID3D12GraphicsCommandList4* cl) {
            auto d = valid_blas_desc();
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC pb{
                post_va + 4, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_CURRENT_SIZE};
            cl->BuildRaytracingAccelerationStructure(&d, 1, &pb);
        })) return 1;

    if (!expect_fail("Build unknown Type=99", [&](ID3D12GraphicsCommandList4* cl) {
            auto d = valid_blas_desc();
            d.Inputs.Type = static_cast<D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE>(99);
            cl->BuildRaytracingAccelerationStructure(&d, 0, nullptr);
        })) return 1;

    if (!expect_fail("Build BLAS unknown DescsLayout=99 Num=1", [&](ID3D12GraphicsCommandList4* cl) {
            auto d = valid_blas_desc();
            d.Inputs.DescsLayout = static_cast<D3D12_ELEMENTS_LAYOUT>(99);
            cl->BuildRaytracingAccelerationStructure(&d, 0, nullptr);
        })) return 1;

    if (!expect_fail("Build BLAS unknown DescsLayout=99 Num=0", [&](ID3D12GraphicsCommandList4* cl) {
            auto d = valid_blas_desc();
            d.Inputs.DescsLayout = static_cast<D3D12_ELEMENTS_LAYOUT>(99);
            d.Inputs.NumDescs = 0;
            d.Inputs.pGeometryDescs = nullptr;
            cl->BuildRaytracingAccelerationStructure(&d, 0, nullptr);
        })) return 1;

    if (!expect_fail("Build TLAS unknown DescsLayout=99 Num=1", [&](ID3D12GraphicsCommandList4* cl) {
            auto t = valid_tlas_desc();
            t.Inputs.DescsLayout = static_cast<D3D12_ELEMENTS_LAYOUT>(99);
            cl->BuildRaytracingAccelerationStructure(&t, 0, nullptr);
        })) return 1;

    if (!expect_fail("Build TLAS unknown DescsLayout=99 Num=0", [&](ID3D12GraphicsCommandList4* cl) {
            auto t = valid_tlas_desc();
            t.Inputs.DescsLayout = static_cast<D3D12_ELEMENTS_LAYOUT>(99);
            t.Inputs.NumDescs = 0;
            t.Inputs.InstanceDescs = 0;
            cl->BuildRaytracingAccelerationStructure(&t, 0, nullptr);
        })) return 1;

    if (!expect_fail("Build dst_va=0", [&](ID3D12GraphicsCommandList4* cl) {
            auto d = valid_blas_desc();
            d.DestAccelerationStructureData = 0;
            cl->BuildRaytracingAccelerationStructure(&d, 0, nullptr);
        })) return 1;

    if (!expect_fail("Build dst_va+128 misaligned", [&](ID3D12GraphicsCommandList4* cl) {
            auto d = valid_blas_desc();
            d.DestAccelerationStructureData = blas_va + 128;
            cl->BuildRaytracingAccelerationStructure(&d, 0, nullptr);
        })) return 1;

    if (!expect_fail("Build scratch_va=0", [&](ID3D12GraphicsCommandList4* cl) {
            auto d = valid_blas_desc();
            d.ScratchAccelerationStructureData = 0;
            cl->BuildRaytracingAccelerationStructure(&d, 0, nullptr);
        })) return 1;

    if (!expect_fail("Build scratch_va+128 misaligned", [&](ID3D12GraphicsCommandList4* cl) {
            auto d = valid_blas_desc();
            d.ScratchAccelerationStructureData = scr_va + 128;
            cl->BuildRaytracingAccelerationStructure(&d, 0, nullptr);
        })) return 1;

    if (!expect_fail("Build PERFORM_UPDATE src_va=0", [&](ID3D12GraphicsCommandList4* cl) {
            auto d = valid_blas_desc();
            d.Inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE |
                             D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
            d.SourceAccelerationStructureData = 0;
            cl->BuildRaytracingAccelerationStructure(&d, 0, nullptr);
        })) return 1;

    if (!expect_fail("Build PERFORM_UPDATE src_va+128 misaligned", [&](ID3D12GraphicsCommandList4* cl) {
            auto d = valid_blas_desc();
            d.Inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE |
                             D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
            d.SourceAccelerationStructureData = src_va + 128;
            cl->BuildRaytracingAccelerationStructure(&d, 0, nullptr);
        })) return 1;

    // EmitRaytracingAccelerationStructurePostbuildInfo rejections
    if (!expect_fail("Emit desc=null", [&](ID3D12GraphicsCommandList4* cl) {
            cl->EmitRaytracingAccelerationStructurePostbuildInfo(nullptr, 1, &blas_va);
        })) return 1;

    if (!expect_fail("Emit count=0", [&](ID3D12GraphicsCommandList4* cl) {
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC pb{
                post_va, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_CURRENT_SIZE};
            cl->EmitRaytracingAccelerationStructurePostbuildInfo(&pb, 0, &blas_va);
        })) return 1;

    if (!expect_fail("Emit sources=null", [&](ID3D12GraphicsCommandList4* cl) {
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC pb{
                post_va, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_CURRENT_SIZE};
            cl->EmitRaytracingAccelerationStructurePostbuildInfo(&pb, 1, nullptr);
        })) return 1;

    if (!expect_fail("Emit InfoType=4", [&](ID3D12GraphicsCommandList4* cl) {
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC pb{
                post_va, static_cast<D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_TYPE>(4)};
            cl->EmitRaytracingAccelerationStructurePostbuildInfo(&pb, 1, &blas_va);
        })) return 1;

    if (!expect_fail("Emit InfoType=-1", [&](ID3D12GraphicsCommandList4* cl) {
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC pb{
                post_va, static_cast<D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_TYPE>(-1)};
            cl->EmitRaytracingAccelerationStructurePostbuildInfo(&pb, 1, &blas_va);
        })) return 1;

    if (!expect_fail("Emit DestBuffer=0", [&](ID3D12GraphicsCommandList4* cl) {
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC pb{
                0, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_CURRENT_SIZE};
            cl->EmitRaytracingAccelerationStructurePostbuildInfo(&pb, 1, &blas_va);
        })) return 1;

    if (!expect_fail("Emit DestBuffer+4", [&](ID3D12GraphicsCommandList4* cl) {
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC pb{
                post_va + 4, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_CURRENT_SIZE};
            cl->EmitRaytracingAccelerationStructurePostbuildInfo(&pb, 1, &blas_va);
        })) return 1;

    if (!expect_fail("Emit source=0", [&](ID3D12GraphicsCommandList4* cl) {
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC pb{
                post_va, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_CURRENT_SIZE};
            const D3D12_GPU_VIRTUAL_ADDRESS zero = 0;
            cl->EmitRaytracingAccelerationStructurePostbuildInfo(&pb, 1, &zero);
        })) return 1;

    if (!expect_fail("Emit source+128", [&](ID3D12GraphicsCommandList4* cl) {
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC pb{
                post_va, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_CURRENT_SIZE};
            const D3D12_GPU_VIRTUAL_ADDRESS mis = blas_va + 128;
            cl->EmitRaytracingAccelerationStructurePostbuildInfo(&pb, 1, &mis);
        })) return 1;

    // CopyRaytracingAccelerationStructure rejections
    if (!expect_fail("Copy mode=5", [&](ID3D12GraphicsCommandList4* cl) {
            cl->CopyRaytracingAccelerationStructure(
                blas_va, src_va, static_cast<D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE>(5));
        })) return 1;

    if (!expect_fail("Copy mode=-1", [&](ID3D12GraphicsCommandList4* cl) {
            cl->CopyRaytracingAccelerationStructure(
                blas_va, src_va, static_cast<D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE>(-1));
        })) return 1;

    if (!expect_fail("Copy dst=0", [&](ID3D12GraphicsCommandList4* cl) {
            cl->CopyRaytracingAccelerationStructure(
                0, src_va, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_CLONE);
        })) return 1;

    if (!expect_fail("Copy dst+128", [&](ID3D12GraphicsCommandList4* cl) {
            cl->CopyRaytracingAccelerationStructure(
                blas_va + 128, src_va, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_CLONE);
        })) return 1;

    if (!expect_fail("Copy src=0", [&](ID3D12GraphicsCommandList4* cl) {
            cl->CopyRaytracingAccelerationStructure(
                blas_va, 0, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_CLONE);
        })) return 1;

    if (!expect_fail("Copy src+128", [&](ID3D12GraphicsCommandList4* cl) {
            cl->CopyRaytracingAccelerationStructure(
                blas_va, src_va + 128, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_CLONE);
        })) return 1;

    std::printf("%s (tier=%d): valid prebuild, GPU BLAS+TLAS build, and all strict validation checks OK\n",
                name, static_cast<int>(o5.RaytracingTier));
    return 0;
}
} // namespace

int main() {
    // 1. Differential host comparators: Windows WARP and native HW D3D12 (if available and DXR-capable).
    Ref<IDXGIFactory4> factory;
    if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(factory.put())))) {
        Ref<IDXGIAdapter> warp;
        Ref<ID3D12Device> warp_dev;
        if (SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(warp.put()))) &&
            SUCCEEDED(D3D12CreateDevice(warp.p, D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(warp_dev.put())))) {
            const int r = run_suite(warp_dev.p, "native WARP", false);
            if (r != 0 && r != 77) return r;
        }

        Ref<IDXGIAdapter1> hw_adapter;
        for (UINT i = 0; SUCCEEDED(factory->EnumAdapters1(i, hw_adapter.put())); ++i) {
            DXGI_ADAPTER_DESC1 desc{};
            hw_adapter->GetDesc1(&desc);
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
                hw_adapter.p->Release();
                hw_adapter.p = nullptr;
                continue;
            }
            Ref<ID3D12Device> hw_dev;
            if (SUCCEEDED(D3D12CreateDevice(hw_adapter.p, D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(hw_dev.put())))) {
                char adapter_name[128]{};
                WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, adapter_name, sizeof(adapter_name), nullptr,
                                    nullptr);
                char label[160]{};
                std::snprintf(label, sizeof(label), "native HW (%s)", adapter_name);
                const int r = run_suite(hw_dev.p, label, false);
                if (r != 0 && r != 77) return r;
                break;
            }
            hw_adapter.p->Release();
            hw_adapter.p = nullptr;
        }
    }

    // 2. Leonida Vulkan D3D12 device (strict validation + GPU BLAS/TLAS build under LEONIDA_VK_VALIDATION=1).
    Ref<ID3D12Device> layer;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(layer.put())))) return 77;
    return run_suite(layer.p, "Leonida Vulkan", true);
}
