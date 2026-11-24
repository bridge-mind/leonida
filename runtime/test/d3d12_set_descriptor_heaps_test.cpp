// ID3D12GraphicsCommandList::SetDescriptorHeaps validation, bound-heap / GPU-base tracking across Reset,
// ClearState and ExecuteBundle, COM lifetime pinning, and directly-indexed root-signature flag preservation,
// compared against native Windows D3D12 on the same adapter. No game data.
#include "vk_test_device.h"
#include "d3d12_copy_compare.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(cond)                                                                         \
    do {                                                                                    \
        if (!(cond)) {                                                                      \
            std::fprintf(stderr, "line %d: CHECK failed: %s\n", __LINE__, #cond);           \
            return false;                                                                   \
        }                                                                                   \
    } while (0)

#define REQUIRE(cond)                                                                       \
    do {                                                                                    \
        if (!(cond)) {                                                                      \
            std::fprintf(stderr, "line %d: REQUIRE failed: %s\n", __LINE__, #cond);         \
            return 1;                                                                       \
        }                                                                                   \
    } while (0)

using copycmp::Ptr;

namespace {

struct DeviceHeaps {
    ID3D12Device* device = nullptr;
    Ptr<ID3D12DescriptorHeap> cbv1, cbv2, smp1, smp2, cbv_nonsv, smp_nonsv, rtv, dsv;

    bool init(ID3D12Device* d) {
        device = d;
        const auto mk = [&](D3D12_DESCRIPTOR_HEAP_TYPE type, UINT count, D3D12_DESCRIPTOR_HEAP_FLAGS flags,
                            Ptr<ID3D12DescriptorHeap>& out) {
            const D3D12_DESCRIPTOR_HEAP_DESC hd{type, count, flags, 0};
            return SUCCEEDED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&out.p)));
        };
        CHECK(mk(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, cbv1));
        CHECK(mk(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, cbv2));
        CHECK(mk(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 4, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, smp1));
        CHECK(mk(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 4, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, smp2));
        CHECK(mk(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, cbv_nonsv));
        CHECK(mk(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 4, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, smp_nonsv));
        CHECK(mk(D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, rtv));
        CHECK(mk(D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, dsv));
        return true;
    }
};

bool check_single_list_cases(DeviceHeaps& ours, ID3D12DescriptorHeap* foreign_heap, DeviceHeaps* host,
                             const std::string& host_name) {
    struct Case {
        const char* label;
        D3D12_COMMAND_LIST_TYPE list_type;
        bool want_ok;
        bool check_host;  // release d3d12Core.dll faults on null array and accepts foreign-device heaps
        UINT num;
        std::vector<int> slots;  // -1 = nullptr, 0..7 = DeviceHeaps member, 8 = foreign_heap
    };
    const auto resolve = [](const DeviceHeaps& h, ID3D12DescriptorHeap* foreign, int slot) -> ID3D12DescriptorHeap* {
        switch (slot) {
        case 0: return h.cbv1.p;
        case 1: return h.cbv2.p;
        case 2: return h.smp1.p;
        case 3: return h.smp2.p;
        case 4: return h.cbv_nonsv.p;
        case 5: return h.smp_nonsv.p;
        case 6: return h.rtv.p;
        case 7: return h.dsv.p;
        case 8: return foreign;
        default: return nullptr;
        }
    };

    const Case cases[] = {
        {"direct (0, null)", D3D12_COMMAND_LIST_TYPE_DIRECT, true, true, 0, {}},
        {"direct (1, cbv1)", D3D12_COMMAND_LIST_TYPE_DIRECT, true, true, 1, {0}},
        {"direct (1, smp1)", D3D12_COMMAND_LIST_TYPE_DIRECT, true, true, 1, {2}},
        {"direct (2, cbv1+smp1)", D3D12_COMMAND_LIST_TYPE_DIRECT, true, true, 2, {0, 2}},
        {"direct (2, smp1+cbv1)", D3D12_COMMAND_LIST_TYPE_DIRECT, true, true, 2, {2, 0}},
        {"direct (3, arr)", D3D12_COMMAND_LIST_TYPE_DIRECT, false, true, 3, {0, 2, 1}},
        {"direct (1, nullptr)", D3D12_COMMAND_LIST_TYPE_DIRECT, false, false, 1, {}},
        {"direct (1, {nullptr})", D3D12_COMMAND_LIST_TYPE_DIRECT, false, true, 1, {-1}},
        {"direct non-sv cbv", D3D12_COMMAND_LIST_TYPE_DIRECT, false, true, 1, {4}},
        {"direct non-sv sampler", D3D12_COMMAND_LIST_TYPE_DIRECT, false, true, 1, {5}},
        {"direct rtv heap", D3D12_COMMAND_LIST_TYPE_DIRECT, false, true, 1, {6}},
        {"direct dsv heap", D3D12_COMMAND_LIST_TYPE_DIRECT, false, true, 1, {7}},
        {"direct duplicate cbv", D3D12_COMMAND_LIST_TYPE_DIRECT, false, true, 2, {0, 1}},
        {"direct duplicate sampler", D3D12_COMMAND_LIST_TYPE_DIRECT, false, true, 2, {2, 3}},
        {"direct foreign heap", D3D12_COMMAND_LIST_TYPE_DIRECT, false, false, 1, {8}},
        {"compute (2, cbv1+smp1)", D3D12_COMMAND_LIST_TYPE_COMPUTE, true, true, 2, {0, 2}},
        {"compute duplicate cbv", D3D12_COMMAND_LIST_TYPE_COMPUTE, false, true, 2, {0, 1}},
        {"copy (0, null)", D3D12_COMMAND_LIST_TYPE_COPY, false, true, 0, {}},
        {"copy (1, cbv1)", D3D12_COMMAND_LIST_TYPE_COPY, false, true, 1, {0}},
        {"bundle (1, nullptr)", D3D12_COMMAND_LIST_TYPE_BUNDLE, false, false, 1, {}},
        {"bundle non-sv cbv", D3D12_COMMAND_LIST_TYPE_BUNDLE, false, true, 1, {4}},
        {"bundle duplicate cbv", D3D12_COMMAND_LIST_TYPE_BUNDLE, false, true, 2, {0, 1}},
        {"bundle foreign heap", D3D12_COMMAND_LIST_TYPE_BUNDLE, false, false, 1, {8}},
    };

    const auto run_one = [&](ID3D12Device* dev, const DeviceHeaps& h, ID3D12DescriptorHeap* foreign,
                             const Case& c, HRESULT& out_hr) -> bool {
        Ptr<ID3D12CommandAllocator> al;
        Ptr<ID3D12GraphicsCommandList> list;
        CHECK(SUCCEEDED(dev->CreateCommandAllocator(c.list_type, IID_PPV_ARGS(&al.p))));
        CHECK(SUCCEEDED(dev->CreateCommandList(0, c.list_type, al.p, nullptr, IID_PPV_ARGS(&list.p))));
        std::vector<ID3D12DescriptorHeap*> ptrs;
        for (int s : c.slots) ptrs.push_back(resolve(h, foreign, s));
        list->SetDescriptorHeaps(c.num, ptrs.empty() ? nullptr : ptrs.data());
        out_hr = list->Close();
        return true;
    };

    for (const Case& c : cases) {
        HRESULT hr_ours = E_FAIL;
        CHECK(run_one(ours.device, ours, foreign_heap, c, hr_ours));
        const bool ok_ours = SUCCEEDED(hr_ours);
        if (ok_ours != c.want_ok || (!c.want_ok && hr_ours != E_INVALIDARG)) {
            std::fprintf(stderr, "SetDescriptorHeaps (%s) layer Close=0x%08lx want_ok=%d\n",
                         c.label, static_cast<unsigned long>(hr_ours), int(c.want_ok));
            return false;
        }
        if (host && c.check_host) {
            HRESULT hr_host = E_FAIL;
            CHECK(run_one(host->device, *host, nullptr, c, hr_host));
            if (SUCCEEDED(hr_host) != ok_ours) {
                std::fprintf(stderr, "SetDescriptorHeaps (%s) differed from host D3D12 (%s): layer=0x%08lx host=0x%08lx\n",
                             c.label, host_name.c_str(), static_cast<unsigned long>(hr_ours),
                             static_cast<unsigned long>(hr_host));
                return false;
            }
        }
    }
    return true;
}

bool check_bundle_parity(DeviceHeaps& ours, DeviceHeaps* host, const std::string& host_name) {
    enum class Action { match_both, parent_both_bundle_cbv_then_smp, parent_cbv_bundle_null_then_cbv,
                        mismatch_cbv, parent_none_bundle_cbv, clear_state_then_bundle_cbv,
                        rebind_single_drops_sampler_then_bundle_smp };
    struct BCase {
        const char* label;
        Action action;
        bool want_ok;
    };
    const BCase bcases[] = {
        {"bundle matches parent both", Action::match_both, true},
        {"parent both -> b(cbv) then b(smp)", Action::parent_both_bundle_cbv_then_smp, true},
        {"parent cbv -> b(0,null) then b(cbv)", Action::parent_cbv_bundle_null_then_cbv, true},
        {"bundle mismatched cbv", Action::mismatch_cbv, false},
        {"parent unbound -> b(cbv)", Action::parent_none_bundle_cbv, false},
        {"parent ClearState -> b(cbv)", Action::clear_state_then_bundle_cbv, false},
        {"parent (2,both)->(1,cbv2) drops sampler", Action::rebind_single_drops_sampler_then_bundle_smp, false},
    };

    const auto run_action = [](const DeviceHeaps& h, Action a, HRESULT& out_pclose) -> bool {
        Ptr<ID3D12CommandAllocator> pa, ba1, ba2, ba0;
        Ptr<ID3D12GraphicsCommandList> p, b1, b2, b0;
        CHECK(SUCCEEDED(h.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&pa.p))));
        CHECK(SUCCEEDED(h.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_BUNDLE, IID_PPV_ARGS(&ba1.p))));
        CHECK(SUCCEEDED(h.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_BUNDLE, IID_PPV_ARGS(&ba2.p))));
        CHECK(SUCCEEDED(h.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_BUNDLE, IID_PPV_ARGS(&ba0.p))));
        CHECK(SUCCEEDED(h.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, pa.p, nullptr, IID_PPV_ARGS(&p.p))));
        CHECK(SUCCEEDED(h.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_BUNDLE, ba1.p, nullptr, IID_PPV_ARGS(&b1.p))));
        CHECK(SUCCEEDED(h.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_BUNDLE, ba2.p, nullptr, IID_PPV_ARGS(&b2.p))));
        CHECK(SUCCEEDED(h.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_BUNDLE, ba0.p, nullptr, IID_PPV_ARGS(&b0.p))));

        ID3D12DescriptorHeap* both[2] = {h.cbv1.p, h.smp1.p};
        switch (a) {
        case Action::match_both:
            b1->SetDescriptorHeaps(2, both);
            CHECK(SUCCEEDED(b1->Close()));
            p->SetDescriptorHeaps(2, both);
            p->ExecuteBundle(b1.p);
            break;
        case Action::parent_both_bundle_cbv_then_smp:
            b1->SetDescriptorHeaps(1, &h.cbv1.p);
            b2->SetDescriptorHeaps(1, &h.smp1.p);
            CHECK(SUCCEEDED(b1->Close()) && SUCCEEDED(b2->Close()));
            p->SetDescriptorHeaps(2, both);
            p->ExecuteBundle(b1.p);
            p->ExecuteBundle(b2.p);
            break;
        case Action::parent_cbv_bundle_null_then_cbv:
            b0->SetDescriptorHeaps(0, nullptr);
            b1->SetDescriptorHeaps(1, &h.cbv1.p);
            CHECK(SUCCEEDED(b0->Close()) && SUCCEEDED(b1->Close()));
            p->SetDescriptorHeaps(1, &h.cbv1.p);
            p->ExecuteBundle(b0.p);
            p->ExecuteBundle(b1.p);
            break;
        case Action::mismatch_cbv:
            b1->SetDescriptorHeaps(1, &h.cbv2.p);
            CHECK(SUCCEEDED(b1->Close()));
            p->SetDescriptorHeaps(1, &h.cbv1.p);
            p->ExecuteBundle(b1.p);
            break;
        case Action::parent_none_bundle_cbv:
            b1->SetDescriptorHeaps(1, &h.cbv1.p);
            CHECK(SUCCEEDED(b1->Close()));
            p->ExecuteBundle(b1.p);
            break;
        case Action::clear_state_then_bundle_cbv:
            b1->SetDescriptorHeaps(1, &h.cbv1.p);
            CHECK(SUCCEEDED(b1->Close()));
            p->SetDescriptorHeaps(1, &h.cbv1.p);
            p->ClearState(nullptr);
            p->ExecuteBundle(b1.p);
            break;
        case Action::rebind_single_drops_sampler_then_bundle_smp:
            b2->SetDescriptorHeaps(1, &h.smp1.p);
            CHECK(SUCCEEDED(b2->Close()));
            p->SetDescriptorHeaps(2, both);
            p->SetDescriptorHeaps(1, &h.cbv2.p);
            p->ExecuteBundle(b2.p);
            break;
        }
        out_pclose = p->Close();
        return true;
    };

    for (const BCase& c : bcases) {
        HRESULT hr_ours = E_FAIL;
        CHECK(run_action(ours, c.action, hr_ours));
        if (SUCCEEDED(hr_ours) != c.want_ok || (!c.want_ok && hr_ours != E_INVALIDARG)) {
            std::fprintf(stderr, "bundle parity (%s) layer Close=0x%08lx want_ok=%d\n",
                         c.label, static_cast<unsigned long>(hr_ours), int(c.want_ok));
            return false;
        }
        if (host) {
            HRESULT hr_host = E_FAIL;
            CHECK(run_action(*host, c.action, hr_host));
            if (SUCCEEDED(hr_host) != SUCCEEDED(hr_ours)) {
                std::fprintf(stderr, "bundle parity (%s) differed from host (%s): layer=0x%08lx host=0x%08lx\n",
                             c.label, host_name.c_str(), static_cast<unsigned long>(hr_ours),
                             static_cast<unsigned long>(hr_host));
                return false;
            }
        }
    }
    return true;
}

bool check_tracking_lifetime_and_direct_indexing(DeviceHeaps& ours) {
    Ptr<ID3D12CommandAllocator> al;
    Ptr<ID3D12GraphicsCommandList> list;
    Ptr<ID3D12CommandQueue> queue;
    Ptr<ID3D12Fence> fence;
    const D3D12_COMMAND_QUEUE_DESC qd{};
    CHECK(SUCCEEDED(ours.device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue.p))));
    CHECK(SUCCEEDED(ours.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
    CHECK(SUCCEEDED(ours.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&al.p))));
    CHECK(SUCCEEDED(ours.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, al.p, nullptr, IID_PPV_ARGS(&list.p))));

    const D3D12_GPU_VIRTUAL_ADDRESS cbv1_va = ours.cbv1->GetGPUDescriptorHandleForHeapStart().ptr;
    const D3D12_GPU_VIRTUAL_ADDRESS cbv2_va = ours.cbv2->GetGPUDescriptorHandleForHeapStart().ptr;
    const D3D12_GPU_VIRTUAL_ADDRESS smp1_va = ours.smp1->GetGPUDescriptorHandleForHeapStart().ptr;
    CHECK(cbv1_va && cbv2_va && smp1_va && cbv1_va != cbv2_va && cbv1_va != smp1_va);

    D3D12_GPU_VIRTUAL_ADDRESS got_cbv = ~0ull, got_smp = ~0ull;
    CHECK(rt::d3d12::bound_descriptor_heap_bases(list.p, got_cbv, got_smp));
    CHECK(got_cbv == 0 && got_smp == 0);

    ID3D12DescriptorHeap* both_rev[2] = {ours.smp1.p, ours.cbv1.p};
    list->SetDescriptorHeaps(2, both_rev);
    CHECK(rt::d3d12::bound_descriptor_heap_bases(list.p, got_cbv, got_smp));
    CHECK(got_cbv == cbv1_va && got_smp == smp1_va);

    list->SetDescriptorHeaps(1, &ours.cbv2.p);
    CHECK(rt::d3d12::bound_descriptor_heap_bases(list.p, got_cbv, got_smp));
    CHECK(got_cbv == cbv2_va && got_smp == 0);

    list->ClearState(nullptr);
    CHECK(rt::d3d12::bound_descriptor_heap_bases(list.p, got_cbv, got_smp));
    CHECK(got_cbv == 0 && got_smp == 0);

    // Invalid call does not clobber already-bound valid heaps, while marking list invalid.
    list->SetDescriptorHeaps(2, both_rev);
    list->SetDescriptorHeaps(1, &ours.rtv.p);
    CHECK(rt::d3d12::bound_descriptor_heap_bases(list.p, got_cbv, got_smp));
    CHECK(got_cbv == cbv1_va && got_smp == smp1_va);
    CHECK(list->Close() == E_INVALIDARG);

    CHECK(SUCCEEDED(al->Reset()) && SUCCEEDED(list->Reset(al.p, nullptr)));
    CHECK(rt::d3d12::bound_descriptor_heap_bases(list.p, got_cbv, got_smp));
    CHECK(got_cbv == 0 && got_smp == 0);

    // COM lifetime: caller releases its heap reference after SetDescriptorHeaps; list keeps it pinned until Reset().
    ID3D12DescriptorHeap* raw_ephemeral = nullptr;
    {
        Ptr<ID3D12DescriptorHeap> eph;
        const D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 8,
                                           D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        CHECK(SUCCEEDED(ours.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&eph.p))));
        raw_ephemeral = eph.p;
        const D3D12_GPU_VIRTUAL_ADDRESS eph_va = eph->GetGPUDescriptorHandleForHeapStart().ptr;
        CHECK(eph_va != 0);
        list->SetDescriptorHeaps(1, &eph.p);
        CHECK(rt::d3d12::bound_descriptor_heap_bases(list.p, got_cbv, got_smp));
        CHECK(got_cbv == eph_va && got_smp == 0);
        // Bundle matching also sees the live heap before eph leaves scope.
        Ptr<ID3D12CommandAllocator> ba;
        Ptr<ID3D12GraphicsCommandList> b;
        CHECK(SUCCEEDED(ours.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_BUNDLE, IID_PPV_ARGS(&ba.p))));
        CHECK(SUCCEEDED(ours.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_BUNDLE, ba.p, nullptr, IID_PPV_ARGS(&b.p))));
        b->SetDescriptorHeaps(1, &eph.p);
        CHECK(SUCCEEDED(b->Close()));
        list->ExecuteBundle(b.p);
    }
    // Caller's Ptr destroyed; list must still hold >= 1 ref so AddRef() returns 2+.
    CHECK(raw_ephemeral->AddRef() >= 2);
    raw_ephemeral->Release();

    CHECK(SUCCEEDED(list->Close()));
    ID3D12CommandList* exec[1] = {list.p};
    queue->ExecuteCommandLists(1, exec);
    CHECK(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
    CHECK(SUCCEEDED(queue->Signal(fence.p, 1)) && SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));

    CHECK(SUCCEEDED(al->Reset()) && SUCCEEDED(list->Reset(al.p, nullptr)));
    CHECK(rt::d3d12::bound_descriptor_heap_bases(list.p, got_cbv, got_smp));
    CHECK(got_cbv == 0 && got_smp == 0);

    // Direct heap indexing root-signature flags are preserved on RootArgumentLayout.
    D3D12_ROOT_PARAMETER p{};
    p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    p.Constants = {0, 0, 2};
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC rs_desc{};
    rs_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    const auto di_flags = D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED |
                          D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED;
    rs_desc.Desc_1_0 = {1, &p, 0, nullptr, di_flags};
    const auto blob = d3d12test::serialize_root_signature(rs_desc);
    CHECK(!blob.empty());
    Ptr<ID3D12RootSignature> root;
    CHECK(SUCCEEDED(ours.device->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(&root.p))));
    const rt::d3d12::RootArgumentLayout* layout = rt::d3d12::root_signature_layout(root.p);
    CHECK(layout && (layout->flags & di_flags) == di_flags);

    CHECK(!rt::d3d12::bound_descriptor_heap_bases(nullptr, got_cbv, got_smp));
    return true;
}

}  // namespace

int main() {
    vktest::Device d;
    d.request_buffer_device_address = true;
    if (!d.create()) {
        std::puts("SKIP: no suitable Vulkan 1.3 (or 1.2 + KHR) device");
        return 77;
    }
    if (!d.buffer_device_address) {
        std::puts("SKIP: bufferDeviceAddress is required");
        return 77;
    }

    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = d.instance_proc;
    info.instance = d.instance;
    info.physical = d.physical;
    info.device = d.device;
    info.queue = d.queue;
    info.queue_family = d.family;
    info.buffer_device_address = d.buffer_device_address;
    info.features = d.enabled;
    info.subgroup = d.subgroup;

    int rc = 0;
    {
        Ptr<ID3D12Device> ours_dev, other_dev;
        REQUIRE(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours_dev.p))));
        REQUIRE(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&other_dev.p))));

        DeviceHeaps ours_heaps;
        REQUIRE(ours_heaps.init(ours_dev.p));
        Ptr<ID3D12DescriptorHeap> foreign_heap;
        const D3D12_DESCRIPTOR_HEAP_DESC fhd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1,
                                             D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        REQUIRE(SUCCEEDED(other_dev->CreateDescriptorHeap(&fhd, IID_PPV_ARGS(&foreign_heap.p))));

        if (const auto* actual_info = rt::d3d12::vulkan_device_info(ours_dev.p)) {
            VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
            VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id};
            reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
                d.instance_proc(d.instance, "vkGetPhysicalDeviceProperties2"))(actual_info->physical, &props2);
            if (id.deviceLUIDValid) {
                LUID luid{};
                std::memcpy(&luid, id.deviceLUID, sizeof(luid));
                copycmp::layer_adapter() = luid;
            }
        }
        Ptr<ID3D12Device> host_dev;
        DeviceHeaps host_heaps;
        std::string host_name;
        const bool has_host = copycmp::create_host(host_dev, host_name) && host_heaps.init(host_dev.p);

        if (!check_single_list_cases(ours_heaps, foreign_heap.p, has_host ? &host_heaps : nullptr, host_name))
            rc = 1;
        if (!check_bundle_parity(ours_heaps, has_host ? &host_heaps : nullptr, host_name))
            rc = 1;
        if (!check_tracking_lifetime_and_direct_indexing(ours_heaps))
            rc = 1;

        if (rc == 0) {
            std::printf("SetDescriptorHeaps: 23 single-list cases, 7 bundle parity cases, ClearState/Reset, "
                        "lifetime pinning, direct-index flags%s passed\n",
                        has_host ? (" (matched host " + host_name + ")").c_str() : "");
        }
    }

    if (d.validation_errors()) {
        std::fprintf(stderr, "%u Vulkan validation error(s)\n", d.validation_errors());
        return 1;
    }
    return rc;
}
