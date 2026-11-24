#include "gpu.h"

#include <cstdio>
#include <vector>

#include "diagnostics.h"

namespace {

// One enumerated adapter. The index is kept so `--adapter N` matches the printed list
// even after software adapters are filtered out of automatic selection.
struct AdapterCandidate {
    int index = -1;
    bool software = false;
    SIZE_T dedicated_video_memory = 0;
    std::string description;
    ComPtr<IDXGIAdapter1> adapter;
};

// Converts a driver-supplied UTF-16 name to UTF-8. Returns false rather than storing a
// truncated name, so a broken name shows up as a reported failure, not a blank field.
bool Utf8FromWide(const wchar_t* wide, std::string& text_out) {
    const int byte_count = WideCharToMultiByte(CP_UTF8, 0, wide, -1, nullptr, 0, nullptr, nullptr);
    if (byte_count <= 1) {
        return false;
    }
    std::vector<char> buffer(static_cast<size_t>(byte_count));
    const int written =
        WideCharToMultiByte(CP_UTF8, 0, wide, -1, buffer.data(), byte_count, nullptr, nullptr);
    if (written != byte_count) {
        return false;
    }
    text_out.assign(buffer.data());
    return true;
}

// Enumerates every DXGI adapter. DXGI_ERROR_NOT_FOUND ends the list; any other failure
// on a later index would silently truncate the list, so it is reported.
bool EnumerateAdapters(IDXGIFactory4& factory, std::vector<AdapterCandidate>& candidates) {
    for (int index = 0;; ++index) {
        ComPtr<IDXGIAdapter1> adapter;
        const HRESULT result = factory.EnumAdapters1(static_cast<UINT>(index), adapter.GetAddressOf());
        if (result == DXGI_ERROR_NOT_FOUND) {
            return true;
        }
        if (!HrSucceeded(result, __FILE__, __LINE__, "IDXGIFactory4::EnumAdapters1",
                         "a broken display driver; try another adapter with --adapter")) {
            return false;
        }

        DXGI_ADAPTER_DESC1 description{};
        if (!HrSucceeded(adapter->GetDesc1(&description), __FILE__, __LINE__,
                         "IDXGIAdapter1::GetDesc1", "a display driver that does not expose a "
                         "description; try another adapter with --adapter")) {
            return false;
        }

        AdapterCandidate candidate;
        candidate.index = index;
        candidate.software = (description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0;
        candidate.dedicated_video_memory = description.DedicatedVideoMemory;
        if (!Utf8FromWide(description.Description, candidate.description)) {
            ReportFailure(__FILE__, __LINE__, "WideCharToMultiByte",
                          "the adapter description could not be converted to UTF-8",
                          "the driver reports an empty adapter name");
            return false;
        }
        candidate.adapter = adapter;
        candidates.push_back(candidate);
    }
}

// Prints the candidate list so a surprising choice can be diagnosed from the log alone.
void PrintAdapterCandidates(const std::vector<AdapterCandidate>& candidates) {
    for (const AdapterCandidate& candidate : candidates) {
        std::printf("adapter[%d] %s (%llu MB dedicated)%s\n", candidate.index,
                    candidate.description.c_str(),
                    static_cast<unsigned long long>(candidate.dedicated_video_memory /
                                                    (1024ull * 1024ull)),
                    candidate.software ? " [software]" : "");
    }
}

// Chooses the adapter: an explicit index wins verbatim, otherwise the non-software
// adapter with the most dedicated memory (the discrete GPU on a hybrid machine).
bool SelectAdapter(const std::vector<AdapterCandidate>& candidates, int requested_index,
                   const AdapterCandidate*& selected_out) {
    if (candidates.empty()) {
        ReportFailure(__FILE__, __LINE__, "SelectAdapter", "DXGI enumerated no adapters",
                      "the GPU driver is installed and the machine has a display adapter");
        return false;
    }
    if (requested_index >= 0) {
        for (const AdapterCandidate& candidate : candidates) {
            if (candidate.index == requested_index) {
                selected_out = &candidate;
                return true;
            }
        }
        ReportFailure(__FILE__, __LINE__, "SelectAdapter", "--adapter names an adapter that does "
                      "not exist", "pick an index from the adapter list printed above");
        return false;
    }

    const AdapterCandidate* best = nullptr;
    for (const AdapterCandidate& candidate : candidates) {
        if (candidate.software) {
            continue;
        }
        if (best == nullptr || candidate.dedicated_video_memory > best->dedicated_video_memory) {
            best = &candidate;
        }
    }
    if (best == nullptr) {
        ReportFailure(__FILE__, __LINE__, "SelectAdapter",
                      "every enumerated adapter is software-only",
                      "install a GPU driver, or pass --adapter with the WARP index if WARP is "
                      "acceptable for a reference image");
        return false;
    }
    selected_out = best;
    return true;
}

// Turns on the D3D12 debug layer. Reports success only when the layer was actually
// enabled: the caller must not claim validation ran when this returns false.
bool EnableDebugLayer() {
    ComPtr<ID3D12Debug> debug_controller;
    const HRESULT result = D3D12GetDebugInterface(IID_PPV_ARGS(debug_controller.GetAddressOf()));
    if (FAILED(result)) {
        std::fprintf(stderr,
                     "note: --debug-layer requested but the D3D12 debug layer is not installed "
                     "(HRESULT 0x%08lX); continuing without validation\n",
                     static_cast<unsigned long>(result));
        return false;
    }
    debug_controller->EnableDebugLayer();
    // GPU-based validation is left at its default (off): it changes nothing about the
    // output but makes a 256x256 reference render take minutes on some drivers. Nothing
    // else is changed here; the layer is used exactly as Microsoft configured it.
    return true;
}

}  // namespace

GpuContext::~GpuContext() { DestroyGpuContext(*this); }

bool CreateGpuContext(GpuContext& gpu, int requested_adapter_index, bool enable_debug_layer) {
    const bool debug_layer_active = enable_debug_layer ? EnableDebugLayer() : false;
    // Recorded on the context so the caller can tell "validation ran and found nothing"
    // from "validation was asked for and never ran".
    gpu.validation_active = debug_layer_active;
    const UINT factory_flags = debug_layer_active ? DXGI_CREATE_FACTORY_DEBUG : 0;

    if (!HrSucceeded(CreateDXGIFactory2(factory_flags, IID_PPV_ARGS(gpu.dxgi_factory.GetAddressOf())),
                     __FILE__, __LINE__, "CreateDXGIFactory2",
                     "dxgi.dll is present and, with --debug-layer, the DXGI debug layer is "
                     "installed")) {
        return false;
    }

    std::vector<AdapterCandidate> candidates;
    if (!EnumerateAdapters(*gpu.dxgi_factory.Get(), candidates)) {
        return false;
    }
    PrintAdapterCandidates(candidates);

    const AdapterCandidate* selected = nullptr;
    if (!SelectAdapter(candidates, requested_adapter_index, selected)) {
        return false;
    }
    gpu.adapter_description = selected->description;
    gpu.adapter_is_software = selected->software;

    // Feature level 11_0 is the floor the scene contract needs; the shaders are SM 6.6,
    // which the PSO creation below validates against the actual device.
    if (!HrSucceeded(D3D12CreateDevice(selected->adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                      IID_PPV_ARGS(gpu.device.GetAddressOf())),
                     __FILE__, __LINE__, "D3D12CreateDevice",
                     "the adapter supports D3D12; an adapter listed as [software] or an old "
                     "driver is the usual cause")) {
        return false;
    }

    D3D12_COMMAND_QUEUE_DESC queue_description{};
    queue_description.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    queue_description.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    queue_description.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    if (!HrSucceeded(gpu.device->CreateCommandQueue(&queue_description, IID_PPV_ARGS(
                         gpu.command_queue.GetAddressOf())),
                     __FILE__, __LINE__, "ID3D12Device::CreateCommandQueue",
                     "a direct queue is available; a device lost at creation is the usual cause")) {
        return false;
    }

    if (!HrSucceeded(gpu.device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                             IID_PPV_ARGS(gpu.fence.GetAddressOf())),
                     __FILE__, __LINE__, "ID3D12Device::CreateFence",
                     "the device is still present after queue creation")) {
        return false;
    }

    gpu.fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!Win32Succeeded(gpu.fence_event != nullptr, __FILE__, __LINE__, "CreateEventW",
                        "the process can create kernel events; a restricted job object is the "
                        "usual cause")) {
        return false;
    }
    return true;
}

void DestroyGpuContext(GpuContext& gpu) {
    if (gpu.fence_event != nullptr) {
        CloseHandle(gpu.fence_event);
        gpu.fence_event = nullptr;
    }
    gpu.fence.Reset();
    gpu.command_queue.Reset();
    gpu.device.Reset();
    gpu.dxgi_factory.Reset();
}

bool WaitForFenceValue(GpuContext& gpu, uint64_t fence_value) {
    if (gpu.fence->GetCompletedValue() >= fence_value) {
        return true;
    }
    if (!HrSucceeded(gpu.fence->SetEventOnCompletion(fence_value, gpu.fence_event), __FILE__,
                     __LINE__, "ID3D12Fence::SetEventOnCompletion",
                     "the fence event handle is valid and the queue has not been destroyed")) {
        return false;
    }
    const DWORD wait_result = WaitForSingleObject(gpu.fence_event, 60000);
    if (wait_result != WAIT_OBJECT_0) {
        ReportFailure(__FILE__, __LINE__, "WaitForSingleObject",
                      wait_result == WAIT_TIMEOUT ? "the GPU did not finish within 60 s"
                                                  : "waiting for the fence failed",
                      "Event Viewer for a display driver reset (TDR) on the adapter named above; "
                      "a hung draw or an unbound resource is the usual cause");
        return false;
    }
    return true;
}

bool SignalAndWaitForGpu(GpuContext& gpu) {
    const uint64_t fence_value = gpu.next_fence_value;
    gpu.next_fence_value += 1;
    if (!HrSucceeded(gpu.command_queue->Signal(gpu.fence.Get(), fence_value), __FILE__, __LINE__,
                     "ID3D12CommandQueue::Signal", "the queue is still open and the fence valid")) {
        return false;
    }
    return WaitForFenceValue(gpu, fence_value);
}

bool DeviceStillPresent(GpuContext& gpu) {
    const HRESULT reason = gpu.device->GetDeviceRemovedReason();
    if (SUCCEEDED(reason)) {
        return true;
    }
    return HrSucceeded(reason, __FILE__, __LINE__, "ID3D12Device::GetDeviceRemovedReason",
                       "Event Viewer for a TDR on the adapter named above; the HRESULT names "
                       "the removal reason (DEVICE_HUNG, DEVICE_RESET, OUT_OF_MEMORY)");
}

int ReportValidationMessages(ID3D12Device& device) {
    ComPtr<ID3D12InfoQueue> info_queue;
    if (FAILED(device.QueryInterface(IID_PPV_ARGS(info_queue.GetAddressOf())))) {
        return 0;
    }
    const UINT64 message_count = info_queue->GetNumStoredMessages();
    int error_count = 0;
    int unreadable_count = 0;
    for (UINT64 index = 0; index < message_count; ++index) {
        SIZE_T message_size = 0;
        if (FAILED(info_queue->GetMessage(index, nullptr, &message_size)) || message_size == 0) {
            ++unreadable_count;
            continue;
        }
        std::vector<char> storage(message_size);
        D3D12_MESSAGE* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
        if (FAILED(info_queue->GetMessage(index, message, &message_size))) {
            ++unreadable_count;
            continue;
        }
        // Severity counts down as severity rises (CORRUPTION 0 ... MESSAGE 4).
        if (message->Severity <= D3D12_MESSAGE_SEVERITY_WARNING) {
            std::fprintf(stderr, "validation: %.*s\n", static_cast<int>(message->DescriptionByteLength),
                         message->pDescription);
        }
        if (message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) {
            ++error_count;
        }
    }
    if (unreadable_count > 0) {
        std::fprintf(stderr, "validation: %d message(s) could not be retrieved from the info queue\n",
                     unreadable_count);
    }
    std::fflush(stderr);
    return error_count;
}
