#pragma once

#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <cstdint>
#include <string>

using Microsoft::WRL::ComPtr;

// Everything the renderer needs from the GPU: a device, a direct queue, and the fence
// used to know when readback is safe. Adapter selection happens here and the chosen
// adapter's description is kept for the manifest.
struct GpuContext {
    ComPtr<IDXGIFactory4> dxgi_factory;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> command_queue;
    ComPtr<ID3D12Fence> fence;
    HANDLE fence_event = nullptr;
    uint64_t next_fence_value = 1;
    std::string adapter_description;  // UTF-8 copy of DXGI_ADAPTER_DESC1::Description
    bool adapter_is_software = false;
    // True only when --debug-layer was requested AND the layer actually came up. A run
    // that asked for validation but did not get it must never report "0 errors".
    bool validation_active = false;

    // Releases the COM objects in the right order when the context goes out of scope, so
    // a caller cannot forget DestroyGpuContext on an early return.
    ~GpuContext();
};

// Creates the DXGI factory, picks an adapter, creates the device, queue and fence.
// Does not create any command lists or resources; that is the renderer's job.
// adapter_index < 0 selects the non-software adapter with the most dedicated memory.
bool CreateGpuContext(GpuContext& gpu, int requested_adapter_index, bool enable_debug_layer);

// Releases the COM objects and closes the fence event. Safe on a partially built
// context, which is why failure paths call it before returning.
void DestroyGpuContext(GpuContext& gpu);

// Blocks until the GPU has finished the given fence value, or 60 s pass. A timeout is
// reported as a failure rather than waited out forever.
bool WaitForFenceValue(GpuContext& gpu, uint64_t fence_value);

// Signals the queue's fence and waits for it. Call after ExecuteCommandLists whenever
// the next step touches the result on the CPU.
bool SignalAndWaitForGpu(GpuContext& gpu);

// True when the device has not been removed. A removed device silently produces zeroed
// or stale readback, so this is checked after every wait.
bool DeviceStillPresent(GpuContext& gpu);

// Prints validation-layer messages of WARNING severity or worse and returns the number
// of ERROR or CORRUPTION messages. Zero when the debug layer was not enabled.
//
// D3D12_MESSAGE_SEVERITY counts DOWN as severity goes up: CORRUPTION is 0, ERROR 1,
// WARNING 2, INFO 3, MESSAGE 4. So "severity <= WARNING" means "warning or worse", and
// "severity <= ERROR" means "error or corruption".
int ReportValidationMessages(ID3D12Device& device);
