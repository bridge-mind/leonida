#include "dxgi_com.h"
#include "d3d12_stubs.h"

#include <winternl.h>  // NTSTATUS, which d3dkmthk.h needs and WIN32_LEAN_AND_MEAN leaves out
#include <d3dkmthk.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace rt::dxgi {
namespace {

// Native DXGI accepts up to 40 queued frames on a waitable swapchain.
constexpr UINT max_waitable_frame_latency = 40;

void stub(const char* method) {
    static std::mutex mutex;
    static std::set<std::string> seen;
    std::lock_guard lock(mutex);
    if (seen.insert(method).second) std::fprintf(stderr, "[rt] dxgi: unimplemented %s\n", method);
}

// IDXGIObject private data, shared by the factory, adapter and swapchain. SetPrivateData keeps a copy of the bytes;
// SetPrivateDataInterface holds one reference until the entry is replaced, removed or the owner dies (Microsoft:
// IDXGIObject::SetPrivateDataInterface). Zero bytes remove the entry. GetPrivateData gives an interface back with a
// reference added, and a buffer too small for the entry gets DXGI_ERROR_MORE_DATA and the size to retry with.
class PrivateDataStore {
public:
    PrivateDataStore() = default;
    PrivateDataStore(const PrivateDataStore&) = delete;
    PrivateDataStore& operator=(const PrivateDataStore&) = delete;
    ~PrivateDataStore() {
        for (auto& entry : data_)
            if (entry.object) entry.object->Release();
    }

    HRESULT get(REFGUID guid, UINT* size, void* data) {
        if (!size) return E_INVALIDARG;
        std::lock_guard lock(mutex_);
        for (const auto& entry : data_) {
            if (entry.key != guid) continue;
            const UINT bytes = entry.object ? UINT(sizeof(IUnknown*)) : UINT(entry.bytes.size());
            if (data && *size < bytes) {
                *size = bytes;
                return DXGI_ERROR_MORE_DATA;
            }
            if (data) {
                if (entry.object) {
                    entry.object->AddRef();
                    std::memcpy(data, &entry.object, sizeof(IUnknown*));
                } else {
                    std::memcpy(data, entry.bytes.data(), bytes);
                }
            }
            *size = bytes;
            return S_OK;
        }
        *size = 0;
        return DXGI_ERROR_NOT_FOUND;
    }

    HRESULT set(REFGUID guid, UINT size, const void* data) {
        if (size && !data) return E_INVALIDARG;
        Entry entry{guid, {}, nullptr};
        if (size) {
            const auto* bytes = static_cast<const std::uint8_t*>(data);
            entry.bytes.assign(bytes, bytes + size);
        }
        store(std::move(entry), size != 0);
        return S_OK;
    }

    HRESULT set_interface(REFGUID guid, const IUnknown* object) {
        IUnknown* held = const_cast<IUnknown*>(object);
        if (held) held->AddRef();
        store(Entry{guid, {}, held}, held != nullptr);
        return S_OK;
    }

private:
    struct Entry {
        GUID key;
        std::vector<std::uint8_t> bytes;
        IUnknown* object;
    };

    // The replaced interface is released after the lock is dropped, since its Release may run arbitrary code.
    void store(Entry entry, bool keep) {
        IUnknown* released = nullptr;
        {
            std::lock_guard lock(mutex_);
            for (auto it = data_.begin(); it != data_.end(); ++it) {
                if (it->key != entry.key) continue;
                released = it->object;
                data_.erase(it);
                break;
            }
            if (keep) data_.push_back(std::move(entry));
        }
        if (released) released->Release();
    }

    std::mutex mutex_;
    std::vector<Entry> data_;
};

bool valid_flip_buffer_count(UINT count) {
    return count >= 2 && count <= DXGI_MAX_SWAP_CHAIN_BUFFERS;
}

bool is_direct_present_queue(IUnknown* candidate) {
    ID3D12CommandQueue* queue = nullptr;
    if (!candidate || FAILED(candidate->QueryInterface(IID_PPV_ARGS(&queue))) || !queue) return false;
    const bool direct = queue->GetDesc().Type == D3D12_COMMAND_LIST_TYPE_DIRECT;
    queue->Release();
    return direct;
}

VkFormat dxgi_to_vk_format(DXGI_FORMAT fmt) {
    switch (fmt) {
    case DXGI_FORMAT_B8G8R8A8_UNORM: return VK_FORMAT_B8G8R8A8_UNORM;
    case DXGI_FORMAT_R8G8B8A8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
    case DXGI_FORMAT_R10G10B10A2_UNORM: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case DXGI_FORMAT_R16G16B16A16_FLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;
    default: return VK_FORMAT_UNDEFINED;
    }
}

DXGI_FORMAT vk_to_dxgi_format(VkFormat fmt) {
    switch (fmt) {
    case VK_FORMAT_B8G8R8A8_UNORM: return DXGI_FORMAT_B8G8R8A8_UNORM;
    case VK_FORMAT_B8G8R8A8_SRGB: return DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
    case VK_FORMAT_R8G8B8A8_UNORM: return DXGI_FORMAT_R8G8B8A8_UNORM;
    case VK_FORMAT_R8G8B8A8_SRGB: return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: return DXGI_FORMAT_R10G10B10A2_UNORM;
    case VK_FORMAT_R16G16B16A16_SFLOAT: return DXGI_FORMAT_R16G16B16A16_FLOAT;
    default: return DXGI_FORMAT_UNKNOWN;
    }
}

// Real ID3D12Resource COM object representing a swapchain backbuffer. Its pointer value is the
// leonida::ResourceId registered in VulkanContext, matching d3d12_com.h's resource_id(ID3D12Resource*) contract.
class BackBufferResource final : public ID3D12ResourceStubs {
public:
    explicit BackBufferResource(const D3D12_RESOURCE_DESC& desc) : desc_(desc) {}

    ULONG STDMETHODCALLTYPE AddRef() override { return ++refs_; }
    ULONG STDMETHODCALLTYPE Release() override {
        const ULONG left = --refs_;
        if (!left) delete this;
        return left;
    }
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (riid == __uuidof(IUnknown) || riid == __uuidof(ID3D12Object) || riid == __uuidof(ID3D12DeviceChild) ||
            riid == __uuidof(ID3D12Pageable) || riid == __uuidof(ID3D12Resource)) {
            *out = static_cast<ID3D12Resource*>(this);
            AddRef();
            return S_OK;
        }
        return E_NOINTERFACE;
    }

    HRESULT STDMETHODCALLTYPE SetName(LPCWSTR) override { return S_OK; }

#ifdef _MSC_VER
    D3D12_RESOURCE_DESC STDMETHODCALLTYPE GetDesc(void) override { return desc_; }
#else
    D3D12_RESOURCE_DESC* STDMETHODCALLTYPE GetDesc(D3D12_RESOURCE_DESC* ret) override {
        if (ret) *ret = desc_;
        return ret;
    }
#endif

    D3D12_GPU_VIRTUAL_ADDRESS STDMETHODCALLTYPE GetGPUVirtualAddress(void) override { return 0; }

    HRESULT STDMETHODCALLTYPE GetHeapProperties(D3D12_HEAP_PROPERTIES* props, D3D12_HEAP_FLAGS* flags) override {
        if (props) {
            *props = {};
            props->Type = D3D12_HEAP_TYPE_DEFAULT;
            props->CreationNodeMask = 1;
            props->VisibleNodeMask = 1;
        }
        if (flags) *flags = D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES;
        return S_OK;
    }

    leonida::ResourceId resource_id() const { return reinterpret_cast<leonida::ResourceId>(this); }
    bool held_by_app() const { return refs_ > 1; }

private:
    std::atomic<ULONG> refs_{1};
    D3D12_RESOURCE_DESC desc_{};
};

class SwapChain final : public IDXGISwapChain3 {
public:
    SwapChain(IDXGIFactory4* parent, IUnknown* device, leonida::VulkanContext& context,
              PFN_vkGetInstanceProcAddr gipa, VkInstance instance, VkPhysicalDevice physical,
              PFN_vkGetDeviceProcAddr gdpa, VkQueue queue, std::uint32_t queue_family)
        : parent_(parent),
          device_(device),
          context_(context),
          swapchain_(gipa, instance, physical, gdpa, context.device(), queue, queue_family, context.submit_mutex()) {
        if (parent_) parent_->AddRef();
        if (device_) device_->AddRef();
    }

    ~SwapChain() {
        unregister_buffers();
        if (restrict_to_output_) restrict_to_output_->Release();
        if (waitable_) CloseHandle(waitable_);
        if (device_) device_->Release();
        if (parent_) parent_->Release();
    }

    bool init(HWND hwnd, const DXGI_SWAP_CHAIN_DESC1& desc, VkFormat vk_fmt) {
        hwnd_ = hwnd;
        flags_ = desc.Flags;
        const VkExtent2D size = size_or_client(desc.Width, desc.Height, {256u, 256u});
        const std::uint32_t w = size.width;
        const std::uint32_t h = size.height;
        const bool allow_tearing = (flags_ & DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING) != 0;
        if (!swapchain_.create(hwnd, w, h, vk_fmt, desc.BufferCount, allow_tearing)) return false;
        source_size_ = {w, h};
        if (flags_ & DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT) {
            // DXGI: waitable swapchains default to maximum frame latency 1.
            max_frame_latency_.store(1, std::memory_order_relaxed);
            waitable_ = CreateSemaphoreW(nullptr, 1, max_waitable_frame_latency, nullptr);
            if (!waitable_) return false;
        }
        register_buffers();
        std::uint32_t idx = 0;
        return swapchain_.acquire_next_image(idx);
    }

    // IUnknown
    ULONG STDMETHODCALLTYPE AddRef() override { return ++refs_; }
    ULONG STDMETHODCALLTYPE Release() override {
        const ULONG left = --refs_;
        if (!left) delete this;
        return left;
    }
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (riid == __uuidof(IUnknown) || riid == __uuidof(IDXGIObject) || riid == __uuidof(IDXGIDeviceSubObject) ||
            riid == __uuidof(IDXGISwapChain) || riid == __uuidof(IDXGISwapChain1) ||
            riid == __uuidof(IDXGISwapChain2) || riid == __uuidof(IDXGISwapChain3)) {
            *out = static_cast<IDXGISwapChain3*>(this);
            AddRef();
            return S_OK;
        }
        return E_NOINTERFACE;
    }

    // IDXGIObject
    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID guid, UINT size, const void* data) override {
        return private_data_.set(guid, size, data);
    }
    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID guid, const IUnknown* object) override {
        return private_data_.set_interface(guid, object);
    }
    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID guid, UINT* size, void* data) override {
        return private_data_.get(guid, size, data);
    }
    HRESULT STDMETHODCALLTYPE GetParent(REFIID riid, void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (!parent_) return E_NOINTERFACE;
        return parent_->QueryInterface(riid, out);
    }

    // IDXGIDeviceSubObject
    HRESULT STDMETHODCALLTYPE GetDevice(REFIID riid, void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (!device_) return E_NOINTERFACE;

        ID3D12DeviceChild* device_child = nullptr;
        if (SUCCEEDED(device_->QueryInterface(IID_PPV_ARGS(&device_child)))) {
            const HRESULT hr = device_child->GetDevice(riid, out);
            device_child->Release();
            return hr;
        }
        return device_->QueryInterface(riid, out);
    }

    // IDXGISwapChain
    HRESULT STDMETHODCALLTYPE Present(UINT sync, UINT flags) override {
        if (sync > 4) return DXGI_ERROR_INVALID_CALL;
        if ((flags & DXGI_PRESENT_ALLOW_TEARING) &&
            (sync != 0 || !(flags_ & DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING)))
            return DXGI_ERROR_INVALID_CALL;
        if (flags & DXGI_PRESENT_TEST) return S_OK;  // only asks whether a present would succeed
        const std::uint32_t cur = swapchain_.current_back_buffer_index();
        if (!swapchain_.present(cur)) return E_FAIL;
        ++present_count_;
        LARGE_INTEGER qpc{};
        if (QueryPerformanceCounter(&qpc)) last_present_qpc_.store(qpc.QuadPart, std::memory_order_relaxed);
        std::uint32_t next = 0;
        if (!swapchain_.acquire_next_image(next)) return E_FAIL;
        if (waitable_) ReleaseSemaphore(waitable_, 1, nullptr);
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE GetBuffer(UINT buffer, REFIID riid, void** surface) override {
        if (!surface) return E_POINTER;
        *surface = nullptr;
        if (buffer >= buffers_.size()) return DXGI_ERROR_INVALID_CALL;
        return buffers_[buffer]->QueryInterface(riid, surface);
    }

    HRESULT STDMETHODCALLTYPE SetFullscreenState(BOOL, IDXGIOutput*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetFullscreenState(BOOL* fs, IDXGIOutput** out) override {
        if (fs) *fs = FALSE;
        if (out) *out = nullptr;
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE GetDesc(DXGI_SWAP_CHAIN_DESC* desc) override {
        if (!desc) return E_INVALIDARG;
        std::memset(desc, 0, sizeof(*desc));
        desc->BufferDesc.Width = swapchain_.extent().width;
        desc->BufferDesc.Height = swapchain_.extent().height;
        desc->BufferDesc.Format = vk_to_dxgi_format(swapchain_.format());
        desc->SampleDesc.Count = 1;
        desc->BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        desc->BufferCount = swapchain_.image_count();
        desc->OutputWindow = hwnd_;
        desc->Windowed = TRUE;
        desc->SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        desc->Flags = flags_;
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE ResizeBuffers(UINT count, UINT width, UINT height, DXGI_FORMAT fmt, UINT flags) override {
        return resize_buffers(count, width, height, fmt, flags, nullptr, nullptr);
    }

    HRESULT resize_buffers(UINT count, UINT width, UINT height, DXGI_FORMAT fmt, UINT flags,
                           const UINT* node_masks, IUnknown* const* present_queues) {
        if (count != 0 && !valid_flip_buffer_count(count)) return E_INVALIDARG;
        VkFormat vk_fmt = VK_FORMAT_UNDEFINED;
        if (fmt != DXGI_FORMAT_UNKNOWN) {
            vk_fmt = dxgi_to_vk_format(fmt);
            if (vk_fmt == VK_FORMAT_UNDEFINED) return E_INVALIDARG;
        }
        constexpr UINT creation_only_flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT |
                                             DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
        if ((flags & creation_only_flags) != (flags_ & creation_only_flags)) return E_INVALIDARG;
        // DXGI refuses to resize while the app still holds a back buffer reference.
        for (auto* bb : buffers_)
            if (bb->held_by_app()) return DXGI_ERROR_INVALID_CALL;
        const std::size_t target_count = count ? count : buffers_.size();
        if (node_masks && !std::all_of(node_masks, node_masks + target_count, [](UINT mask) { return mask <= 1; }))
            return DXGI_ERROR_INVALID_CALL;
        if (present_queues && !std::all_of(present_queues, present_queues + target_count, is_direct_present_queue))
            return DXGI_ERROR_INVALID_CALL;
        unregister_buffers();
        const VkExtent2D size = size_or_client(width, height, swapchain_.extent());
        const std::uint32_t w = size.width;
        const std::uint32_t h = size.height;
        const UINT next_flags = flags ? flags : flags_;
        const bool allow_tearing = (next_flags & DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING) != 0;
        if (!swapchain_.resize(w, h, vk_fmt, count, allow_tearing)) return E_FAIL;
        source_size_ = {w, h};  // host-measured: ResizeBuffers resets the source region to the new back buffers
        statistics_disjoint_.store(true, std::memory_order_relaxed);
        flags_ = next_flags;
        register_buffers();
        std::uint32_t next = 0;
        return swapchain_.acquire_next_image(next) ? S_OK : E_FAIL;
    }

    HRESULT STDMETHODCALLTYPE ResizeTarget(const DXGI_MODE_DESC* mode) override {
        if (!mode) return DXGI_ERROR_INVALID_CALL;  // host-measured, not E_INVALIDARG
        // Windowed: the target output's client area takes the mode's size, and the back buffers keep theirs until
        // ResizeBuffers (where 0 means the client area). The window's own style gives the frame around it. A zero
        // size is accepted, as on the host, and the window's minimum tracking size decides what is left of it.
        RECT frame{0, 0, static_cast<LONG>(mode->Width), static_cast<LONG>(mode->Height)};
        const auto style = static_cast<DWORD>(GetWindowLongW(hwnd_, GWL_STYLE));
        const auto ex_style = static_cast<DWORD>(GetWindowLongW(hwnd_, GWL_EXSTYLE));
        if (!AdjustWindowRectEx(&frame, style, FALSE, ex_style)) return E_INVALIDARG;
        if (!SetWindowPos(hwnd_, nullptr, 0, 0, frame.right - frame.left, frame.bottom - frame.top,
                          SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE))
            return DXGI_ERROR_INVALID_CALL;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE GetContainingOutput(IDXGIOutput** out) override {
        if (!out) return DXGI_ERROR_INVALID_CALL;
        *out = nullptr;
        if (!parent_) return DXGI_ERROR_NOT_CURRENTLY_AVAILABLE;
        IDXGIAdapter* adapter = nullptr;
        if (FAILED(parent_->EnumAdapters(0, &adapter)) || !adapter) return DXGI_ERROR_NOT_CURRENTLY_AVAILABLE;

        const HMONITOR target = hwnd_ ? MonitorFromWindow(hwnd_, MONITOR_DEFAULTTONEAREST) : nullptr;
        IDXGIOutput* fallback = nullptr;
        for (UINT i = 0;; ++i) {
            IDXGIOutput* candidate = nullptr;
            if (adapter->EnumOutputs(i, &candidate) == DXGI_ERROR_NOT_FOUND) break;
            if (!candidate) continue;
            DXGI_OUTPUT_DESC desc{};
            if (target && SUCCEEDED(candidate->GetDesc(&desc)) && desc.Monitor == target) {
                if (fallback) fallback->Release();
                adapter->Release();
                *out = candidate;
                return S_OK;
            }
            if (!fallback) {
                fallback = candidate;
            } else {
                candidate->Release();
            }
        }
        adapter->Release();
        if (!fallback) return DXGI_ERROR_NOT_CURRENTLY_AVAILABLE;
        *out = fallback;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE GetFrameStatistics(DXGI_FRAME_STATISTICS* stats) override {
        if (!stats) return DXGI_ERROR_INVALID_CALL;
        std::memset(stats, 0, sizeof(*stats));
        // Native DXGI reports the discontinuity once, on the first valid query after creation or resize.
        if (statistics_disjoint_.exchange(false, std::memory_order_relaxed))
            return DXGI_ERROR_FRAME_STATISTICS_DISJOINT;
        const UINT count = present_count_.load(std::memory_order_relaxed);
        stats->PresentCount = count;
        stats->PresentRefreshCount = count;
        stats->SyncRefreshCount = count;
        stats->SyncQPCTime.QuadPart = last_present_qpc_.load(std::memory_order_relaxed);
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE GetLastPresentCount(UINT* p) override {
        if (!p) return E_INVALIDARG;
        *p = present_count_.load(std::memory_order_relaxed);
        return S_OK;
    }

    // IDXGISwapChain1
    HRESULT STDMETHODCALLTYPE GetDesc1(DXGI_SWAP_CHAIN_DESC1* desc) override {
        if (!desc) return E_INVALIDARG;
        std::memset(desc, 0, sizeof(*desc));
        desc->Width = swapchain_.extent().width;
        desc->Height = swapchain_.extent().height;
        desc->Format = vk_to_dxgi_format(swapchain_.format());
        desc->SampleDesc.Count = 1;
        desc->BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        desc->BufferCount = swapchain_.image_count();
        desc->Scaling = DXGI_SCALING_STRETCH;
        desc->SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        desc->AlphaMode = DXGI_ALPHA_MODE_IGNORE;
        desc->Flags = flags_;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE GetFullscreenDesc(DXGI_SWAP_CHAIN_FULLSCREEN_DESC* desc) override {
        if (!desc) return E_INVALIDARG;
        std::memset(desc, 0, sizeof(*desc));
        desc->Windowed = TRUE;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE GetHwnd(HWND* out) override { if (!out) return E_INVALIDARG; *out = hwnd_; return S_OK; }
    HRESULT STDMETHODCALLTYPE GetCoreWindow(REFIID, void** out) override {
        if (!out) return DXGI_ERROR_INVALID_CALL;
        *out = nullptr;
        return DXGI_ERROR_INVALID_CALL;
    }
    HRESULT STDMETHODCALLTYPE Present1(UINT sync, UINT flags, const DXGI_PRESENT_PARAMETERS*) override {
        return Present(sync, flags);
    }
    BOOL STDMETHODCALLTYPE IsTemporaryMonoSupported() override { return FALSE; }
    HRESULT STDMETHODCALLTYPE GetRestrictToOutput(IDXGIOutput** out) override {
        if (!out) return E_INVALIDARG;
        if (restrict_to_output_) {
            *out = restrict_to_output_;
            restrict_to_output_->AddRef();
            return S_OK;
        }
        *out = nullptr;
        return DXGI_ERROR_NOT_FOUND;
    }
    void set_restrict_to_output(IDXGIOutput* out) {
        if (restrict_to_output_) restrict_to_output_->Release();
        restrict_to_output_ = out;
        if (restrict_to_output_) restrict_to_output_->AddRef();
    }
    HRESULT STDMETHODCALLTYPE SetBackgroundColor(const DXGI_RGBA* color) override {
        if (!color) return E_INVALIDARG;
        background_color_ = *color;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE GetBackgroundColor(DXGI_RGBA* color) override {
        if (!color) return E_INVALIDARG;
        *color = background_color_;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE SetRotation(DXGI_MODE_ROTATION rotation) override {
        rotation_ = rotation;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE GetRotation(DXGI_MODE_ROTATION* rotation) override {
        if (!rotation) return E_INVALIDARG;
        *rotation = rotation_;
        return S_OK;
    }

    // IDXGISwapChain2
    HRESULT STDMETHODCALLTYPE SetSourceSize(UINT width, UINT height) override {
        // Host-measured: a region larger than either back-buffer dimension is refused and keeps the stored one;
        // zero is accepted. Present does not scale by it yet (documented difference).
        const VkExtent2D extent = swapchain_.extent();
        if (width > extent.width || height > extent.height) return DXGI_ERROR_INVALID_CALL;
        source_size_ = {width, height};
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE GetSourceSize(UINT* w, UINT* h) override {
        if (!w || !h) return DXGI_ERROR_INVALID_CALL;  // host-measured: both are required
        *w = source_size_.width;
        *h = source_size_.height;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE SetMaximumFrameLatency(UINT max_latency) override {
        if (!waitable_ || max_latency == 0 || max_latency > max_waitable_frame_latency)
            return DXGI_ERROR_INVALID_CALL;
        const UINT prev = max_frame_latency_.exchange(max_latency, std::memory_order_relaxed);
        if (max_latency > prev)
            ReleaseSemaphore(waitable_, static_cast<LONG>(max_latency - prev), nullptr);
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE GetMaximumFrameLatency(UINT* n) override {
        if (!n) return E_INVALIDARG;
        if (!waitable_) return DXGI_ERROR_INVALID_CALL;
        *n = max_frame_latency_.load(std::memory_order_relaxed);
        return S_OK;
    }
    HANDLE STDMETHODCALLTYPE GetFrameLatencyWaitableObject() override {
        if (!waitable_) return nullptr;
        HANDLE dup = nullptr;
        if (!DuplicateHandle(GetCurrentProcess(), waitable_, GetCurrentProcess(), &dup, 0, FALSE,
                             DUPLICATE_SAME_ACCESS))
            return nullptr;
        return dup;
    }
    // A matrix transform belongs to composition swapchains only (CreateSwapChainForComposition, E_NOTIMPL here).
    // Host DXGI answers DXGI_ERROR_INVALID_CALL to both methods on an HWND swapchain, with a matrix or with null
    // (Windows 11, RTX 5060 Ti, driver 621.14); every swapchain the layer creates gets that answer.
    HRESULT STDMETHODCALLTYPE SetMatrixTransform(const DXGI_MATRIX_3X2_F*) override { return DXGI_ERROR_INVALID_CALL; }
    HRESULT STDMETHODCALLTYPE GetMatrixTransform(DXGI_MATRIX_3X2_F*) override { return DXGI_ERROR_INVALID_CALL; }

    // IDXGISwapChain3
    UINT STDMETHODCALLTYPE GetCurrentBackBufferIndex() override { return swapchain_.current_back_buffer_index(); }
    HRESULT STDMETHODCALLTYPE CheckColorSpaceSupport(DXGI_COLOR_SPACE_TYPE, UINT* s) override {
        if (s) *s = DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE SetColorSpace1(DXGI_COLOR_SPACE_TYPE) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE ResizeBuffers1(UINT count, UINT w, UINT h, DXGI_FORMAT fmt, UINT flags,
                                             const UINT* node_masks, IUnknown* const* present_queues) override {
        return resize_buffers(count, w, h, fmt, flags, node_masks, present_queues);
    }

private:
    // DXGI: a zero width or height takes the whole size from the window's client area (host-measured).
    VkExtent2D size_or_client(UINT width, UINT height, VkExtent2D fallback) const {
        if (width && height) return {width, height};
        RECT rc{};
        if (GetClientRect(hwnd_, &rc) && rc.right > 0 && rc.bottom > 0)
            return {static_cast<std::uint32_t>(rc.right), static_cast<std::uint32_t>(rc.bottom)};
        return fallback;
    }

    void register_buffers() {
        const std::uint32_t count = swapchain_.image_count();
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width = swapchain_.extent().width;
        desc.Height = swapchain_.extent().height;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = vk_to_dxgi_format(swapchain_.format());
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        const VkImageLayout initial_layout =
            swapchain_.is_offscreen() ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        buffers_.reserve(count);
        for (std::uint32_t i = 0; i < count; ++i) {
            auto* bb = new BackBufferResource(desc);
            context_.add_resource(bb->resource_id(), swapchain_.resource_binding(i), initial_layout);
            buffers_.push_back(bb);
        }
    }

    void unregister_buffers() {
        for (auto* bb : buffers_) {
            context_.remove_resource(bb->resource_id());
            bb->Release();
        }
        buffers_.clear();
    }

    std::atomic<ULONG> refs_{1};
    PrivateDataStore private_data_;
    IDXGIFactory4* parent_ = nullptr;
    IUnknown* device_ = nullptr;
    leonida::VulkanContext& context_;
    leonida::VulkanSwapchain swapchain_;
    HWND hwnd_ = nullptr;
    IDXGIOutput* restrict_to_output_ = nullptr;
    UINT flags_ = 0;
    HANDLE waitable_ = nullptr;
    std::vector<BackBufferResource*> buffers_;
    std::atomic<UINT> present_count_{0};
    std::atomic<bool> statistics_disjoint_{true};
    std::atomic<LONGLONG> last_present_qpc_{0};
    std::atomic<UINT> max_frame_latency_{2};
    DXGI_RGBA background_color_{0, 0, 0, 0};
    DXGI_MODE_ROTATION rotation_ = DXGI_MODE_ROTATION_IDENTITY;
    VkExtent2D source_size_{0, 0};
};

struct ReservationState {
    std::atomic<UINT64> local{0};
    std::atomic<UINT64> non_local{0};
};

// Video-memory reservations are scoped per physical adapter in the process (matching native DXGI,
// where separate IDXGIAdapter3 handles opened for the same LUID observe each other's reservations).
std::shared_ptr<ReservationState> adapter_reservation_state(LUID luid, VkPhysicalDevice physical) {
    static std::mutex mutex;
    static std::map<std::tuple<std::uint32_t, std::int32_t, std::uintptr_t>, std::shared_ptr<ReservationState>> table;
    const bool has_luid = (luid.LowPart != 0 || luid.HighPart != 0);
    const auto key =
        has_luid ? std::make_tuple(std::uint32_t(luid.LowPart), std::int32_t(luid.HighPart), std::uintptr_t(0))
                 : std::make_tuple(std::uint32_t(0), std::int32_t(0), reinterpret_cast<std::uintptr_t>(physical));
    std::lock_guard lock(mutex);
    auto& entry = table[key];
    if (!entry) entry = std::make_shared<ReservationState>();
    return entry;
}

// On discrete GPUs without Resizable BAR, Vulkan exposes a second smaller DEVICE_LOCAL heap for the
// CPU-visible BAR1 aperture (typically 256 MiB) that aliases the same physical VRAM as the primary heap.
// Pick the largest DEVICE_LOCAL heap so DedicatedVideoMemory and LOCAL budget/usage match Windows DXGI
// instead of double-counting BAR1.
std::uint32_t primary_device_local_heap(const VkPhysicalDeviceMemoryProperties& mem) {
    std::uint32_t best = UINT32_MAX;
    VkDeviceSize best_size = 0;
    for (std::uint32_t i = 0; i < mem.memoryHeapCount; ++i) {
        if ((mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0) continue;
        if (best == UINT32_MAX || mem.memoryHeaps[i].size > best_size) {
            best = i;
            best_size = mem.memoryHeaps[i].size;
        }
    }
    return best;
}

// Snapshot of a Win32 monitor (or synthetic headless fallback) exposed through IDXGIOutput*.
// Defaults target a 1920x1080@60 Hz primary output; Win32 enumeration must also fill desktop_coords explicitly.
struct DisplayInfo {
    HMONITOR monitor = nullptr;
    WCHAR device_name[CCHDEVICENAME] = L"\\\\.\\DISPLAY1";  // matches DXGI_OUTPUT_DESC::DeviceName (32 WCHARs, NUL-terminated)
    RECT desktop_coords{0, 0, 1920, 1080};  // Default desktop coordinates
    DXGI_MODE_ROTATION rotation = DXGI_MODE_ROTATION_IDENTITY;
    UINT current_width = 1920;
    UINT current_height = 1080;
    UINT current_refresh_hz = 60;
    bool primary = true;
};

static_assert(sizeof(DisplayInfo{}.device_name) == sizeof(DXGI_OUTPUT_DESC{}.DeviceName));

// Maps Win32 DEVMODEW::dmDisplayOrientation (valid only when DM_DISPLAYORIENTATION is set in dmFields)
// to DXGI_MODE_ROTATION.
constexpr DXGI_MODE_ROTATION rotation_from_dm(const DEVMODEW& dm) noexcept {
    if ((dm.dmFields & DM_DISPLAYORIENTATION) == 0)  // dmDisplayOrientation is in a union and undefined without this flag
        return DXGI_MODE_ROTATION_IDENTITY;
    switch (dm.dmDisplayOrientation) {
    case DMDO_90: return DXGI_MODE_ROTATION_ROTATE90;
    case DMDO_180: return DXGI_MODE_ROTATION_ROTATE180;
    case DMDO_270: return DXGI_MODE_ROTATION_ROTATE270;
    default: return DXGI_MODE_ROTATION_IDENTITY;  // covers DMDO_DEFAULT and unexpected values
    }
}

BOOL CALLBACK collect_monitor_cb(HMONITOR hmon, HDC, LPRECT, LPARAM lparam) {
    auto* displays = reinterpret_cast<std::vector<DisplayInfo>*>(lparam);
    MONITORINFOEXW mi{};
    mi.cbSize = sizeof(mi);
    if (!GetMonitorInfoW(hmon, &mi)) return TRUE;

    DisplayInfo info{};
    info.monitor = hmon;
    info.desktop_coords = mi.rcMonitor;
    info.primary = (mi.dwFlags & MONITORINFOF_PRIMARY) != 0;
    // szDevice is CCHDEVICENAME (32) WCHARs; _TRUNCATE guarantees a trailing NUL even on unexpected input.
    wcsncpy_s(info.device_name, mi.szDevice, _TRUNCATE);

    DEVMODEW dm{};
    dm.dmSize = sizeof(dm);
    if (EnumDisplaySettingsW(mi.szDevice, ENUM_CURRENT_SETTINGS, &dm)) {
        if ((dm.dmFields & (DM_PELSWIDTH | DM_PELSHEIGHT)) == (DM_PELSWIDTH | DM_PELSHEIGHT) &&
            dm.dmPelsWidth > 0 && dm.dmPelsHeight > 0) {
            info.current_width = dm.dmPelsWidth;
            info.current_height = dm.dmPelsHeight;
        }
        // Win32 drivers may report 0 or 1 as "hardware default" rather than an actual Hz rate.
        if ((dm.dmFields & DM_DISPLAYFREQUENCY) != 0 && dm.dmDisplayFrequency > 1)
            info.current_refresh_hz = dm.dmDisplayFrequency;
        info.rotation = rotation_from_dm(dm);
    }

    displays->push_back(info);
    return TRUE;
}

// Enumerates attached Win32 displays with the primary monitor at index 0 (matching DXGI's EnumOutputs contract),
// falling back to a single 1920x1080@60 output in headless environments with no desktop monitors.
std::vector<DisplayInfo> enumerate_displays() {
    std::vector<DisplayInfo> displays;
    EnumDisplayMonitors(nullptr, nullptr, &collect_monitor_cb, reinterpret_cast<LPARAM>(&displays));
    if (displays.empty()) {
        displays.push_back(DisplayInfo{});
        return displays;
    }
    // Keep enumeration order among secondary monitors while ensuring primary is output 0.
    for (std::size_t i = 1; i < displays.size(); ++i) {
        if (displays[i].primary) {
            DisplayInfo primary_info = displays[i];
            for (std::size_t j = i; j > 0; --j) displays[j] = displays[j - 1];
            displays[0] = primary_info;
            break;
        }
    }
    return displays;
}

bool is_supported_display_format(DXGI_FORMAT fmt) noexcept {
    switch (fmt) {
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
    case DXGI_FORMAT_R10G10B10A2_UNORM:
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
        return true;
    default:
        return false;
    }
}

std::vector<DXGI_MODE_DESC> collect_modes(const DisplayInfo& info, DXGI_FORMAT fmt) {
    std::vector<DXGI_MODE_DESC> modes;
    if (!is_supported_display_format(fmt)) return modes;

    std::set<std::tuple<UINT, UINT, UINT>> seen;
    auto push_mode = [&](UINT w, UINT h, UINT hz) {
        if (!w || !h || !hz) return;
        if (!seen.emplace(w, h, hz).second) return;
        DXGI_MODE_DESC m{};
        m.Width = w;
        m.Height = h;
        m.RefreshRate = {hz, 1};
        m.Format = fmt;
        m.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_PROGRESSIVE;
        m.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;
        modes.push_back(m);
    };

    if (info.monitor) {
        // Bound the driver enumeration loop defensively in case a faulty driver never stops returning modes.
        for (DWORD i = 0; i < 4096; ++i) {
            DEVMODEW dm{};
            dm.dmSize = sizeof(dm);
            if (!EnumDisplaySettingsW(info.device_name, i, &dm)) break;
            if ((dm.dmFields & (DM_PELSWIDTH | DM_PELSHEIGHT)) != (DM_PELSWIDTH | DM_PELSHEIGHT) ||
                dm.dmPelsWidth == 0 || dm.dmPelsHeight == 0)
                continue;
            const UINT hz =
                ((dm.dmFields & DM_DISPLAYFREQUENCY) != 0 && dm.dmDisplayFrequency > 1) ? dm.dmDisplayFrequency
                                                                                        : info.current_refresh_hz;
            push_mode(dm.dmPelsWidth, dm.dmPelsHeight, hz);
        }
    }
    push_mode(info.current_width, info.current_height, info.current_refresh_hz);

    std::sort(modes.begin(), modes.end(), [](const DXGI_MODE_DESC& a, const DXGI_MODE_DESC& b) {
        return std::tie(a.Width, a.Height, a.RefreshRate.Numerator) <
               std::tie(b.Width, b.Height, b.RefreshRate.Numerator);
    });
    return modes;
}

constexpr DXGI_MODE_DESC1 to_mode_desc1(const DXGI_MODE_DESC& m) noexcept {
    return {m.Width, m.Height, m.RefreshRate, m.Format, m.ScanlineOrdering, m.Scaling, FALSE};
}

constexpr DXGI_MODE_DESC from_mode_desc1(const DXGI_MODE_DESC1& m) noexcept {
    return {m.Width, m.Height, m.RefreshRate, m.Format, m.ScanlineOrdering, m.Scaling};
}

template <class DescT, class ConvertFn>
HRESULT fill_mode_list(const DisplayInfo& info, DXGI_FORMAT fmt, UINT* num_modes, DescT* out, ConvertFn convert) {
    if (!num_modes) return DXGI_ERROR_INVALID_CALL;
    const std::vector<DXGI_MODE_DESC> modes = collect_modes(info, fmt);
    const UINT total = static_cast<UINT>(modes.size());
    if (!out) {
        *num_modes = total;
        return S_OK;
    }
    const UINT to_copy = std::min(*num_modes, total);
    for (UINT i = 0; i < to_copy; ++i) out[i] = convert(modes[i]);
    // Host DXGI writes the number of copied entries back into *pNumModes when returning MORE_DATA.
    *num_modes = to_copy;
    return (to_copy < total) ? DXGI_ERROR_MORE_DATA : S_OK;
}

HRESULT find_closest_mode(const DisplayInfo& info, const DXGI_MODE_DESC& req, IUnknown* concerned_device,
                          DXGI_MODE_DESC& out) {
    out = {};
    if (req.Format == DXGI_FORMAT_UNKNOWN && !concerned_device) return DXGI_ERROR_INVALID_CALL;
    const DXGI_FORMAT target_fmt = (req.Format == DXGI_FORMAT_UNKNOWN) ? DXGI_FORMAT_B8G8R8A8_UNORM : req.Format;
    const std::vector<DXGI_MODE_DESC> modes = collect_modes(info, target_fmt);
    if (modes.empty()) return DXGI_ERROR_NOT_FOUND;

    const UINT target_w = req.Width ? req.Width : info.current_width;
    const UINT target_h = req.Height ? req.Height : info.current_height;
    const bool has_refresh = (req.RefreshRate.Numerator > 0 && req.RefreshRate.Denominator > 0);
    const std::uint64_t target_mhz =
        has_refresh ? (std::uint64_t(req.RefreshRate.Numerator) * 1000ull) / req.RefreshRate.Denominator
                    : std::uint64_t(info.current_refresh_hz) * 1000ull;

    auto abs_diff = [](std::uint64_t a, std::uint64_t b) -> std::uint64_t { return a > b ? (a - b) : (b - a); };
    auto score = [&](const DXGI_MODE_DESC& m) {
        const std::uint64_t area_diff = abs_diff(std::uint64_t(m.Width) * m.Height, std::uint64_t(target_w) * target_h);
        const std::uint64_t dim_diff = abs_diff(m.Width, target_w) + abs_diff(m.Height, target_h);
        const std::uint64_t m_mhz =
            m.RefreshRate.Denominator ? (std::uint64_t(m.RefreshRate.Numerator) * 1000ull) / m.RefreshRate.Denominator : 0;
        const std::uint64_t hz_diff = abs_diff(m_mhz, target_mhz);
        return std::make_tuple(dim_diff, area_diff, hz_diff);
    };

    const DXGI_MODE_DESC* best = &modes[0];
    auto best_score = score(*best);
    for (std::size_t i = 1; i < modes.size(); ++i) {
        const auto s = score(modes[i]);
        if (s < best_score) {
            best = &modes[i];
            best_score = s;
        }
    }
    out = *best;
    return S_OK;
}

void wait_for_vblank() {
    using DwmFlushFn = HRESULT(WINAPI*)();
    static const DwmFlushFn dwm_flush = []() -> DwmFlushFn {
        // Resolve dynamically so dxgi_com.cpp needs no extra dwmapi.lib link entry in CMakeLists.txt.
        const HMODULE mod = LoadLibraryW(L"dwmapi.dll");
        return mod ? reinterpret_cast<DwmFlushFn>(GetProcAddress(mod, "DwmFlush")) : nullptr;
    }();
    if (dwm_flush && SUCCEEDED(dwm_flush())) return;
    Sleep(16);
}

// Win32/DXGI-backed IDXGIOutput6 reporting desktop monitor descriptors and display modes.
class Output final : public IDXGIOutput6 {
public:
    Output(IDXGIAdapter* parent, DisplayInfo info) : parent_(parent), info_(info) {
        if (parent_) parent_->AddRef();
    }

    ~Output() {
        if (parent_) parent_->Release();
    }

    // IUnknown
    ULONG STDMETHODCALLTYPE AddRef() override { return ++refs_; }
    ULONG STDMETHODCALLTYPE Release() override {
        const ULONG left = --refs_;
        if (!left) delete this;
        return left;
    }
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (riid == __uuidof(IUnknown) || riid == __uuidof(IDXGIObject) || riid == __uuidof(IDXGIOutput) ||
            riid == __uuidof(IDXGIOutput1) || riid == __uuidof(IDXGIOutput2) || riid == __uuidof(IDXGIOutput3) ||
            riid == __uuidof(IDXGIOutput4) || riid == __uuidof(IDXGIOutput5) || riid == __uuidof(IDXGIOutput6)) {
            *out = static_cast<IDXGIOutput6*>(this);
            AddRef();
            return S_OK;
        }
        return E_NOINTERFACE;
    }

    // IDXGIObject
    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID, UINT, const void*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID, const IUnknown*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID, UINT*, void*) override { return DXGI_ERROR_NOT_FOUND; }
    HRESULT STDMETHODCALLTYPE GetParent(REFIID riid, void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (!parent_) return E_NOINTERFACE;
        return parent_->QueryInterface(riid, out);
    }

    // IDXGIOutput
    HRESULT STDMETHODCALLTYPE GetDesc(DXGI_OUTPUT_DESC* desc) override {
        if (!desc) return E_INVALIDARG;
        fill_base_desc(*desc);
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE GetDisplayModeList(DXGI_FORMAT fmt, UINT, UINT* num_modes, DXGI_MODE_DESC* desc) override {
        return fill_mode_list(info_, fmt, num_modes, desc, [](const DXGI_MODE_DESC& m) { return m; });
    }
    HRESULT STDMETHODCALLTYPE FindClosestMatchingMode(const DXGI_MODE_DESC* mode_to_match, DXGI_MODE_DESC* closest,
                                                      IUnknown* concerned_device) override {
        if (!mode_to_match || !closest) return DXGI_ERROR_INVALID_CALL;
        return find_closest_mode(info_, *mode_to_match, concerned_device, *closest);
    }
    HRESULT STDMETHODCALLTYPE WaitForVBlank() override {
        wait_for_vblank();
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE TakeOwnership(IUnknown* device, BOOL) override {
        return device ? S_OK : E_INVALIDARG;
    }
    void STDMETHODCALLTYPE ReleaseOwnership() override {}
    HRESULT STDMETHODCALLTYPE GetGammaControlCapabilities(DXGI_GAMMA_CONTROL_CAPABILITIES* caps) override {
        if (caps) std::memset(caps, 0, sizeof(*caps));
        return DXGI_ERROR_INVALID_CALL;
    }
    HRESULT STDMETHODCALLTYPE SetGammaControl(const DXGI_GAMMA_CONTROL*) override { return DXGI_ERROR_INVALID_CALL; }
    HRESULT STDMETHODCALLTYPE GetGammaControl(DXGI_GAMMA_CONTROL*) override { return DXGI_ERROR_INVALID_CALL; }
    HRESULT STDMETHODCALLTYPE SetDisplaySurface(IDXGISurface*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetDisplaySurfaceData(IDXGISurface*) override { return DXGI_ERROR_INVALID_CALL; }
    HRESULT STDMETHODCALLTYPE GetFrameStatistics(DXGI_FRAME_STATISTICS* stats) override {
        if (!stats) return DXGI_ERROR_INVALID_CALL;
        std::memset(stats, 0, sizeof(*stats));
        LARGE_INTEGER qpc{};
        if (QueryPerformanceCounter(&qpc)) stats->SyncQPCTime = qpc;
        return S_OK;
    }

    // IDXGIOutput1
    HRESULT STDMETHODCALLTYPE GetDisplayModeList1(DXGI_FORMAT fmt, UINT, UINT* num_modes,
                                                  DXGI_MODE_DESC1* desc) override {
        return fill_mode_list(info_, fmt, num_modes, desc, to_mode_desc1);
    }
    HRESULT STDMETHODCALLTYPE FindClosestMatchingMode1(const DXGI_MODE_DESC1* mode_to_match, DXGI_MODE_DESC1* closest,
                                                       IUnknown* concerned_device) override {
        if (!mode_to_match || !closest) return DXGI_ERROR_INVALID_CALL;
        DXGI_MODE_DESC best{};
        const HRESULT hr = find_closest_mode(info_, from_mode_desc1(*mode_to_match), concerned_device, best);
        *closest = SUCCEEDED(hr) ? to_mode_desc1(best) : DXGI_MODE_DESC1{};
        return hr;
    }
    HRESULT STDMETHODCALLTYPE GetDisplaySurfaceData1(IDXGIResource*) override { return DXGI_ERROR_INVALID_CALL; }
    HRESULT STDMETHODCALLTYPE DuplicateOutput(IUnknown* device, IDXGIOutputDuplication** dup) override {
        if (dup) *dup = nullptr;
        if (!device || !dup) return E_INVALIDARG;
        return DXGI_ERROR_UNSUPPORTED;
    }

    // IDXGIOutput2
    BOOL STDMETHODCALLTYPE SupportsOverlays() override { return FALSE; }

    // IDXGIOutput3
    HRESULT STDMETHODCALLTYPE CheckOverlaySupport(DXGI_FORMAT, IUnknown* device, UINT* flags) override {
        if (flags) *flags = 0;
        return (device && flags) ? S_OK : E_INVALIDARG;
    }

    // IDXGIOutput4
    HRESULT STDMETHODCALLTYPE CheckOverlayColorSpaceSupport(DXGI_FORMAT, DXGI_COLOR_SPACE_TYPE, IUnknown* device,
                                                            UINT* flags) override {
        if (flags) *flags = 0;
        return (device && flags) ? S_OK : E_INVALIDARG;
    }

    // IDXGIOutput5
    HRESULT STDMETHODCALLTYPE DuplicateOutput1(IUnknown* device, UINT, UINT, const DXGI_FORMAT*,
                                               IDXGIOutputDuplication** dup) override {
        if (dup) *dup = nullptr;
        if (!device || !dup) return E_INVALIDARG;
        return DXGI_ERROR_UNSUPPORTED;
    }

    // IDXGIOutput6
    HRESULT STDMETHODCALLTYPE GetDesc1(DXGI_OUTPUT_DESC1* desc) override {
        if (!desc) return E_INVALIDARG;
        std::memset(desc, 0, sizeof(*desc));
        DXGI_OUTPUT_DESC base{};
        fill_base_desc(base);
        std::memcpy(desc->DeviceName, base.DeviceName, sizeof(desc->DeviceName));
        desc->DesktopCoordinates = base.DesktopCoordinates;
        desc->AttachedToDesktop = base.AttachedToDesktop;
        desc->Rotation = base.Rotation;
        desc->Monitor = base.Monitor;
        desc->BitsPerColor = 8;
        desc->ColorSpace = DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
        // Standard sRGB / ITU-R BT.709 primaries and baseline SDR luminance.
        desc->RedPrimary[0] = 0.64f;
        desc->RedPrimary[1] = 0.33f;
        desc->GreenPrimary[0] = 0.30f;
        desc->GreenPrimary[1] = 0.60f;
        desc->BluePrimary[0] = 0.15f;
        desc->BluePrimary[1] = 0.06f;
        desc->WhitePoint[0] = 0.3127f;
        desc->WhitePoint[1] = 0.3290f;
        desc->MinLuminance = 0.0f;
        desc->MaxLuminance = 80.0f;
        desc->MaxFullFrameLuminance = 80.0f;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE CheckHardwareCompositionSupport(UINT* flags) override {
        if (!flags) return E_INVALIDARG;
        *flags = 0;
        return S_OK;
    }

private:
    void fill_base_desc(DXGI_OUTPUT_DESC& desc) const noexcept {
        std::memset(&desc, 0, sizeof(desc));
        std::memcpy(desc.DeviceName, info_.device_name, sizeof(desc.DeviceName));
        desc.DesktopCoordinates = info_.desktop_coords;
        desc.AttachedToDesktop = TRUE;
        desc.Rotation = info_.rotation;
        desc.Monitor = info_.monitor;
    }

    std::atomic<ULONG> refs_{1};
    IDXGIAdapter* parent_ = nullptr;
    DisplayInfo info_{};
};


// Vulkan-backed IDXGIAdapter3 reporting the physical device's name, IDs, LUID (matching
// ID3D12Device::GetAdapterLuid), memory heap sizes, and video memory budget info.

// The adapter's user-mode driver version, as DXGI's CheckInterfaceSupport(IDXGIDevice) reports it: what
// D3DKMTQueryAdapterInfo(KMTQAITYPE_UMD_DRIVER_VERSION) gives for the adapter's LUID (measured equal to host DXGI on
// an RTX 5060 Ti, 0x0020000000100842 for driver 621.14, and on WARP). gdi32's entry points are looked up at run
// time and the host dxgi.dll isn't used; 0 when they are missing (Wine) or the query fails.
LARGE_INTEGER umd_driver_version(LUID luid) {
    LARGE_INTEGER version{};
    if (!luid.LowPart && !luid.HighPart) return version;
    HMODULE gdi32 = GetModuleHandleW(L"gdi32.dll");
    if (!gdi32) gdi32 = LoadLibraryW(L"gdi32.dll");
    if (!gdi32) return version;
    const auto open = reinterpret_cast<decltype(&D3DKMTOpenAdapterFromLuid)>(
        GetProcAddress(gdi32, "D3DKMTOpenAdapterFromLuid"));
    const auto query = reinterpret_cast<decltype(&D3DKMTQueryAdapterInfo)>(
        GetProcAddress(gdi32, "D3DKMTQueryAdapterInfo"));
    const auto close = reinterpret_cast<decltype(&D3DKMTCloseAdapter)>(GetProcAddress(gdi32, "D3DKMTCloseAdapter"));
    if (!open || !query || !close) return version;
    D3DKMT_OPENADAPTERFROMLUID adapter{};
    adapter.AdapterLuid = luid;
    if (open(&adapter) != 0) return version;
    D3DKMT_UMD_DRIVER_VERSION umd{};
    D3DKMT_QUERYADAPTERINFO info{};
    info.hAdapter = adapter.hAdapter;
    info.Type = KMTQAITYPE_UMD_DRIVER_VERSION;
    info.pPrivateDriverData = &umd;
    info.PrivateDriverDataSize = sizeof(umd);
    if (query(&info) == 0) version = umd.DriverVersion;
    D3DKMT_CLOSEADAPTER closing{adapter.hAdapter};
    close(&closing);
    return version;
}

class Adapter final : public IDXGIAdapter3 {
public:
    Adapter(IDXGIFactory4* parent, PFN_vkGetInstanceProcAddr gipa, VkInstance instance, VkPhysicalDevice physical)
        : parent_(parent), gipa_(gipa), instance_(instance), physical_(physical) {
        if (parent_) parent_->AddRef();
        populate_desc();
        umd_version_ = umd_driver_version(desc_.AdapterLuid);
        reservations_ = adapter_reservation_state(desc_.AdapterLuid, physical_);
    }

    ~Adapter() {
        if (parent_) parent_->Release();
    }

    LUID luid() const { return desc_.AdapterLuid; }

    // IUnknown
    ULONG STDMETHODCALLTYPE AddRef() override { return ++refs_; }
    ULONG STDMETHODCALLTYPE Release() override {
        const ULONG left = --refs_;
        if (!left) delete this;
        return left;
    }
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (riid == __uuidof(IUnknown) || riid == __uuidof(IDXGIObject) || riid == __uuidof(IDXGIAdapter) ||
            riid == __uuidof(IDXGIAdapter1) || riid == __uuidof(IDXGIAdapter2) || riid == __uuidof(IDXGIAdapter3)) {
            *out = static_cast<IDXGIAdapter3*>(this);
            AddRef();
            return S_OK;
        }
        return E_NOINTERFACE;
    }

    // IDXGIObject
    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID guid, UINT size, const void* data) override {
        return private_data_.set(guid, size, data);
    }
    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID guid, const IUnknown* object) override {
        return private_data_.set_interface(guid, object);
    }
    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID guid, UINT* size, void* data) override {
        return private_data_.get(guid, size, data);
    }
    HRESULT STDMETHODCALLTYPE GetParent(REFIID riid, void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (!parent_) return E_NOINTERFACE;
        return parent_->QueryInterface(riid, out);
    }

    // IDXGIAdapter
    HRESULT STDMETHODCALLTYPE EnumOutputs(UINT index, IDXGIOutput** out) override {
        if (!out) return E_INVALIDARG;
        *out = nullptr;
        const std::vector<DisplayInfo> displays = enumerate_displays();
        if (index >= displays.size()) return DXGI_ERROR_NOT_FOUND;
        *out = new Output(static_cast<IDXGIAdapter*>(this), displays[index]);
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE GetDesc(DXGI_ADAPTER_DESC* desc) override {
        if (!desc) return E_INVALIDARG;
        std::memcpy(desc->Description, desc_.Description, sizeof(desc->Description));
        desc->VendorId = desc_.VendorId;
        desc->DeviceId = desc_.DeviceId;
        desc->SubSysId = desc_.SubSysId;
        desc->Revision = desc_.Revision;
        desc->DedicatedVideoMemory = desc_.DedicatedVideoMemory;
        desc->DedicatedSystemMemory = desc_.DedicatedSystemMemory;
        desc->SharedSystemMemory = desc_.SharedSystemMemory;
        desc->AdapterLuid = desc_.AdapterLuid;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE CheckInterfaceSupport(REFGUID interface_name, LARGE_INTEGER* umd_version) override {
        if (interface_name == __uuidof(IDXGIDevice)) {
            if (umd_version) *umd_version = umd_version_;
            return S_OK;
        }
        return DXGI_ERROR_UNSUPPORTED;
    }

    // IDXGIAdapter1
    HRESULT STDMETHODCALLTYPE GetDesc1(DXGI_ADAPTER_DESC1* desc) override {
        if (!desc) return E_INVALIDARG;
        std::memcpy(desc->Description, desc_.Description, sizeof(desc->Description));
        desc->VendorId = desc_.VendorId;
        desc->DeviceId = desc_.DeviceId;
        desc->SubSysId = desc_.SubSysId;
        desc->Revision = desc_.Revision;
        desc->DedicatedVideoMemory = desc_.DedicatedVideoMemory;
        desc->DedicatedSystemMemory = desc_.DedicatedSystemMemory;
        desc->SharedSystemMemory = desc_.SharedSystemMemory;
        desc->AdapterLuid = desc_.AdapterLuid;
        desc->Flags = desc_.Flags;
        return S_OK;
    }

    // IDXGIAdapter2
    HRESULT STDMETHODCALLTYPE GetDesc2(DXGI_ADAPTER_DESC2* desc) override {
        if (!desc) return E_INVALIDARG;
        *desc = desc_;
        return S_OK;
    }

    // IDXGIAdapter3
    HRESULT STDMETHODCALLTYPE RegisterHardwareContentProtectionTeardownStatusEvent(HANDLE, DWORD* cookie) override {
        if (cookie) *cookie = 0;
        return E_NOTIMPL;
    }
    void STDMETHODCALLTYPE UnregisterHardwareContentProtectionTeardownStatus(DWORD) override {}
    HRESULT STDMETHODCALLTYPE QueryVideoMemoryInfo(UINT node_index, DXGI_MEMORY_SEGMENT_GROUP group,
                                                   DXGI_QUERY_VIDEO_MEMORY_INFO* info) override {
        if (node_index != 0 || !info) return E_INVALIDARG;
        if (group != DXGI_MEMORY_SEGMENT_GROUP_LOCAL && group != DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL)
            return E_INVALIDARG;
        std::memset(info, 0, sizeof(*info));
        const bool want_local = (group == DXGI_MEMORY_SEGMENT_GROUP_LOCAL);
        if (has_memory_budget_ && gipa_ && instance_ && physical_) {
            if (const auto get_mem2 = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties2>(
                    gipa_(instance_, "vkGetPhysicalDeviceMemoryProperties2"))) {
                VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
                VkPhysicalDeviceMemoryProperties2 mem2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, &budget};
                get_mem2(physical_, &mem2);
                const auto& mem = mem2.memoryProperties;
                if (want_local) {
                    const std::uint32_t local_heap = primary_device_local_heap(mem);
                    if (local_heap != UINT32_MAX) {
                        info->Budget = budget.heapBudget[local_heap];
                        info->CurrentUsage = budget.heapUsage[local_heap];
                    }
                } else {
                    for (std::uint32_t i = 0; i < mem.memoryHeapCount; ++i) {
                        if ((mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0) {
                            info->Budget += budget.heapBudget[i];
                            info->CurrentUsage += budget.heapUsage[i];
                        }
                    }
                }
            }
        }
        if (info->Budget == 0) {
            info->Budget = want_local ? desc_.DedicatedVideoMemory : desc_.SharedSystemMemory;
        }
        info->AvailableForReservation = info->Budget / 2;
        info->CurrentReservation = want_local ? reservations_->local.load(std::memory_order_relaxed)
                                              : reservations_->non_local.load(std::memory_order_relaxed);
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE SetVideoMemoryReservation(UINT node_index, DXGI_MEMORY_SEGMENT_GROUP group,
                                                        UINT64 reservation) override {
        if (node_index != 0) return E_INVALIDARG;
        if (group == DXGI_MEMORY_SEGMENT_GROUP_LOCAL) {
            reservations_->local.store(reservation, std::memory_order_relaxed);
            return S_OK;
        }
        if (group == DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL) {
            reservations_->non_local.store(reservation, std::memory_order_relaxed);
            return S_OK;
        }
        return E_INVALIDARG;
    }
    // Microsoft documents only "an HRESULT success or error code" and a cookie for
    // UnregisterVideoMemoryBudgetChangeNotification. The budget here is recomputed on every QueryVideoMemoryInfo and
    // never changes by itself, so the event is registered and never signalled.
    HRESULT STDMETHODCALLTYPE RegisterVideoMemoryBudgetChangeNotificationEvent(HANDLE event, DWORD* cookie) override {
        if (!event || !cookie) return E_INVALIDARG;
        *cookie = next_budget_cookie_.fetch_add(1, std::memory_order_relaxed);
        return S_OK;
    }
    void STDMETHODCALLTYPE UnregisterVideoMemoryBudgetChangeNotification(DWORD) override {}

private:
    void populate_desc() {
        std::memset(&desc_, 0, sizeof(desc_));
        desc_.GraphicsPreemptionGranularity = DXGI_GRAPHICS_PREEMPTION_DMA_BUFFER_BOUNDARY;
        desc_.ComputePreemptionGranularity = DXGI_COMPUTE_PREEMPTION_DMA_BUFFER_BOUNDARY;
        if (!gipa_ || !instance_ || !physical_) return;

        VkPhysicalDeviceProperties props{};
        bool got_props = false;
        if (const auto get_props2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
                gipa_(instance_, "vkGetPhysicalDeviceProperties2"))) {
            VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
            VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id};
            get_props2(physical_, &props2);
            props = props2.properties;
            got_props = true;
            static_assert(sizeof(LUID) == VK_LUID_SIZE);
            if (id.deviceLUIDValid) std::memcpy(&desc_.AdapterLuid, id.deviceLUID, sizeof(LUID));
        } else if (const auto get_props = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
                       gipa_(instance_, "vkGetPhysicalDeviceProperties"))) {
            get_props(physical_, &props);
            got_props = true;
        }
        if (got_props) {
            MultiByteToWideChar(CP_UTF8, 0, props.deviceName, -1, desc_.Description,
                                static_cast<int>(std::size(desc_.Description) - 1));
            desc_.Description[std::size(desc_.Description) - 1] = L'\0';
            desc_.VendorId = props.vendorID;
            desc_.DeviceId = props.deviceID;
            desc_.Revision = props.driverVersion;
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) desc_.Flags = DXGI_ADAPTER_FLAG_SOFTWARE;
        }

        if (const auto get_mem = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
                gipa_(instance_, "vkGetPhysicalDeviceMemoryProperties"))) {
            VkPhysicalDeviceMemoryProperties mem{};
            get_mem(physical_, &mem);
            const std::uint32_t local_heap = primary_device_local_heap(mem);
            const std::uint64_t dedicated = (local_heap != UINT32_MAX) ? mem.memoryHeaps[local_heap].size : 0;
            std::uint64_t shared = 0;
            for (std::uint32_t i = 0; i < mem.memoryHeapCount; ++i) {
                if ((mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0)
                    shared += mem.memoryHeaps[i].size;
            }
            desc_.DedicatedVideoMemory = static_cast<SIZE_T>(dedicated);
            desc_.SharedSystemMemory = static_cast<SIZE_T>(shared);
        }

        if (const auto enum_exts = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
                gipa_(instance_, "vkEnumerateDeviceExtensionProperties"))) {
            std::uint32_t count = 0;
            if (enum_exts(physical_, nullptr, &count, nullptr) == VK_SUCCESS && count > 0) {
                std::vector<VkExtensionProperties> exts(count);
                if (enum_exts(physical_, nullptr, &count, exts.data()) == VK_SUCCESS) {
                    for (const auto& e : exts) {
                        if (std::strcmp(e.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0) {
                            has_memory_budget_ = true;
                            break;
                        }
                    }
                }
            }
        }
    }

    std::atomic<DWORD> next_budget_cookie_{1};
    std::atomic<ULONG> refs_{1};
    PrivateDataStore private_data_;
    IDXGIFactory4* parent_ = nullptr;
    PFN_vkGetInstanceProcAddr gipa_ = nullptr;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    DXGI_ADAPTER_DESC2 desc_{};
    LARGE_INTEGER umd_version_{};  // CheckInterfaceSupport(IDXGIDevice): the adapter's user-mode driver version
    bool has_memory_budget_ = false;
    std::shared_ptr<ReservationState> reservations_;
};

class Factory final : public IDXGIFactory7 {
public:
    Factory(leonida::VulkanContext& context, PFN_vkGetInstanceProcAddr gipa, VkInstance instance,
            VkPhysicalDevice physical, PFN_vkGetDeviceProcAddr gdpa, VkQueue queue, std::uint32_t queue_family)
        : context_(context), gipa_(gipa), instance_(instance), physical_(physical), gdpa_(gdpa),
          queue_(queue), queue_family_(queue_family) {}

    ULONG STDMETHODCALLTYPE AddRef() override { return ++refs_; }
    ULONG STDMETHODCALLTYPE Release() override {
        const ULONG left = --refs_;
        if (!left) delete this;
        return left;
    }
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (riid == __uuidof(IUnknown) || riid == __uuidof(IDXGIObject) || riid == __uuidof(IDXGIFactory) ||
            riid == __uuidof(IDXGIFactory1) || riid == __uuidof(IDXGIFactory2) ||
            riid == __uuidof(IDXGIFactory3) || riid == __uuidof(IDXGIFactory4) ||
            riid == __uuidof(IDXGIFactory5) || riid == __uuidof(IDXGIFactory6) ||
            riid == __uuidof(IDXGIFactory7)) {
            *out = static_cast<IDXGIFactory7*>(this);
            AddRef();
            return S_OK;
        }
        return E_NOINTERFACE;
    }

    // IDXGIObject
    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID guid, UINT size, const void* data) override {
        return private_data_.set(guid, size, data);
    }
    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID guid, const IUnknown* object) override {
        return private_data_.set_interface(guid, object);
    }
    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID guid, UINT* size, void* data) override {
        return private_data_.get(guid, size, data);
    }
    HRESULT STDMETHODCALLTYPE GetParent(REFIID, void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        return E_NOINTERFACE;
    }

    // IDXGIFactory
    HRESULT STDMETHODCALLTYPE EnumAdapters(UINT index, IDXGIAdapter** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (index != 0) return DXGI_ERROR_NOT_FOUND;
        auto* adapter = new Adapter(this, gipa_, instance_, physical_);
        *out = static_cast<IDXGIAdapter*>(adapter);
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE MakeWindowAssociation(HWND hwnd, UINT flags) override {
        // Host-measured: only the three documented bits are accepted; any other bit is DXGI_ERROR_INVALID_CALL.
        constexpr UINT known = DXGI_MWA_NO_WINDOW_CHANGES | DXGI_MWA_NO_ALT_ENTER | DXGI_MWA_NO_PRINT_SCREEN;
        if (flags & ~known) return DXGI_ERROR_INVALID_CALL;
        assoc_hwnd_ = hwnd;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE GetWindowAssociation(HWND* hwnd) override {
        if (!hwnd) return E_INVALIDARG;
        *hwnd = assoc_hwnd_;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE CreateSwapChain(IUnknown* device, DXGI_SWAP_CHAIN_DESC* desc, IDXGISwapChain** out) override {
        if (!out || !device || !desc) return DXGI_ERROR_INVALID_CALL;
        *out = nullptr;
        DXGI_SWAP_CHAIN_DESC1 desc1{};
        desc1.Width = desc->BufferDesc.Width;
        desc1.Height = desc->BufferDesc.Height;
        desc1.Format = desc->BufferDesc.Format;
        desc1.SampleDesc = desc->SampleDesc;
        desc1.BufferUsage = desc->BufferUsage;
        desc1.BufferCount = desc->BufferCount;
        desc1.Scaling = DXGI_SCALING_STRETCH;
        desc1.SwapEffect = desc->SwapEffect;
        desc1.Flags = desc->Flags;
        DXGI_SWAP_CHAIN_FULLSCREEN_DESC fs_desc{};
        fs_desc.Windowed = desc->Windowed;
        fs_desc.RefreshRate = desc->BufferDesc.RefreshRate;
        fs_desc.Scaling = desc->BufferDesc.Scaling;
        fs_desc.ScanlineOrdering = desc->BufferDesc.ScanlineOrdering;
        IDXGISwapChain1* sc1 = nullptr;
        const HRESULT hr = CreateSwapChainForHwnd(device, desc->OutputWindow, &desc1, &fs_desc, nullptr, &sc1);
        if (SUCCEEDED(hr) && sc1) {
            *out = static_cast<IDXGISwapChain*>(sc1);
            if (!desc->Windowed) {
                // When exclusive fullscreen is unavailable, native DXGI creates the
                // windowed chain and returns DXGI_STATUS_OCCLUDED.
                return DXGI_STATUS_OCCLUDED;
            }
        }
        return hr;
    }
    HRESULT STDMETHODCALLTYPE CreateSoftwareAdapter(HMODULE, IDXGIAdapter**) override { return E_NOTIMPL; }

    // IDXGIFactory1
    HRESULT STDMETHODCALLTYPE EnumAdapters1(UINT index, IDXGIAdapter1** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (index != 0) return DXGI_ERROR_NOT_FOUND;
        auto* adapter = new Adapter(this, gipa_, instance_, physical_);
        *out = static_cast<IDXGIAdapter1*>(adapter);
        return S_OK;
    }
    BOOL STDMETHODCALLTYPE IsCurrent() override { return TRUE; }

    // IDXGIFactory2
    BOOL STDMETHODCALLTYPE IsWindowedStereoEnabled() override { return FALSE; }
    HRESULT STDMETHODCALLTYPE CreateSwapChainForHwnd(IUnknown* device, HWND hwnd, const DXGI_SWAP_CHAIN_DESC1* desc,
                                                     const DXGI_SWAP_CHAIN_FULLSCREEN_DESC*, IDXGIOutput* restrict_to_output,
                                                     IDXGISwapChain1** out) override {
        if (out) *out = nullptr;
        if (const HRESULT hr = check_hwnd_swap_chain(device, hwnd, desc, out); FAILED(hr)) return hr;
        return create_swap_chain(device, hwnd, desc, restrict_to_output, out);
    }

    // What the host runtime's DXGI refuses for a D3D12 swap chain on a window (dxgi_swapchain_test compares these,
    // RTX 5070 / 621.14): a null queue, window, description or output, an ID3D12Device in place of the queue, the
    // bitblt effects, a scaling or alpha mode a window can't present, a flag outside the ones a D3D12 window swap chain
    // takes (MODE_SWITCH, DISPLAY_ONLY, FRAME_LATENCY_WAITABLE_OBJECT, ALLOW_TEARING, the holographic restriction and
    // 0x2000), a shared, read-only, discard-on-present or unordered-access buffer usage, MSAA, a buffer count outside
    // 2..16 and a format the swap chain can't take are DXGI_ERROR_INVALID_CALL; a side above 16384 is E_INVALIDARG,
    // only when nothing else is wrong.
    static HRESULT check_hwnd_swap_chain(IUnknown* device, HWND hwnd, const DXGI_SWAP_CHAIN_DESC1* desc,
                                         IDXGISwapChain1** out) {
        if (!device || !hwnd || !desc || !out) return DXGI_ERROR_INVALID_CALL;
        ID3D12CommandQueue* queue = nullptr;
        if (FAILED(device->QueryInterface(IID_PPV_ARGS(&queue))) || !queue) return DXGI_ERROR_INVALID_CALL;
        queue->Release();
        if (desc->SwapEffect != DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL && desc->SwapEffect != DXGI_SWAP_EFFECT_FLIP_DISCARD)
            return DXGI_ERROR_INVALID_CALL;
        if (desc->Scaling != DXGI_SCALING_STRETCH && desc->Scaling != DXGI_SCALING_NONE) return DXGI_ERROR_INVALID_CALL;
        if (desc->AlphaMode != DXGI_ALPHA_MODE_UNSPECIFIED && desc->AlphaMode != DXGI_ALPHA_MODE_IGNORE)
            return DXGI_ERROR_INVALID_CALL;
        constexpr UINT kWindowFlags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH | DXGI_SWAP_CHAIN_FLAG_DISPLAY_ONLY |
                                      DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT |
                                      DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING | 0x1000u | 0x2000u;
        if (desc->Flags & ~kWindowFlags) return DXGI_ERROR_INVALID_CALL;
        constexpr DXGI_USAGE kRefusedUsage =
            DXGI_USAGE_SHARED | DXGI_USAGE_READ_ONLY | DXGI_USAGE_DISCARD_ON_PRESENT | DXGI_USAGE_UNORDERED_ACCESS;
        if (desc->BufferUsage & kRefusedUsage) return DXGI_ERROR_INVALID_CALL;
        // The host checks these before the size: MSAA 4 or one buffer at width 16385 is still INVALID_CALL.
        if (desc->SampleDesc.Count != 1 || desc->SampleDesc.Quality != 0) return DXGI_ERROR_INVALID_CALL;
        if (!valid_flip_buffer_count(desc->BufferCount)) return DXGI_ERROR_INVALID_CALL;
        if (dxgi_to_vk_format(desc->Format) == VK_FORMAT_UNDEFINED) return DXGI_ERROR_INVALID_CALL;
        if (desc->Width > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION || desc->Height > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION)
            return E_INVALIDARG;
        return S_OK;
    }

    // The swap chain itself, for a window or (through its HWND) a CoreWindow. The CoreWindow path keeps only these
    // checks: the host's rules for it (ASPECT_RATIO_STRETCH, for one) can't be compared without a real CoreWindow.
    HRESULT create_swap_chain(IUnknown* device, HWND hwnd, const DXGI_SWAP_CHAIN_DESC1* desc,
                              IDXGIOutput* restrict_to_output, IDXGISwapChain1** out) {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (!hwnd || !desc) return E_INVALIDARG;
        if (desc->SampleDesc.Count != 1 || desc->SampleDesc.Quality != 0) return DXGI_ERROR_INVALID_CALL;
        if (!valid_flip_buffer_count(desc->BufferCount)) return DXGI_ERROR_INVALID_CALL;
        const VkFormat vk_fmt = dxgi_to_vk_format(desc->Format);
        if (vk_fmt == VK_FORMAT_UNDEFINED) return DXGI_ERROR_INVALID_CALL;
        auto* sc = new SwapChain(this, device, context_, gipa_, instance_, physical_, gdpa_, queue_, queue_family_);
        if (restrict_to_output) sc->set_restrict_to_output(restrict_to_output);
        if (!sc->init(hwnd, *desc, vk_fmt)) {
            sc->Release();
            return E_FAIL;
        }
        *out = static_cast<IDXGISwapChain1*>(sc);
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE CreateSwapChainForCoreWindow(IUnknown* device, IUnknown* window,
                                                           const DXGI_SWAP_CHAIN_DESC1* desc,
                                                           IDXGIOutput* restrict_to_output,
                                                           IDXGISwapChain1** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (!device || !window || !desc) return E_INVALIDARG;
        // UWP / Game Core Scarlett titles present to an ICoreWindow. Extract the underlying
        // Win32 HWND via ICoreWindowInterop.
        static constexpr GUID kIID_ICoreWindowInterop = {
            0x45D64A29, 0xA63E, 0x4CB6, {0xB4, 0x98, 0x57, 0x81, 0xD2, 0x98, 0xCB, 0x4F}};
        struct ICoreWindowInterop : public IUnknown {
            virtual HRESULT STDMETHODCALLTYPE get_WindowHandle(HWND* hwnd) = 0;
            virtual HRESULT STDMETHODCALLTYPE put_MessageHandled(unsigned char value) = 0;
        };
        ICoreWindowInterop* interop = nullptr;
        if (FAILED(window->QueryInterface(kIID_ICoreWindowInterop, reinterpret_cast<void**>(&interop))) || !interop) {
            return DXGI_ERROR_INVALID_CALL;
        }
        HWND hwnd = nullptr;
        const HRESULT hr = interop->get_WindowHandle(&hwnd);
        interop->Release();
        if (FAILED(hr) || !hwnd) return DXGI_ERROR_INVALID_CALL;
        return create_swap_chain(device, hwnd, desc, restrict_to_output, out);
    }
    HRESULT STDMETHODCALLTYPE GetSharedResourceAdapterLuid(HANDLE, LUID*) override { return E_NOTIMPL; }
    // Microsoft documents S_OK with a cookie for UnregisterStereoStatus (E_NOTIMPL is only the answer of the
    // Windows 7 Platform Update). Stereo is never enabled here, so the window or event is never notified.
    HRESULT STDMETHODCALLTYPE RegisterStereoStatusWindow(HWND hwnd, UINT msg, DWORD* cookie) override {
        (void)msg;
        if (!hwnd || !cookie) return DXGI_ERROR_INVALID_CALL;
        *cookie = next_stereo_cookie_.fetch_add(1, std::memory_order_relaxed);
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE RegisterStereoStatusEvent(HANDLE event, DWORD* cookie) override {
        if (!event || !cookie) return DXGI_ERROR_INVALID_CALL;
        *cookie = next_stereo_cookie_.fetch_add(1, std::memory_order_relaxed);
        return S_OK;
    }
    void STDMETHODCALLTYPE UnregisterStereoStatus(DWORD) override {}
    HRESULT STDMETHODCALLTYPE RegisterOcclusionStatusWindow(HWND hwnd, UINT msg, DWORD* cookie) override {
        (void)msg;
        if (!hwnd || !cookie) return DXGI_ERROR_INVALID_CALL;
        *cookie = next_occlusion_cookie_.fetch_add(1, std::memory_order_relaxed);
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE RegisterOcclusionStatusEvent(HANDLE event, DWORD* cookie) override {
        if (!event || !cookie) return DXGI_ERROR_INVALID_CALL;
        *cookie = next_occlusion_cookie_.fetch_add(1, std::memory_order_relaxed);
        return S_OK;
    }
    void STDMETHODCALLTYPE UnregisterOcclusionStatus(DWORD) override {}
    HRESULT STDMETHODCALLTYPE CreateSwapChainForComposition(IUnknown*, const DXGI_SWAP_CHAIN_DESC1*, IDXGIOutput*,
                                                            IDXGISwapChain1**) override {
        return E_NOTIMPL;
    }

    // IDXGIFactory3
    UINT STDMETHODCALLTYPE GetCreationFlags() override { return 0; }

    // IDXGIFactory4
    HRESULT STDMETHODCALLTYPE EnumAdapterByLuid(LUID luid, REFIID riid, void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        auto* adapter = new Adapter(this, gipa_, instance_, physical_);
        const LUID actual = adapter->luid();
        if (actual.LowPart != luid.LowPart || actual.HighPart != luid.HighPart) {
            adapter->Release();
            return DXGI_ERROR_NOT_FOUND;
        }
        const HRESULT hr = adapter->QueryInterface(riid, out);
        adapter->Release();
        return hr;
    }
    HRESULT STDMETHODCALLTYPE EnumWarpAdapter(REFIID, void**) override { return DXGI_ERROR_NOT_FOUND; }

    // IDXGIFactory5
    HRESULT STDMETHODCALLTYPE CheckFeatureSupport(DXGI_FEATURE feature, void* data, UINT size) override {
        if (!data) return E_INVALIDARG;
        if (feature == DXGI_FEATURE_PRESENT_ALLOW_TEARING) {
            if (size != sizeof(BOOL)) return E_INVALIDARG;
            std::call_once(tearing_once_, [this] { tearing_supported_ = probe_immediate_present_mode(); });
            *static_cast<BOOL*>(data) = tearing_supported_ ? TRUE : FALSE;
            return S_OK;
        }
        return E_INVALIDARG;
    }

    // IDXGIFactory6. The factory has one adapter, so every preference orders the same list. Measured on the host's
    // dxgi.dll (Windows 11, RTX 3070): a preference outside the enum is accepted, a null `out` is
    // DXGI_ERROR_INVALID_CALL, and an index past the end is DXGI_ERROR_NOT_FOUND with *out cleared.
    HRESULT STDMETHODCALLTYPE EnumAdapterByGpuPreference(UINT index, DXGI_GPU_PREFERENCE, REFIID riid,
                                                         void** out) override {
        if (!out) return DXGI_ERROR_INVALID_CALL;
        *out = nullptr;
        if (index != 0) return DXGI_ERROR_NOT_FOUND;
        auto* adapter = new Adapter(this, gipa_, instance_, physical_);
        const HRESULT hr = adapter->QueryInterface(riid, out);
        adapter->Release();
        return hr;
    }

    // IDXGIFactory7. The adapter list never changes, so a registered event is never signalled; the registry only
    // keeps the host's answers (same measurement): a null event or cookie and an event registered twice are
    // DXGI_ERROR_INVALID_CALL, cookies start at 1 and are separate from the occlusion cookies, and an unknown
    // cookie is DXGI_ERROR_NOT_FOUND.
    HRESULT STDMETHODCALLTYPE RegisterAdaptersChangedEvent(HANDLE event, DWORD* cookie) override {
        if (!event || !cookie) return DXGI_ERROR_INVALID_CALL;
        std::lock_guard lock(adapters_changed_mutex_);
        for (const auto& [registered_cookie, registered_event] : adapters_changed_events_)
            if (registered_event == event) return DXGI_ERROR_INVALID_CALL;
        const DWORD next = next_adapters_changed_cookie_++;
        adapters_changed_events_.emplace(next, event);
        *cookie = next;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE UnregisterAdaptersChangedEvent(DWORD cookie) override {
        std::lock_guard lock(adapters_changed_mutex_);
        return adapters_changed_events_.erase(cookie) ? S_OK : DXGI_ERROR_NOT_FOUND;
    }

private:
    bool probe_immediate_present_mode() const {
        if (!gipa_ || !instance_ || !physical_) return false;
        const auto create_surface =
            reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(gipa_(instance_, "vkCreateWin32SurfaceKHR"));
        const auto destroy_surface =
            reinterpret_cast<PFN_vkDestroySurfaceKHR>(gipa_(instance_, "vkDestroySurfaceKHR"));
        const auto present_modes = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfacePresentModesKHR>(
            gipa_(instance_, "vkGetPhysicalDeviceSurfacePresentModesKHR"));
        if (!create_surface || !destroy_surface || !present_modes) return false;
        const HINSTANCE hinst = GetModuleHandleW(nullptr);
        HWND probe_hwnd = CreateWindowExW(0, L"STATIC", L"", WS_POPUP, 0, 0, 1, 1, nullptr, nullptr, hinst, nullptr);
        if (!probe_hwnd) return false;
        VkWin32SurfaceCreateInfoKHR sci{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
        sci.hinstance = hinst;
        sci.hwnd = probe_hwnd;
        VkSurfaceKHR surf = VK_NULL_HANDLE;
        bool has_immediate = false;
        if (create_surface(instance_, &sci, nullptr, &surf) == VK_SUCCESS && surf) {
            std::uint32_t count = 0;
            if (present_modes(physical_, surf, &count, nullptr) == VK_SUCCESS && count > 0) {
                std::vector<VkPresentModeKHR> modes(count);
                if (present_modes(physical_, surf, &count, modes.data()) == VK_SUCCESS) {
                    for (VkPresentModeKHR m : modes) {
                        if (m == VK_PRESENT_MODE_IMMEDIATE_KHR) {
                            has_immediate = true;
                            break;
                        }
                    }
                }
            }
            destroy_surface(instance_, surf, nullptr);
        }
        DestroyWindow(probe_hwnd);
        return has_immediate;
    }

    std::atomic<ULONG> refs_{1};
    PrivateDataStore private_data_;
    leonida::VulkanContext& context_;
    PFN_vkGetInstanceProcAddr gipa_;
    VkInstance instance_;
    VkPhysicalDevice physical_;
    PFN_vkGetDeviceProcAddr gdpa_;
    VkQueue queue_;
    std::uint32_t queue_family_;
    HWND assoc_hwnd_ = nullptr;
    mutable std::once_flag tearing_once_;
    bool tearing_supported_ = false;
    std::atomic<DWORD> next_occlusion_cookie_{1};
    std::mutex adapters_changed_mutex_;
    std::map<DWORD, HANDLE> adapters_changed_events_;
    DWORD next_adapters_changed_cookie_ = 1;
    std::atomic<DWORD> next_stereo_cookie_{1};
};

}  // namespace

HRESULT create_factory(leonida::VulkanContext& context, PFN_vkGetInstanceProcAddr gipa, VkInstance instance,
                       VkPhysicalDevice physical, PFN_vkGetDeviceProcAddr gdpa, VkQueue queue,
                       std::uint32_t queue_family, REFIID riid, void** out) {
    if (!out) return E_POINTER;
    *out = nullptr;
    auto* factory = new Factory(context, gipa, instance, physical, gdpa, queue, queue_family);
    const HRESULT hr = factory->QueryInterface(riid, out);
    factory->Release();
    return hr;
}

}  // namespace rt::dxgi
