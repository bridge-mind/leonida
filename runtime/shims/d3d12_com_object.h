// Shared IUnknown/ID3D12Object/ID3D12DeviceChild implementation for the Vulkan-backed D3D12 COM objects
// (d3d12_com.cpp, d3d12_vulkan_device.cpp). Private to the runtime; include after the generated d3d12_stubs.h.
#pragma once

#include <windows.h>
#include <d3d12.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <mutex>
#include <utility>
#include <vector>

namespace rt::d3d12 {

void stub(const char* method);

// IUnknown and ID3D12Object for one implementation class. QueryInterface answers the listed public interfaces
// and the class's private Derived::kTag (which returns the implementation itself); anything else is logged with
// its IID and refused, as docs/d3d12_vulkan.md section 3.1 asks.
template<class Derived, class Stubs, class... Interfaces>
class Object : public Stubs {
public:
    ULONG STDMETHODCALLTYPE AddRef() override { return ++refs_; }
    ULONG STDMETHODCALLTYPE Release() override {
        const ULONG left = --refs_;
        if (!left) delete static_cast<Derived*>(this);
        return left;
    }

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (riid == Derived::kTag) {
            *out = static_cast<Derived*>(this);
        } else if (riid == __uuidof(IUnknown) || ((riid == __uuidof(Interfaces)) || ...)) {
            *out = static_cast<Stubs*>(this);  // one inheritance chain: every interface shares this pointer
        } else {
            std::fprintf(stderr,
                         "[rt] d3d12: QueryInterface: unknown IID {%08lx-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x}\n",
                         static_cast<unsigned long>(riid.Data1), riid.Data2, riid.Data3, riid.Data4[0], riid.Data4[1],
                         riid.Data4[2], riid.Data4[3], riid.Data4[4], riid.Data4[5], riid.Data4[6], riid.Data4[7]);
            return E_NOINTERFACE;
        }
        AddRef();
        return S_OK;
    }

    // Private data (ID3D12Object): bytes, or with SetPrivateDataInterface an interface the object holds a reference
    // to until the entry is replaced, removed or the object dies. GetPrivateData hands such an interface back with a
    // reference added, as D3D12 does. The answers are the host runtime's (d3d12_private_data_test, RTX 3060 /
    // 610.88, on the device and four child objects): a buffer too small for the entry gets DXGI_ERROR_MORE_DATA and
    // the size it needs; SetPrivateData with null data removes the entry whatever the size, S_OK when there was one
    // and S_FALSE when there wasn't; zero bytes with data keep an empty entry; SetPrivateDataInterface with null
    // keeps a pointer-sized entry that reads back as null.
    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID guid, UINT* size, void* data) override {
        if (!size) return E_INVALIDARG;
        std::lock_guard lock(data_mutex_);
        for (const auto& entry : data_) {
            if (entry.key != guid) continue;
            const UINT bytes = entry.is_interface ? UINT(sizeof(IUnknown*)) : UINT(entry.bytes.size());
            if (data && *size < bytes) {
                *size = bytes;  // the size to retry with
                return DXGI_ERROR_MORE_DATA;
            }
            if (data) {
                if (entry.is_interface) {
                    if (entry.object) entry.object->AddRef();
                    std::memcpy(data, &entry.object, sizeof(IUnknown*));
                } else if (bytes) {
                    std::memcpy(data, entry.bytes.data(), bytes);
                }
            }
            *size = bytes;
            return S_OK;
        }
        *size = 0;
        return DXGI_ERROR_NOT_FOUND;
    }

    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID guid, UINT size, const void* data) override {
        if (!data) return remove(guid) ? S_OK : S_FALSE;
        PrivateData entry{guid, {}, nullptr, false};
        const auto* bytes = static_cast<const std::uint8_t*>(data);
        entry.bytes.assign(bytes, bytes + size);
        store(std::move(entry));
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID guid, const IUnknown* object) override {
        IUnknown* held = const_cast<IUnknown*>(object);
        if (held) held->AddRef();
        store(PrivateData{guid, {}, held, true});
        return S_OK;
    }

    // SetName is SetPrivateData with WKPDID_D3DDebugObjectNameW and the name with its terminator, as D3D12 documents
    // and the host runtime does: GetPrivateData reads it back (an empty name is a 2-byte entry), and the narrow
    // WKPDID_D3DDebugObjectName is not touched. A null name, which the host runtime faults on, changes nothing.
    HRESULT STDMETHODCALLTYPE SetName(LPCWSTR name) override {
        if (!name) return E_INVALIDARG;
        return SetPrivateData(kDebugObjectNameW, UINT((std::wcslen(name) + 1) * sizeof(wchar_t)), name);
    }

    // WKPDID_D3DDebugObjectNameW (d3dcommon.h), spelled out so nothing needs dxguid.lib.
    static constexpr GUID kDebugObjectNameW{0x4cca5fd8, 0x921f, 0x42c8,
                                            {0x85, 0x66, 0x70, 0xca, 0xf2, 0xa9, 0xb7, 0x41}};

    // The implementation behind a public interface pointer, or null when it is not a T. No reference is added.
    template<class T>
    static T* from(IUnknown* object) {
        if (!object) return nullptr;
        void* implementation = nullptr;
        if (FAILED(object->QueryInterface(T::kTag, &implementation))) return nullptr;
        static_cast<T*>(implementation)->Release();  // the caller holds its own reference
        return static_cast<T*>(implementation);
    }

protected:
    virtual ~Object() {
        for (auto& entry : data_)
            if (entry.object) entry.object->Release();
    }

private:
    struct PrivateData {
        GUID key;
        std::vector<std::uint8_t> bytes;
        IUnknown* object;  // SetPrivateDataInterface: one reference held (null for a null interface)
        bool is_interface;  // set by SetPrivateDataInterface, even with null: reads back as a pointer
    };

    // Takes the entry for `key` out; true when there was one. The interface it held is released after the lock is
    // dropped, since its Release may run arbitrary code.
    bool remove(REFGUID key) { return replace(key, nullptr); }
    void store(PrivateData entry) {
        const GUID key = entry.key;
        replace(key, &entry);
    }
    bool replace(REFGUID key, PrivateData* entry) {
        IUnknown* released = nullptr;
        bool found = false;
        {
            std::lock_guard lock(data_mutex_);
            for (auto it = data_.begin(); it != data_.end(); ++it) {
                if (it->key != key) continue;
                released = it->object;
                data_.erase(it);
                found = true;
                break;
            }
            if (entry) data_.push_back(std::move(*entry));
        }
        if (released) released->Release();
        return found;
    }

    std::atomic<ULONG> refs_{1};
    std::mutex data_mutex_;
    std::vector<PrivateData> data_;
};

// An ID3D12DeviceChild: holds a reference to its device, as D3D12 children do, and GetDevice returns it. When the
// device sits behind the Xbox-ABI proxy, the reference is to the proxy, so the game never sees the inner device.
template<class Derived, class Stubs, class... Interfaces>
class Child : public Object<Derived, Stubs, Interfaces...> {
public:
    void attach_device(IUnknown* device) {
        if (device) device->AddRef();
        if (device_) device_->Release();
        device_ = device;
    }

    IUnknown* attached_device() const { return device_; }

    HRESULT STDMETHODCALLTYPE GetDevice(REFIID riid, void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        return device_ ? device_->QueryInterface(riid, out) : E_NOINTERFACE;
    }

protected:
    ~Child() override {
        if (device_) device_->Release();
    }

private:
    IUnknown* device_ = nullptr;
};

template<class T>
HRESULT hand_out(T* object, REFIID riid, void** out) {
    const HRESULT hr = object->QueryInterface(riid, out);
    object->Release();
    return hr;
}

}  // namespace rt::d3d12
