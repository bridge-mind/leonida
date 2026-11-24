// xgameruntime: XStoreImpl, the class xgameruntime.lib's XStore* functions reach through QueryApiImpl, offline.
//
// A GDK title checks its license at boot (XStoreCreateContext, XStoreQueryGameLicenseAsync) and often its add-ons
// and package updates. Each XStore* function in xgameruntime.lib asks QueryApiImpl for XStoreImpl and calls one slot
// with its own arguments after `this`. The class and interface ids and the method order are xodus-gaming/
// xgameruntime-docs' COM/XStoreImpl pages (GPL-3.0; facts only, nothing copied); IUnknown takes slots 0-2, so the
// docs' first method is slot 3. Signatures follow Microsoft's public GDK reference (learn.microsoft.com/gaming/gdk,
// XStore pages). No GDK header or library code is used.
//
// Offline the game is owned: the game license is active, not a trial; a package's license is acquirable and valid;
// there are no add-on licenses, no products in any query and no package updates. Everything that needs the store
// service or a store UI (purchases, consumables, tokens, collections ids, downloads) fails before starting, so no
// XAsyncBlock waits, and logs once. License-changed and license-lost callbacks are never called: nothing changes.
//
// XStoreGameLicense's layout is measured on the retail executable (docs/runtime_imports.md): the title's
// XStoreQueryGameLicenseResult call passes a 104-byte stack buffer and reads the SKU id as a C string at +0x00, the
// three booleans at +0x13..+0x15, the trial id as a C string at +0x1C and the expiration time at +0x60. That fixes
// STORE_SKU_ID_SIZE = 18 and the struct's 104-byte total.
//
// UNVERIFIED: SKU_ID_SIZE = 5 and the exact TRIAL_UNIQUE_ID_MAX_SIZE = 64. The executable does not call either
// can-acquire slot (IXStoreImpl slots 24-27), so XStoreCanAcquireLicenseResult is unmeasured. It bounds the trial id
// to at most 68 bytes (the expiration time is at +0x60) but reads it as a C string, so it does not name the bound;
// 64 is the public source's value.
#include <windows.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <iterator>
#include <mutex>
#include <unordered_set>

#include <XAsync.h>
#include <XAsyncProvider.h>
#include <XTaskQueue.h>

namespace {

// ---- Public GDK types (XStore.h, as the GDK reference documents them) ----

using XUserHandle = void*;
struct XStoreContext {
    uint32_t unused;
};
struct XStoreLicense {
    uint32_t unused;
};
struct XStoreProductQuery {
    uint32_t unused;
};
using XStoreContextHandle = XStoreContext*;
using XStoreLicenseHandle = XStoreLicense*;
using XStoreProductQueryHandle = XStoreProductQuery*;

constexpr size_t STORE_SKU_ID_SIZE = 18;          // measured, see the header
constexpr size_t SKU_ID_SIZE = 5;                 // UNVERIFIED; the executable has no call site for it
constexpr size_t TRIAL_UNIQUE_ID_MAX_SIZE = 64;   // public source; the executable bounds it to <= 68

struct XStoreGameLicense {
    char skuStoreId[STORE_SKU_ID_SIZE];
    bool isActive;
    bool isTrialOwnedByThisUser;
    bool isDiscLicense;
    bool isTrial;
    uint32_t trialTimeRemainingInSeconds;
    char trialUniqueId[TRIAL_UNIQUE_ID_MAX_SIZE];
    time_t expirationDate;
};
static_assert(sizeof(XStoreGameLicense) == 104, "the layout the executable's result buffer gives");
// The offsets the executable's XStoreQueryGameLicenseResult caller reads (docs/runtime_imports.md).
static_assert(offsetof(XStoreGameLicense, isTrialOwnedByThisUser) == 0x13, "measured flag offset");
static_assert(offsetof(XStoreGameLicense, trialUniqueId) == 0x1C, "measured trial-id offset");
static_assert(offsetof(XStoreGameLicense, expirationDate) == 0x60, "measured expiration offset");

enum class XStoreCanLicenseStatus : uint32_t { NotLicensableToUser = 0, Licensable = 1, LicenseActionNotApplicableToProduct = 2 };
struct XStoreCanAcquireLicenseResult {
    char licensableSku[SKU_ID_SIZE];
    XStoreCanLicenseStatus status;
};
static_assert(sizeof(XStoreCanAcquireLicenseResult) == 12, "the layout the UNVERIFIED constant gives");

using LicenseChangedCallback = void(CALLBACK*)(void* context);

// The SKU the offline license names. ASSUMPTION: a title compares it with nothing offline.
constexpr char kOfflineSku[] = "0010";
static_assert(sizeof(kOfflineSku) <= SKU_ID_SIZE, "fits licensableSku");

// ---- Handles ----

std::mutex g_mutex;
std::unordered_set<const void*> g_contexts, g_licenses, g_queries;
uint64_t g_next_token = 1;

template <class T> T* new_handle(std::unordered_set<const void*>& set) {
    auto* h = new T{};
    std::lock_guard<std::mutex> lock(g_mutex);
    set.insert(h);
    return h;
}
bool live(const std::unordered_set<const void*>& set, const void* h) {
    if (!h) return false;
    std::lock_guard<std::mutex> lock(g_mutex);
    return set.count(h) != 0;
}
template <class T> void close_handle(std::unordered_set<const void*>& set, T* h) {
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!h || set.erase(h) == 0) return;   // closed already, or not ours
    }
    delete h;
}

// ---- Slots that are not available offline ----

// An operation that needs the store service or a store UI. The *Async function fails before starting, so the
// XAsyncBlock is never used and no callback runs; its Result/ResultSize partners return the same error. Their
// arguments don't matter, so the functions take none (x64: the caller owns the argument area).
template <int Slot> HRESULT STDMETHODCALLTYPE offline_failure(void*) {
    static std::atomic<bool> s_logged{false};
    if (!s_logged.exchange(true))
        std::fprintf(stderr, "[rt] xgameruntime: XStoreImpl slot %d is not available offline (E_FAIL)\n", Slot);
    return E_FAIL;
}

// A slot the docs don't name, or name without a signature: E_NOTIMPL, so nothing reads outputs it didn't write.
template <int Slot> HRESULT STDMETHODCALLTYPE unknown_slot(void*) {
    static std::atomic<bool> s_logged{false};
    if (!s_logged.exchange(true))
        std::fprintf(stderr, "[rt] xgameruntime: XStoreImpl slot %d (no documented signature) called; E_NOTIMPL\n", Slot);
    return E_NOTIMPL;
}

// ---- Async: every served *Async completes at once on the title's queue ----

const char kProductsIdentity = 0, kLicenseIdentity = 0, kCanAcquireIdentity = 0, kGameLicenseIdentity = 0,
           kAddOnsIdentity = 0, kUpdatesIdentity = 0;

size_t result_size(const void* identity) {
    if (identity == &kProductsIdentity) return sizeof(XStoreProductQueryHandle);
    if (identity == &kLicenseIdentity) return sizeof(XStoreLicenseHandle);
    if (identity == &kCanAcquireIdentity) return sizeof(XStoreCanAcquireLicenseResult);
    if (identity == &kGameLicenseIdentity) return sizeof(XStoreGameLicense);
    return 0;   // add-ons, updates: a count of 0, no buffer
}

// The context is the identity, so one provider serves every kind; handles are made when the title collects the
// result, so a result never collected leaks nothing.
HRESULT CALLBACK store_provider(XAsyncOp op, const XAsyncProviderData* data) {
    const void* identity = data->context;
    switch (op) {
    case XAsyncOp::Begin:
        return XAsyncSchedule(data->async, 0);
    case XAsyncOp::DoWork:
        XAsyncComplete(data->async, S_OK, result_size(identity));
        return S_OK;
    case XAsyncOp::GetResult:
        if (!data->buffer || data->bufferSize < result_size(identity)) return E_INVALIDARG;
        if (identity == &kProductsIdentity) {
            *static_cast<XStoreProductQueryHandle*>(data->buffer) = new_handle<XStoreProductQuery>(g_queries);
        } else if (identity == &kLicenseIdentity) {
            *static_cast<XStoreLicenseHandle*>(data->buffer) = new_handle<XStoreLicense>(g_licenses);
        } else if (identity == &kCanAcquireIdentity) {
            auto* r = static_cast<XStoreCanAcquireLicenseResult*>(data->buffer);
            *r = XStoreCanAcquireLicenseResult{};
            std::memcpy(r->licensableSku, kOfflineSku, sizeof(kOfflineSku));
            r->status = XStoreCanLicenseStatus::Licensable;
        } else if (identity == &kGameLicenseIdentity) {
            auto* l = static_cast<XStoreGameLicense*>(data->buffer);
            *l = XStoreGameLicense{};
            std::memcpy(l->skuStoreId, kOfflineSku, sizeof(kOfflineSku));
            l->isActive = true;   // owned; not a trial, not a disc license, no expiry
        }
        return S_OK;
    default:
        return S_OK;
    }
}

HRESULT begin(XStoreContextHandle context, XAsyncBlock* async, const char* identity, const char* name) {
    if (!async) return E_POINTER;
    if (!live(g_contexts, context)) return E_INVALIDARG;
    return XAsyncBegin(async, const_cast<char*>(identity), identity, name, store_provider);
}

// ---- IXStoreImpl (slots 3-69), IXStoreImpl2-6 (70-83) ----

// 3: HRESULT XStoreCreateContext(const XUserHandle user, XStoreContextHandle* storeContextHandle). The user may be
// null (PC: the store's user).
HRESULT STDMETHODCALLTYPE store_create_context(void*, XUserHandle, XStoreContextHandle* out) {
    if (!out) return E_POINTER;
    *out = new_handle<XStoreContext>(g_contexts);
    static std::atomic<bool> s_logged{false};
    if (!s_logged.exchange(true))
        std::fprintf(stderr, "[rt] xgameruntime: XStore offline: the game license is active, no add-ons, no updates\n");
    return S_OK;
}

// 4: void XStoreCloseContextHandle(XStoreContextHandle storeContextHandle)
void STDMETHODCALLTYPE store_close_context(void*, XStoreContextHandle h) { close_handle(g_contexts, h); }

// Product queries (5-14, 77-78): every one completes with an empty query: no products, no further pages.
// 5: XStoreQueryAssociatedProductsAsync(context, XStoreProductKind, uint32_t maxItemsToRetrievePerPage, XAsyncBlock*)
HRESULT STDMETHODCALLTYPE store_query_associated_async(void*, XStoreContextHandle c, uint32_t, uint32_t, XAsyncBlock* a) {
    return begin(c, a, &kProductsIdentity, "XStoreQueryAssociatedProductsAsync");
}
// 7: XStoreQueryProductsAsync(context, kinds, const char** storeIds, size_t, const char** actionFilters, size_t, async)
HRESULT STDMETHODCALLTYPE store_query_products_async(void*, XStoreContextHandle c, uint32_t, const char**, size_t,
                                                     const char**, size_t, XAsyncBlock* a) {
    return begin(c, a, &kProductsIdentity, "XStoreQueryProductsAsync");
}
// 9: XStoreQueryEntitledProductsAsync(context, kinds, uint32_t maxItemsToRetrievePerPage, XAsyncBlock*)
HRESULT STDMETHODCALLTYPE store_query_entitled_async(void*, XStoreContextHandle c, uint32_t, uint32_t, XAsyncBlock* a) {
    return begin(c, a, &kProductsIdentity, "XStoreQueryEntitledProductsAsync");
}
// 11: XStoreQueryProductForCurrentGameAsync(context, XAsyncBlock*)
HRESULT STDMETHODCALLTYPE store_query_current_game_async(void*, XStoreContextHandle c, XAsyncBlock* a) {
    return begin(c, a, &kProductsIdentity, "XStoreQueryProductForCurrentGameAsync");
}
// 13: XStoreQueryProductForPackageAsync(context, kinds, const char* packageIdentifier, XAsyncBlock*)
HRESULT STDMETHODCALLTYPE store_query_for_package_async(void*, XStoreContextHandle c, uint32_t, const char*,
                                                        XAsyncBlock* a) {
    return begin(c, a, &kProductsIdentity, "XStoreQueryProductForPackageAsync");
}
// 77: XStoreQueryAssociatedProductsForStoreIdAsync(context, const char* storeId, kinds, uint32_t maxItems, async)
HRESULT STDMETHODCALLTYPE store_query_associated_for_id_async(void*, XStoreContextHandle c, const char*, uint32_t,
                                                              uint32_t, XAsyncBlock* a) {
    return begin(c, a, &kProductsIdentity, "XStoreQueryAssociatedProductsForStoreIdAsync");
}
// 6, 8, 10, 12, 14, 18, 78: HRESULT ...Result(XAsyncBlock* async, XStoreProductQueryHandle* productQueryHandle)
HRESULT STDMETHODCALLTYPE store_products_result(void*, XAsyncBlock* async, XStoreProductQueryHandle* out) {
    if (!out) return E_POINTER;
    *out = nullptr;
    return XAsyncGetResult(async, &kProductsIdentity, sizeof(*out), out, nullptr);
}

// 15: HRESULT XStoreEnumerateProductsQuery(productQueryHandle, void* context, XStoreProductQueryCallback* callback)
HRESULT STDMETHODCALLTYPE store_enumerate_products(void*, XStoreProductQueryHandle q, void*, void* callback) {
    if (!callback) return E_INVALIDARG;
    return live(g_queries, q) ? S_OK : E_INVALIDARG;   // no products: the callback is never called
}

// 16: bool XStoreProductsQueryHasMorePages(const XStoreProductQueryHandle productQueryHandle)
bool STDMETHODCALLTYPE store_has_more_pages(void*, XStoreProductQueryHandle) { return false; }

// 17: HRESULT XStoreProductsQueryNextPageAsync(productQueryHandle, XAsyncBlock*): there is no next page.
HRESULT STDMETHODCALLTYPE store_next_page_async(void*, XStoreProductQueryHandle, XAsyncBlock* async) {
    return async ? E_INVALIDARG : E_POINTER;
}

// 19: void XStoreCloseProductsQueryHandle(XStoreProductQueryHandle productQueryHandle)
void STDMETHODCALLTYPE store_close_query(void*, XStoreProductQueryHandle h) { close_handle(g_queries, h); }

// 20: HRESULT XStoreAcquireLicenseForPackageAsync(context, const char* packageIdentifier, XAsyncBlock*)
HRESULT STDMETHODCALLTYPE store_acquire_package_async(void*, XStoreContextHandle c, const char* package, XAsyncBlock* a) {
    if (!package) return E_INVALIDARG;
    return begin(c, a, &kLicenseIdentity, "XStoreAcquireLicenseForPackageAsync");
}
// 21: HRESULT XStoreAcquireLicenseForPackageResult(XAsyncBlock* async, XStoreLicenseHandle* storeLicenseHandle)
HRESULT STDMETHODCALLTYPE store_acquire_package_result(void*, XAsyncBlock* async, XStoreLicenseHandle* out) {
    if (!out) return E_POINTER;
    *out = nullptr;
    return XAsyncGetResult(async, &kLicenseIdentity, sizeof(*out), out, nullptr);
}

// 22: bool XStoreIsLicenseValid(const XStoreLicenseHandle storeLicenseHandle)
bool STDMETHODCALLTYPE store_is_license_valid(void*, XStoreLicenseHandle h) { return live(g_licenses, h); }

// 23: void XStoreCloseLicenseHandle(XStoreLicenseHandle storeLicenseHandle)
void STDMETHODCALLTYPE store_close_license(void*, XStoreLicenseHandle h) { close_handle(g_licenses, h); }

// 24: HRESULT XStoreCanAcquireLicenseForStoreIdAsync(context, const char* storeProductId, XAsyncBlock*)
// 26: HRESULT XStoreCanAcquireLicenseForPackageAsync(context, const char* packageIdentifier, XAsyncBlock*)
HRESULT STDMETHODCALLTYPE store_can_acquire_async(void*, XStoreContextHandle c, const char* id, XAsyncBlock* a) {
    if (!id) return E_INVALIDARG;
    return begin(c, a, &kCanAcquireIdentity, "XStoreCanAcquireLicenseAsync");
}
// 25, 27: HRESULT ...Result(XAsyncBlock* async, XStoreCanAcquireLicenseResult* storeCanAcquireLicense)
HRESULT STDMETHODCALLTYPE store_can_acquire_result(void*, XAsyncBlock* async, XStoreCanAcquireLicenseResult* out) {
    if (!out) return E_POINTER;
    return XAsyncGetResult(async, &kCanAcquireIdentity, sizeof(*out), out, nullptr);
}

// 28: HRESULT XStoreQueryGameLicenseAsync(const XStoreContextHandle storeContextHandle, XAsyncBlock* async)
HRESULT STDMETHODCALLTYPE store_game_license_async(void*, XStoreContextHandle c, XAsyncBlock* a) {
    return begin(c, a, &kGameLicenseIdentity, "XStoreQueryGameLicenseAsync");
}
// 29: HRESULT XStoreQueryGameLicenseResult(XAsyncBlock* async, XStoreGameLicense* license)
HRESULT STDMETHODCALLTYPE store_game_license_result(void*, XAsyncBlock* async, XStoreGameLicense* out) {
    if (!out) return E_POINTER;
    return XAsyncGetResult(async, &kGameLicenseIdentity, sizeof(*out), out, nullptr);
}

// 30: HRESULT XStoreQueryAddOnLicensesAsync(const XStoreContextHandle storeContextHandle, XAsyncBlock* async)
HRESULT STDMETHODCALLTYPE store_addons_async(void*, XStoreContextHandle c, XAsyncBlock* a) {
    return begin(c, a, &kAddOnsIdentity, "XStoreQueryAddOnLicensesAsync");
}
// 55: HRESULT XStoreQueryGameAndDlcPackageUpdatesAsync(const XStoreContextHandle storeContextHandle, XAsyncBlock*)
HRESULT STDMETHODCALLTYPE store_updates_async(void*, XStoreContextHandle c, XAsyncBlock* a) {
    return begin(c, a, &kUpdatesIdentity, "XStoreQueryGameAndDlcPackageUpdatesAsync");
}
// 79: HRESULT XStoreQueryPackageUpdatesAsync(context, const char** packageIdentifiers, size_t count, XAsyncBlock*)
HRESULT STDMETHODCALLTYPE store_package_updates_async(void*, XStoreContextHandle c, const char**, size_t, XAsyncBlock* a) {
    return begin(c, a, &kUpdatesIdentity, "XStoreQueryPackageUpdatesAsync");
}

// 31: HRESULT XStoreQueryAddOnLicensesResultCount(XAsyncBlock* async, uint32_t* count)
HRESULT STDMETHODCALLTYPE store_addons_count(void*, XAsyncBlock* async, uint32_t* count) {
    if (!count) return E_POINTER;
    *count = 0;
    return XAsyncGetResult(async, &kAddOnsIdentity, 0, nullptr, nullptr);
}
// 56, 80: HRESULT ...ResultCount(XAsyncBlock* async, uint32_t* count)
HRESULT STDMETHODCALLTYPE store_updates_count(void*, XAsyncBlock* async, uint32_t* count) {
    if (!count) return E_POINTER;
    *count = 0;
    return XAsyncGetResult(async, &kUpdatesIdentity, 0, nullptr, nullptr);
}
// 32: HRESULT XStoreQueryAddOnLicensesResult(XAsyncBlock* async, uint32_t count, XStoreAddonLicense* addOnLicenses)
HRESULT STDMETHODCALLTYPE store_addons_result(void*, XAsyncBlock* async, uint32_t count, void*) {
    if (count != 0) return E_INVALIDARG;   // the count was 0
    return XAsyncGetResult(async, &kAddOnsIdentity, 0, nullptr, nullptr);
}
// 57, 81: HRESULT ...Result(XAsyncBlock* async, uint32_t count, XStorePackageUpdate* packageUpdates)
HRESULT STDMETHODCALLTYPE store_updates_result(void*, XAsyncBlock* async, uint32_t count, void*) {
    if (count != 0) return E_INVALIDARG;
    return XAsyncGetResult(async, &kUpdatesIdentity, 0, nullptr, nullptr);
}

// 65: HRESULT XStoreQueryPackageIdentifier(const char* storeId, size_t size, char* packageIdentifier): no store
// product is installed as a package offline.
HRESULT STDMETHODCALLTYPE store_query_package_identifier(void*, const char* store_id, size_t size, char* out) {
    if (!store_id || !out) return E_INVALIDARG;
    if (size) out[0] = '\0';
    return HRESULT_FROM_WIN32(ERROR_NOT_FOUND);
}

// 66: HRESULT XStoreRegisterGameLicenseChanged(context, XTaskQueueHandle, void* context, callback, token*)
// 68: HRESULT XStoreRegisterPackageLicenseLost(licenseHandle, XTaskQueueHandle, void* context, callback, token*)
// Nothing changes offline: the token is handed out, the callback is never called.
HRESULT STDMETHODCALLTYPE store_register(void*, void* handle, XTaskQueueHandle, void*, LicenseChangedCallback callback,
                                         XTaskQueueRegistrationToken* token) {
    if (!handle || !callback || !token) return E_INVALIDARG;
    std::lock_guard<std::mutex> lock(g_mutex);
    token->token = g_next_token++;
    return S_OK;
}
// 67, 69: bool ...Unregister...(handle, XTaskQueueRegistrationToken token, bool wait): never pending, so true.
bool STDMETHODCALLTYPE store_unregister(void*, void*, XTaskQueueRegistrationToken, bool) { return true; }

// ---- The object ----

constexpr GUID kStoreClass = {0x0dd112ac, 0x7c24, 0x448c, {0xb9, 0x2b, 0x39, 0x60, 0xfb, 0x5b, 0xd3, 0x0c}};
const GUID kStoreInterfaces[] = {
    kStoreClass,                                                                       // IXStoreImpl
    {0x60b09f4e, 0x1b85, 0x45b1, {0x82, 0x6c, 0x16, 0x91, 0x18, 0xe2, 0x30, 0xe1}},    // IXStoreImpl2
    {0x2d42fea5, 0xe71d, 0x4b76, {0x97, 0xcd, 0xc5, 0x0a, 0xfb, 0xb3, 0xae, 0x5d}},    // IXStoreImpl3
    {0xde3dbdd4, 0x0b37, 0x4bdb, {0xa1, 0x0e, 0xac, 0xf3, 0xa3, 0x54, 0xd0, 0x6a}},    // IXStoreImpl4
    {0x5c48dedf, 0x0b67, 0x4492, {0xa4, 0xb5, 0x68, 0x29, 0xb8, 0xe7, 0x96, 0xe1}},    // IXStoreImpl5
    {0xb09d803c, 0x2414, 0x4a05, {0x82, 0xc6, 0x66, 0xdf, 0xdc, 0x9e, 0x9a, 0x44}},    // IXStoreImpl6
};
constexpr GUID kIUnknown = {0x00000000, 0x0000, 0x0000, {0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46}};

struct StoreObject {
    const void* const* vtable;
};
StoreObject* store_object();

HRESULT query(REFIID iid, void** out) {
    if (!out) return E_POINTER;
    *out = nullptr;
    if (iid == kIUnknown) {
        *out = store_object();
        return S_OK;
    }
    for (const GUID& id : kStoreInterfaces)
        if (iid == id) {
            *out = store_object();
            return S_OK;
        }
    return E_NOINTERFACE;
}

// A process-wide singleton, as the other runtime classes: AddRef and Release don't free.
HRESULT STDMETHODCALLTYPE object_query_interface(void*, REFIID iid, void** out) { return query(iid, out); }
ULONG STDMETHODCALLTYPE object_add_ref(void*) { return 2; }
ULONG STDMETHODCALLTYPE object_release(void*) { return 1; }

#define FN(f) reinterpret_cast<const void*>(&f)

const void* const kStoreVtable[] = {
    FN(object_query_interface), FN(object_add_ref), FN(object_release),   // 0-2  IUnknown
    FN(store_create_context),                // 3  XStoreCreateContext
    FN(store_close_context),                 // 4  XStoreCloseContextHandle
    FN(store_query_associated_async),        // 5  XStoreQueryAssociatedProductsAsync
    FN(store_products_result),               // 6  XStoreQueryAssociatedProductsResult
    FN(store_query_products_async),          // 7  XStoreQueryProductsAsync
    FN(store_products_result),               // 8  XStoreQueryProductsResult
    FN(store_query_entitled_async),          // 9  XStoreQueryEntitledProductsAsync
    FN(store_products_result),               // 10 XStoreQueryEntitledProductsResult
    FN(store_query_current_game_async),      // 11 XStoreQueryProductForCurrentGameAsync
    FN(store_products_result),               // 12 XStoreQueryProductForCurrentGameResult
    FN(store_query_for_package_async),       // 13 XStoreQueryProductForPackageAsync
    FN(store_products_result),               // 14 XStoreQueryProductForPackageResult
    FN(store_enumerate_products),            // 15 XStoreEnumerateProductsQuery
    FN(store_has_more_pages),                // 16 XStoreProductsQueryHasMorePages
    FN(store_next_page_async),               // 17 XStoreProductsQueryNextPageAsync
    FN(store_products_result),               // 18 XStoreProductsQueryNextPageResult
    FN(store_close_query),                   // 19 XStoreCloseProductsQueryHandle
    FN(store_acquire_package_async),         // 20 XStoreAcquireLicenseForPackageAsync
    FN(store_acquire_package_result),        // 21 XStoreAcquireLicenseForPackageResult
    FN(store_is_license_valid),              // 22 XStoreIsLicenseValid
    FN(store_close_license),                 // 23 XStoreCloseLicenseHandle
    FN(store_can_acquire_async),             // 24 XStoreCanAcquireLicenseForStoreIdAsync
    FN(store_can_acquire_result),            // 25 XStoreCanAcquireLicenseForStoreIdResult
    FN(store_can_acquire_async),             // 26 XStoreCanAcquireLicenseForPackageAsync
    FN(store_can_acquire_result),            // 27 XStoreCanAcquireLicenseForPackageResult
    FN(store_game_license_async),            // 28 XStoreQueryGameLicenseAsync
    FN(store_game_license_result),           // 29 XStoreQueryGameLicenseResult
    FN(store_addons_async),                  // 30 XStoreQueryAddOnLicensesAsync
    FN(store_addons_count),                  // 31 XStoreQueryAddOnLicensesResultCount
    FN(store_addons_result),                 // 32 XStoreQueryAddOnLicensesResult
    FN(offline_failure<33>),                 // 33 XStoreQueryConsumableBalanceRemainingAsync
    FN(offline_failure<34>),                 // 34 XStoreQueryConsumableBalanceRemainingResult
    FN(offline_failure<35>),                 // 35 XStoreReportConsumableFulfillmentAsync
    FN(offline_failure<36>),                 // 36 XStoreReportConsumableFulfillmentResult
    FN(offline_failure<37>),                 // 37 XStoreGetUserCollectionsIdAsync
    FN(offline_failure<38>),                 // 38 XStoreGetUserCollectionsIdResultSize
    FN(offline_failure<39>),                 // 39 XStoreGetUserCollectionsIdResult
    FN(offline_failure<40>),                 // 40 XStoreGetUserPurchaseIdAsync
    FN(offline_failure<41>),                 // 41 XStoreGetUserPurchaseIdResultSize
    FN(offline_failure<42>),                 // 42 XStoreGetUserPurchaseIdResult
    FN(offline_failure<43>),                 // 43 XStoreQueryLicenseTokenAsync
    FN(offline_failure<44>),                 // 44 XStoreQueryLicenseTokenResultSize
    FN(offline_failure<45>),                 // 45 XStoreQueryLicenseTokenResult
    FN(unknown_slot<46>),                    // 46 (unknown in the docs)
    FN(unknown_slot<47>),                    // 47 (unknown in the docs)
    FN(unknown_slot<48>),                    // 48 (unknown in the docs)
    FN(offline_failure<49>),                 // 49 XStoreShowPurchaseUIAsync
    FN(offline_failure<50>),                 // 50 XStoreShowPurchaseUIResult
    FN(offline_failure<51>),                 // 51 XStoreShowRateAndReviewUIAsync
    FN(offline_failure<52>),                 // 52 XStoreShowRateAndReviewUIResult
    FN(offline_failure<53>),                 // 53 XStoreShowRedeemTokenUIAsync
    FN(offline_failure<54>),                 // 54 XStoreShowRedeemTokenUIResult
    FN(store_updates_async),                 // 55 XStoreQueryGameAndDlcPackageUpdatesAsync
    FN(store_updates_count),                 // 56 XStoreQueryGameAndDlcPackageUpdatesResultCount
    FN(store_updates_result),                // 57 XStoreQueryGameAndDlcPackageUpdatesResult
    FN(offline_failure<58>),                 // 58 XStoreDownloadPackageUpdatesAsync
    FN(offline_failure<59>),                 // 59 XStoreDownloadPackageUpdatesResult
    FN(offline_failure<60>),                 // 60 XStoreDownloadAndInstallPackageUpdatesAsync
    FN(offline_failure<61>),                 // 61 XStoreDownloadAndInstallPackageUpdatesResult
    FN(offline_failure<62>),                 // 62 XStoreDownloadAndInstallPackagesAsync
    FN(offline_failure<63>),                 // 63 XStoreDownloadAndInstallPackagesResultCount
    FN(offline_failure<64>),                 // 64 XStoreDownloadAndInstallPackagesResult
    FN(store_query_package_identifier),      // 65 XStoreQueryPackageIdentifier
    FN(store_register),                      // 66 XStoreRegisterGameLicenseChanged
    FN(store_unregister),                    // 67 XStoreUnregisterGameLicenseChanged
    FN(store_register),                      // 68 XStoreRegisterPackageLicenseLost
    FN(store_unregister),                    // 69 XStoreUnregisterPackageLicenseLost
    FN(unknown_slot<70>),                    // 70 XStoreIsAvailabilityPurchasable (2; no documented signature)
    FN(offline_failure<71>),                 // 71 XStoreAcquireLicenseForDurablesAsync (3)
    FN(offline_failure<72>),                 // 72 XStoreAcquireLicenseForDurablesResult (3)
    FN(offline_failure<73>),                 // 73 XStoreShowAssociatedProductsUIAsync (4)
    FN(offline_failure<74>),                 // 74 XStoreShowAssociatedProductsUIResult (4)
    FN(offline_failure<75>),                 // 75 XStoreShowProductPageUIAsync (4)
    FN(offline_failure<76>),                 // 76 XStoreShowProductPageUIResult (4)
    FN(store_query_associated_for_id_async), // 77 XStoreQueryAssociatedProductsForStoreIdAsync (5)
    FN(store_products_result),               // 78 XStoreQueryAssociatedProductsForStoreIdResult (5)
    FN(store_package_updates_async),         // 79 XStoreQueryPackageUpdatesAsync (5)
    FN(store_updates_count),                 // 80 XStoreQueryPackageUpdatesResultCount (5)
    FN(store_updates_result),                // 81 XStoreQueryPackageUpdatesResult (5)
    FN(offline_failure<82>),                 // 82 XStoreShowGiftingUIAsync (6)
    FN(offline_failure<83>),                 // 83 XStoreShowGiftingUIResult (6)
};
static_assert(std::size(kStoreVtable) == 84, "IUnknown + IXStoreImpl's 67 methods + 1 + 2 + 4 + 5 + 2");

#undef FN

StoreObject* store_object() {
    static StoreObject o{kStoreVtable};
    return &o;
}

}  // namespace

namespace rt::shims {

// Called by xgameruntime.cpp's QueryApiImpl for every class: true when the class is XStoreImpl, with *hr set.
bool xgameruntime_store_query(const GUID& class_id, REFIID interface_id, void** out, HRESULT* hr) {
    if (class_id != kStoreClass) return false;
    *hr = query(interface_id, out);
    return true;
}

// For tests: the context, license and product-query handles a title holds open.
long xgameruntime_store_open_handles() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return static_cast<long>(g_contexts.size() + g_licenses.size() + g_queries.size());
}

}  // namespace rt::shims
