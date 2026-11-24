// XStoreImpl through QueryApiImpl, called the way xgameruntime.lib's XStore* thunks call it: QueryApiImpl with the
// class id (or a later interface id), one vtable slot with `this` first. Slots follow xgameruntime-docs' layout;
// arguments follow the public GDK reference. No game data; nothing online.
#include <windows.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>

#include <XAsync.h>
#include <XAsyncProvider.h>
#include <XTaskQueue.h>

#include "shims.h"

namespace rt::shims {
long xgameruntime_store_open_handles();
}

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

using Handle = void*;
struct GameLicense {   // XStoreGameLicense with the measured sizes of xgameruntime_store.cpp
    char skuStoreId[18];
    bool isActive;
    bool isTrialOwnedByThisUser;
    bool isDiscLicense;
    bool isTrial;
    uint32_t trialTimeRemainingInSeconds;
    char trialUniqueId[64];
    time_t expirationDate;
};
static_assert(sizeof(GameLicense) == 104, "XStoreGameLicense is 104 bytes");
static_assert(offsetof(GameLicense, isTrialOwnedByThisUser) == 0x13, "measured flag offset");
static_assert(offsetof(GameLicense, trialUniqueId) == 0x1C, "measured trial-id offset");
static_assert(offsetof(GameLicense, expirationDate) == 0x60, "measured expiration offset");
struct CanAcquire {
    char licensableSku[5];
    uint32_t status;
};

constexpr GUID kStoreClass = {0x0dd112ac, 0x7c24, 0x448c, {0xb9, 0x2b, 0x39, 0x60, 0xfb, 0x5b, 0xd3, 0x0c}};
constexpr GUID kStoreImpl6 = {0xb09d803c, 0x2414, 0x4a05, {0x82, 0xc6, 0x66, 0xdf, 0xdc, 0x9e, 0x9a, 0x44}};
constexpr GUID kStoreImpl2 = {0x60b09f4e, 0x1b85, 0x45b1, {0x82, 0x6c, 0x16, 0x91, 0x18, 0xe2, 0x30, 0xe1}};
constexpr GUID kOther = {0x973a344e, 0x24bf, 0x4d0f, {0x84, 0x57, 0x56, 0xc5, 0x34, 0x89, 0x2b, 0x29}};  // XGameImpl

using Query = HRESULT(WINAPI*)(const GUID*, REFIID, void**);
using Fn = void*;
Fn slot(void* obj, int index) { return (*static_cast<Fn* const*>(obj))[index]; }
template <class F> F at(void* obj, int index) { return reinterpret_cast<F>(slot(obj, index)); }

bool finished(XAsyncBlock* async) { return XAsyncGetStatus(async, true) != E_PENDING; }

using CreateContext = HRESULT(STDMETHODCALLTYPE*)(void*, Handle, Handle*);
using Close = void(STDMETHODCALLTYPE*)(void*, Handle);
using CtxAsync = HRESULT(STDMETHODCALLTYPE*)(void*, Handle, XAsyncBlock*);
using CtxIdAsync = HRESULT(STDMETHODCALLTYPE*)(void*, Handle, const char*, XAsyncBlock*);
using LicenseResult = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*, GameLicense*);
using CanResult = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*, CanAcquire*);
using HandleResult = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*, Handle*);
using Count = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*, uint32_t*);
using Items = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*, uint32_t, void*);
using IsValid = bool(STDMETHODCALLTYPE*)(void*, Handle);
using ProductsAsync = HRESULT(STDMETHODCALLTYPE*)(void*, Handle, uint32_t, const char**, size_t, const char**, size_t,
                                                  XAsyncBlock*);
using Enumerate = HRESULT(STDMETHODCALLTYPE*)(void*, Handle, void*, void*);
using Register = HRESULT(STDMETHODCALLTYPE*)(void*, Handle, XTaskQueueHandle, void*, void(CALLBACK*)(void*),
                                             XTaskQueueRegistrationToken*);
using Unregister = bool(STDMETHODCALLTYPE*)(void*, Handle, XTaskQueueRegistrationToken, bool);
using NoArgs = HRESULT(STDMETHODCALLTYPE*)(void*);

int g_callbacks = 0;
void CALLBACK on_change(void*) { ++g_callbacks; }
bool CALLBACK on_product(void*, void*) {
    ++g_callbacks;
    return true;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("xgameruntime_store_test: XStoreImpl through QueryApiImpl\n");
    auto query = reinterpret_cast<Query>(rt::shims::find("xgameruntime.dll", "QueryApiImpl"));
    check(query != nullptr, "QueryApiImpl is registered");
    if (!query) return 1;

    void* store = nullptr;
    check(query(&kStoreClass, kStoreClass, &store) == S_OK && store, "QueryApiImpl hands out XStoreImpl for its class id");
    if (!store) return 1;
    void* store6 = nullptr;
    void* store2 = nullptr;
    check(query(&kStoreClass, kStoreImpl6, &store6) == S_OK && store6 == store &&
              query(&kStoreClass, kStoreImpl2, &store2) == S_OK && store2 == store,
          "IXStoreImpl2 and IXStoreImpl6 answer with the same object");
    void* none = reinterpret_cast<void*>(1);
    check(query(&kStoreClass, kOther, &none) == E_NOINTERFACE && none == nullptr, "another interface id is refused");

    Handle context = nullptr;
    check(at<CreateContext>(store, 3)(store, reinterpret_cast<Handle>(0x1000), &context) == S_OK && context,
          "XStoreCreateContext");

    // ---- the game license, as a title checks it at boot ----
    XAsyncBlock async{};
    GameLicense license;
    std::memset(&license, 0xCD, sizeof(license));
    check(at<CtxAsync>(store, 28)(store, context, &async) == S_OK && finished(&async),
          "XStoreQueryGameLicenseAsync completes");
    size_t license_size = 0;
    check(XAsyncGetResultSize(&async, &license_size) == S_OK && license_size == 104,
          "the game license result size is the measured 104 bytes");
    check(at<LicenseResult>(store, 29)(store, &async, &license) == S_OK, "XStoreQueryGameLicenseResult");
    check(license.isActive && !license.isTrial && !license.isTrialOwnedByThisUser && !license.isDiscLicense &&
              license.trialTimeRemainingInSeconds == 0 && license.trialUniqueId[0] == '\0' &&
              std::strcmp(license.skuStoreId, "0010") == 0,
          "the game license is active, not a trial, every other field cleared");
    check(at<CtxAsync>(store, 28)(store, nullptr, &async) == E_INVALIDARG, "a null store context is refused");

    // ---- add-ons and updates: none ----
    async = XAsyncBlock{};
    uint32_t count = 99;
    check(at<CtxAsync>(store, 30)(store, context, &async) == S_OK && finished(&async) &&
              at<Count>(store, 31)(store, &async, &count) == S_OK && count == 0 &&
              at<Items>(store, 32)(store, &async, 0, nullptr) == S_OK,
          "no add-on licenses");
    async = XAsyncBlock{};
    count = 99;
    check(at<CtxAsync>(store, 55)(store, context, &async) == S_OK && finished(&async) &&
              at<Count>(store, 56)(store, &async, &count) == S_OK && count == 0,
          "no game or DLC package updates");

    // ---- package licenses ----
    async = XAsyncBlock{};
    Handle package_license = nullptr;
    check(at<CtxIdAsync>(store, 20)(store, context, "SomePackage", &async) == S_OK && finished(&async) &&
              at<HandleResult>(store, 21)(store, &async, &package_license) == S_OK && package_license,
          "XStoreAcquireLicenseForPackage hands out a license");
    check(at<IsValid>(store, 22)(store, package_license), "the package license is valid");
    async = XAsyncBlock{};
    CanAcquire can{};
    check(at<CtxIdAsync>(store, 26)(store, context, "SomePackage", &async) == S_OK && finished(&async) &&
              at<CanResult>(store, 27)(store, &async, &can) == S_OK && can.status == 1,
          "XStoreCanAcquireLicenseForPackage: Licensable");
    XTaskQueueRegistrationToken token{};
    check(at<Register>(store, 68)(store, package_license, nullptr, nullptr, on_change, &token) == S_OK &&
              token.token != 0 && at<Unregister>(store, 69)(store, package_license, token, false),
          "license-lost registration hands out a token; unregistering is never pending");
    at<Close>(store, 23)(store, package_license);
    check(!at<IsValid>(store, 22)(store, package_license), "a closed license is not valid");

    // ---- product queries are empty ----
    async = XAsyncBlock{};
    Handle products = nullptr;
    const char* ids[] = {"9NBLGGH4R315"};
    check(at<ProductsAsync>(store, 7)(store, context, 1, ids, 1, nullptr, 0, &async) == S_OK && finished(&async) &&
              at<HandleResult>(store, 8)(store, &async, &products) == S_OK && products,
          "XStoreQueryProducts completes with a query handle");
    check(at<Enumerate>(store, 15)(store, products, nullptr, reinterpret_cast<void*>(&on_product)) == S_OK &&
              !at<IsValid>(store, 16)(store, products),
          "the query has no products and no further pages");
    at<Close>(store, 19)(store, products);

    // ---- what needs the service fails before starting ----
    check(at<NoArgs>(store, 49)(store) == E_FAIL, "XStoreShowPurchaseUIAsync is not available offline");
    check(at<NoArgs>(store, 46)(store) == E_NOTIMPL, "an undocumented slot answers E_NOTIMPL");

    check(at<Register>(store, 66)(store, context, nullptr, nullptr, on_change, &token) == S_OK &&
              at<Unregister>(store, 67)(store, context, token, true),
          "license-changed registration");
    check(g_callbacks == 0, "no callback ran: nothing changes offline");
    at<Close>(store, 4)(store, context);
    at<Close>(store, 4)(store, context);   // twice: ignored
    check(rt::shims::xgameruntime_store_open_handles() == 0, "every handle the test opened is closed");

    std::printf("xgameruntime_store_test: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
