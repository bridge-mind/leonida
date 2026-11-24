// xgameruntime: XGameEventImpl, XGameUiImpl, XGameStreamingImpl, XDisplayImpl, XUserDeviceImpl, XAccessibilityImpl and
// XLauncherImpl, offline: small classes a title calls early.
//
// Each XGameEvent*, XGameUi* and XGameStreaming* function in xgameruntime.lib asks QueryApiImpl for its class and
// calls one slot with its own arguments after `this`. The class and interface ids and the method order are
// xodus-gaming/xgameruntime-docs' COM pages (GPL-3.0; facts only, nothing copied); IUnknown takes slots 0-2, so each
// class's first method is slot 3. Signatures follow Microsoft's public GDK reference (learn.microsoft.com/gaming/gdk).
// No GDK header or library code is used.
//
//   XGameEventImpl      XGameEventWrite (telemetry) succeeds and is dropped; the first event's name is logged, and
//                       LEONIDA_LOG_GAME_EVENTS=1 logs every one.
//   XGameUiImpl         a message dialog or error dialog has no screen: its title and text go to the log and it
//                       completes at once with the dialog's default button (error dialogs: dismissed). Every other UI
//                       (profile cards, pickers, text entry, web authentication, invites) fails before starting.
//   XGameStreamingImpl  the game is never streamed: initialising succeeds, IsStreaming is false, there are no
//                       clients, the connection-state registration hands out a token that never fires.
//   XDisplayImpl        XDisplayTryEnableHdrMode: Disabled (the host output is SDR), Enabled with LEONIDA_HDR=1.
//   XUserDeviceImpl     every device belongs to the offline user; associations and audio endpoints never change; the
//                       user has no audio endpoint of its own; finding a controller shows no UI (as on desktop).
//   XAccessibilityImpl  high contrast is Off, closed captions can't be configured, no voice is installed (enumerating
//                       calls back zero times), XSpeechSynthesizerCreate fails (E_FAIL) with a null handle, the other
//                       speech calls refuse the handle they were never given (E_INVALIDARG), speech-to-text display
//                       calls fail (E_FAIL).
//   XLauncherImpl       XLaunchUri and the display timeout deferral calls fail (E_FAIL); nothing is launched.
//
// Slots whose signature the public pages don't give answer with a value that is safe whichever way the caller reads
// it: see `client_not_connected` (XGameStreaming) and `offline_failure`.
#include <windows.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <mutex>

#include <XAsync.h>
#include <XAsyncProvider.h>
#include <XTaskQueue.h>

namespace rt::shims {
void* xgameruntime_user_new_handle();   // xgameruntime_user.cpp
}

namespace {

using XUserHandle = void*;

// XGameErr.h values, from the GDK "Error codes" page.
constexpr HRESULT E_GAMESTREAMING_NOT_INITIALIZED = static_cast<HRESULT>(0x89245400);

constexpr GUID kIUnknown = {0x00000000, 0x0000, 0x0000, {0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46}};

#define FN(f) reinterpret_cast<const void*>(&f)

// A process-wide singleton per class, as the other runtime classes: AddRef and Release don't free.
ULONG STDMETHODCALLTYPE object_add_ref(void*) { return 2; }
ULONG STDMETHODCALLTYPE object_release(void*) { return 1; }

// One class: its object (a pointer to its vtable) and the interface ids it answers with that vtable.
struct ClassObject {
    const void* const* vtable;
};

HRESULT answer(ClassObject* object, const GUID* begin, const GUID* end, REFIID iid, void** out) {
    if (!out) return E_POINTER;
    *out = nullptr;
    if (iid == kIUnknown) {
        *out = object;
        return S_OK;
    }
    for (const GUID* id = begin; id != end; ++id)
        if (iid == *id) {
            *out = object;
            return S_OK;
        }
    return E_NOINTERFACE;
}

// A slot the docs don't name: E_NOTIMPL, so nothing reads outputs it didn't write.
template <const char* Class, int Slot> HRESULT STDMETHODCALLTYPE unknown_slot(void*) {
    static std::atomic<bool> s_logged{false};
    if (!s_logged.exchange(true))
        std::fprintf(stderr, "[rt] xgameruntime: %s slot %d (unknown in the docs) called; E_NOTIMPL\n", Class, Slot);
    return E_NOTIMPL;
}

// A UI the offline runtime doesn't have. The *Async function fails before starting, so the XAsyncBlock is never used
// and no callback runs; its Result partners return the same error. Arguments don't matter (x64: the caller owns the
// argument area).
template <const char* Class, int Slot> HRESULT STDMETHODCALLTYPE offline_failure(void*) {
    static std::atomic<bool> s_logged{false};
    if (!s_logged.exchange(true))
        std::fprintf(stderr, "[rt] xgameruntime: %s slot %d is not available offline (E_FAIL)\n", Class, Slot);
    return E_FAIL;
}

// ======================= XGameEventImpl =======================

constexpr char kEventName[] = "XGameEventImpl";
constexpr GUID kEventClass = {0xbbfbdcc7, 0xbfe7, 0x409b, {0xa5, 0xca, 0xed, 0xf0, 0x54, 0x96, 0x0b, 0x4d}};
std::atomic<long> g_events{0};

bool log_every_event() {
    static const bool on = [] {
        char value[4] = {};
        size_t length = 0;
        return getenv_s(&length, value, "LEONIDA_LOG_GAME_EVENTS") == 0 && length == 2 && value[0] == '1';
    }();
    return on;
}

// 3: HRESULT XGameEventWrite(XUserHandle user, const char* serviceConfigId, const char* playSessionId,
//                            const char* eventName, const char* dimensionsJson, const char* measurementsJson)
HRESULT STDMETHODCALLTYPE event_write(void*, XUserHandle, const char*, const char*, const char* name,
                                      const char* dimensions, const char* measurements) {
    if (!name) return E_INVALIDARG;
    const long n = g_events.fetch_add(1);
    if (log_every_event())
        std::fprintf(stderr, "[rt] game event %s dimensions=%s measurements=%s\n", name, dimensions ? dimensions : "",
                     measurements ? measurements : "");
    else if (n == 0)
        std::fprintf(stderr, "[rt] xgameruntime: game events are dropped offline (first: %s; "
                             "LEONIDA_LOG_GAME_EVENTS=1 logs them)\n", name);
    return S_OK;
}

HRESULT STDMETHODCALLTYPE event_query_interface(void*, REFIID iid, void** out);
const void* const kEventVtable[] = {
    FN(event_query_interface), FN(object_add_ref), FN(object_release),   // 0-2 IUnknown
    FN(event_write),                                                     // 3 XGameEventWrite
};
ClassObject g_event_object{kEventVtable};
const GUID kEventInterfaces[] = {kEventClass};
HRESULT STDMETHODCALLTYPE event_query_interface(void*, REFIID iid, void** out) {
    return answer(&g_event_object, std::begin(kEventInterfaces), std::end(kEventInterfaces), iid, out);
}

// ======================= XGameUiImpl =======================

constexpr char kUiName[] = "XGameUiImpl";
constexpr GUID kUiClass = {0xdfcd4649, 0x4ff8, 0x4043, {0xba, 0x07, 0x35, 0xd6, 0x07, 0xdf, 0x98, 0xb0}};
const GUID kUiInterfaces[] = {
    kUiClass,   // the class id, which a thunk may pass for its implemented interface (IXGameUiImpl4)
    {0x6eeaa73e, 0x9669, 0x43ad, {0xa2, 0xc7, 0xd0, 0xda, 0x4e, 0x1f, 0x50, 0xa1}},   // IXGameUiImpl
    {0x36a03122, 0x9ea3, 0x4a3a, {0xa8, 0xa4, 0x89, 0x9c, 0xfd, 0x85, 0xd7, 0xdb}},   // IXGameUiImpl2
    {0xade7eba1, 0x2093, 0x42ce, {0xa5, 0x44, 0xa5, 0x23, 0xa6, 0x67, 0x90, 0xe0}},   // IXGameUiImpl3
    {0xeaf669df, 0x5542, 0x4590, {0x99, 0xa3, 0x8d, 0xc0, 0x61, 0xf8, 0x37, 0xcc}},   // IXGameUiImpl4
};

enum class XGameUiMessageDialogButton : uint32_t { First = 0, Second = 1, Third = 2 };

const char kDialogIdentity = 0, kErrorDialogIdentity = 0;

// The answer of a message dialog rides in the async context: the button, which never needs more.
HRESULT CALLBACK dialog_provider(XAsyncOp op, const XAsyncProviderData* data) {
    switch (op) {
    case XAsyncOp::Begin:
        return XAsyncSchedule(data->async, 0);
    case XAsyncOp::DoWork:
        XAsyncComplete(data->async, S_OK, sizeof(XGameUiMessageDialogButton));
        return S_OK;
    case XAsyncOp::GetResult:
        if (!data->buffer || data->bufferSize < sizeof(XGameUiMessageDialogButton)) return E_INVALIDARG;
        *static_cast<XGameUiMessageDialogButton*>(data->buffer) =
            static_cast<XGameUiMessageDialogButton>(reinterpret_cast<uintptr_t>(data->context));
        return S_OK;
    default:
        return S_OK;
    }
}

HRESULT CALLBACK error_dialog_provider(XAsyncOp op, const XAsyncProviderData* data) {
    if (op == XAsyncOp::Begin) return XAsyncSchedule(data->async, 0);
    if (op == XAsyncOp::DoWork) XAsyncComplete(data->async, S_OK, 0);
    return S_OK;
}

// 3: HRESULT XGameUiShowMessageDialogAsync(XAsyncBlock* async, const char* titleText, const char* contentText,
//        const char* firstButtonText, const char* secondButtonText, const char* thirdButtonText,
//        XGameUiMessageDialogButton defaultButton, XGameUiMessageDialogButton cancelButton)
// The parameter list is the one the GDK reference gives; only the result-page and the enum were checked for this
// file. No caller confirms it: tools/gdk_runtime_calls.py finds no thunk for this slot in the executable
// (docs/runtime_imports.md). The dialog is "answered" with its default button.
HRESULT STDMETHODCALLTYPE ui_message_dialog_async(void*, XAsyncBlock* async, const char* title, const char* content,
                                                  const char* first, const char* second, const char* third,
                                                  XGameUiMessageDialogButton default_button, XGameUiMessageDialogButton) {
    if (!async) return E_POINTER;
    const auto button = static_cast<uint32_t>(default_button);
    if (button > 2) return E_INVALIDARG;
    const char* labels[] = {first, second, third};
    std::fprintf(stderr, "[rt] game dialog: \"%s\": %s [buttons: %s%s%s%s%s; answered: %s]\n", title ? title : "",
                 content ? content : "", first ? first : "", second ? " / " : "", second ? second : "",
                 third ? " / " : "", third ? third : "", labels[button] ? labels[button] : "(default)");
    return XAsyncBegin(async, reinterpret_cast<void*>(static_cast<uintptr_t>(button)), &kDialogIdentity,
                       "XGameUiShowMessageDialogAsync", dialog_provider);
}

// 4: HRESULT XGameUiShowMessageDialogResult(XAsyncBlock* async, XGameUiMessageDialogButton* resultButton)
HRESULT STDMETHODCALLTYPE ui_message_dialog_result(void*, XAsyncBlock* async, XGameUiMessageDialogButton* button) {
    if (!button) return E_POINTER;
    return XAsyncGetResult(async, &kDialogIdentity, sizeof(*button), button, nullptr);
}

// 14: HRESULT XGameUiShowErrorDialogAsync(XAsyncBlock* async, HRESULT errorCode, const char* context)
HRESULT STDMETHODCALLTYPE ui_error_dialog_async(void*, XAsyncBlock* async, HRESULT code, const char* context) {
    if (!async) return E_POINTER;
    std::fprintf(stderr, "[rt] game error dialog: HRESULT 0x%08lX%s%s [dismissed]\n", static_cast<unsigned long>(code),
                 context ? " context " : "", context ? context : "");
    return XAsyncBegin(async, nullptr, &kErrorDialogIdentity, "XGameUiShowErrorDialogAsync", error_dialog_provider);
}

// 15: HRESULT XGameUiShowErrorDialogResult(XAsyncBlock* async)
HRESULT STDMETHODCALLTYPE ui_error_dialog_result(void*, XAsyncBlock* async) {
    return XAsyncGetResult(async, &kErrorDialogIdentity, 0, nullptr, nullptr);
}

// 16: XGameUiSetNotificationPositionHint(position): there are no notifications to place.
HRESULT STDMETHODCALLTYPE ui_set_notification_hint(void*) { return S_OK; }

template <int Slot> HRESULT STDMETHODCALLTYPE ui_offline(void* self) { return offline_failure<kUiName, Slot>(self); }
template <int Slot> HRESULT STDMETHODCALLTYPE ui_unknown(void* self) { return unknown_slot<kUiName, Slot>(self); }

HRESULT STDMETHODCALLTYPE ui_query_interface(void*, REFIID iid, void** out);
const void* const kUiVtable[] = {
    FN(ui_query_interface), FN(object_add_ref), FN(object_release),   // 0-2 IUnknown
    FN(ui_message_dialog_async),       // 3  XGameUiShowMessageDialogAsync
    FN(ui_message_dialog_result),      // 4  XGameUiShowMessageDialogResult
    FN(ui_offline<5>),                 // 5  XGameUiShowSendGameInviteAsync
    FN(ui_offline<6>),                 // 6  XGameUiShowSendGameInviteResult
    FN(ui_offline<7>),                 // 7  XGameUiShowPlayerProfileCardAsync
    FN(ui_offline<8>),                 // 8  XGameUiShowPlayerProfileCardResult
    FN(ui_offline<9>),                 // 9  XGameUiShowAchievementsAsync
    FN(ui_offline<10>),                // 10 XGameUiShowAchievementsResult
    FN(ui_offline<11>),                // 11 XGameUiShowPlayerPickerAsync
    FN(ui_offline<12>),                // 12 XGameUiShowPlayerPickerResultCount
    FN(ui_offline<13>),                // 13 XGameUiShowPlayerPickerResult
    FN(ui_error_dialog_async),         // 14 XGameUiShowErrorDialogAsync
    FN(ui_error_dialog_result),        // 15 XGameUiShowErrorDialogResult
    FN(ui_set_notification_hint),      // 16 XGameUiSetNotificationPositionHint
    FN(ui_offline<17>),                // 17 XGameUiShowTextEntryAsync
    FN(ui_offline<18>),                // 18 XGameUiShowTextEntryResultSize
    FN(ui_offline<19>),                // 19 XGameUiShowTextEntryResult
    FN(ui_unknown<20>),                // 20 (unknown in the docs)
    FN(ui_unknown<21>),                // 21 (unknown in the docs)
    FN(ui_unknown<22>),                // 22 (unknown in the docs)
    FN(ui_unknown<23>),                // 23 (unknown in the docs)
    FN(ui_offline<24>),                // 24 XGameUiShowWebAuthenticationAsync
    FN(ui_offline<25>),                // 25 XGameUiShowWebAuthenticationResultSize
    FN(ui_offline<26>),                // 26 XGameUiShowWebAuthenticationResult
    FN(ui_offline<27>),                // 27 XGameUiShowWebAuthenticationWithOptionsAsync
    FN(ui_unknown<28>),                // 28 (unknown in the docs)
    FN(ui_unknown<29>),                // 29 (unknown in the docs)
    FN(ui_offline<30>),                // 30 XGameUiShowMultiplayerActivityGameInviteAsync (2)
    FN(ui_offline<31>),                // 31 XGameUiShowMultiplayerActivityGameInviteResult (2)
    FN(ui_unknown<32>),                // 32 (unknown in the docs) (2)
    FN(ui_unknown<33>),                // 33 (unknown in the docs) (2)
    FN(ui_offline<34>),                // 34 XGameUiTextEntryOpen (2)
    FN(ui_offline<35>),                // 35 XGameUiTextEntryClose (2)
    FN(ui_offline<36>),                // 36 XGameUiTextEntryGetState (2)
    FN(ui_offline<37>),                // 37 XGameUiTextEntryGetExtents (2)
    FN(ui_offline<38>),                // 38 XGameUiTextEntryUpdatePositionHint (2)
    FN(ui_offline<39>),                // 39 XGameUiTextEntryUpdateVisibility (2)
    FN(ui_offline<40>),                // 40 XGameUiShowStateShareAsync (3)
    FN(ui_offline<41>),                // 41 XGameUiShowStateShareResult (3)
    FN(ui_unknown<42>),                // 42 XGameUiSetUiCallbacks (4, PC-hosted UI; signature not given)
    FN(ui_unknown<43>),                // 43 XGameUiSetMessageDialogUiResponse (4)
    FN(ui_unknown<44>),                // 44 XGameUiSetPlayerPickerUiResponse (4)
    FN(ui_unknown<45>),                // 45 XGameUiSetTextEntryUiResponse (4)
    FN(ui_unknown<46>),                // 46 XGameUiSetPlayerProfileCardUiResponse (4)
    FN(ui_unknown<47>),                // 47 XGameUiSetSendGameInviteUiResponse (4)
    FN(ui_unknown<48>),                // 48 XGameUiSetAchievementsUiResponse (4)
    FN(ui_unknown<49>),                // 49 XGameUiSetMultiplayerActivityGameInviteUiResponse (4)
    FN(ui_unknown<50>),                // 50 XGameUiSetErrorDialogUiResponse (4)
};
static_assert(std::size(kUiVtable) == 51, "IUnknown + IXGameUiImpl's 27 methods + 10 + 2 + 9");
ClassObject g_ui_object{kUiVtable};
HRESULT STDMETHODCALLTYPE ui_query_interface(void*, REFIID iid, void** out) {
    return answer(&g_ui_object, std::begin(kUiInterfaces), std::end(kUiInterfaces), iid, out);
}

// ======================= XGameStreamingImpl =======================

constexpr GUID kStreamingClass = {0x0a2192aa, 0xb2d5, 0x4d58, {0x83, 0xbe, 0x38, 0x3b, 0x6d, 0x80, 0x79, 0x9e}};
const GUID kStreamingInterfaces[] = {
    kStreamingClass,   // the class id, for its implemented interface (IXGameStreamingImpl3)
    {0x8aff07f5, 0xa1bf, 0x4db8, {0x80, 0xa5, 0x31, 0xcc, 0xa0, 0xde, 0x51, 0xb7}},   // IXGameStreamingImpl
    {0x5f5e5169, 0x746c, 0x4001, {0xad, 0x1c, 0xda, 0x72, 0x8d, 0x01, 0xc9, 0xeb}},   // IXGameStreamingImpl2
    {0x57786622, 0x6605, 0x46d0, {0xb9, 0x17, 0x0f, 0x22, 0xbb, 0xcd, 0x9c, 0x52}},   // IXGameStreamingImpl3
};

std::mutex g_streaming_mutex;
uint64_t g_next_token = 1;

// 3: HRESULT XGameStreamingInitialize()
HRESULT STDMETHODCALLTYPE streaming_initialize(void*) { return S_OK; }
// 4: void XGameStreamingUninitialize()
void STDMETHODCALLTYPE streaming_uninitialize(void*) {}
// 5: bool XGameStreamingIsStreaming(): no client is ever connected.
bool STDMETHODCALLTYPE streaming_is_streaming(void*) { return false; }
// 9: uint32_t XGameStreamingGetClientCount()
uint32_t STDMETHODCALLTYPE streaming_client_count(void*) { return 0; }

// 12: HRESULT XGameStreamingRegisterConnectionStateChanged(XTaskQueueHandle queue, void* context,
//        XGameStreamingConnectionStateChangedCallback* callback, XTaskQueueRegistrationToken* token)
// That parameter list is the GDK reference's, not re-checked for this file, and no caller confirms it: the executable
// links no XGameStreamingImpl thunk (docs/runtime_imports.md). No client connects, so the callback never runs.
HRESULT STDMETHODCALLTYPE streaming_register_connection(void*, XTaskQueueHandle, void*, void* callback,
                                                        XTaskQueueRegistrationToken* token) {
    if (!callback || !token) return E_INVALIDARG;
    std::lock_guard<std::mutex> lock(g_streaming_mutex);
    token->token = g_next_token++;
    return S_OK;
}

// Unregistering (7, 13): nothing is ever pending. Returns true, in case the caller reads a bool; a void caller
// ignores it.
bool STDMETHODCALLTYPE streaming_unregister(void*) { return true; }

// Every other slot asks about a client, its touch controls or its stream, and there is none. The answer is
// E_GAMESTREAMING_NOT_INITIALIZED (0x89245400), chosen for its shape: a caller reading an HRESULT sees a failure, and
// one reading a bool (al) sees false, since its low byte is 0. The slots returning an enum or a count are the ones
// above, answered explicitly.
template <int Slot> HRESULT STDMETHODCALLTYPE client_not_connected(void*) {
    static std::atomic<bool> s_logged{false};
    if (!s_logged.exchange(true))
        std::fprintf(stderr, "[rt] xgameruntime: XGameStreamingImpl slot %d: no streaming client offline\n", Slot);
    return E_GAMESTREAMING_NOT_INITIALIZED;
}
static_assert((static_cast<uint32_t>(E_GAMESTREAMING_NOT_INITIALIZED) & 0xFF) == 0, "reads as false in al");

// 11: XGameStreamingConnectionState XGameStreamingGetConnectionState(XGameStreamingClientId client): 0, the first
// enumerator, taken to be Disconnected; there is no client to ask about either way, and the executable has no caller.
uint32_t STDMETHODCALLTYPE streaming_connection_state(void*) { return 0; }

HRESULT STDMETHODCALLTYPE streaming_query_interface(void*, REFIID iid, void** out);
const void* const kStreamingVtable[] = {
    FN(streaming_query_interface), FN(object_add_ref), FN(object_release),   // 0-2 IUnknown
    FN(streaming_initialize),              // 3  XGameStreamingInitialize
    FN(streaming_uninitialize),            // 4  XGameStreamingUninitialize
    FN(streaming_is_streaming),            // 5  XGameStreamingIsStreaming
    FN(client_not_connected<6>),           // 6  XGameStreamingRegisterClientPropertiesChanged (per client)
    FN(streaming_unregister),              // 7  XGameStreamingUnregisterClientPropertiesChanged
    FN(client_not_connected<8>),           // 8  XGameStreamingGetStreamPhysicalDimensions
    FN(streaming_client_count),            // 9  XGameStreamingGetClientCount
    FN(client_not_connected<10>),          // 10 XGameStreamingGetClients
    FN(streaming_connection_state),        // 11 XGameStreamingGetConnectionState
    FN(streaming_register_connection),     // 12 XGameStreamingRegisterConnectionStateChanged
    FN(streaming_unregister),              // 13 XGameStreamingUnregisterConnectionStateChanged
    FN(client_not_connected<14>),          // 14 XGameStreamingGetStreamAddedLatency
    FN(client_not_connected<15>),          // 15 XGameStreamingGetServerLocationNameSize
    FN(client_not_connected<16>),          // 16 XGameStreamingGetServerLocationName
    FN(client_not_connected<17>),          // 17 XGameStreamingHideTouchControls
    FN(client_not_connected<18>),          // 18 XGameStreamingShowTouchControlLayout
    FN(client_not_connected<19>),          // 19 XGameStreamingHideTouchControlsOnClient
    FN(client_not_connected<20>),          // 20 XGameStreamingShowTouchControlLayoutOnClient
    FN(client_not_connected<21>),          // 21 XGameStreamingIsTouchInputEnabled (reads as false)
    FN(client_not_connected<22>),          // 22 XGameStreamingGetLastFrameDisplayed
    FN(client_not_connected<23>),          // 23 XGameStreamingGetAssociatedFrame
    FN(client_not_connected<24>),          // 24 XGameStreamingGetGamepadPhysicality
    FN(client_not_connected<25>),          // 25 XGameStreamingUpdateTouchControlsState
    FN(client_not_connected<26>),          // 26 XGameStreamingUpdateTouchControlsStateOnClient
    FN(client_not_connected<27>),          // 27 XGameStreamingShowTouchControlsWithStateUpdate
    FN(client_not_connected<28>),          // 28 XGameStreamingShowTouchControlsWithStateUpdateOnClient
    FN(client_not_connected<29>),          // 29 XGameStreamingGetTouchBundleVersionNameSize
    FN(client_not_connected<30>),          // 30 XGameStreamingGetTouchBundleVersion
    FN(client_not_connected<31>),          // 31 XGameStreamingGetClientIPAddress
    FN(client_not_connected<32>),          // 32 XGameStreamingGetSessionId (2)
    FN(client_not_connected<33>),          // 33 XGameStreamingGetDisplayDetails (3)
    FN(client_not_connected<34>),          // 34 XGameStreamingSetResolution (3)
};
static_assert(std::size(kStreamingVtable) == 35, "IUnknown + IXGameStreamingImpl's 29 methods + 1 + 2");
ClassObject g_streaming_object{kStreamingVtable};
HRESULT STDMETHODCALLTYPE streaming_query_interface(void*, REFIID iid, void** out) {
    return answer(&g_streaming_object, std::begin(kStreamingInterfaces), std::end(kStreamingInterfaces), iid, out);
}


// ======================= XDisplayImpl =======================

constexpr char kDisplayName[] = "XDisplayImpl";
constexpr GUID kDisplayClass = {0x03f0fe74, 0xfdd9, 0x4e5c, {0xb6, 0x30, 0xf9, 0x33, 0x9c, 0x47, 0xac, 0xc5}};
const GUID kDisplayInterfaces[] = {
    kDisplayClass,
    {0x35f07670, 0x706e, 0x4bfb, {0x94, 0x76, 0x09, 0x07, 0x98, 0xc5, 0xeb, 0xf3}},   // IXDisplayImpl
};

enum class XDisplayHdrModePreference : uint32_t { PreferHdr = 0, PreferRefreshRate = 1 };
enum class XDisplayHdrModeResult : uint32_t { Unknown = 0, Enabled = 1, Disabled = 2 };
struct XDisplayHdrModeInfo {
    float minToneMapLuminance;            // nits
    float maxToneMapLuminance;
    float maxFullFrameToneMapLuminance;
};

bool hdr_requested() {
    static const bool on = [] {
        char value[4] = {};
        size_t length = 0;
        return getenv_s(&length, value, "LEONIDA_HDR") == 0 && length == 2 && value[0] == '1';
    }();
    return on;
}

// 4: XDisplayHdrModeResult XDisplayTryEnableHdrMode(XDisplayHdrModePreference preference, XDisplayHdrModeInfo* info)
// The host swapchain is SDR, so HDR stays off unless LEONIDA_HDR=1; then it is "enabled" with the documented
// defaults (0.01 / 1000 / 1000 nits).
uint32_t STDMETHODCALLTYPE display_try_enable_hdr(void*, XDisplayHdrModePreference, XDisplayHdrModeInfo* info) {
    const bool on = hdr_requested();
    if (info) *info = on ? XDisplayHdrModeInfo{0.01f, 1000.0f, 1000.0f} : XDisplayHdrModeInfo{};
    static std::atomic<bool> s_logged{false};
    if (!s_logged.exchange(true))
        std::fprintf(stderr, "[rt] xgameruntime: XDisplayTryEnableHdrMode: HDR %s\n",
                     on ? "enabled (LEONIDA_HDR=1)" : "disabled (SDR output; LEONIDA_HDR=1 asks for HDR)");
    return static_cast<uint32_t>(on ? XDisplayHdrModeResult::Enabled : XDisplayHdrModeResult::Disabled);
}

template <int Slot> HRESULT STDMETHODCALLTYPE display_unknown(void* self) { return unknown_slot<kDisplayName, Slot>(self); }

HRESULT STDMETHODCALLTYPE display_query_interface(void*, REFIID iid, void** out);
const void* const kDisplayVtable[] = {
    FN(display_query_interface), FN(object_add_ref), FN(object_release),   // 0-2 IUnknown
    FN(display_unknown<3>),                 // 3 (unknown in the docs)
    FN(display_try_enable_hdr),             // 4 XDisplayTryEnableHdrMode
};
ClassObject g_display_object{kDisplayVtable};
HRESULT STDMETHODCALLTYPE display_query_interface(void*, REFIID iid, void** out) {
    return answer(&g_display_object, std::begin(kDisplayInterfaces), std::end(kDisplayInterfaces), iid, out);
}

// ======================= XUserDeviceImpl =======================

constexpr GUID kUserDeviceClass = {0x7d824997, 0x10dc, 0x45ab, {0x86, 0xb7, 0x27, 0x37, 0x76, 0x7c, 0x0b, 0xf1}};
const GUID kUserDeviceInterfaces[] = {
    kUserDeviceClass,                                                                  // IXUserDeviceImpl
    {0x0cc6a956, 0xe7e1, 0x4fdf, {0x93, 0x41, 0x9d, 0x5d, 0xa6, 0x49, 0xeb, 0xc8}},    // IXUserDeviceImpl2 (no methods)
};

// APP_LOCAL_DEVICE_ID is taken to be 32 bytes (APP_LOCAL_DEVICE_ID_SIZE), not on the pages read for this file. No
// caller confirms it: the executable links no XUserDeviceImpl thunk (docs/runtime_imports.md).
struct APP_LOCAL_DEVICE_ID {
    uint8_t value[32];
};

const char kFindControllerIdentity = 0;

// 3: HRESULT XUserFindForDevice(const APP_LOCAL_DEVICE_ID* deviceId, XUserHandle* handle): every device belongs to
// the one offline user.
HRESULT STDMETHODCALLTYPE device_find_user(void*, const APP_LOCAL_DEVICE_ID* device, XUserHandle* out) {
    if (!device || !out) return E_INVALIDARG;
    *out = rt::shims::xgameruntime_user_new_handle();
    return S_OK;
}

// 4: HRESULT XUserRegisterForDeviceAssociationChanged(XTaskQueueHandle queue, void* context,
//        XUserDeviceAssociationChangedCallback* callback, XTaskQueueRegistrationToken* token)
// 7: HRESULT XUserRegisterForDefaultAudioEndpointUtf16Changed(XTaskQueueHandle, void*, callback, token*)
// Associations and endpoints never change offline: a token, no callback.
HRESULT STDMETHODCALLTYPE device_register(void*, XTaskQueueHandle, void*, void* callback,
                                          XTaskQueueRegistrationToken* token) {
    if (!callback || !token) return E_INVALIDARG;
    std::lock_guard<std::mutex> lock(g_streaming_mutex);
    token->token = g_next_token++;
    return S_OK;
}

// 5, 8: bool ...Unregister...(XTaskQueueRegistrationToken token, bool wait): never pending.
bool STDMETHODCALLTYPE device_unregister(void*, XTaskQueueRegistrationToken, bool) { return true; }

// 6: HRESULT XUserGetDefaultAudioEndpointUtf16(XUserLocalId user, XUserDefaultAudioEndpointKind kind,
//        size_t endpointIdUtf16Count, wchar_t* endpointIdUtf16, size_t* endpointIdUtf16Used)
// The user has no audio endpoint of its own (no headset): ERROR_NOT_FOUND, with an empty string written.
HRESULT STDMETHODCALLTYPE device_audio_endpoint(void*, uint64_t, uint32_t, size_t count, wchar_t* id, size_t* used) {
    if (id && count) id[0] = L'\0';
    if (used) *used = 0;
    return HRESULT_FROM_WIN32(ERROR_NOT_FOUND);
}

HRESULT CALLBACK find_controller_provider(XAsyncOp op, const XAsyncProviderData* data) {
    switch (op) {
    case XAsyncOp::Begin:
        return XAsyncSchedule(data->async, 0);
    case XAsyncOp::DoWork:
        XAsyncComplete(data->async, S_OK, sizeof(APP_LOCAL_DEVICE_ID));
        return S_OK;
    case XAsyncOp::GetResult:
        if (!data->buffer || data->bufferSize < sizeof(APP_LOCAL_DEVICE_ID)) return E_INVALIDARG;
        *static_cast<APP_LOCAL_DEVICE_ID*>(data->buffer) = APP_LOCAL_DEVICE_ID{};
        return S_OK;
    default:
        return S_OK;
    }
}

// 9: HRESULT XUserFindControllerForUserWithUiAsync(XUserHandle user, XAsyncBlock* async): no UI, as on desktop.
HRESULT STDMETHODCALLTYPE device_find_controller_async(void*, XUserHandle user, XAsyncBlock* async) {
    if (!user) return E_INVALIDARG;
    if (!async) return E_POINTER;
    return XAsyncBegin(async, nullptr, &kFindControllerIdentity, "XUserFindControllerForUserWithUiAsync",
                       find_controller_provider);
}

// 10: HRESULT XUserFindControllerForUserWithUiResult(XAsyncBlock* async, APP_LOCAL_DEVICE_ID* deviceId): all zero,
// as the GDK documents for desktop.
HRESULT STDMETHODCALLTYPE device_find_controller_result(void*, XAsyncBlock* async, APP_LOCAL_DEVICE_ID* device) {
    if (!device) return E_POINTER;
    return XAsyncGetResult(async, &kFindControllerIdentity, sizeof(*device), device, nullptr);
}

HRESULT STDMETHODCALLTYPE device_query_interface(void*, REFIID iid, void** out);
const void* const kUserDeviceVtable[] = {
    FN(device_query_interface), FN(object_add_ref), FN(object_release),   // 0-2 IUnknown
    FN(device_find_user),                   // 3  XUserFindForDevice
    FN(device_register),                    // 4  XUserRegisterForDeviceAssociationChanged
    FN(device_unregister),                  // 5  XUserUnregisterForDeviceAssociationChanged
    FN(device_audio_endpoint),              // 6  XUserGetDefaultAudioEndpointUtf16
    FN(device_register),                    // 7  XUserRegisterForDefaultAudioEndpointUtf16Changed
    FN(device_unregister),                  // 8  XUserUnregisterForDefaultAudioEndpointUtf16Changed
    FN(device_find_controller_async),       // 9  XUserFindControllerForUserWithUiAsync
    FN(device_find_controller_result),      // 10 XUserFindControllerForUserWithUiResult
};
static_assert(std::size(kUserDeviceVtable) == 11, "IUnknown + IXUserDeviceImpl's 8 methods");
ClassObject g_user_device_object{kUserDeviceVtable};
HRESULT STDMETHODCALLTYPE device_query_interface(void*, REFIID iid, void** out) {
    return answer(&g_user_device_object, std::begin(kUserDeviceInterfaces), std::end(kUserDeviceInterfaces), iid, out);
}

// ======================= XAccessibilityImpl =======================

constexpr char kAccessibilityName[] = "XAccessibilityImpl";
constexpr GUID kAccessibilityClass = {0x3e241977, 0x6237, 0x41e9, {0x85, 0x59, 0x10, 0x2c, 0x2d, 0x99, 0x83, 0xf1}};
const GUID kAccessibilityInterfaces[] = {
    kAccessibilityClass,   // IXAccessibilityImpl: the class id
    {0xd722b373, 0x8c4d, 0x4692, {0x9e, 0x51, 0xd6, 0xad, 0x9b, 0x37, 0xaa, 0x7d}},   // IXAccessibilityImpl2
};

// 5: HRESULT XHighContrastGetMode(XHighContrastMode* mode): Off, the first enumerator. ASSUMPTION: the signature and
// that Off is 0 (GDK reference, not re-checked for this file).
HRESULT STDMETHODCALLTYPE accessibility_high_contrast(void*, uint32_t* mode) {
    if (!mode) return E_POINTER;
    *mode = 0;
    return S_OK;
}

// 4: HRESULT XClosedCaptionSetEnabled(bool enabled): there are no captions to show; accepted and dropped.
HRESULT STDMETHODCALLTYPE accessibility_captions_set_enabled(void*, bool) { return S_OK; }

// 8: HRESULT XSpeechSynthesizerEnumerateInstalledVoices(callback, void* context): no voice is installed, so the
// callback runs zero times. ASSUMPTION: the argument order (GDK reference), only the count (2) was measured.
HRESULT STDMETHODCALLTYPE speech_enumerate_voices(void*, void* callback, void*) {
    return callback ? S_OK : E_INVALIDARG;
}

// 9: HRESULT XSpeechSynthesizerCreate(XSpeechSynthesizerHandle* handle): no synthesizer offline.
HRESULT STDMETHODCALLTYPE speech_create(void* self, void** handle) {
    if (handle) *handle = nullptr;
    return offline_failure<kAccessibilityName, 9>(self);
}

// 10-12, 14: a synthesizer or stream handle is the first argument, and none was ever handed out: E_INVALIDARG.
HRESULT STDMETHODCALLTYPE speech_bad_handle(void*) { return E_INVALIDARG; }

// 13, 21: HRESULT XSpeechSynthesizerCreateStreamFromText/Ssml(synthesizer, const char* text,
//                                                             XSpeechSynthesizerStreamHandle* stream)
HRESULT STDMETHODCALLTYPE speech_create_stream(void*, void*, const char*, void** stream) {
    if (stream) *stream = nullptr;
    return E_INVALIDARG;
}

// 15: HRESULT XSpeechSynthesizerGetStreamDataSize(stream, size_t* size)
HRESULT STDMETHODCALLTYPE speech_stream_size(void*, void*, size_t* size) {
    if (size) *size = 0;
    return E_INVALIDARG;
}

// 16: HRESULT XSpeechSynthesizerGetStreamData(stream, size_t size, void* buffer, size_t* used)
HRESULT STDMETHODCALLTYPE speech_stream_data(void*, void*, size_t, void*, size_t* used) {
    if (used) *used = 0;
    return E_INVALIDARG;
}

// The slots with no offline counterpart whose arguments the layout doesn't give: the caption properties (3) and
// speech-to-text display (6, 7, 17-20). Nothing is written.
template <int Slot> HRESULT STDMETHODCALLTYPE accessibility_offline(void* self) {
    return offline_failure<kAccessibilityName, Slot>(self);
}

HRESULT STDMETHODCALLTYPE accessibility_query_interface(void*, REFIID iid, void** out);
const void* const kAccessibilityVtable[] = {
    FN(accessibility_query_interface), FN(object_add_ref), FN(object_release),   // 0-2 IUnknown
    FN(accessibility_offline<3>),            // 3  XClosedCaptionGetProperties
    FN(accessibility_captions_set_enabled),  // 4  XClosedCaptionSetEnabled
    FN(accessibility_high_contrast),         // 5  XHighContrastGetMode
    FN(accessibility_offline<6>),            // 6  XSpeechToTextSetPositionHint
    FN(accessibility_offline<7>),            // 7  XSpeechToTextSendString
    FN(speech_enumerate_voices),             // 8  XSpeechSynthesizerEnumerateInstalledVoices
    FN(speech_create),                       // 9  XSpeechSynthesizerCreate
    FN(speech_bad_handle),                   // 10 XSpeechSynthesizerCloseHandle
    FN(speech_bad_handle),                   // 11 XSpeechSynthesizerSetDefaultVoice
    FN(speech_bad_handle),                   // 12 XSpeechSynthesizerSetCustomVoice
    FN(speech_create_stream),                // 13 XSpeechSynthesizerCreateStreamFromText
    FN(speech_bad_handle),                   // 14 XSpeechSynthesizerCloseStreamHandle
    FN(speech_stream_size),                  // 15 XSpeechSynthesizerGetStreamDataSize
    FN(speech_stream_data),                  // 16 XSpeechSynthesizerGetStreamData
    FN(accessibility_offline<17>),           // 17 XSpeechToTextBeginHypothesisString
    FN(accessibility_offline<18>),           // 18 XSpeechToTextUpdateHypothesisString
    FN(accessibility_offline<19>),           // 19 XSpeechToTextFinalizeHypothesisString
    FN(accessibility_offline<20>),           // 20 XSpeechToTextCancelHypothesisString
    FN(speech_create_stream),                // 21 XSpeechSynthesizerCreateStreamFromSsml
};
static_assert(std::size(kAccessibilityVtable) == 22, "IUnknown + IXAccessibilityImpl's 19 methods");
ClassObject g_accessibility_object{kAccessibilityVtable};
HRESULT STDMETHODCALLTYPE accessibility_query_interface(void*, REFIID iid, void** out) {
    return answer(&g_accessibility_object, std::begin(kAccessibilityInterfaces), std::end(kAccessibilityInterfaces),
                  iid, out);
}

// ======================= XLauncherImpl =======================

constexpr char kLauncherName[] = "XLauncherImpl";
constexpr GUID kLauncherClass = {0x1b339674, 0x328d, 0x4283, {0xa2, 0x00, 0x31, 0x71, 0xf1, 0x8d, 0x36, 0x39}};

// 3: HRESULT XLaunchUri(XUserHandle user, const char* uri): nothing is launched; E_FAIL.
// 4: HRESULT XDisplayAcquireTimeoutDeferral(XDisplayTimeoutDeferralHandle* handle): none is handed out.
// 5: XDisplayCloseTimeoutDeferralHandle(handle): no handle exists; E_FAIL, which a void caller ignores.
HRESULT STDMETHODCALLTYPE launcher_acquire_deferral(void* self, void** handle) {
    if (handle) *handle = nullptr;
    return offline_failure<kLauncherName, 4>(self);
}

HRESULT STDMETHODCALLTYPE launcher_query_interface(void*, REFIID iid, void** out);
const void* const kLauncherVtable[] = {
    FN(launcher_query_interface), FN(object_add_ref), FN(object_release),   // 0-2 IUnknown
    FN((offline_failure<kLauncherName, 3>)),   // 3 XLaunchUri
    FN(launcher_acquire_deferral),             // 4 XDisplayAcquireTimeoutDeferral
    FN((offline_failure<kLauncherName, 5>)),   // 5 XDisplayCloseTimeoutDeferralHandle
};
static_assert(std::size(kLauncherVtable) == 6, "IUnknown + IXLauncherImpl's 3 methods");
ClassObject g_launcher_object{kLauncherVtable};
HRESULT STDMETHODCALLTYPE launcher_query_interface(void*, REFIID iid, void** out) {
    // IXLauncherImpl's id is the class id.
    return answer(&g_launcher_object, &kLauncherClass, &kLauncherClass + 1, iid, out);
}

#undef FN

}  // namespace

namespace rt::shims {

// Called by xgameruntime.cpp's QueryApiImpl for every class: true when the class is one of this file's, with *hr set.
bool xgameruntime_offline_query(const GUID& class_id, REFIID interface_id, void** out, HRESULT* hr) {
    if (class_id == kEventClass) {
        *hr = event_query_interface(nullptr, interface_id, out);
        return true;
    }
    if (class_id == kUiClass) {
        *hr = ui_query_interface(nullptr, interface_id, out);
        return true;
    }
    if (class_id == kStreamingClass) {
        *hr = streaming_query_interface(nullptr, interface_id, out);
        return true;
    }
    if (class_id == kDisplayClass) {
        *hr = display_query_interface(nullptr, interface_id, out);
        return true;
    }
    if (class_id == kUserDeviceClass) {
        *hr = device_query_interface(nullptr, interface_id, out);
        return true;
    }
    if (class_id == kAccessibilityClass) {
        *hr = accessibility_query_interface(nullptr, interface_id, out);
        return true;
    }
    if (class_id == kLauncherClass) {
        *hr = launcher_query_interface(nullptr, interface_id, out);
        return true;
    }
    return false;
}

// For tests: how many game events were written.
long xgameruntime_offline_game_events() { return g_events.load(); }

}  // namespace rt::shims
