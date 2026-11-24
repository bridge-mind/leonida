// GameInput v0 ABI: the part of Microsoft's GameInput API a title uses for gamepads.
//
// These are Leonida's own declarations, written to match the binary layout of the public v0 GameInput API (the
// in-box GDK version, also shipped as the Microsoft.GameInput 0.2303.x NuGet package). Only what the shim implements
// or must lay out is declared; interface methods are in vtable order and every struct is checked with static_asserts
// below. Names follow the public API so the docs at learn.microsoft.com/gaming/gdk/docs/reference/input/gameinput
// apply directly.
//
// VERIFY: GameInput v1 and later (the 1.x-4.x NuGet packages) changed the interfaces. Which version the console
// executable was built against is unknown until it is available; this file is v0.
#pragma once
#include <windows.h>
#include <unknwn.h>

#include <cstddef>
#include <cstdint>

namespace gi {

enum GameInputKind : uint32_t {
    GameInputKindUnknown = 0x00000000,
    GameInputKindRawDeviceReport = 0x00000001,
    GameInputKindControllerAxis = 0x00000002,
    GameInputKindControllerButton = 0x00000004,
    GameInputKindControllerSwitch = 0x00000008,
    GameInputKindController = 0x0000000E,
    GameInputKindKeyboard = 0x00000010,
    GameInputKindMouse = 0x00000020,
    GameInputKindTouch = 0x00000100,
    GameInputKindMotion = 0x00001000,
    GameInputKindArcadeStick = 0x00010000,
    GameInputKindFlightStick = 0x00020000,
    GameInputKindGamepad = 0x00040000,
    GameInputKindRacingWheel = 0x00080000,
    GameInputKindUiNavigation = 0x01000000,
};

enum GameInputEnumerationKind : int32_t {
    GameInputNoEnumeration = 0,
    GameInputAsyncEnumeration = 1,
    GameInputBlockingEnumeration = 2,
};

enum GameInputGamepadButtons : uint32_t {
    GameInputGamepadNone = 0x00000000,
    GameInputGamepadMenu = 0x00000001,
    GameInputGamepadView = 0x00000002,
    GameInputGamepadA = 0x00000004,
    GameInputGamepadB = 0x00000008,
    GameInputGamepadX = 0x00000010,
    GameInputGamepadY = 0x00000020,
    GameInputGamepadDPadUp = 0x00000040,
    GameInputGamepadDPadDown = 0x00000080,
    GameInputGamepadDPadLeft = 0x00000100,
    GameInputGamepadDPadRight = 0x00000200,
    GameInputGamepadLeftShoulder = 0x00000400,
    GameInputGamepadRightShoulder = 0x00000800,
    GameInputGamepadLeftThumbstick = 0x00001000,
    GameInputGamepadRightThumbstick = 0x00002000,
};

enum GameInputUiNavigationButtons : uint32_t {
    GameInputUiNavigationNone = 0x00000000,
    GameInputUiNavigationMenu = 0x00000001,
    GameInputUiNavigationView = 0x00000002,
    GameInputUiNavigationAccept = 0x00000004,
    GameInputUiNavigationCancel = 0x00000008,
    GameInputUiNavigationUp = 0x00000010,
    GameInputUiNavigationDown = 0x00000020,
    GameInputUiNavigationLeft = 0x00000040,
    GameInputUiNavigationRight = 0x00000080,
    GameInputUiNavigationContext1 = 0x00000100,
    GameInputUiNavigationContext2 = 0x00000200,
    GameInputUiNavigationContext3 = 0x00000400,
    GameInputUiNavigationContext4 = 0x00000800,
    GameInputUiNavigationPageUp = 0x00001000,
    GameInputUiNavigationPageDown = 0x00002000,
    GameInputUiNavigationPageLeft = 0x00004000,
    GameInputUiNavigationPageRight = 0x00008000,
    GameInputUiNavigationScrollUp = 0x00010000,
    GameInputUiNavigationScrollDown = 0x00020000,
    GameInputUiNavigationScrollLeft = 0x00040000,
    GameInputUiNavigationScrollRight = 0x00080000,
};

enum GameInputDeviceStatus : uint32_t {
    GameInputDeviceNoStatus = 0x00000000,
    GameInputDeviceConnected = 0x00000001,
    GameInputDeviceInputEnabled = 0x00000002,
    GameInputDeviceOutputEnabled = 0x00000004,
    GameInputDeviceRawIoEnabled = 0x00000008,
    GameInputDeviceAudioCapture = 0x00000010,
    GameInputDeviceAudioRender = 0x00000020,
    GameInputDeviceSynchronized = 0x00000040,
    GameInputDeviceWireless = 0x00000080,
    GameInputDeviceUserIdle = 0x00100000,
    GameInputDeviceAnyStatus = 0x00FFFFFF,
};

enum GameInputBatteryStatus : int32_t {
    GameInputBatteryUnknown = -1,
    GameInputBatteryNotPresent = 0,
    GameInputBatteryDischarging = 1,
    GameInputBatteryIdle = 2,
    GameInputBatteryCharging = 3,
};

enum GameInputDeviceFamily : int32_t {
    GameInputFamilyVirtual = -1,
    GameInputFamilyAggregate = 0,
    GameInputFamilyXboxOne = 1,
    GameInputFamilyXbox360 = 2,
    GameInputFamilyHid = 3,
    GameInputFamilyI8042 = 4,
};

enum GameInputDeviceCapabilities : uint32_t {
    GameInputDeviceCapabilityNone = 0x00000000,
    GameInputDeviceCapabilityAudio = 0x00000001,
    GameInputDeviceCapabilityPluginModule = 0x00000002,
    GameInputDeviceCapabilityPowerOff = 0x00000004,
    GameInputDeviceCapabilitySynchronization = 0x00000008,
    GameInputDeviceCapabilityWireless = 0x00000010,
};

enum GameInputRumbleMotors : uint32_t {
    GameInputRumbleNone = 0x00000000,
    GameInputRumbleLowFrequency = 0x00000001,
    GameInputRumbleHighFrequency = 0x00000002,
    GameInputRumbleLeftTrigger = 0x00000004,
    GameInputRumbleRightTrigger = 0x00000008,
};

// Where a motor sits on the device. v0 carries it in the motor infos below.
enum GameInputLocation : int32_t {
    GameInputLocationUnknown = -1,
    GameInputLocationChassis = 0,
    GameInputLocationDisplay = 1,
    GameInputLocationAxis = 2,
    GameInputLocationButton = 3,
    GameInputLocationSwitch = 4,
    GameInputLocationKey = 5,
    GameInputLocationTouchPad = 6,
};

// The axes a force-feedback motor can drive, and the kinds of effect it can play.
enum GameInputFeedbackAxes : uint32_t {
    GameInputFeedbackAxisNone = 0x00000000,
    GameInputFeedbackAxisLinearX = 0x00000001,
    GameInputFeedbackAxisLinearY = 0x00000002,
    GameInputFeedbackAxisLinearZ = 0x00000004,
    GameInputFeedbackAxisAngularX = 0x00000008,
    GameInputFeedbackAxisAngularY = 0x00000010,
    GameInputFeedbackAxisAngularZ = 0x00000020,
    GameInputFeedbackAxisNormal = 0x00000040,
};

enum GameInputFeedbackEffectState : int32_t {
    GameInputFeedbackStopped = 0,
    GameInputFeedbackRunning = 1,
    GameInputFeedbackPaused = 2,
};

enum GameInputForceFeedbackEffectKind : int32_t {
    GameInputForceFeedbackConstant = 0,
    GameInputForceFeedbackRamp = 1,
    GameInputForceFeedbackSineWave = 2,
    GameInputForceFeedbackSquareWave = 3,
    GameInputForceFeedbackTriangleWave = 4,
    GameInputForceFeedbackSawtoothUpWave = 5,
    GameInputForceFeedbackSawtoothDownWave = 6,
    GameInputForceFeedbackSpring = 7,
    GameInputForceFeedbackFriction = 8,
    GameInputForceFeedbackDamper = 9,
    GameInputForceFeedbackInertia = 10,
};

enum GameInputSwitchPosition : int32_t { GameInputSwitchCenter = 0 };
enum GameInputFocusPolicy : uint32_t { GameInputDefaultFocusPolicy = 0 };

using GameInputCallbackToken = uint64_t;
constexpr GameInputCallbackToken kInvalidCallbackToken = 0;

// Error codes (facility 0x38A).
constexpr HRESULT GAMEINPUT_E_DEVICE_DISCONNECTED = static_cast<HRESULT>(0x838A0001L);
constexpr HRESULT GAMEINPUT_E_DEVICE_NOT_FOUND = static_cast<HRESULT>(0x838A0002L);
constexpr HRESULT GAMEINPUT_E_READING_NOT_FOUND = static_cast<HRESULT>(0x838A0003L);
constexpr HRESULT GAMEINPUT_E_REFERENCE_READING_TOO_OLD = static_cast<HRESULT>(0x838A0004L);
constexpr HRESULT GAMEINPUT_E_TIMESTAMP_OUT_OF_RANGE = static_cast<HRESULT>(0x838A0005L);
constexpr HRESULT GAMEINPUT_E_INSUFFICIENT_FORCE_FEEDBACK_RESOURCES = static_cast<HRESULT>(0x838A0006L);

struct AppLocalDeviceId {  // APP_LOCAL_DEVICE_ID
    uint8_t value[32];
};

struct GameInputGamepadState {
    uint32_t buttons;  // GameInputGamepadButtons
    float leftTrigger;
    float rightTrigger;
    float leftThumbstickX;
    float leftThumbstickY;
    float rightThumbstickX;
    float rightThumbstickY;
};

struct GameInputUiNavigationState {
    uint32_t buttons;  // GameInputUiNavigationButtons
};

struct GameInputBatteryState {
    float chargeRate;
    float maxChargeRate;
    float remainingCapacity;
    float fullChargeCapacity;
    GameInputBatteryStatus status;
};

struct GameInputUsage {
    uint16_t page;
    uint16_t id;
};

struct GameInputRumbleParams {
    float lowFrequency;
    float highFrequency;
    float leftTrigger;
    float rightTrigger;
};

// ---- force feedback and haptic motors ------------------------------------------------------------
// One entry of GameInputDeviceInfo::forceFeedbackMotorInfo per motor.
struct GameInputForceFeedbackMotorInfo {
    GameInputFeedbackAxes supportedAxes;
    GameInputLocation location;
    uint32_t locationId;
    uint32_t maxSimultaneousEffects;
    bool isConstantEffectSupported;
    bool isRampEffectSupported;
    bool isSineWaveEffectSupported;
    bool isSquareWaveEffectSupported;
    bool isTriangleWaveEffectSupported;
    bool isSawtoothUpWaveEffectSupported;
    bool isSawtoothDownWaveEffectSupported;
    bool isSpringEffectSupported;
    bool isFrictionEffectSupported;
    bool isDamperEffectSupported;
    bool isInertiaEffectSupported;
};

// One entry of GameInputHapticFeedbackMotorInfo::waveformInfo per waveform the motor can play.
struct GameInputHapticWaveformInfo {
    GameInputUsage usage;  // the HID usage of the waveform
    bool isDurationSupported;
    bool isIntensitySupported;
    bool isRepeatSupported;
    bool isRepeatDelaySupported;
    uint64_t defaultDuration;  // microseconds
};

// One entry of GameInputDeviceInfo::hapticFeedbackMotorInfo per haptic motor.
struct GameInputHapticFeedbackMotorInfo {
    GameInputRumbleMotors mappedRumbleMotors;  // the rumble motors this haptic motor sits on
    GameInputLocation location;
    uint32_t locationId;
    uint32_t waveformCount;
    const GameInputHapticWaveformInfo* waveformInfo;
};

// An effect's duration and gain over time. The durations are microseconds; 0 everywhere means the effect has no
// envelope and lasts until it is stopped, and playCount 0 means the same.
struct GameInputForceFeedbackEnvelope {
    uint64_t attackDuration;
    uint64_t sustainDuration;
    uint64_t releaseDuration;
    float attackGain;
    float sustainGain;
    float releaseGain;
    uint32_t playCount;
    uint64_t repeatDelay;
};

// An effect's strength per axis, -1..1 (the axis is chosen by the motor's supportedAxes).
struct GameInputForceFeedbackMagnitude {
    float linearX;
    float linearY;
    float linearZ;
    float angularX;
    float angularY;
    float angularZ;
    float normal;
};

struct GameInputForceFeedbackConditionParams {
    GameInputForceFeedbackMagnitude magnitude;
    float positiveCoefficient;
    float negativeCoefficient;
    float maxPositiveMagnitude;
    float maxNegativeMagnitude;
    float deadZone;
    float bias;
};

struct GameInputForceFeedbackConstantParams {
    GameInputForceFeedbackEnvelope envelope;
    GameInputForceFeedbackMagnitude magnitude;
};

// Sine, square, triangle and both sawtooth waves. `frequency` is Hz and `phase` degrees (the shim's reading of the
// GDK remarks: 0 is no displacement); SDL takes a period in ms and a phase in hundredths of a degree.
struct GameInputForceFeedbackPeriodicParams {
    GameInputForceFeedbackEnvelope envelope;
    GameInputForceFeedbackMagnitude magnitude;
    float frequency;
    float phase;
    float bias;
};

struct GameInputForceFeedbackRampParams {
    GameInputForceFeedbackEnvelope envelope;
    GameInputForceFeedbackMagnitude startMagnitude;
    GameInputForceFeedbackMagnitude endMagnitude;
};

// What CreateForceFeedbackEffect and IGameInputForceFeedbackEffect::SetParams take: `kind` chooses the union member.
struct GameInputForceFeedbackParams {
    GameInputForceFeedbackEffectKind kind;
    union {
        GameInputForceFeedbackConstantParams constant;
        GameInputForceFeedbackRampParams ramp;
        GameInputForceFeedbackPeriodicParams sineWave;
        GameInputForceFeedbackPeriodicParams squareWave;
        GameInputForceFeedbackPeriodicParams triangleWave;
        GameInputForceFeedbackPeriodicParams sawtoothUpWave;
        GameInputForceFeedbackPeriodicParams sawtoothDownWave;
        GameInputForceFeedbackConditionParams spring;
        GameInputForceFeedbackConditionParams friction;
        GameInputForceFeedbackConditionParams damper;
        GameInputForceFeedbackConditionParams inertia;
    } data;
};

// What SetHapticMotorState takes: one waveform of the motor's waveformInfo, played `playCount` times. `duration` and
// `repeatDelay` are microseconds; playCount 0 plays until the state is set again.
struct GameInputHapticFeedbackParams {
    uint32_t waveformIndex;
    uint64_t duration;
    float intensity;
    uint32_t playCount;
    uint64_t repeatDelay;
};

struct GameInputString {
    uint32_t sizeInBytes;
    uint32_t codePointCount;
    const char* data;  // UTF-8, zero-terminated
};

struct GameInputVersion {
    uint16_t major, minor, build, revision;
};

// Raw HID reports (RT-175). Layouts and order match Microsoft's v0 header (NuGet Microsoft.GameInput
// 0.2303.22621.3038), checked with offsetof against it (docs/gameinput.md, "ABI"); the shim fills reportInfo with
// each report's kind, id and size and no items (itemCount 0, see docs/gameinput.md "Raw HID").
enum GameInputRawDeviceReportKind : int32_t {
    GameInputRawInputReport = 0,
    GameInputRawOutputReport = 1,
    GameInputRawFeatureReport = 2,
};

struct GameInputRawDeviceItemCollectionInfo {
    int32_t kind;  // GameInputRawDeviceItemCollectionKind
    uint32_t childCount;
    uint32_t siblingCount;
    uint32_t usageCount;
    const GameInputUsage* usages;
    const GameInputRawDeviceItemCollectionInfo* parent;
    const GameInputRawDeviceItemCollectionInfo* firstSibling;
    const GameInputRawDeviceItemCollectionInfo* previousSibling;
    const GameInputRawDeviceItemCollectionInfo* nextSibling;
    const GameInputRawDeviceItemCollectionInfo* lastSibling;
    const GameInputRawDeviceItemCollectionInfo* firstChild;
    const GameInputRawDeviceItemCollectionInfo* lastChild;
};

struct GameInputRawDeviceReportItemInfo {
    uint32_t bitOffset;
    uint32_t bitSize;
    int64_t logicalMin;
    int64_t logicalMax;
    double physicalMin;
    double physicalMax;
    int32_t physicalUnits;  // GameInputRawDevicePhysicalUnitKind
    uint32_t rawPhysicalUnits;
    int32_t rawPhysicalUnitsExponent;
    uint32_t flags;  // GameInputRawDeviceReportItemFlags
    uint32_t usageCount;
    const GameInputUsage* usages;
    const GameInputRawDeviceItemCollectionInfo* collection;
    const GameInputString* itemString;
};

struct GameInputRawDeviceReportInfo {
    GameInputRawDeviceReportKind kind;
    uint32_t id;
    uint32_t size;
    uint32_t itemCount;
    const GameInputRawDeviceReportItemInfo* items;
};

struct GameInputControllerAxisInfo {
    uint32_t mappedInputKinds;  // GameInputKind
    bool isContinuous;
    bool isNonlinear;
    bool isQuantized;
    bool hasRestValue;
    float restValue;
    uint64_t resolution;
    uint16_t legacyDInputIndex;
    uint16_t legacyHidIndex;
    uint32_t rawReportIndex;
    const GameInputRawDeviceReportInfo* inputReport;
    const GameInputRawDeviceReportItemInfo* inputReportItem;
};

struct GameInputControllerButtonInfo {
    uint32_t mappedInputKinds;  // GameInputKind
    uint16_t legacyDInputIndex;
    uint16_t legacyHidIndex;
    uint32_t rawReportIndex;
    const GameInputRawDeviceReportInfo* inputReport;
    const GameInputRawDeviceReportItemInfo* inputReportItem;
};

// Empty in v0 (one byte in C++); a title only checks the pointer.
struct GameInputGamepadInfo {};
struct GameInputUiNavigationInfo {};

// Motion: layouts from Microsoft's v0 documentation (learn.microsoft.com GameInputMotionInfo / GameInputMotionState /
// GameInputMotionAccuracy; the GDK public header spells the same members). Accuracy is deprecated only in GameInput
// v1 and later; v0 still uses it in GameInputMotionState.
enum GameInputMotionAccuracy : int32_t {
    GameInputMotionAccuracyUnknown = -1,
    GameInputMotionAccuracyUnreliable = 0,
    GameInputMotionAccuracyLow = 1,
    GameInputMotionAccuracyMedium = 2,
    GameInputMotionAccuracyHigh = 3,
};

struct GameInputMotionInfo {
    float maxAcceleration;           // m/s^2; 0 says acceleration input is not available
    float maxAngularVelocity;        // rad/s; 0 says angular-velocity input is not available
    float maxMagneticFieldStrength;  // teslas; 0 says magnetic-field input is not available
};

// The state GetMotionState writes. Each float is the signed fraction of the GameInputMotionInfo maximum for its
// sensor (1.0 = full magnitude); orientation is a quaternion. SDL delivers accel in m/s^2 (gravity included) and
// gyro in rad/s, so the shim divides by the maxima it reports in motionInfo.
struct GameInputMotionState {
    float accelerationX;
    float accelerationY;
    float accelerationZ;
    float angularVelocityX;
    float angularVelocityY;
    float angularVelocityZ;
    float magneticFieldX;
    float magneticFieldY;
    float magneticFieldZ;
    float orientationW;
    float orientationX;
    float orientationY;
    float orientationZ;
    GameInputMotionAccuracy accelerometerAccuracy;
    GameInputMotionAccuracy gyroscopeAccuracy;
    GameInputMotionAccuracy magnetometerAccuracy;
    GameInputMotionAccuracy orientationAccuracy;
};

struct GameInputDeviceInfo {
    uint32_t infoSize;
    uint16_t vendorId;
    uint16_t productId;
    uint16_t revisionNumber;
    uint8_t interfaceNumber;
    uint8_t collectionNumber;
    GameInputUsage usage;
    GameInputVersion hardwareVersion;
    GameInputVersion firmwareVersion;
    AppLocalDeviceId deviceId;
    AppLocalDeviceId deviceRootId;
    GameInputDeviceFamily deviceFamily;
    uint32_t capabilities;           // GameInputDeviceCapabilities
    uint32_t supportedInput;         // GameInputKind
    uint32_t supportedRumbleMotors;  // GameInputRumbleMotors
    uint32_t inputReportCount;
    uint32_t outputReportCount;
    uint32_t featureReportCount;
    uint32_t controllerAxisCount;
    uint32_t controllerButtonCount;
    uint32_t controllerSwitchCount;
    uint32_t touchPointCount;
    uint32_t touchSensorCount;
    uint32_t forceFeedbackMotorCount;
    uint32_t hapticFeedbackMotorCount;
    uint32_t deviceStringCount;
    uint32_t deviceDescriptorSize;
    const GameInputRawDeviceReportInfo* inputReportInfo;
    const GameInputRawDeviceReportInfo* outputReportInfo;
    const GameInputRawDeviceReportInfo* featureReportInfo;
    const GameInputControllerAxisInfo* controllerAxisInfo;
    const GameInputControllerButtonInfo* controllerButtonInfo;
    const void* controllerSwitchInfo;
    const void* keyboardInfo;
    const void* mouseInfo;
    const void* touchSensorInfo;
    const void* motionInfo;
    const void* arcadeStickInfo;
    const void* flightStickInfo;
    const GameInputGamepadInfo* gamepadInfo;
    const void* racingWheelInfo;
    const GameInputUiNavigationInfo* uiNavigationInfo;
    const void* forceFeedbackMotorInfo;
    const void* hapticFeedbackMotorInfo;
    const GameInputString* displayName;
    const GameInputString* deviceStrings;
    const void* deviceDescriptorData;
};

// Layout checks. The offsets were compared with the public v0 header by compiling both side by side (see
// docs/gameinput.md); they hold for x64 MSVC.
static_assert(sizeof(GameInputGamepadState) == 28);
static_assert(sizeof(GameInputRumbleParams) == 16);
static_assert(sizeof(GameInputBatteryState) == 20);
static_assert(sizeof(GameInputString) == 16);
static_assert(sizeof(GameInputMotionInfo) == 12);
static_assert(sizeof(GameInputMotionState) == 68);
static_assert(sizeof(GameInputControllerAxisInfo) == 48);
static_assert(sizeof(GameInputControllerButtonInfo) == 32);
static_assert(offsetof(GameInputDeviceInfo, deviceId) == 32);
static_assert(offsetof(GameInputDeviceInfo, deviceFamily) == 96);
static_assert(offsetof(GameInputDeviceInfo, inputReportCount) == 112);
static_assert(offsetof(GameInputDeviceInfo, inputReportInfo) == 160);
static_assert(offsetof(GameInputDeviceInfo, gamepadInfo) == 256);
static_assert(offsetof(GameInputDeviceInfo, displayName) == 296);
static_assert(sizeof(GameInputDeviceInfo) == 320);
static_assert(sizeof(GameInputRawDeviceItemCollectionInfo) == 80);
static_assert(offsetof(GameInputRawDeviceItemCollectionInfo, usages) == 16);
static_assert(offsetof(GameInputRawDeviceReportItemInfo, physicalUnits) == 40);
static_assert(offsetof(GameInputRawDeviceReportItemInfo, usages) == 64);
static_assert(sizeof(GameInputRawDeviceReportItemInfo) == 88);
static_assert(offsetof(GameInputRawDeviceReportInfo, items) == 16);
static_assert(sizeof(GameInputRawDeviceReportInfo) == 24);

// The force-feedback and haptic layouts, transcribed from the public v0 header (GameInput.h, the same GDK release
// as the rest of this file). They were not compiled against that header here — it is not in the repository — so
// the sizes and offsets below are the check: they are what x64 MSVC lays out for those declarations.
static_assert(sizeof(GameInputForceFeedbackMotorInfo) == 28);
static_assert(offsetof(GameInputForceFeedbackMotorInfo, location) == 4);
static_assert(offsetof(GameInputForceFeedbackMotorInfo, maxSimultaneousEffects) == 12);
static_assert(offsetof(GameInputForceFeedbackMotorInfo, isConstantEffectSupported) == 16);
static_assert(sizeof(GameInputHapticWaveformInfo) == 16);
static_assert(offsetof(GameInputHapticWaveformInfo, defaultDuration) == 8);
static_assert(sizeof(GameInputHapticFeedbackMotorInfo) == 24);
static_assert(offsetof(GameInputHapticFeedbackMotorInfo, waveformInfo) == 16);
static_assert(sizeof(GameInputForceFeedbackEnvelope) == 48);
static_assert(offsetof(GameInputForceFeedbackEnvelope, repeatDelay) == 40);
static_assert(sizeof(GameInputForceFeedbackMagnitude) == 28);
static_assert(sizeof(GameInputForceFeedbackConditionParams) == 52);
static_assert(sizeof(GameInputForceFeedbackConstantParams) == 80);  // 76 rounded up to the envelope's 8-byte alignment
static_assert(offsetof(GameInputForceFeedbackConstantParams, magnitude) == 48);
static_assert(sizeof(GameInputForceFeedbackPeriodicParams) == 88);
static_assert(sizeof(GameInputForceFeedbackRampParams) == 104);
static_assert(sizeof(GameInputForceFeedbackParams) == 112);
static_assert(offsetof(GameInputForceFeedbackParams, data) == 8);
static_assert(sizeof(GameInputHapticFeedbackParams) == 32);
static_assert(offsetof(GameInputHapticFeedbackParams, repeatDelay) == 24);

struct IGameInputDevice;
struct IGameInputReading;
struct IGameInputDispatcher;

using GameInputReadingCallback = void(CALLBACK*)(GameInputCallbackToken, void* context, IGameInputReading* reading,
                                                 bool hasOverrunOccurred);
using GameInputDeviceCallback = void(CALLBACK*)(GameInputCallbackToken, void* context, IGameInputDevice* device,
                                                uint64_t timestamp, uint32_t currentStatus, uint32_t previousStatus);
using GameInputGuideButtonCallback = void(CALLBACK*)(GameInputCallbackToken, void* context, IGameInputDevice* device,
                                                     uint64_t timestamp, bool isPressed);
using GameInputKeyboardLayoutCallback = void(CALLBACK*)(GameInputCallbackToken, void* context,
                                                        IGameInputDevice* device, uint64_t timestamp,
                                                        uint32_t currentLayout, uint32_t previousLayout);

// {11BE2A7E-4254-445A-9C09-FFC40F006918}
constexpr GUID IID_IGameInput = {0x11BE2A7E, 0x4254, 0x445A, {0x9C, 0x09, 0xFF, 0xC4, 0x0F, 0x00, 0x69, 0x18}};
// {2156947A-E1FA-4DE0-A30B-D812931DBD8D}
constexpr GUID IID_IGameInputReading = {0x2156947A, 0xE1FA, 0x4DE0, {0xA3, 0x0B, 0xD8, 0x12, 0x93, 0x1D, 0xBD, 0x8D}};
// {31DD86FB-4C1B-408A-868F-439B3CD47125}
constexpr GUID IID_IGameInputDevice = {0x31DD86FB, 0x4C1B, 0x408A, {0x86, 0x8F, 0x43, 0x9B, 0x3C, 0xD4, 0x71, 0x25}};
// {61F08CF1-1FFC-40CA-A2B8-E1AB8BC5B6DC}
constexpr GUID IID_IGameInputRawDeviceReport = {0x61F08CF1, 0x1FFC, 0x40CA, {0xA2, 0xB8, 0xE1, 0xAB, 0x8B, 0xC5, 0xB6, 0xDC}};
// {415EED2E-98CB-42C2-8F28-B94601074E31}
constexpr GUID IID_IGameInputDispatcher = {0x415EED2E, 0x98CB, 0x42C2, {0x8F, 0x28, 0xB9, 0x46, 0x01, 0x07, 0x4E, 0x31}};
// {51BDA05E-F742-45D9-B085-9444AE48381D}
constexpr GUID IID_IGameInputForceFeedbackEffect = {0x51BDA05E, 0xF742, 0x45D9, {0xB0, 0x85, 0x94, 0x44, 0xAE, 0x48, 0x38, 0x1D}};

// Interfaces, methods in vtable order. Pointer types the shim never implements are void*.
struct IGameInput : IUnknown {
    virtual uint64_t STDMETHODCALLTYPE GetCurrentTimestamp() = 0;
    virtual HRESULT STDMETHODCALLTYPE GetCurrentReading(uint32_t inputKind, IGameInputDevice* device,
                                                        IGameInputReading** reading) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetNextReading(IGameInputReading* referenceReading, uint32_t inputKind,
                                                     IGameInputDevice* device, IGameInputReading** reading) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetPreviousReading(IGameInputReading* referenceReading, uint32_t inputKind,
                                                         IGameInputDevice* device, IGameInputReading** reading) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetTemporalReading(uint64_t timestamp, IGameInputDevice* device,
                                                         IGameInputReading** reading) = 0;
    virtual HRESULT STDMETHODCALLTYPE RegisterReadingCallback(IGameInputDevice* device, uint32_t inputKind,
                                                              float analogThreshold, void* context,
                                                              GameInputReadingCallback callbackFunc,
                                                              GameInputCallbackToken* callbackToken) = 0;
    virtual HRESULT STDMETHODCALLTYPE RegisterDeviceCallback(IGameInputDevice* device, uint32_t inputKind,
                                                             uint32_t statusFilter, GameInputEnumerationKind enumerationKind,
                                                             void* context, GameInputDeviceCallback callbackFunc,
                                                             GameInputCallbackToken* callbackToken) = 0;
    virtual HRESULT STDMETHODCALLTYPE RegisterGuideButtonCallback(IGameInputDevice* device, void* context,
                                                                  GameInputGuideButtonCallback callbackFunc,
                                                                  GameInputCallbackToken* callbackToken) = 0;
    virtual HRESULT STDMETHODCALLTYPE RegisterKeyboardLayoutCallback(IGameInputDevice* device, void* context,
                                                                     GameInputKeyboardLayoutCallback callbackFunc,
                                                                     GameInputCallbackToken* callbackToken) = 0;
    virtual void STDMETHODCALLTYPE StopCallback(GameInputCallbackToken callbackToken) = 0;
    virtual bool STDMETHODCALLTYPE UnregisterCallback(GameInputCallbackToken callbackToken,
                                                      uint64_t timeoutInMicroseconds) = 0;
    virtual HRESULT STDMETHODCALLTYPE CreateDispatcher(IGameInputDispatcher** dispatcher) = 0;
    virtual HRESULT STDMETHODCALLTYPE CreateAggregateDevice(uint32_t inputKind, IGameInputDevice** device) = 0;
    virtual HRESULT STDMETHODCALLTYPE FindDeviceFromId(const AppLocalDeviceId* value, IGameInputDevice** device) = 0;
    virtual HRESULT STDMETHODCALLTYPE FindDeviceFromObject(IUnknown* value, IGameInputDevice** device) = 0;
    virtual HRESULT STDMETHODCALLTYPE FindDeviceFromPlatformHandle(HANDLE value, IGameInputDevice** device) = 0;
    virtual HRESULT STDMETHODCALLTYPE FindDeviceFromPlatformString(LPCWSTR value, IGameInputDevice** device) = 0;
    virtual HRESULT STDMETHODCALLTYPE EnableOemDeviceSupport(uint16_t vendorId, uint16_t productId,
                                                             uint8_t interfaceNumber, uint8_t collectionNumber) = 0;
    virtual void STDMETHODCALLTYPE SetFocusPolicy(uint32_t policy) = 0;
};

struct IGameInputReading : IUnknown {
    virtual uint32_t STDMETHODCALLTYPE GetInputKind() = 0;
    virtual uint64_t STDMETHODCALLTYPE GetSequenceNumber(uint32_t inputKind) = 0;
    virtual uint64_t STDMETHODCALLTYPE GetTimestamp() = 0;
    virtual void STDMETHODCALLTYPE GetDevice(IGameInputDevice** device) = 0;
    virtual bool STDMETHODCALLTYPE GetRawReport(void** report) = 0;
    virtual uint32_t STDMETHODCALLTYPE GetControllerAxisCount() = 0;
    virtual uint32_t STDMETHODCALLTYPE GetControllerAxisState(uint32_t stateArrayCount, float* stateArray) = 0;
    virtual uint32_t STDMETHODCALLTYPE GetControllerButtonCount() = 0;
    virtual uint32_t STDMETHODCALLTYPE GetControllerButtonState(uint32_t stateArrayCount, bool* stateArray) = 0;
    virtual uint32_t STDMETHODCALLTYPE GetControllerSwitchCount() = 0;
    virtual uint32_t STDMETHODCALLTYPE GetControllerSwitchState(uint32_t stateArrayCount,
                                                                GameInputSwitchPosition* stateArray) = 0;
    virtual uint32_t STDMETHODCALLTYPE GetKeyCount() = 0;
    virtual uint32_t STDMETHODCALLTYPE GetKeyState(uint32_t stateArrayCount, void* stateArray) = 0;
    virtual bool STDMETHODCALLTYPE GetMouseState(void* state) = 0;
    virtual uint32_t STDMETHODCALLTYPE GetTouchCount() = 0;
    virtual uint32_t STDMETHODCALLTYPE GetTouchState(uint32_t stateArrayCount, void* stateArray) = 0;
    virtual bool STDMETHODCALLTYPE GetMotionState(void* state) = 0;
    virtual bool STDMETHODCALLTYPE GetArcadeStickState(void* state) = 0;
    virtual bool STDMETHODCALLTYPE GetFlightStickState(void* state) = 0;
    virtual bool STDMETHODCALLTYPE GetGamepadState(GameInputGamepadState* state) = 0;
    virtual bool STDMETHODCALLTYPE GetRacingWheelState(void* state) = 0;
    virtual bool STDMETHODCALLTYPE GetUiNavigationState(GameInputUiNavigationState* state) = 0;
};

struct IGameInputDevice : IUnknown {
    virtual const GameInputDeviceInfo* STDMETHODCALLTYPE GetDeviceInfo() = 0;
    virtual uint32_t STDMETHODCALLTYPE GetDeviceStatus() = 0;
    virtual void STDMETHODCALLTYPE GetBatteryState(GameInputBatteryState* state) = 0;
    virtual HRESULT STDMETHODCALLTYPE CreateForceFeedbackEffect(uint32_t motorIndex, const void* params,
                                                                void** effect) = 0;
    virtual bool STDMETHODCALLTYPE IsForceFeedbackMotorPoweredOn(uint32_t motorIndex) = 0;
    virtual void STDMETHODCALLTYPE SetForceFeedbackMotorGain(uint32_t motorIndex, float masterGain) = 0;
    virtual HRESULT STDMETHODCALLTYPE SetHapticMotorState(uint32_t motorIndex, const void* params) = 0;
    virtual void STDMETHODCALLTYPE SetRumbleState(const GameInputRumbleParams* params) = 0;
    virtual void STDMETHODCALLTYPE SetInputSynchronizationState(bool enabled) = 0;
    virtual void STDMETHODCALLTYPE SendInputSynchronizationHint() = 0;
    virtual void STDMETHODCALLTYPE PowerOff() = 0;
    virtual HRESULT STDMETHODCALLTYPE CreateRawDeviceReport(uint32_t reportId, int32_t reportKind, void** report) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetRawDeviceFeature(uint32_t reportId, void** report) = 0;
    virtual HRESULT STDMETHODCALLTYPE SetRawDeviceFeature(void* report) = 0;
    virtual HRESULT STDMETHODCALLTYPE SendRawDeviceOutput(void* report) = 0;
    virtual HRESULT STDMETHODCALLTYPE SendRawDeviceOutputWithResponse(void* requestReport, void** responseReport) = 0;
    virtual HRESULT STDMETHODCALLTYPE ExecuteRawDeviceIoControl(uint32_t controlCode, size_t inputBufferSize,
                                                                const void* inputBuffer, size_t outputBufferSize,
                                                                void* outputBuffer, size_t* outputSize) = 0;
    virtual bool STDMETHODCALLTYPE AcquireExclusiveRawDeviceAccess(uint64_t timeoutInMicroseconds) = 0;
    virtual void STDMETHODCALLTYPE ReleaseExclusiveRawDeviceAccess() = 0;
};

struct IGameInputRawDeviceReport : IUnknown {
    virtual void STDMETHODCALLTYPE GetDevice(IGameInputDevice** device) = 0;
    virtual const GameInputRawDeviceReportInfo* STDMETHODCALLTYPE GetReportInfo() = 0;
    virtual size_t STDMETHODCALLTYPE GetRawDataSize() = 0;
    virtual size_t STDMETHODCALLTYPE GetRawData(size_t bufferSize, void* buffer) = 0;
    virtual bool STDMETHODCALLTYPE SetRawData(size_t bufferSize, const void* buffer) = 0;
    virtual bool STDMETHODCALLTYPE GetItemValue(uint32_t itemIndex, int64_t* value) = 0;
    virtual bool STDMETHODCALLTYPE SetItemValue(uint32_t itemIndex, int64_t value) = 0;
    virtual bool STDMETHODCALLTYPE ResetItemValue(uint32_t itemIndex) = 0;
    virtual bool STDMETHODCALLTYPE ResetAllItems() = 0;
};

// What CreateForceFeedbackEffect returns: the effect's parameters, gain and run state, and one reference on the
// device it belongs to.
struct IGameInputForceFeedbackEffect : IUnknown {
    virtual void STDMETHODCALLTYPE GetDevice(IGameInputDevice** device) = 0;
    virtual uint32_t STDMETHODCALLTYPE GetMotorIndex() = 0;
    virtual float STDMETHODCALLTYPE GetGain() = 0;
    virtual void STDMETHODCALLTYPE SetGain(float gain) = 0;
    virtual void STDMETHODCALLTYPE GetParams(GameInputForceFeedbackParams* params) = 0;
    virtual bool STDMETHODCALLTYPE SetParams(const GameInputForceFeedbackParams* params) = 0;
    virtual GameInputFeedbackEffectState STDMETHODCALLTYPE GetState() = 0;
    virtual void STDMETHODCALLTYPE SetState(GameInputFeedbackEffectState state) = 0;
};

struct IGameInputDispatcher : IUnknown {
    virtual bool STDMETHODCALLTYPE Dispatch(uint64_t quotaInMicroseconds) = 0;
    virtual HRESULT STDMETHODCALLTYPE OpenWaitHandle(HANDLE* waitHandle) = 0;
};

using GameInputCreateFn = HRESULT(WINAPI*)(IGameInput** gameInput);

}  // namespace gi
