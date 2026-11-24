cmake_minimum_required(VERSION 3.24)
project(gta6rt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The sources are UTF-8. Without /utf-8, MSVC reads them in the system's ANSI code page, so on code page 936 (or
# any other non-1252 locale) it prints C4819 for every non-ASCII comment. Many targets set /utf-8 themselves; this
# covers every target, including those that don't. clang-cl accepts it too. Only C and C++: ml64 (abi_probe.asm) has
# no /utf-8 and warns A4018 for it.
if(MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>)
endif()

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR "gta6rt must be built for x64 (run from an x64 Native Tools prompt or vcvars64.bat)")
endif()

option(LEONIDA_BUILD_WINDOWS_RUNTIME "Build the Windows runtime, its shims and their tests" ${WIN32})
if(WIN32 AND LEONIDA_BUILD_WINDOWS_RUNTIME)
  # Custom CMAKE_CXX_FLAGS can replace MSVC's default /EHsc. Require C++ stack unwinding so error
  # paths in the runtime and its libraries still destroy automatic objects when an exception leaves a frame.
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/EHsc>)
  endif()

  # The d3d12_xs shim and its tests use version 1.2 root signatures (D3D12_ROOT_SIGNATURE_DESC2), which the Windows
  # SDK declares from 10.0.26100 on. Stop here with the fix instead of failing halfway through the build.
  include(CheckCXXSourceCompiles)
  check_cxx_source_compiles([=[
    #include <d3d12.h>
    int main() { D3D12_VERSIONED_ROOT_SIGNATURE_DESC d{}; d.Version = D3D_ROOT_SIGNATURE_VERSION_1_2;
                 return static_cast<int>(sizeof(d.Desc_1_2)); }
  ]=] GTA6RT_SDK_HAS_ROOT_SIGNATURE_1_2)
  if(NOT GTA6RT_SDK_HAS_ROOT_SIGNATURE_1_2)
    if(CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION)
      set(_gta6rt_sdk "${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}")
    elseif(DEFINED ENV{WindowsSDKVersion})
      string(REGEX REPLACE "[\\/]+$" "" _gta6rt_sdk "$ENV{WindowsSDKVersion}")
    else()
      set(_gta6rt_sdk "an older one")
    endif()
    message(FATAL_ERROR
      "gta6rt needs the Windows SDK 10.0.26100 or newer (version 1.2 root signatures); "
      "this build uses ${_gta6rt_sdk}. Install it with "
      "`winget install Microsoft.WindowsSDK.10.0.26100` or the Visual Studio Installer, then delete the build "
      "folder and configure again.")
  endif()

  # Root-signature flag fix, shared by the d3d12_xs shim and its tests.
  add_library(gta6rt_d3d12_xs_rootsig STATIC
    shims/d3d12_xs_rootsig.cpp
  )
  target_include_directories(gta6rt_d3d12_xs_rootsig PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/shims)
  target_compile_definitions(gta6rt_d3d12_xs_rootsig PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
  target_compile_options(gta6rt_d3d12_xs_rootsig PRIVATE /W4 /permissive- /utf-8)

  # Portable root-signature model + packer to user SGPRs (no d3d12.h, no windows.h).
  add_library(gta6rt_root_signature STATIC
    src/root_signature.cpp
  )
  target_include_directories(gta6rt_root_signature PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_compile_definitions(gta6rt_root_signature PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
  target_compile_options(gta6rt_root_signature PRIVATE /W4 /permissive- /utf-8)

  # Portable contract test: model, packer, canonical codec and RTS0 hypothesis.
  # Needs neither d3d12.lib nor the game; runs on Windows CI like the rest here.
  add_executable(root_signature_test
    test/root_signature_test.cpp
  )
  target_link_libraries(root_signature_test PRIVATE gta6rt_root_signature)
  target_compile_definitions(root_signature_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
  target_compile_options(root_signature_test PRIVATE /W4 /permissive- /utf-8)
  target_link_options(root_signature_test PRIVATE /OPT:NOREF)

  # XTaskQueue/XAsync from microsoft/libHttpClient (MIT), vendored in third_party/xtaskqueue; shims/xtaskqueue/pch.h
  # stands in for libHttpClient's pch.h so only the task queue builds. Used by the xgameruntime shim.
  set(XTASKQUEUE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/xtaskqueue)
  add_library(gta6rt_xtaskqueue STATIC
    ${XTASKQUEUE_DIR}/Source/Task/TaskQueue.cpp
    ${XTASKQUEUE_DIR}/Source/Task/AsyncLib.cpp
    ${XTASKQUEUE_DIR}/Source/Task/ThreadPool_win32.cpp
    ${XTASKQUEUE_DIR}/Source/Task/WaitTimer_win32.cpp
  )
  target_include_directories(gta6rt_xtaskqueue
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/shims/xtaskqueue ${XTASKQUEUE_DIR}/Source/Task ${XTASKQUEUE_DIR}/Source/Common
    PUBLIC ${XTASKQUEUE_DIR}/Include)
  target_compile_definitions(gta6rt_xtaskqueue PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
  target_compile_options(gta6rt_xtaskqueue PRIVATE /utf-8)  # third-party code: its own warning level
  target_link_libraries(gta6rt_xtaskqueue PUBLIC ole32)

  # Reserves one loader-assigned module TLS index for mapped images. Loaded from the executable's directory by
  # Loader::initialize_tls; its trivial DllMain only keeps the TLS anchor in the image. Built in the same directory as
  # the executables that map images, because the loader looks for it next to the running EXE.
  add_library(gta6rt_tls_host SHARED src/tls_host.cpp)
  target_compile_options(gta6rt_tls_host PRIVATE /W4 /permissive- /utf-8 /GS-)
  target_compile_definitions(gta6rt_tls_host PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
  set_target_properties(gta6rt_tls_host PROPERTIES
    OUTPUT_NAME gta6rt_tls_host
    PREFIX ""
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
  )

  add_executable(gta6rt
    src/main.cpp
    src/pe_loader.cpp
    shims/registry.cpp
    shims/file_system.cpp
    shims/winhttp_offline.cpp
    shims/xmem.cpp
    shims/xg_xs.cpp
    shims/d3d12_xs.cpp
    shims/d3d12_xs_device.cpp
    shims/xgameruntime.cpp
    shims/xgameruntime_user.cpp
    shims/xgameruntime_gamesave.cpp
    shims/xgameruntime_store.cpp
    shims/xgameruntime_offline.cpp
    shims/library_loader.cpp
    shims/process.cpp
  )
  add_dependencies(gta6rt gta6rt_tls_host)
  target_compile_definitions(gta6rt PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX UNICODE _UNICODE)
  target_compile_options(gta6rt PRIVATE /W4 /permissive- /utf-8)
  # Static shim tables register themselves at startup; keep the linker from dropping their objects.
  # GTA6RT_HOST_BASE keeps the loader's own image off the default x64 EXE base (0x140000000), where a mapped EXE
  # without relocations wants to go. Windows moves the EXE with ASLR anyway; Wine loads it at its preferred base.
  set(GTA6RT_HOST_BASE /BASE:0x10000000000)
  # BUILD-1457: a thread that asks for no stack size gets the reserve in this program's PE header, 1 MB by default:
  # a thread the game starts with dwStackSize 0, a thread-pool thread, a std::thread of the shims. The D3D12 shim
  # creates Vulkan pipelines on the calling thread, the driver compiles there, and AMD's Windows compiler (26.11.2)
  # overflows 1 MB on some of the game's shaders (docs/pipeline_check.md, "Stack reserve"). run_entry already gives
  # the entry thread 8 MB (shims/process_shim.h); this is the same number for the others. Keep it in step with
  # scripts/build_runtime.ps1.
  set(GTA6RT_STACK 8388608)
  target_link_options(gta6rt PRIVATE /OPT:NOREF ${GTA6RT_HOST_BASE} /STACK:${GTA6RT_STACK})
  target_link_libraries(gta6rt PRIVATE gta6rt_d3d12_xs_rootsig gta6rt_xtaskqueue psapi)

  # Root-signature forwarding test: needs a real PC d3d12.lib to serialize the synthetic descriptors.
  # find_library searches %LIB% (a Developer Prompt), not the Windows SDK the Visual Studio generator links
  # against, so also look in that SDK's um/x64 folder.
  set(_sdk_lib_hints)
  if(DEFINED ENV{WindowsSdkDir} AND DEFINED ENV{WindowsSDKLibVersion})
    list(APPEND _sdk_lib_hints "$ENV{WindowsSdkDir}Lib/$ENV{WindowsSDKLibVersion}um/x64")
  endif()
  get_filename_component(_kits_root
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows Kits\\Installed Roots;KitsRoot10]" ABSOLUTE)
  if(CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION AND EXISTS "${_kits_root}")
    list(APPEND _sdk_lib_hints "${_kits_root}/Lib/${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}/um/x64")
  endif()
  find_library(D3D12_LIBRARY NAMES d3d12 HINTS ${_sdk_lib_hints})
  find_library(DXGI_LIBRARY NAMES dxgi HINTS ${_sdk_lib_hints})
  if(D3D12_LIBRARY)
    add_executable(root_sig_test
      test/root_sig_test.cpp
      shims/registry.cpp
      shims/d3d12_xs.cpp
      shims/d3d12_xs_device.cpp
    )
    target_link_libraries(root_sig_test PRIVATE gta6rt_d3d12_xs_rootsig ${D3D12_LIBRARY})
    target_compile_definitions(root_sig_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(root_sig_test PRIVATE /W4 /permissive- /utf-8)
    target_link_options(root_sig_test PRIVATE /OPT:NOREF)

    # RTS0 parser oracle: synthetic descriptors only, serialized by the real PC
    # d3d12.dll and parsed back by rt::rootsig::parse_rts0. A layout mismatch
    # fails here with the blob bytes, never silently in a draw.
    add_executable(root_signature_oracle_test
      test/root_signature_oracle_test.cpp
    )
    target_link_libraries(root_signature_oracle_test PRIVATE gta6rt_root_signature ${D3D12_LIBRARY})
    target_compile_definitions(root_signature_oracle_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(root_signature_oracle_test PRIVATE /W4 /permissive- /utf-8)
    target_link_options(root_signature_oracle_test PRIVATE /OPT:NOREF)

    # Xbox-ABI device proxy test: WARP PC device, no game executable. Needs DXGI for the WARP adapter.
    if(DXGI_LIBRARY)
      add_executable(device_proxy_test
        test/device_proxy_test.cpp
        shims/registry.cpp
        shims/d3d12_xs.cpp
        shims/d3d12_xs_device.cpp
      )
      target_link_libraries(device_proxy_test PRIVATE gta6rt_d3d12_xs_rootsig ${D3D12_LIBRARY} ${DXGI_LIBRARY})
      target_compile_definitions(device_proxy_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
      target_compile_options(device_proxy_test PRIVATE /W4 /permissive- /utf-8)
      target_link_options(device_proxy_test PRIVATE /OPT:NOREF)
    else()
      message(STATUS "dxgi.lib not found: device_proxy_test is not built")
    endif()
    # The queue and fence proxies over synthetic COM objects: no D3D12 runtime, no GPU. Replaces operator new and
    # patches its own import table to inject failures, so the proxy source is compiled into it alone.
    add_executable(xs_proxy_mock_test
      test/xs_proxy_mock_test.cpp
      shims/d3d12_xs_device.cpp
    )
    target_compile_definitions(xs_proxy_mock_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(xs_proxy_mock_test PRIVATE /W4 /permissive- /utf-8 /EHsc)
  else()
    message(STATUS "d3d12.lib not found: root_sig_test and device_proxy_test are not built")
  endif()

  # xmem shim ABI test: resolves the registered xmem.dll exports itself, so xmem.cpp and registry.cpp are compiled
  # into this executable instead of reusing gta6rt's objects (which would duplicate their symbols).
  add_executable(xmem_test
    test/xmem_test.cpp
    shims/xmem.cpp
    shims/registry.cpp
  )
  target_include_directories(xmem_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/shims)
  target_compile_definitions(xmem_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
  target_compile_options(xmem_test PRIVATE /W4 /permissive- /utf-8)
  target_link_options(xmem_test PRIVATE /OPT:NOREF)
  target_link_libraries(xmem_test PRIVATE psapi)

  # xg_xs shim: the two exports and the texture computer's slots against the measured call shapes.
  add_executable(xg_xs_test
    test/xg_xs_test.cpp
    shims/xg_xs.cpp
    shims/registry.cpp
  )
  target_include_directories(xg_xs_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/shims)
  target_compile_definitions(xg_xs_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
  target_compile_options(xg_xs_test PRIVATE /W4 /permissive- /utf-8)
  target_link_options(xg_xs_test PRIVATE /OPT:NOREF)

  # The process shim's command line: GetCommandLineW must parse (CommandLineToArgvW) back to its __wargv.
  add_executable(process_cmdline_test
    test/process_cmdline_test.cpp
    shims/process.cpp
    shims/registry.cpp
  )
  target_include_directories(process_cmdline_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/shims)
  target_compile_definitions(process_cmdline_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
  target_compile_options(process_cmdline_test PRIVATE /W4 /permissive- /utf-8)
  target_link_options(process_cmdline_test PRIVATE /OPT:NOREF)
  target_link_libraries(process_cmdline_test PRIVATE shell32)

  # Console paths (GAME:) and the write overlay: the file shims resolved through the registry, over temporary folders.
  add_executable(file_system_test
    test/file_system_test.cpp
    shims/file_system.cpp
    shims/registry.cpp
  )
  target_include_directories(file_system_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/shims)
  target_compile_definitions(file_system_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
  target_compile_options(file_system_test PRIVATE /W4 /permissive- /utf-8)
  target_link_options(file_system_test PRIVATE /OPT:NOREF)
  target_link_libraries(file_system_test PRIVATE advapi32)  # the ACL calls of the unlistable-marker-folder check

  # LEONIDA_SKIP_INT3 (shims/startup_trace.h): its own executable, the variable is read once per process.
  add_executable(int3_skip_test
    test/int3_skip_test.cpp
    shims/registry.cpp
  )
  target_include_directories(int3_skip_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/shims)
  target_compile_definitions(int3_skip_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
  target_compile_options(int3_skip_test PRIVATE /W4 /permissive- /utf-8)
  target_link_options(int3_skip_test PRIVATE /OPT:NOREF)

  # Offline WinHTTP for mapped images (the package's XCurl.dll and libHttpClient.GDK.dll); no network is touched.
  add_executable(winhttp_offline_test
    test/winhttp_offline_test.cpp
    shims/winhttp_offline.cpp
    shims/registry.cpp
  )
  target_include_directories(winhttp_offline_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/shims)
  target_compile_definitions(winhttp_offline_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
  target_compile_options(winhttp_offline_test PRIVATE /W4 /permissive- /utf-8)
  target_link_options(winhttp_offline_test PRIVATE /OPT:NOREF)

  # Synthetic tests for the atomic API reference lifecycle.
  add_executable(api_ref_state_test test/api_ref_state_test.cpp)
  target_include_directories(api_ref_state_test PRIVATE
    ${XTASKQUEUE_DIR}/Source/Task)
  target_compile_definitions(api_ref_state_test PRIVATE
    WIN32_LEAN_AND_MEAN NOMINMAX)
  target_compile_options(api_ref_state_test PRIVATE
    /W4 /permissive- /utf-8)
  # Verify global reference accounting during task queue teardown.
  add_executable(api_global_refs_test test/api_global_refs_test.cpp)
  target_compile_definitions(api_global_refs_test PRIVATE
    WIN32_LEAN_AND_MEAN NOMINMAX)
  target_compile_options(api_global_refs_test PRIVATE
    /W4 /permissive- /utf-8)
  target_link_libraries(api_global_refs_test PRIVATE
    gta6rt_xtaskqueue)
  # Offline GDK runtime shim (XGameRuntime, XSystem, XTaskQueue, XAsync), driven through the shim registry the way
  # a title calls it at boot. Like xmem_test it compiles its own registry.cpp.
  add_executable(xgameruntime_test
    test/xgameruntime_test.cpp
    shims/xgameruntime.cpp
    shims/xgameruntime_user.cpp
    shims/xgameruntime_gamesave.cpp
    shims/xgameruntime_store.cpp
    shims/xgameruntime_offline.cpp
    shims/library_loader.cpp
    shims/registry.cpp
    shims/file_system.cpp
  )
  target_include_directories(xgameruntime_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/shims)
  target_compile_definitions(xgameruntime_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
  target_compile_options(xgameruntime_test PRIVATE /W4 /permissive- /utf-8)
  target_link_options(xgameruntime_test PRIVATE /OPT:NOREF)
  target_link_libraries(xgameruntime_test PRIVATE gta6rt_xtaskqueue)

  # XUserImpl (offline local user) through QueryApiImpl, slot by slot as xgameruntime.lib's thunks call it.
  add_executable(xgameruntime_user_test
    test/xgameruntime_user_test.cpp
    shims/xgameruntime.cpp
    shims/xgameruntime_user.cpp
    shims/xgameruntime_gamesave.cpp
    shims/xgameruntime_store.cpp
    shims/xgameruntime_offline.cpp
    shims/library_loader.cpp
    shims/registry.cpp
    shims/file_system.cpp
  )
  target_include_directories(xgameruntime_user_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/shims)
  target_compile_definitions(xgameruntime_user_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
  target_compile_options(xgameruntime_user_test PRIVATE /W4 /permissive- /utf-8)
  target_link_options(xgameruntime_user_test PRIVATE /OPT:NOREF)
  target_link_libraries(xgameruntime_user_test PRIVATE gta6rt_xtaskqueue)

  # XGameSaveImpl (offline saves in LEONIDA_SAVE_DIR) through QueryApiImpl, slot by slot.
  add_executable(xgameruntime_gamesave_test
    test/xgameruntime_gamesave_test.cpp
    shims/xgameruntime.cpp
    shims/xgameruntime_user.cpp
    shims/xgameruntime_gamesave.cpp
    shims/xgameruntime_store.cpp
    shims/xgameruntime_offline.cpp
    shims/library_loader.cpp
    shims/registry.cpp
    shims/file_system.cpp
  )
  target_include_directories(xgameruntime_gamesave_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/shims)
  target_compile_definitions(xgameruntime_gamesave_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
  target_compile_options(xgameruntime_gamesave_test PRIVATE /W4 /permissive- /utf-8)
  target_link_options(xgameruntime_gamesave_test PRIVATE /OPT:NOREF)
  target_link_libraries(xgameruntime_gamesave_test PRIVATE gta6rt_xtaskqueue)

  # XStoreImpl (offline: the game is owned, no add-ons, no updates) through QueryApiImpl, slot by slot.
  add_executable(xgameruntime_store_test
    test/xgameruntime_store_test.cpp
    shims/xgameruntime.cpp
    shims/xgameruntime_user.cpp
    shims/xgameruntime_gamesave.cpp
    shims/xgameruntime_store.cpp
    shims/xgameruntime_offline.cpp
    shims/library_loader.cpp
    shims/registry.cpp
    shims/file_system.cpp
  )
  target_include_directories(xgameruntime_store_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/shims)
  target_compile_definitions(xgameruntime_store_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
  target_compile_options(xgameruntime_store_test PRIVATE /W4 /permissive- /utf-8)
  target_link_options(xgameruntime_store_test PRIVATE /OPT:NOREF)
  target_link_libraries(xgameruntime_store_test PRIVATE gta6rt_xtaskqueue)

  # XGameEventImpl, XGameUiImpl, XGameStreamingImpl (offline) through QueryApiImpl.
  add_executable(xgameruntime_offline_test
    test/xgameruntime_offline_test.cpp
    shims/xgameruntime.cpp
    shims/xgameruntime_user.cpp
    shims/xgameruntime_gamesave.cpp
    shims/xgameruntime_store.cpp
    shims/xgameruntime_offline.cpp
    shims/library_loader.cpp
    shims/registry.cpp
    shims/file_system.cpp
  )
  target_include_directories(xgameruntime_offline_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/shims)
  target_compile_definitions(xgameruntime_offline_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
  target_compile_options(xgameruntime_offline_test PRIVATE /W4 /permissive- /utf-8)
  target_link_options(xgameruntime_offline_test PRIVATE /OPT:NOREF)
  target_link_libraries(xgameruntime_offline_test PRIVATE gta6rt_xtaskqueue)

  # GameInput shim on SDL3 (zlib licence): any controller SDL supports, DualSense trigger effects and lightbar.
  # Nothing is downloaded unless asked. The shim is built when SDL3 is available:
  #   - an installed SDL3 3.2+ that find_package finds (SDL3_DIR / CMAKE_PREFIX_PATH), or
  #   - -DLEONIDA_SDL3_FETCH=ON: download the pinned release below once at configure time (hash-checked) and build it
  #     static, or -DFETCHCONTENT_SOURCE_DIR_SDL3=<extracted SDL3-3.4.20> to build that copy without downloading.
  # Otherwise, or with -DLEONIDA_SDL3=OFF, gta6rt is built without the shim and the loader binds GameInputCreate to the
  # host's GameInput.dll or the unimplemented thunk as before. docs/gameinput.md has the details.
  option(LEONIDA_SDL3 "Build the GameInput shim when SDL3 is available" ON)
  option(LEONIDA_SDL3_FETCH "Download the pinned SDL3 release if no installed SDL3 is found" OFF)
  if(LEONIDA_SDL3)
    find_package(SDL3 3.2 CONFIG QUIET COMPONENTS SDL3-static NO_SYSTEM_ENVIRONMENT_PATH)
    if(NOT TARGET SDL3::SDL3-static AND (LEONIDA_SDL3_FETCH OR FETCHCONTENT_SOURCE_DIR_SDL3))
      include(FetchContent)
      FetchContent_Declare(SDL3
        URL https://github.com/libsdl-org/SDL/releases/download/release-3.4.20/SDL3-3.4.20.tar.gz
        URL_HASH SHA256=f3bb06f55bed9557227a700af2ea62e49497ebdf58037732a731a5fb3ef8581f
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
      # Only what gamepads need: the joystick, HIDAPI, haptic, sensor and power code, in a static library.
      set(SDL_SHARED OFF CACHE BOOL "" FORCE)
      set(SDL_STATIC ON CACHE BOOL "" FORCE)
      set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
      set(SDL_TESTS OFF CACHE BOOL "" FORCE)
      set(SDL_EXAMPLES OFF CACHE BOOL "" FORCE)
      set(SDL_INSTALL OFF CACHE BOOL "" FORCE)
      set(SDL_DISABLE_INSTALL ON CACHE BOOL "" FORCE)
      foreach(_sdl_off SDL_AUDIO SDL_VIDEO SDL_GPU SDL_RENDER SDL_CAMERA SDL_DIALOG SDL_TRAY)
        set(${_sdl_off} OFF CACHE BOOL "" FORCE)
      endforeach()
      FetchContent_MakeAvailable(SDL3)
    endif()
  endif()
  if(LEONIDA_SDL3 AND TARGET SDL3::SDL3-static)
    target_sources(gta6rt PRIVATE shims/gameinput.cpp)
    target_link_libraries(gta6rt PRIVATE SDL3::SDL3-static)

    # GameInput through the shim registry on SDL virtual joysticks: no controller, no game data.
    add_executable(gameinput_test test/gameinput_test.cpp shims/gameinput.cpp shims/registry.cpp)
    # gameinput_probe: the same shim on a real controller, run by hand (not a CTest).
    add_executable(gameinput_probe test/gameinput_probe.cpp shims/gameinput.cpp shims/registry.cpp)
    foreach(_gi gameinput_test gameinput_probe)
      target_include_directories(${_gi} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/shims)
      target_compile_definitions(${_gi} PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
      target_compile_options(${_gi} PRIVATE /W4 /permissive- /utf-8)
      target_link_options(${_gi} PRIVATE /OPT:NOREF)
      target_link_libraries(${_gi} PRIVATE SDL3::SDL3-static)
    endforeach()
  else()
    if(LEONIDA_SDL3)
      message(STATUS "GameInput shim not built: no installed SDL3 found. Set SDL3_DIR, or pass -DLEONIDA_SDL3_FETCH=ON "
                     "to download the pinned SDL3 release (see docs/gameinput.md)")
    else()
      message(STATUS "GameInput shim not built (LEONIDA_SDL3=OFF)")
    endif()
  endif()

elseif(NOT WIN32)
  message(STATUS "gta6rt requires Windows; only Vulkan tools are available on Linux")
endif()

include(CTest)
if(BUILD_TESTING)
  add_executable(gfx10_tiling_test test/gfx10_tiling_test.cpp)
  add_test(NAME gfx10_tiling COMMAND gfx10_tiling_test)

  add_executable(retire_tickets_test test/retire_tickets_test.cpp)
  add_test(NAME retire_tickets COMMAND retire_tickets_test)
  # VulkanContext::buffer_at's VA index (runtime/src/buffer_va_index.h): pure stdlib, no Vulkan instance.
  add_executable(buffer_va_index_test test/buffer_va_index_test.cpp)
  add_test(NAME buffer_va_index COMMAND buffer_va_index_test)

  add_executable(xbox_present_readback_size_test test/xbox_present_readback_size_test.cpp)
  add_test(NAME xbox_present_readback_size COMMAND xbox_present_readback_size_test)

  add_executable(pipeline_cache_file_test test/pipeline_cache_file_test.cpp)
  add_test(NAME pipeline_cache_file COMMAND pipeline_cache_file_test)

  add_executable(xdxr_build_output_test test/xdxr_build_output_test.cpp)
  add_test(NAME xdxr_build_output COMMAND xdxr_build_output_test)

  # vk_pipeline_check's memory budget for large modules: pure stdlib, no Vulkan device.
  add_executable(large_module_gate_test test/large_module_gate_test.cpp)
  find_package(Threads REQUIRED)
  target_link_libraries(large_module_gate_test PRIVATE Threads::Threads)
  add_test(NAME large_module_gate COMMAND large_module_gate_test)
  # vk_pipeline_check's isolate summary stage set: pure stdlib, no Vulkan device.
  add_executable(pipeline_summary_test test/pipeline_summary_test.cpp)
  add_test(NAME pipeline_summary COMMAND pipeline_summary_test)
  # XDXR build descriptions (docs/xdxr.md); compared with d3d12.h where the Windows SDK provides it.
  add_executable(xdxr_descs_test test/xdxr_descs_test.cpp)
  add_test(NAME xdxr_descs COMMAND xdxr_descs_test)
  # LEONIDA_VK_DEVICE name matching (runtime/src/vulkan_device_filter.h): pure stdlib, no Vulkan instance.
  add_executable(vulkan_device_filter_test test/vulkan_device_filter_test.cpp)
  target_include_directories(vulkan_device_filter_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  add_test(NAME vulkan_device_filter COMMAND vulkan_device_filter_test)
  add_executable(vk_presenter_capture_colour_test test/vk_presenter_capture_colour_test.cpp)
  add_test(NAME vk_presenter_capture_colour COMMAND vk_presenter_capture_colour_test)
  if(TARGET gta6rt)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      add_executable(cpp_unwind_test test/cpp_unwind_test.cpp)
      add_test(NAME cpp_unwind COMMAND cpp_unwind_test)
    endif()

    # Compile-time ABI checks (D3D12/XMem structure sizes, alignments and member offsets against the Windows SDK).
    # The file existed but no target built it, so none of its static_asserts ever ran.
    add_executable(runtime_types_test test/test_runtime_types.cpp)
    target_include_directories(runtime_types_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/shims)
    add_test(NAME runtime_types COMMAND runtime_types_test)

    # Loader tests. They build their own PE images, so they need neither the game nor any other input.
    add_executable(loader_test
      test/loader_test.cpp
      src/pe_loader.cpp
      shims/registry.cpp
    )
    target_compile_definitions(loader_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX UNICODE _UNICODE)
    target_compile_options(loader_test PRIVATE /W4 /permissive- /utf-8 /EHsc)
    target_link_options(loader_test PRIVATE /OPT:NOREF ${GTA6RT_HOST_BASE})
    add_dependencies(loader_test gta6rt_tls_host)
    add_test(NAME loader COMMAND loader_test)
    # PE loader tests use two locally built fixtures: a DLL with exports, imports and relocations, and an EXE that
    # imports it so Loader::load exercises recursive module loading. They contain no game data.
    set(TINY_FIXTURE_DIR "${CMAKE_CURRENT_BINARY_DIR}/fixtures/$<CONFIG>")

    add_library(tiny_dll SHARED test/fixtures/tiny_dll.cpp)
    target_compile_definitions(tiny_dll PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(tiny_dll PRIVATE /W4 /permissive- /utf-8 /GS-)
    # tiny_fault_caught faults inside its own __try. MSVC always catches that; clang-cl only with asynchronous
    # exceptions, otherwise the access violation escapes and the unwind-registration check fails.
    target_compile_options(tiny_dll PRIVATE $<$<CXX_COMPILER_ID:Clang>:/EHa>)
    set_target_properties(tiny_dll PROPERTIES
      OUTPUT_NAME tiny_dll
      PREFIX ""
      RUNTIME_OUTPUT_DIRECTORY "${TINY_FIXTURE_DIR}"
      ARCHIVE_OUTPUT_DIRECTORY "${TINY_FIXTURE_DIR}"
      MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
    )

    add_executable(tiny_exe test/fixtures/tiny_exe.cpp)
    target_link_libraries(tiny_exe PRIVATE tiny_dll)
    target_compile_definitions(tiny_exe PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(tiny_exe PRIVATE /W4 /permissive- /utf-8 /GS-)
    set_target_properties(tiny_exe PROPERTIES
      OUTPUT_NAME tiny_exe
      RUNTIME_OUTPUT_DIRECTORY "${TINY_FIXTURE_DIR}"
      MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
    )

    add_executable(module_names_test test/module_names_test.cpp src/pe_loader.cpp shims/registry.cpp)
    target_include_directories(module_names_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(module_names_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(module_names_test PRIVATE /W4 /permissive- /utf-8 /EHsc)
    target_link_options(module_names_test PRIVATE /OPT:NOREF)
    add_dependencies(module_names_test gta6rt_tls_host tiny_dll)
    add_test(NAME module_names COMMAND module_names_test "$<TARGET_FILE:tiny_dll>")

    add_executable(tiny_module_names test/fixtures/tiny_module_names.cpp)
    target_compile_definitions(tiny_module_names PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(tiny_module_names PRIVATE /W4 /permissive- /utf-8 /EHsc)
    set_target_properties(tiny_module_names PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${TINY_FIXTURE_DIR}")

    # Native Windows and --dry-run/--init/--run share original renamed fixtures, including an ACP-named import.
    if(NOT CMAKE_CROSSCOMPILING)
      find_package(Python3 COMPONENTS Interpreter QUIET)
      if(Python3_FOUND)
        add_test(NAME run_module_names COMMAND "${Python3_EXECUTABLE}" -I
          "${CMAKE_CURRENT_SOURCE_DIR}/test/check_module_names.py"
          "--loader=$<TARGET_FILE:gta6rt>" "--guest=$<TARGET_FILE:tiny_module_names>"
          "--importer=$<TARGET_FILE:tiny_exe>" "--dependency=$<TARGET_FILE:tiny_dll>"
          "--work-dir=${CMAKE_CURRENT_BINARY_DIR}"
          "--report=${CMAKE_CURRENT_BINARY_DIR}/module-names-results.json")
        set_tests_properties(run_module_names PROPERTIES TIMEOUT 90)
      endif()
    endif()

    # pe_loader.cpp needs the shim registry for shims::find/make_unimplemented_thunk. The xmem and d3d12_xs tables
    # are linked too so the optional LEONIDA_GAME_DIR run classifies the shipped modules' imports exactly like
    # gta6rt does (no d3d12.lib is needed: d3d12_xs resolves its host entry points at run time).
    add_executable(pe_loader_test
      test/pe_loader_test.cpp
      src/pe_loader.cpp
      shims/registry.cpp
      shims/xmem.cpp
      shims/d3d12_xs.cpp
      shims/d3d12_xs_device.cpp
    )
    target_include_directories(pe_loader_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_SOURCE_DIR}/shims
    )
    target_compile_definitions(pe_loader_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(pe_loader_test PRIVATE /W4 /permissive- /utf-8 /EHsc)
    target_link_options(pe_loader_test PRIVATE /OPT:NOREF)
    target_link_libraries(pe_loader_test PRIVATE gta6rt_d3d12_xs_rootsig psapi)
    add_dependencies(pe_loader_test tiny_dll tiny_exe gta6rt_tls_host)

    # `gta6rt --run` rehearsal: a stand-in game EXE (CRT startup, statics, heap, files, threads) that imports
    # xgameruntime.dll from a link-time stub; under --run the shim registry answers those imports instead.
    add_library(xgameruntime_stub SHARED test/fixtures/xgameruntime_stub.cpp test/fixtures/xgameruntime_stub.def)
    target_include_directories(xgameruntime_stub PUBLIC ${XTASKQUEUE_DIR}/Include)
    target_compile_definitions(xgameruntime_stub PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    set_target_properties(xgameruntime_stub PROPERTIES
      OUTPUT_NAME xgameruntime
      PREFIX ""
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fixtures/game/$<CONFIG>"
      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fixtures/game/$<CONFIG>"
    )
    add_executable(tiny_async test/fixtures/tiny_async.cpp)
    target_link_libraries(tiny_async PRIVATE xgameruntime_stub)
    set_target_properties(tiny_async PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fixtures/game/$<CONFIG>"
    )
    # Reaches xgameruntime.dll through LoadLibraryW/GetProcAddress, as xgameruntime.lib's thunks do (no import).
    add_executable(tiny_gdk_load test/fixtures/tiny_gdk_load.cpp)
    target_compile_definitions(tiny_gdk_load PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(tiny_gdk_load PRIVATE /W4 /permissive- /utf-8)
    set_target_properties(tiny_gdk_load PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fixtures/game/$<CONFIG>"
    )
    # Run-time LoadLibrary of a bundled Xbox DLL (importing xmem.dll) through rt::Loader.
    add_library(xmem_stub SHARED test/fixtures/xmem_stub.cpp test/fixtures/xmem_stub.def)
    target_compile_definitions(xmem_stub PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(xmem_stub PRIVATE /W4 /permissive- /utf-8)
    set_target_properties(xmem_stub PROPERTIES
      OUTPUT_NAME xmem
      PREFIX ""
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fixtures/stubs/$<CONFIG>"
      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fixtures/stubs/$<CONFIG>"
    )
    add_library(tiny_bundled_dll SHARED test/fixtures/tiny_bundled_dll.cpp test/fixtures/tiny_bundled_dll.def)
    target_link_libraries(tiny_bundled_dll PRIVATE xmem_stub)
    target_compile_definitions(tiny_bundled_dll PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(tiny_bundled_dll PRIVATE /W4 /permissive- /utf-8)
    set_target_properties(tiny_bundled_dll PROPERTIES
      OUTPUT_NAME tiny_bundled_dll
      PREFIX ""
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fixtures/game/$<CONFIG>"
      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fixtures/stubs/$<CONFIG>"
    )
    add_executable(tiny_bundled_load test/fixtures/tiny_bundled_load.cpp)
    target_compile_definitions(tiny_bundled_load PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(tiny_bundled_load PRIVATE /W4 /permissive- /utf-8)
    add_dependencies(tiny_bundled_load tiny_bundled_dll)
    set_target_properties(tiny_bundled_load PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fixtures/game/$<CONFIG>"
    )
    add_library(tiny_startup_cmdline_dll SHARED test/fixtures/tiny_startup_cmdline.cpp)
    target_compile_definitions(tiny_startup_cmdline_dll PRIVATE STARTUP_CMDLINE_DLL)
    add_executable(tiny_startup_cmdline test/fixtures/tiny_startup_cmdline.cpp)
    target_link_libraries(tiny_startup_cmdline PRIVATE tiny_startup_cmdline_dll)
    foreach(target tiny_startup_cmdline_dll tiny_startup_cmdline)
      target_compile_definitions(${target} PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
      target_compile_options(${target} PRIVATE /W4 /permissive- /utf-8)
      set_target_properties(${target} PROPERTIES
        PREFIX ""
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fixtures/game/$<CONFIG>"
      )
    endforeach()
    # These linker-selected startup modes are specific to the MSVC CRT.
    if(MSVC)
      foreach(width a w)
        foreach(mode default expand noargs)
          set(target tiny_argv_${mode}_${width})
          add_executable(${target} test/fixtures/tiny_argv_modes.cpp)
          target_compile_definitions(${target} PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
          target_compile_options(${target} PRIVATE /W4 /permissive- /utf-8)
          if(width STREQUAL "w")
            target_compile_definitions(${target} PRIVATE APP_WIDE)
          endif()
          if(mode STREQUAL "expand")
            if(width STREQUAL "w")
              target_link_options(${target} PRIVATE wsetargv.obj)
            else()
              target_link_options(${target} PRIVATE setargv.obj)
            endif()
          elseif(mode STREQUAL "noargs")
            target_link_options(${target} PRIVATE noarg.obj)
          endif()
          set_target_properties(${target} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fixtures/game/$<CONFIG>")
        endforeach()
      endforeach()
    endif()
    add_executable(tiny_atexit test/fixtures/tiny_atexit.cpp)
    target_compile_definitions(tiny_atexit PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(tiny_atexit PRIVATE /W4 /permissive- /utf-8)
    set_target_properties(tiny_atexit PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fixtures/game/$<CONFIG>"
    )
    add_executable(tiny_game test/fixtures/tiny_game.cpp)
    target_link_libraries(tiny_game PRIVATE xgameruntime_stub)
    target_compile_definitions(tiny_game PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX _CRT_SECURE_NO_WARNINGS)  # plain fopen, as games use
    target_compile_options(tiny_game PRIVATE /W4 /permissive- /utf-8 /EHsc)
    set_target_properties(tiny_game PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fixtures/game/$<CONFIG>"
    )
    # The same game with no .reloc, at the default EXE base, as RT-22 asks about the real one: it can only run at
    # 0x140000000, which arm64 Wine on macOS uses for itself (src/base_reservation.h).
    add_executable(tiny_game_fixed test/fixtures/tiny_game.cpp)
    target_link_libraries(tiny_game_fixed PRIVATE xgameruntime_stub)
    target_compile_definitions(tiny_game_fixed PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX _CRT_SECURE_NO_WARNINGS)
    target_compile_options(tiny_game_fixed PRIVATE /W4 /permissive- /utf-8 /EHsc)
    target_link_options(tiny_game_fixed PRIVATE /FIXED /DYNAMICBASE:NO /BASE:0x140000000)
    set_target_properties(tiny_game_fixed PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fixtures/game/$<CONFIG>"
    )
    add_executable(tiny_game_exit_code_test test/tiny_game_exit_code_test.cpp)
    target_compile_definitions(tiny_game_exit_code_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(tiny_game_exit_code_test PRIVATE /W4 /permissive- /utf-8)
    add_dependencies(tiny_game_exit_code_test gta6rt tiny_game tiny_game_fixed)
    # abort() must reach the guest's SIGABRT handler before the run ends with the CRT's abort code (3).
    add_executable(tiny_game_abort_test test/tiny_game_abort_test.cpp)
    target_compile_definitions(tiny_game_abort_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(tiny_game_abort_test PRIVATE /W4 /permissive- /utf-8)
    add_dependencies(tiny_game_abort_test gta6rt tiny_game)
    add_test(NAME pe_loader COMMAND pe_loader_test "$<TARGET_FILE:tiny_dll>" "$<TARGET_FILE:tiny_exe>")

    # Synthetic, hand-built images through Loader::load: range checks, wrapping RVAs, import-table termination and
    # base relocations (HIGHLOW included). No fixture DLL is needed; the images are written to a temporary folder.
    add_executable(test_pe_loader_bounds
      test/test_pe_loader_bounds.cpp
      src/pe_loader.cpp
      shims/registry.cpp
      shims/xmem.cpp
      shims/d3d12_xs.cpp
      shims/d3d12_xs_device.cpp
    )
    target_include_directories(test_pe_loader_bounds PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_SOURCE_DIR}/shims
    )
    target_compile_definitions(test_pe_loader_bounds PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(test_pe_loader_bounds PRIVATE /W4 /permissive- /utf-8 /EHsc)
    target_link_options(test_pe_loader_bounds PRIVATE /OPT:NOREF ${GTA6RT_HOST_BASE})
    target_link_libraries(test_pe_loader_bounds PRIVATE gta6rt_d3d12_xs_rootsig psapi)
    add_dependencies(test_pe_loader_bounds gta6rt_tls_host)
    add_test(NAME pe_loader_bounds COMMAND test_pe_loader_bounds)

    # Static TLS fixture: a DLL with __declspec(thread) and a TLS callback. It is ordinary test code; no game data
    # is involved. loader_tls_test needs gta6rt_tls_host.dll next to its own EXE, because that is where
    # Loader::initialize_tls looks for the reserved module TLS index.
    add_library(tiny_tls SHARED test/fixtures/tiny_tls.cpp)
    target_compile_definitions(tiny_tls PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(tiny_tls PRIVATE /W4 /permissive- /utf-8 /GS-)
    set_target_properties(tiny_tls PROPERTIES
      OUTPUT_NAME tiny_tls
      PREFIX ""
      RUNTIME_OUTPUT_DIRECTORY "${TINY_FIXTURE_DIR}"
      ARCHIVE_OUTPUT_DIRECTORY "${TINY_FIXTURE_DIR}"
    )

    add_library(tiny_tls_cb_only SHARED test/fixtures/tiny_tls_cb_only.cpp)
    target_compile_definitions(tiny_tls_cb_only PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(tiny_tls_cb_only PRIVATE /W4 /permissive- /utf-8 /GS-)
    target_link_options(tiny_tls_cb_only PRIVATE /NODEFAULTLIB /ENTRY:DllMainCRTStartup)
    target_link_libraries(tiny_tls_cb_only PRIVATE kernel32)
    set_target_properties(tiny_tls_cb_only PROPERTIES
      OUTPUT_NAME tiny_tls_cb_only
      PREFIX ""
      RUNTIME_OUTPUT_DIRECTORY "${TINY_FIXTURE_DIR}"
      ARCHIVE_OUTPUT_DIRECTORY "${TINY_FIXTURE_DIR}"
    )

    # pe_loader.cpp needs the shim registry for shims::find/make_unimplemented_thunk. The fixtures import only host
    # DLLs, so no shim tables are linked.
    add_executable(loader_tls_test
      test/loader_tls_test.cpp
      src/pe_loader.cpp
      shims/registry.cpp
    )
    target_include_directories(loader_tls_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(loader_tls_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(loader_tls_test PRIVATE /W4 /permissive- /utf-8 /EHsc)
    add_dependencies(loader_tls_test tiny_tls tiny_tls_cb_only gta6rt_tls_host)
    add_test(NAME loader_tls
      COMMAND loader_tls_test "$<TARGET_FILE:tiny_tls>"
    )

    # DisableThreadLibraryCalls from a mapped DllMain (shims/library_loader.cpp). The fixture is built without the CRT, so
    # DllMain's only work is that one call. No game files.
    add_library(dll_thread_calls SHARED test/fixtures/dll_thread_calls.cpp)
    target_compile_options(dll_thread_calls PRIVATE /W4 /permissive- /utf-8 /GS-)
    target_link_options(dll_thread_calls PRIVATE /NODEFAULTLIB /ENTRY:dll_thread_calls_main)
    target_link_libraries(dll_thread_calls PRIVATE kernel32)
    set_target_properties(dll_thread_calls PROPERTIES
      OUTPUT_NAME dll_thread_calls
      PREFIX ""
      RUNTIME_OUTPUT_DIRECTORY "${TINY_FIXTURE_DIR}"
      ARCHIVE_OUTPUT_DIRECTORY "${TINY_FIXTURE_DIR}"
    )
    add_executable(dll_thread_calls_test
      test/dll_thread_calls_test.cpp
      src/pe_loader.cpp
      shims/library_loader.cpp
      shims/registry.cpp
      shims/xmem.cpp
    )
    target_include_directories(dll_thread_calls_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_SOURCE_DIR}/shims
    )
    target_compile_definitions(dll_thread_calls_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(dll_thread_calls_test PRIVATE /W4 /permissive- /utf-8 /EHsc)
    target_link_options(dll_thread_calls_test PRIVATE /OPT:NOREF)
    target_link_libraries(dll_thread_calls_test PRIVATE psapi)
    add_dependencies(dll_thread_calls_test dll_thread_calls)
    add_test(NAME dll_thread_calls
      COMMAND dll_thread_calls_test "$<TARGET_FILE:dll_thread_calls>"
    )

    # The host GameInput path (docs/gameinput.md): mapped code that imports GameInputCreate from GameInput.dll, with no
    # GameInput shim linked. Needs the SDK's gameinput.lib and GameInput.h to build the fixture, and a GameInput.dll on
    # the machine to run (the test skips with 77 only when Windows cannot load GameInput.dll). No game files.
    find_library(GAMEINPUT_LIBRARY NAMES gameinput HINTS ${_sdk_lib_hints})
    if(GAMEINPUT_LIBRARY)
      add_library(gameinput_host SHARED test/fixtures/gameinput_host.cpp)
      target_compile_options(gameinput_host PRIVATE /W4 /permissive- /utf-8 /GS-)
      target_link_options(gameinput_host PRIVATE /NODEFAULTLIB /ENTRY:gameinput_host_main)
      target_link_libraries(gameinput_host PRIVATE kernel32 ${GAMEINPUT_LIBRARY})
      set_target_properties(gameinput_host PROPERTIES
        OUTPUT_NAME gameinput_host
        PREFIX ""
        RUNTIME_OUTPUT_DIRECTORY "${TINY_FIXTURE_DIR}"
        ARCHIVE_OUTPUT_DIRECTORY "${TINY_FIXTURE_DIR}"
      )
      add_executable(gameinput_host_test
        test/gameinput_host_test.cpp
        src/pe_loader.cpp
        shims/registry.cpp
      )
      target_include_directories(gameinput_host_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
      target_compile_definitions(gameinput_host_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
      target_compile_options(gameinput_host_test PRIVATE /W4 /permissive- /utf-8 /EHsc)
      add_dependencies(gameinput_host_test gameinput_host)
      add_test(NAME gameinput_host COMMAND gameinput_host_test "$<TARGET_FILE:gameinput_host>")
      set_tests_properties(gameinput_host PROPERTIES SKIP_RETURN_CODE 77)  # Windows cannot load GameInput.dll
    endif()

    # LEONIDA_UNIMPLEMENTED=log|fail|stop: builds its own image with two imports that bind to nothing, calls them in
    # process (log, fail) and in child processes (default/env parsing, stop). No game files, no GPU.
    add_executable(unimplemented_policy_test
      test/unimplemented_policy_test.cpp
      src/pe_loader.cpp
      shims/registry.cpp
    )
    # The register-preservation check needs a caller that sets rbx/rbp/rdi/rsi/r12-r15 and xmm6-15, which MSVC x64
    # cannot do inline: test/fixtures/abi_probe.asm (MASM). Without ml64 the rest of the test still runs.
    # Visual Studio locates cl.exe without adding its directory to PATH.
    # Keep looking for ml64 specifically so Ninja cannot select the 32-bit assembler.
    get_filename_component(_gta6rt_compiler_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
    find_program(LEONIDA_ML64 NAMES ml64 HINTS "${_gta6rt_compiler_bin}")
    if(LEONIDA_ML64)
      set(CMAKE_ASM_MASM_COMPILER "${LEONIDA_ML64}")
      enable_language(ASM_MASM)
      target_sources(unimplemented_policy_test PRIVATE test/fixtures/abi_probe.asm)
      target_compile_definitions(unimplemented_policy_test PRIVATE LEONIDA_HAVE_ABI_PROBE=1)
    else()
      message(STATUS "ml64 not found: unimplemented_policy_test skips the register-preservation check")
    endif()
    target_include_directories(unimplemented_policy_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(unimplemented_policy_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX UNICODE _UNICODE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(unimplemented_policy_test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive- /utf-8 /EHsc>)
    target_link_options(unimplemented_policy_test PRIVATE /OPT:NOREF)
    add_dependencies(unimplemented_policy_test gta6rt_tls_host)
    add_test(NAME unimplemented_policy COMMAND unimplemented_policy_test)

    # gta6rt --trace FILE (shims/startup_trace.h): a synthetic image with an unimplemented import, an
    # OutputDebugStringA call and a read of address 0; checks the JSON lines. No game files, no GPU.
    add_executable(startup_trace_test
      test/startup_trace_test.cpp
      src/pe_loader.cpp
      shims/registry.cpp
    )
    target_include_directories(startup_trace_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(startup_trace_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX UNICODE _UNICODE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(startup_trace_test PRIVATE /W4 /permissive- /utf-8 /EHsc)
    target_link_options(startup_trace_test PRIVATE /OPT:NOREF)
    add_dependencies(startup_trace_test gta6rt_tls_host)
    add_test(NAME startup_trace COMMAND startup_trace_test)
  endif()

  if(TARGET device_proxy_test)
    add_test(NAME device_proxy COMMAND device_proxy_test)
    set_tests_properties(device_proxy PROPERTIES SKIP_RETURN_CODE 77)  # no WARP adapter (Wine)
    # With --trace-d3d12 the 24 D3D12XboxCreateDevice parameter words still print (docs/boot.md, question 1),
    # and the line with run-specific pointers does not.
    add_test(NAME device_proxy_trace COMMAND device_proxy_test --trace-d3d12)
    set_tests_properties(device_proxy_trace PROPERTIES SKIP_RETURN_CODE 77
        PASS_REGULAR_EXPRESSION "D3D12XboxCreateDevice params\\[24\\] =1234abcd"
        FAIL_REGULAR_EXPRESSION "FAIL;D3D12XboxCreateDevice adapter=")
  endif()
  if(TARGET xs_proxy_mock_test)
    add_test(NAME xs_proxy_mock COMMAND xs_proxy_mock_test)
    # The guard region's setup failing (one process each: the tables are built once per process).
    add_test(NAME xs_proxy_mock_no_reserve COMMAND xs_proxy_mock_test --fail-virtualalloc)
    add_test(NAME xs_proxy_mock_no_protect COMMAND xs_proxy_mock_test --fail-virtualprotect)
    add_test(NAME xs_proxy_mock_no_handler COMMAND xs_proxy_mock_test --fail-veh)
    # The routes are read once per process, so each override case is its own test.
    add_test(NAME xs_proxy_mock_env_slots COMMAND xs_proxy_mock_test --env-slots)
    add_test(NAME xs_proxy_mock_env_slots_invalid COMMAND xs_proxy_mock_test --env-slots-invalid)
  endif()
  if(TARGET root_sig_test)
    add_test(NAME root_sig COMMAND root_sig_test)
  endif()
  add_executable(command_list_test test/command_list_test.cpp)
  add_test(NAME command_list COMMAND command_list_test)
  # Which SPIR-V modules declare a storage image or buffer they may write (src/spirv_stores.h), on hand-built words.
  add_executable(spirv_stores_test test/spirv_stores_test.cpp)
  add_test(NAME spirv_stores COMMAND spirv_stores_test)
  add_executable(shader_fixture_path_test test/shader_fixture_path_test.cpp)
  add_test(NAME shader_fixture_path COMMAND shader_fixture_path_test)
  add_executable(resource_tiling_test test/resource_tiling_test.cpp)
  add_test(NAME resource_tiling COMMAND resource_tiling_test)
  add_executable(resource_detile_test test/resource_detile_test.cpp)
  add_test(NAME resource_detile COMMAND resource_detile_test)
  add_executable(descriptor_table_test test/descriptor_table_test.cpp)
  add_test(NAME descriptor_table COMMAND descriptor_table_test)
  add_executable(shader_library_test test/shader_library_test.cpp)
  add_test(NAME shader_library COMMAND shader_library_test)
  add_executable(spirv_f16_halves_test test/spirv_f16_halves_test.cpp)
  add_test(NAME spirv_f16_halves COMMAND spirv_f16_halves_test)
  # Optional: skips unless LEONIDA_SHADERS names a translated work folder (<work>/shaders/spirv).
  add_executable(shader_library_corpus_test test/shader_library_corpus_test.cpp)
  add_test(NAME shader_library_corpus COMMAND shader_library_corpus_test)
  # Reads all 26,503 containers and modules: about 9 minutes from a hard disk.
  set_tests_properties(shader_library_corpus PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 3600)
  if(TARGET gta6rt)
    # Optional: skips unless LEONIDA_GAME_DIR and LEONIDA_MOVIE are set.
    add_executable(bink_decode_test
      test/bink_decode_test.cpp
      src/pe_loader.cpp
      shims/registry.cpp
      shims/xmem.cpp
    )
    target_compile_definitions(bink_decode_test PRIVATE
      WIN32_LEAN_AND_MEAN NOMINMAX UNICODE _UNICODE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(bink_decode_test PRIVATE /W4 /permissive- /EHsc /utf-8)
    target_link_options(bink_decode_test PRIVATE /OPT:NOREF)
    target_link_libraries(bink_decode_test PRIVATE psapi)
    add_dependencies(bink_decode_test gta6rt_tls_host)
    add_test(NAME bink_decode COMMAND bink_decode_test)
    set_tests_properties(bink_decode PROPERTIES SKIP_RETURN_CODE 77)

    # Optional: skips unless LEONIDA_GAME_DIR and LEONIDA_BINK_INDEX are set. Measures the HBINK/BINKSUMMARY
    # offsets and the decode return values over every stream of movies.rpf (docs/movies.md section 5).
    add_executable(bink_abi_test
      test/bink_abi_test.cpp
      src/pe_loader.cpp
      shims/registry.cpp
      shims/xmem.cpp
    )
    target_compile_definitions(bink_abi_test PRIVATE
      WIN32_LEAN_AND_MEAN NOMINMAX UNICODE _UNICODE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(bink_abi_test PRIVATE /W4 /permissive- /EHsc /utf-8)
    target_link_options(bink_abi_test PRIVATE /OPT:NOREF)
    target_link_libraries(bink_abi_test PRIVATE psapi)
    add_dependencies(bink_abi_test gta6rt_tls_host)
    add_test(NAME bink_abi COMMAND bink_abi_test)
    set_tests_properties(bink_abi PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 3600)

    # Optional local viewer: plays a .bk2 in a window through the package's Bink2Scarlett.dll (test/bink_play.cpp).
    # The CTest entry is the headless smoke run; it skips unless LEONIDA_GAME_DIR and LEONIDA_MOVIE are set.
    add_executable(bink_play
      test/bink_play.cpp
      src/pe_loader.cpp
      shims/registry.cpp
      shims/xmem.cpp
    )
    target_compile_definitions(bink_play PRIVATE
      WIN32_LEAN_AND_MEAN NOMINMAX UNICODE _UNICODE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(bink_play PRIVATE /W4 /permissive- /EHsc /utf-8)
    target_link_options(bink_play PRIVATE /OPT:NOREF)
    target_link_libraries(bink_play PRIVATE psapi user32 gdi32)
    add_dependencies(bink_play gta6rt_tls_host)
    add_test(NAME bink_play_headless COMMAND bink_play --headless --frames 30)
    set_tests_properties(bink_play_headless PROPERTIES SKIP_RETURN_CODE 77)

    # Optional: skips unless LEONIDA_GAME_DIR is set. Drives the package's FSR DLL on the Xbox-ABI device proxy.
    if(D3D12_LIBRARY AND DXGI_LIBRARY)
      add_executable(fsr_dispatch_test
        test/fsr_dispatch_test.cpp
        src/pe_loader.cpp
        shims/registry.cpp
        shims/xmem.cpp
        shims/d3d12_xs.cpp
        shims/d3d12_xs_device.cpp
      )
      target_compile_definitions(fsr_dispatch_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX _CRT_SECURE_NO_WARNINGS)
      target_compile_options(fsr_dispatch_test PRIVATE /W4 /permissive- /EHsc /utf-8)
      target_link_options(fsr_dispatch_test PRIVATE /OPT:NOREF)
      target_link_libraries(fsr_dispatch_test PRIVATE gta6rt_d3d12_xs_rootsig ${D3D12_LIBRARY} ${DXGI_LIBRARY} psapi)
      add_dependencies(fsr_dispatch_test gta6rt_tls_host)
      add_test(NAME fsr_dispatch COMMAND fsr_dispatch_test)
      set_tests_properties(fsr_dispatch PROPERTIES SKIP_RETURN_CODE 77)

      # Optional: skips unless LEONIDA_GAME_DIR is set. Constructs the package's XboxBvhBuilder on the device proxy.
      add_executable(bvh_builder_test
        test/bvh_builder_test.cpp
        src/pe_loader.cpp
        shims/registry.cpp
        shims/xmem.cpp
        shims/d3d12_xs.cpp
        shims/d3d12_xs_device.cpp
      )
      target_compile_definitions(bvh_builder_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX _CRT_SECURE_NO_WARNINGS)
      target_compile_options(bvh_builder_test PRIVATE /W4 /permissive- /EHsc /utf-8)
      target_link_options(bvh_builder_test PRIVATE /OPT:NOREF)
      target_link_libraries(bvh_builder_test PRIVATE gta6rt_d3d12_xs_rootsig ${D3D12_LIBRARY} ${DXGI_LIBRARY} psapi)
      add_dependencies(bvh_builder_test gta6rt_tls_host)
      add_test(NAME bvh_builder COMMAND bvh_builder_test)
      set_tests_properties(bvh_builder PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    # Optional: skips unless LEONIDA_GAME_DIR and LEONIDA_MOVIE_AUDIO are set.
    add_executable(bink_audio_test
      test/bink_audio_test.cpp
      src/pe_loader.cpp
      shims/registry.cpp
      shims/xmem.cpp
    )
    target_compile_definitions(bink_audio_test PRIVATE
      WIN32_LEAN_AND_MEAN NOMINMAX UNICODE _UNICODE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(bink_audio_test PRIVATE /W4 /permissive- /EHsc /utf-8)
    target_link_options(bink_audio_test PRIVATE /OPT:NOREF)
    target_link_libraries(bink_audio_test PRIVATE psapi)
    add_dependencies(bink_audio_test gta6rt_tls_host)
    add_test(NAME bink_audio COMMAND bink_audio_test)
    set_tests_properties(bink_audio PROPERTIES SKIP_RETURN_CODE 77)
  endif()
  if(WIN32 AND TARGET xmem_test)
    add_test(NAME xmem COMMAND xmem_test)
  endif()
  if(WIN32 AND TARGET xg_xs_test)
    add_test(NAME xg_xs COMMAND xg_xs_test)
  endif()
  if(WIN32 AND TARGET process_cmdline_test)
    add_test(NAME process_cmdline COMMAND process_cmdline_test)
  endif()
  if(WIN32 AND TARGET file_system_test)
    add_test(NAME file_system COMMAND file_system_test)
  endif()
  if(WIN32 AND TARGET int3_skip_test)
    add_test(NAME int3_skip COMMAND int3_skip_test)
  endif()
  if(WIN32 AND TARGET winhttp_offline_test)
    add_test(NAME winhttp_offline COMMAND winhttp_offline_test)
  endif()
  if(WIN32 AND TARGET api_ref_state_test)
    add_test(NAME api_ref_state COMMAND api_ref_state_test)
    set_tests_properties(api_ref_state PROPERTIES TIMEOUT 10)
  endif()
  if(WIN32 AND TARGET api_global_refs_test)
    add_test(NAME api_global_refs COMMAND api_global_refs_test)
    set_tests_properties(api_global_refs PROPERTIES TIMEOUT 10)
  endif()
  # Internal XTaskQueue context pinning and rundown regression tests.
  if(WIN32 AND TARGET gta6rt_xtaskqueue)
    foreach(_context_test context_pinning context_rundown context_reentrant context_concurrent_detach context_attach_detach_stress)
      add_executable(${_context_test}_test
        test/${_context_test}_test.cpp
      )
      target_include_directories(${_context_test}_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/shims/xtaskqueue
        ${XTASKQUEUE_DIR}/Source/Task
        ${XTASKQUEUE_DIR}/Source/Common
      )
      target_compile_definitions(${_context_test}_test PRIVATE
        WIN32_LEAN_AND_MEAN NOMINMAX
      )
      target_compile_options(${_context_test}_test PRIVATE
        /W4 /permissive- /utf-8 /EHsc
      )
      target_link_libraries(${_context_test}_test PRIVATE
        gta6rt_xtaskqueue
      )
      add_test(NAME ${_context_test}
        COMMAND ${_context_test}_test
      )
      set_tests_properties(${_context_test} PROPERTIES TIMEOUT 15)
    endforeach()

    add_executable(wait_registration_test test/wait_registration_test.cpp)
    target_include_directories(wait_registration_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/shims/xtaskqueue ${XTASKQUEUE_DIR}/Source/Task ${XTASKQUEUE_DIR}/Source/Common)
    target_link_libraries(wait_registration_test PRIVATE gta6rt_xtaskqueue)
    foreach(_case unregister undispatched terminate parked self presignaled monitor close composite snapshot
                  cross concurrent held_snapshot admission cancel_unregister reentrant_admission
                  reentrant_vs_external_unregister multi_dispatcher_terminate multi_dispatcher_unregister_replace)
      add_test(NAME wait_registration_${_case} COMMAND wait_registration_test ${_case})
      set_tests_properties(wait_registration_${_case} PROPERTIES TIMEOUT 15)
    endforeach()
  endif()
  if(WIN32 AND TARGET xgameruntime_test)
    add_test(NAME xgameruntime COMMAND xgameruntime_test)
    add_test(NAME xgameruntime_user COMMAND xgameruntime_user_test)
    add_test(NAME xgameruntime_gamesave COMMAND xgameruntime_gamesave_test)
    add_test(NAME xgameruntime_store COMMAND xgameruntime_store_test)
    add_test(NAME xgameruntime_offline COMMAND xgameruntime_offline_test)
    add_test(NAME xgameruntime_offline_hdr COMMAND xgameruntime_offline_test)
    set_tests_properties(xgameruntime_offline_hdr PROPERTIES ENVIRONMENT "LEONIDA_HDR=1")
    add_test(NAME xgameruntime_user_gamertag COMMAND xgameruntime_user_test --expect-gamertag Tester42)
    set_tests_properties(xgameruntime_user_gamertag PROPERTIES ENVIRONMENT "LEONIDA_GAMERTAG=Tester42")
    # Longer than getenv_s's old 64-byte buffer: still the "Player" fallback (and its message), not a silent one.
    add_test(NAME xgameruntime_user_gamertag_long COMMAND xgameruntime_user_test --expect-gamertag Player)
    set_tests_properties(xgameruntime_user_gamertag_long PROPERTIES ENVIRONMENT "LEONIDA_GAMERTAG=GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG")
  endif()
  # The active-input tracker the GameInput shim feeds (src/active_input.h), on synthetic samples: no SDL, no
  # controller, no GPU, so it builds and runs everywhere, with or without SDL3.
  add_executable(active_input_test test/active_input_test.cpp)
  if(MSVC)
    target_compile_options(active_input_test PRIVATE /W4 /permissive- /utf-8)
  endif()
  add_test(NAME active_input COMMAND active_input_test)
  if(WIN32 AND TARGET gameinput_test)
    add_test(NAME gameinput COMMAND gameinput_test)
  elseif(TARGET gta6rt)
    # Built without SDL3: listed as "Not Run (Disabled)" so it is visible that the shim was not tested.
    add_test(NAME gameinput COMMAND ${CMAKE_COMMAND} -E echo "built without SDL3")
    set_tests_properties(gameinput PROPERTIES DISABLED TRUE)
  endif()
  if(WIN32 AND TARGET gta6rt AND TARGET tiny_game)
    if(TARGET tiny_argv_default_a AND NOT CMAKE_CROSSCOMPILING)
      find_package(Python3 COMPONENTS Interpreter QUIET)
      if(Python3_FOUND)
        add_test(NAME run_argv_modes COMMAND "${Python3_EXECUTABLE}" -I
          "${CMAKE_CURRENT_SOURCE_DIR}/test/check_argv_modes.py"
          --loader "$<TARGET_FILE:gta6rt>" --fixtures "$<TARGET_FILE_DIR:tiny_argv_default_a>"
          --work-dir "${CMAKE_CURRENT_BINARY_DIR}")
        set_tests_properties(run_argv_modes PROPERTIES TIMEOUT 90)
      endif()
    endif()
    add_test(NAME run_startup_cmdline COMMAND ${CMAKE_COMMAND}
      "-DLOADER=$<TARGET_FILE:gta6rt>" "-DGUEST=$<TARGET_FILE:tiny_startup_cmdline>"
      "-DRUNNER=${CMAKE_CROSSCOMPILING_EMULATOR}" -P "${CMAKE_CURRENT_SOURCE_DIR}/test/check_startup_cmdline.cmake")
    add_test(NAME run_tiny_async COMMAND ${CMAKE_COMMAND}
      "-DLOADER=$<TARGET_FILE:gta6rt>" "-DGUEST=$<TARGET_FILE:tiny_async>"
      "-DRUNNER=${CMAKE_CROSSCOMPILING_EMULATOR}" -P "${CMAKE_CURRENT_SOURCE_DIR}/test/check_loaded_async.cmake")
    add_test(NAME run_tiny_gdk_load COMMAND ${CMAKE_COMMAND}
      "-DLOADER=$<TARGET_FILE:gta6rt>" "-DGUEST=$<TARGET_FILE:tiny_gdk_load>"
      "-DRUNNER=${CMAKE_CROSSCOMPILING_EMULATOR}" -P "${CMAKE_CURRENT_SOURCE_DIR}/test/check_loaded_gdk.cmake")
    add_test(NAME run_tiny_bundled_load COMMAND ${CMAKE_COMMAND}
      "-DLOADER=$<TARGET_FILE:gta6rt>" "-DGUEST=$<TARGET_FILE:tiny_bundled_load>"
      "-DRUNNER=${CMAKE_CROSSCOMPILING_EMULATOR}" -P "${CMAKE_CURRENT_SOURCE_DIR}/test/check_loaded_bundled.cmake")
    # atexit handlers, static destructors and the ways out, compared with the same program run natively.
    add_test(NAME run_tiny_atexit COMMAND ${CMAKE_COMMAND}
      "-DLOADER=$<TARGET_FILE:gta6rt>" "-DGUEST=$<TARGET_FILE:tiny_atexit>"
      "-DRUNNER=${CMAKE_CROSSCOMPILING_EMULATOR}" -P "${CMAKE_CURRENT_SOURCE_DIR}/test/check_atexit.cmake")
    add_test(NAME run_tiny_game COMMAND gta6rt --run $<TARGET_FILE:tiny_game> -- hello 42)
    add_test(NAME run_tiny_game_console_paths COMMAND ${CMAKE_COMMAND}
      "-DLOADER=$<TARGET_FILE:gta6rt>" "-DGUEST=$<TARGET_FILE:tiny_game>"
      "-DRUNNER=${CMAKE_CROSSCOMPILING_EMULATOR}" -P "${CMAKE_CURRENT_SOURCE_DIR}/test/check_console_paths.cmake")
    add_test(NAME run_tiny_game_exit_code COMMAND tiny_game_exit_code_test
             "$<TARGET_FILE:gta6rt>" "$<TARGET_FILE:tiny_game>")
    add_test(NAME run_tiny_game_abort_exit_codes COMMAND tiny_game_abort_test
             "$<TARGET_FILE:gta6rt>" "$<TARGET_FILE:tiny_game>")
    # Without relocations it can only run at 0x140000000 (base_reservation.h), so passing means it got there; the
    # game's exact exit code comes back through the relaunch.
    add_test(NAME run_tiny_game_fixed COMMAND gta6rt --run $<TARGET_FILE:tiny_game_fixed> -- hello 42)
    add_test(NAME run_tiny_game_fixed_exit_code COMMAND tiny_game_exit_code_test
             "$<TARGET_FILE:gta6rt>" "$<TARGET_FILE:tiny_game_fixed>")
    # Two modes at once are refused before anything is mapped (#886). The stand-in game's own arguments are passed,
    # so a gta6rt that ignored the conflict would run it, print "running" and exit 0.
    foreach(modes "dry-run;run" "dry-run;init" "init;run")
      string(REPLACE ";" "_" name "${modes}")
      list(TRANSFORM modes PREPEND "--")
      add_test(NAME gta6rt_rejects_${name} COMMAND gta6rt ${modes} $<TARGET_FILE:tiny_game> -- hello 42)
      set_tests_properties(gta6rt_rejects_${name} PROPERTIES
        PASS_REGULAR_EXPRESSION "usage: gta6rt"
        FAIL_REGULAR_EXPRESSION "running |module\\(s\\) mapped")
    endforeach()
    # The opt-in checks, one test each: thread_local on the game's threads, a C++ exception caught inside the image.
    add_test(NAME run_tiny_game_tls COMMAND gta6rt --run $<TARGET_FILE:tiny_game> -- hello 42 tls)
    add_test(NAME run_tiny_game_eh COMMAND gta6rt --run $<TARGET_FILE:tiny_game> -- hello 42 eh)
    # BUILD-1457: the stack a game thread gets. gta6rt_stack reads SizeOfStackReserve from the built loader (nothing
    # is run); run_tiny_game_stack asks Windows, on the entry thread and on a thread created with dwStackSize 0.
    add_test(NAME gta6rt_stack COMMAND "${CMAKE_COMMAND}"
             "-DTOOL=$<TARGET_FILE:gta6rt>" "-DWANT=${GTA6RT_STACK}"
             -P "${CMAKE_CURRENT_SOURCE_DIR}/test/check_stack_reserve.cmake")
    add_test(NAME run_tiny_game_stack COMMAND gta6rt --run $<TARGET_FILE:tiny_game> -- hello 42 stack)
    # GAME: is the game folder, absolute, when gta6rt is given a bare image name or `--game-dir .`.
    add_test(NAME run_tiny_game_dir COMMAND ${CMAKE_COMMAND}
      "-DLOADER=$<TARGET_FILE:gta6rt>" "-DGUEST=$<TARGET_FILE:tiny_game>"
      "-DRUNNER=${CMAKE_CROSSCOMPILING_EMULATOR}" -P "${CMAKE_CURRENT_SOURCE_DIR}/test/check_game_dir.cmake")
  endif()
  if(WIN32 AND TARGET root_signature_test)
    add_test(NAME root_signature COMMAND root_signature_test)
  endif()
  if(WIN32 AND TARGET root_signature_oracle_test)
    add_test(NAME root_signature_oracle COMMAND root_signature_oracle_test)
  endif()
endif()

if(TARGET gta6rt_d3d12_xs_rootsig)
option(LEONIDA_D3D12_SHIM_TESTS "Build the standalone D3D12 root-signature shim test" OFF)
if(LEONIDA_D3D12_SHIM_TESTS)
  enable_testing()
  add_executable(d3d12_xs_root_signature_test
    test/d3d12_xs_root_signature_test.cpp
    shims/d3d12_xs.cpp
    shims/registry.cpp
    shims/d3d12_xs_device.cpp
  )
  target_include_directories(d3d12_xs_root_signature_test PRIVATE shims)
  target_compile_definitions(d3d12_xs_root_signature_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
  target_compile_options(d3d12_xs_root_signature_test PRIVATE /W4 /permissive- /EHsc /utf-8)
  target_link_options(d3d12_xs_root_signature_test PRIVATE /OPT:NOREF)
  target_link_libraries(d3d12_xs_root_signature_test PRIVATE gta6rt_d3d12_xs_rootsig d3d12 dxgi)
  add_test(NAME d3d12_xs_root_signature COMMAND d3d12_xs_root_signature_test)
  set_tests_properties(d3d12_xs_root_signature PROPERTIES SKIP_RETURN_CODE 77)  # no WARP adapter (Wine)
endif()
endif()

# CPU-only: which SPIR-V capabilities make the standalone GPU tools ask for shaderInt64 (TEST-468). It includes no
# Vulkan header, so it builds and runs without them, unlike the tools below that use it.
add_executable(spirv_capability_test test/spirv_capability_test.cpp)
if(BUILD_TESTING)
  add_test(NAME spirv_capability COMMAND spirv_capability_test)
endif()

# GPU test and pipeline tools load Vulkan at run time, so only the Vulkan headers are needed at build time.
# Nothing is downloaded unless asked: set VULKAN_SDK, install the Vulkan headers, pass
# -DLEONIDA_VULKAN_HEADERS_FETCH=ON to download the pinned Khronos Vulkan-Headers release once at configure time
# (hash-checked), or pass -DFETCHCONTENT_SOURCE_DIR_VULKANHEADERS=<path> to use a local copy without downloading.
option(LEONIDA_VULKAN_HEADERS_FETCH "Download the pinned Khronos Vulkan-Headers release if no Vulkan headers are found" OFF)
option(LEONIDA_REQUIRE_VULKAN "Fail the configure instead of warning when no Vulkan headers are found" OFF)
find_package(Vulkan QUIET)
if(NOT TARGET Vulkan::Headers)
  find_path(Vulkan_INCLUDE_DIR
    NAMES vulkan/vulkan.h
    HINTS "$ENV{VULKAN_SDK}/Include" "$ENV{VULKAN_SDK}/include"
  )
  if(Vulkan_INCLUDE_DIR AND EXISTS "${Vulkan_INCLUDE_DIR}/vulkan/vulkan.h")
    add_library(Vulkan::Headers INTERFACE IMPORTED)
    set_target_properties(Vulkan::Headers PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${Vulkan_INCLUDE_DIR}"
    )
  elseif(LEONIDA_VULKAN_HEADERS_FETCH OR FETCHCONTENT_SOURCE_DIR_VULKANHEADERS)
    include(FetchContent)
    FetchContent_Declare(VulkanHeaders
      # vulkan-sdk-1.4.363.0 (same commit pinned in .github/workflows/runtime.yml)
      URL https://github.com/KhronosGroup/Vulkan-Headers/archive/6802bb4733b63ed5efd3adb308a6c885ef180ea1.tar.gz
      URL_HASH SHA256=9044fbb356160e43201dc2094acb44c8a906722a63a35f48fc0f051579db9bb6
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    set(VULKAN_HEADERS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(VULKAN_HEADERS_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(VULKAN_HEADERS_ENABLE_MODULE OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(VulkanHeaders)
  endif()
endif()
if(TARGET Vulkan::Headers)
  foreach(tool vk_harness vk_ps_sweep vk_texture_test vk_sampler_test vk_compute_test vk_store_test vk_pipeline_check vk_command_replay_test
          vk_bindless_test vk_image_fuzz vk_descriptor_bench depth24_convert_test vk_predicate_capture_test vk_predicated_copy_test
          vk_xdxr_build_test vk_queue_test vk_sandbox)
    add_executable(${tool} test/${tool}.cpp)
    target_link_libraries(${tool} PRIVATE Vulkan::Headers ${CMAKE_DL_LIBS})
    if(WIN32)
      target_compile_definitions(${tool} PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    endif()
  endforeach()
  # OpFmaKHR width gate of vk_shader_fma.h (shaderFmaFloat32 / shaderFmaFloat64): no Vulkan instance, no GPU.
  add_executable(vk_shader_fma_test test/vk_shader_fma_test.cpp)
  target_link_libraries(vk_shader_fma_test PRIVATE Vulkan::Headers)
  add_test(NAME vk_shader_fma COMMAND vk_shader_fma_test)
  if(MSVC)
    # Keep the checker and its worker threads on the toolchain's C++ runtime under Wine.
    set_property(TARGET vk_pipeline_check PROPERTY
      MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  endif()
  # BUILD-1457: a driver compiles on the thread that creates the pipeline, and its compiler can recurse with the
  # shader's nesting. On Windows a thread that asks for no size gets the reserve in the PE header, 1 MB by default:
  # the main thread and every std::thread worker of the checker. AMD's Windows compiler (26.11.2) overflows that
  # on 26 of the corpus's mesh shaders, and 2 MB is still too little for compute shaders nested deeper than 16
  # (docs/pipeline_check.md, "Stack reserve"). Keep the number in step with scripts/build_runtime.ps1.
  set(LEONIDA_PIPELINE_CHECK_STACK 8388608)
  if(WIN32)
    if(MSVC)
      target_link_options(vk_pipeline_check PRIVATE /STACK:${LEONIDA_PIPELINE_CHECK_STACK})
    else()
      target_link_options(vk_pipeline_check PRIVATE -Wl,--stack,${LEONIDA_PIPELINE_CHECK_STACK})
    endif()
  endif()
  if(BUILD_TESTING)
    if(WIN32 AND LEONIDA_BUILD_WINDOWS_RUNTIME)
      # Real Win32 environment and DXGI names, with D3D12 device creation stubbed: no GPU commands or game files.
      # It includes the runtime's D3D12 shim headers, so it needs the same Windows SDK as the runtime.
      add_executable(host_adapter_test test/host_adapter_test.cpp)
      target_link_libraries(host_adapter_test PRIVATE Vulkan::Headers dxgi)
      target_compile_definitions(host_adapter_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
      if(MSVC)
        target_compile_options(host_adapter_test PRIVATE /W4 /permissive- /EHsc /utf-8)
      endif()
      add_test(NAME host_adapter COMMAND host_adapter_test)
    endif()
    add_test(NAME vk_pipeline_check_cli COMMAND "${CMAKE_COMMAND}"
             "-DTOOL=$<TARGET_FILE:vk_pipeline_check>"
             "-DTEST_DIR=${CMAKE_CURRENT_BINARY_DIR}/vk_pipeline_check_cli_input"
             "-DRUNNER=${CMAKE_CROSSCOMPILING_EMULATOR}"
             -P "${CMAKE_CURRENT_SOURCE_DIR}/test/check_vk_pipeline_cli.cmake")
    add_test(NAME vk_texture_cube_alpha COMMAND vk_texture_test --self-test-cube-alpha)
    if(WIN32)
      # Reads SizeOfStackReserve from the built file; the program is not run, so it needs no emulator.
      add_test(NAME vk_pipeline_check_stack COMMAND "${CMAKE_COMMAND}"
               "-DTOOL=$<TARGET_FILE:vk_pipeline_check>"
               "-DWANT=${LEONIDA_PIPELINE_CHECK_STACK}"
               -P "${CMAKE_CURRENT_SOURCE_DIR}/test/check_stack_reserve.cmake")
    endif()
    add_test(NAME depth24_convert COMMAND depth24_convert_test)
    set_tests_properties(depth24_convert PROPERTIES SKIP_RETURN_CODE 77)
    # Self-checking Vulkan tests with no shaders or game data; each exits 77 without a Vulkan device (TEST-514).
    foreach(test vk_command_replay vk_predicated_copy vk_predicate_capture vk_queue)
      add_test(NAME ${test} COMMAND ${test}_test)
      set_tests_properties(${test} PROPERTIES SKIP_RETURN_CODE 77)
    endforeach()
    # The XDXR BVH builder: the GPU writes every case, then tools/xdxr_bvh.py validates and traces them.
    set(LEONIDA_XDXR_OUT "${CMAKE_CURRENT_BINARY_DIR}/xdxr_build_cases")
    file(MAKE_DIRECTORY "${LEONIDA_XDXR_OUT}")
    add_test(NAME xdxr_build COMMAND vk_xdxr_build_test "${LEONIDA_XDXR_OUT}")
    set_tests_properties(xdxr_build PROPERTIES SKIP_RETURN_CODE 77 FIXTURES_SETUP xdxr_build_cases)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_FOUND)
      add_test(NAME xdxr_build_check COMMAND "${Python3_EXECUTABLE}" -I
               "${CMAKE_CURRENT_SOURCE_DIR}/../tools/xdxr_bvh.py" check-build "${LEONIDA_XDXR_OUT}" --rays 1000)
      set_tests_properties(xdxr_build_check PROPERTIES FIXTURES_REQUIRED xdxr_build_cases SKIP_RETURN_CODE 77)
    endif()
  endif()
  # VulkanQueue/VulkanContext query-state synchronization (remove_query_pool vs commit, cross-queue plan/commit)
  # and pending Wait() preservation across transient vkQueueSubmit failures.
  add_executable(vk_queue_query_sync_test test/vk_queue_query_sync_test.cpp)
  target_link_libraries(vk_queue_query_sync_test PRIVATE Vulkan::Headers ${CMAKE_DL_LIBS})
  if(WIN32)
    target_compile_definitions(vk_queue_query_sync_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
  else()
    find_package(Threads REQUIRED)
    target_link_libraries(vk_queue_query_sync_test PRIVATE Threads::Threads)
  endif()
  if(MSVC)
    target_compile_options(vk_queue_query_sync_test PRIVATE /W4 /permissive- /utf-8)
  endif()
  if(BUILD_TESTING)
    add_test(NAME vk_queue_query_sync COMMAND vk_queue_query_sync_test)
    set_tests_properties(vk_queue_query_sync PROPERTIES SKIP_RETURN_CODE 77)
  endif()
  # With LEONIDA_VK_VALIDATION=1: synchronization validation reports a missing barrier (and not a present one), so
  # the other GPU tests' "0 validation errors" was checked by an active layer. 77 without validation or a device.
  add_executable(vk_sync_validation_selftest test/vk_sync_validation_selftest.cpp)
  target_link_libraries(vk_sync_validation_selftest PRIVATE Vulkan::Headers ${CMAKE_DL_LIBS})
  if(WIN32)
    target_compile_definitions(vk_sync_validation_selftest PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
  endif()
  if(MSVC)
    target_compile_options(vk_sync_validation_selftest PRIVATE /W4 /permissive- /utf-8)
  endif()
  if(BUILD_TESTING)
    add_test(NAME vk_sync_validation COMMAND vk_sync_validation_selftest)
    set_tests_properties(vk_sync_validation PROPERTIES SKIP_RETURN_CODE 77)
  endif()
  if(NOT Vulkan_GLSLANG_VALIDATOR_EXECUTABLE)
    find_program(Vulkan_GLSLANG_VALIDATOR_EXECUTABLE glslangValidator HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
  endif()
  if(Vulkan_GLSLANG_VALIDATOR_EXECUTABLE)
    add_executable(vk_overlapping_occlusion_test test/vk_overlapping_occlusion_test.cpp)
    target_link_libraries(vk_overlapping_occlusion_test PRIVATE Vulkan::Headers ${CMAKE_DL_LIBS})
    if(WIN32)
      target_compile_definitions(vk_overlapping_occlusion_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    endif()
    foreach(stage vert frag)
      set(source "${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/overlapping_occlusion.${stage}")
      set(output "${CMAKE_CURRENT_BINARY_DIR}/overlapping_occlusion.${stage}.spv")
      add_custom_command(OUTPUT "${output}"
                         COMMAND "${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE}" -V --target-env vulkan1.0
                                 -o "${output}" "${source}"
                         DEPENDS "${source}" VERBATIM)
      target_sources(vk_overlapping_occlusion_test PRIVATE "${output}")
    endforeach()
    if(BUILD_TESTING)
      add_test(NAME vk_overlapping_occlusion COMMAND vk_overlapping_occlusion_test
               "${CMAKE_CURRENT_BINARY_DIR}/overlapping_occlusion.vert.spv"
               "${CMAKE_CURRENT_BINARY_DIR}/overlapping_occlusion.frag.spv")
      # The same test as a device without pipelineStatisticsQuery runs it (MoltenVK), on whatever device is here.
      add_test(NAME vk_overlapping_occlusion_no_statistics COMMAND vk_overlapping_occlusion_test
               "${CMAKE_CURRENT_BINARY_DIR}/overlapping_occlusion.vert.spv"
               "${CMAKE_CURRENT_BINARY_DIR}/overlapping_occlusion.frag.spv" --no-statistics)
      set_tests_properties(vk_overlapping_occlusion vk_overlapping_occlusion_no_statistics
                           PROPERTIES SKIP_RETURN_CODE 77)
    endif()
  endif()
  # CPU-only: the ranking that makes these harnesses prefer a native ICD over Mesa's Dozen (#915). No GPU needed.
  add_executable(vk_device_pick_test test/vk_device_pick_test.cpp)
  target_link_libraries(vk_device_pick_test PRIVATE Vulkan::Headers)
  if(BUILD_TESTING)
    add_test(NAME vk_device_pick COMMAND vk_device_pick_test)
  endif()
  # Committed BC* texture zero-initialization via vk.zero_buffer or vk.copy_scratch (D3D-220).
  add_executable(vk_bc_zero_init_test test/vk_bc_zero_init_test.cpp)
  target_link_libraries(vk_bc_zero_init_test PRIVATE Vulkan::Headers ${CMAKE_DL_LIBS})
  if(WIN32)
    target_compile_definitions(vk_bc_zero_init_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
  endif()
  if(MSVC)
    target_compile_options(vk_bc_zero_init_test PRIVATE /W4 /permissive- /utf-8)
  endif()
  if(BUILD_TESTING)
    add_test(NAME vk_bc_zero_init COMMAND vk_bc_zero_init_test)
    set_tests_properties(vk_bc_zero_init PROPERTIES SKIP_RETURN_CODE 77)
  endif()
  # FenceWaitDispatcher concurrency and device-lost regressions (re-entrant callbacks, wakeup on device lost, counter error drain).
  add_executable(vk_fence_dispatcher_sync_test test/vk_fence_dispatcher_sync_test.cpp)
  target_link_libraries(vk_fence_dispatcher_sync_test PRIVATE Vulkan::Headers ${CMAKE_DL_LIBS})
  if(WIN32)
    target_compile_definitions(vk_fence_dispatcher_sync_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
  else()
    find_package(Threads REQUIRED)
    target_link_libraries(vk_fence_dispatcher_sync_test PRIVATE Threads::Threads)
  endif()
  if(MSVC)
    target_compile_options(vk_fence_dispatcher_sync_test PRIVATE /W4 /permissive- /utf-8)
  endif()
  if(BUILD_TESTING)
    add_test(NAME vk_fence_dispatcher_sync COMMAND vk_fence_dispatcher_sync_test)
    set_tests_properties(vk_fence_dispatcher_sync PROPERTIES SKIP_RETURN_CODE 77)
  endif()
  # CPU-only: the bindless set's array sizes from synthetic device limits.
  add_executable(bindless_sizing_test test/bindless_sizing_test.cpp)
  target_link_libraries(bindless_sizing_test PRIVATE Vulkan::Headers)
  if(BUILD_TESTING)
    add_test(NAME bindless_sizing COMMAND bindless_sizing_test)
  endif()
  # CPU-only: Vulkan 1.3 core vs 1.2 KHR/EXT feature probing and proc fallback (runtime/src/vulkan13_features.h).
  add_executable(vulkan13_features_test test/vulkan13_features_test.cpp)
  target_link_libraries(vulkan13_features_test PRIVATE Vulkan::Headers)
  if(MSVC)
    target_compile_options(vulkan13_features_test PRIVATE /W4 /permissive- /utf-8)
  endif()
  if(BUILD_TESTING)
    add_test(NAME vulkan13_features COMMAND vulkan13_features_test)
  endif()
  # CPU-only: ScratchBuffers chunk alignment, retire/collect/forget, and spare cap (runtime/src/vulkan_scratch.h).
  add_executable(vulkan_scratch_test test/vulkan_scratch_test.cpp)
  target_link_libraries(vulkan_scratch_test PRIVATE Vulkan::Headers)
  if(MSVC)
    target_compile_options(vulkan_scratch_test PRIVATE /W4 /permissive- /utf-8)
  endif()
  if(BUILD_TESTING)
    add_test(NAME vulkan_scratch COMMAND vulkan_scratch_test)
  endif()
  if(WIN32 AND BUILD_TESTING)
    # VulkanPresenter tracker lock / failed-submit rollback and capture-buffer WAW synchronization.
    add_executable(vk_presenter_sync_test test/vk_presenter_sync_test.cpp)
    target_link_libraries(vk_presenter_sync_test PRIVATE Vulkan::Headers user32)
    target_compile_definitions(vk_presenter_sync_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    if(MSVC)
      target_compile_options(vk_presenter_sync_test PRIVATE /W4 /permissive- /utf-8)
    endif()
    add_test(NAME vk_presenter_sync COMMAND vk_presenter_sync_test)
    set_tests_properties(vk_presenter_sync PROPERTIES SKIP_RETURN_CODE 77)
  endif()
  if(NOT Vulkan_GLSLANG_VALIDATOR_EXECUTABLE)
    find_program(Vulkan_GLSLANG_VALIDATOR_EXECUTABLE glslangValidator HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
  endif()
  if(Vulkan_GLSLANG_VALIDATOR_EXECUTABLE)
    add_executable(vk_binary_occlusion_test test/vk_binary_occlusion_test.cpp)
    target_link_libraries(vk_binary_occlusion_test PRIVATE Vulkan::Headers ${CMAKE_DL_LIBS})
    if(WIN32)
      target_compile_definitions(vk_binary_occlusion_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    endif()
    foreach(stage vert frag)
      set(source "${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/binary_occlusion.${stage}")
      set(output "${CMAKE_CURRENT_BINARY_DIR}/binary_occlusion.${stage}.spv")
      add_custom_command(OUTPUT "${output}"
                         COMMAND "${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE}" -V --target-env vulkan1.0
                                 -o "${output}" "${source}"
                         DEPENDS "${source}" VERBATIM)
      target_sources(vk_binary_occlusion_test PRIVATE "${output}")
    endforeach()
    if(BUILD_TESTING)
      add_test(NAME vk_binary_occlusion COMMAND vk_binary_occlusion_test
               "${CMAKE_CURRENT_BINARY_DIR}/binary_occlusion.vert.spv"
               "${CMAKE_CURRENT_BINARY_DIR}/binary_occlusion.frag.spv")
      set_tests_properties(vk_binary_occlusion PROPERTIES SKIP_RETURN_CODE 77)
    endif()
  endif()
  # D3D12 COM objects over the Vulkan backend. Their method stubs are generated from the SDK's d3d12.h at build
  # time (tools/gen_d3d12_stubs.py), so nothing derived from the SDK header is checked in.
  find_package(Python3 COMPONENTS Interpreter QUIET)
  # The generator finds d3d12.h through INCLUDE, which a Developer Prompt sets. A cross build (for example
  # runtime/cmake/clang-cl-xwin.cmake) has no such prompt, so it is handed the header CMake finds.
  set(_stubs_header)
  set(_stubs_header_ok TRUE)
  if(TARGET gta6rt AND CMAKE_CROSSCOMPILING)
    find_file(LEONIDA_D3D12_HEADER d3d12.h)
    if(LEONIDA_D3D12_HEADER)
      set(_stubs_header --header "${LEONIDA_D3D12_HEADER}")
    else()
      set(_stubs_header_ok FALSE)
      message(STATUS "d3d12.h not found (set CMAKE_INCLUDE_PATH): the D3D12 COM objects are not built")
    endif()
  endif()
  if(TARGET gta6rt AND Python3_FOUND AND _stubs_header_ok)
    set(LEONIDA_STUBS ${CMAKE_CURRENT_BINARY_DIR}/generated/d3d12_stubs.h)
    add_custom_command(
      OUTPUT ${LEONIDA_STUBS}
      COMMAND "${Python3_EXECUTABLE}" -I ${CMAKE_CURRENT_SOURCE_DIR}/../tools/gen_d3d12_stubs.py --out ${LEONIDA_STUBS}
              ${_stubs_header}
              ID3D12GraphicsCommandList8 ID3D12CommandQueue ID3D12CommandAllocator ID3D12Fence1
              ID3D12Device11 ID3D12DescriptorHeap ID3D12Heap ID3D12Resource ID3D12QueryHeap ID3D12RootSignature
              ID3D12PipelineState ID3D12CommandSignature ID3D12PipelineLibrary1 ID3D12ShaderCacheSession
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../tools/gen_d3d12_stubs.py ${LEONIDA_D3D12_HEADER}
      COMMENT "Generating D3D12 COM stubs"
      VERBATIM)
    add_library(gta6rt_d3d12_com STATIC shims/d3d12_com.cpp shims/d3d12_vulkan_device.cpp shims/d3d12_xs_device.cpp shims/xbox_present.cpp
                shims/dxgi_com.cpp ${LEONIDA_STUBS})   # d3d12_vulkan_device.cpp reads the PC device behind the proxy (#224)
    target_include_directories(gta6rt_d3d12_com PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
    target_link_libraries(gta6rt_d3d12_com PUBLIC Vulkan::Headers gta6rt_root_signature)
    target_compile_definitions(gta6rt_d3d12_com PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    # /bigobj: a Debug build of d3d12_vulkan_device.cpp has more sections than a COFF object holds by default
    # (fatal error C1128), because without optimisation every inline function keeps its own COMDAT.
    target_compile_options(gta6rt_d3d12_com PRIVATE /W4 /permissive- /utf-8 /bigobj)

    add_executable(d3d12_com_test test/d3d12_com_test.cpp)
    target_link_libraries(d3d12_com_test PRIVATE gta6rt_d3d12_com)
    target_compile_definitions(d3d12_com_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_com_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_com COMMAND d3d12_com_test)
    set_tests_properties(d3d12_com PROPERTIES SKIP_RETURN_CODE 77)

    # Multi-threaded concurrent submissions across two D3D12 queues + resource creation/destruction churn.
    add_executable(d3d12_multithread_test test/d3d12_multithread_test.cpp)
    target_link_libraries(d3d12_multithread_test PRIVATE gta6rt_d3d12_com)
    target_compile_definitions(d3d12_multithread_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_multithread_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_multithread COMMAND d3d12_multithread_test)
    # 300 s, not 120: the test's 60 s fence waits need headroom when other GPU tests share the device
    # (two concurrent `ctest -j 6` suites pushed it to 91-120 s; TEST-514 follow-up).
    set_tests_properties(d3d12_multithread PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
    # Several threads make the first CheckFeatureSupport(D3D12_OPTIONS1) call of a new device at the same time.
    add_executable(d3d12_options1_first_use_test test/d3d12_options1_first_use_test.cpp)
    target_link_libraries(d3d12_options1_first_use_test PRIVATE gta6rt_d3d12_com)
    target_compile_definitions(d3d12_options1_first_use_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_options1_first_use_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_options1_first_use COMMAND d3d12_options1_first_use_test)
    set_tests_properties(d3d12_options1_first_use PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
    # A queued Wait must not block a Signal submitted by another D3D12 queue.
    add_executable(d3d12_queue_wait_test test/d3d12_queue_wait_test.cpp)
    target_link_libraries(d3d12_queue_wait_test PRIVATE gta6rt_d3d12_com)
    target_compile_definitions(d3d12_queue_wait_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_queue_wait_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_queue_wait COMMAND d3d12_queue_wait_test)
    add_executable(d3d12_noop_copy_test test/d3d12_noop_copy_test.cpp)
    target_link_libraries(d3d12_noop_copy_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_noop_copy_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_noop_copy_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_noop_copy COMMAND d3d12_noop_copy_test)
    set_tests_properties(d3d12_noop_copy PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 45)
    set_tests_properties(d3d12_queue_wait PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 20)
    add_test(NAME d3d12_queue_wait_timeout COMMAND d3d12_queue_wait_test --timeout-control)
    set_tests_properties(d3d12_queue_wait_timeout PROPERTIES SKIP_RETURN_CODE 77 WILL_FAIL TRUE TIMEOUT 10)
    # ExecuteCommandLists refuses a list whose type isn't the queue's (debug-layer message 836).
    add_executable(d3d12_queue_list_type_test test/d3d12_queue_list_type_test.cpp)
    target_link_libraries(d3d12_queue_list_type_test PRIVATE gta6rt_d3d12_com)
    target_compile_definitions(d3d12_queue_list_type_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_queue_list_type_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_queue_list_type COMMAND d3d12_queue_list_type_test)
    set_tests_properties(d3d12_queue_list_type PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)
    # Wait-before-Signal with a texture and a buffer handed over between the COPY, COMPUTE and DIRECT queues, in every
    # submission order, on the layer and on the host runtime (docs/d3d12_vulkan.md section 4.4).
    add_executable(d3d12_queue_handover_test test/d3d12_queue_handover_test.cpp)
    target_link_libraries(d3d12_queue_handover_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_queue_handover_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_queue_handover_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_queue_handover COMMAND d3d12_queue_handover_test)
    set_tests_properties(d3d12_queue_handover PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 400
                         FAIL_REGULAR_EXPRESSION "Validation Error;VUID-")
    # The same on a caller-owned device: two VkQueues handed to the layer (the D3D12 queues share them), and one
    # (every D3D12 queue shares it, where a Wait ahead of its Signal would never progress on the GPU).
    add_test(NAME d3d12_queue_handover_two_queues COMMAND d3d12_queue_handover_test --two-queues)
    add_test(NAME d3d12_queue_handover_shared_queue COMMAND d3d12_queue_handover_test --shared-queue)
    set_tests_properties(d3d12_queue_handover_two_queues d3d12_queue_handover_shared_queue PROPERTIES
                         SKIP_RETURN_CODE 77 TIMEOUT 400 FAIL_REGULAR_EXPRESSION "Validation Error;VUID-")
    # More D3D12 queues than Vulkan queues: they share VkQueues, keeping fences and Wait-before-Signal (section 4.4).
    # The _one_queue runs make every GPU look like the Intel UHD 770 (one family, one VkQueue): every queue shares it.
    add_executable(d3d12_queue_share_test test/d3d12_queue_share_test.cpp)
    target_link_libraries(d3d12_queue_share_test PRIVATE gta6rt_d3d12_com)
    target_compile_definitions(d3d12_queue_share_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_queue_share_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_queue_share COMMAND d3d12_queue_share_test)
    add_test(NAME d3d12_queue_share_one_queue COMMAND d3d12_queue_share_test)
    set_tests_properties(d3d12_queue_share d3d12_queue_share_one_queue PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300
                         FAIL_REGULAR_EXPRESSION "Validation Error;VUID-")
    add_test(NAME d3d12_queue_wait_one_queue COMMAND d3d12_queue_wait_test)
    set_tests_properties(d3d12_queue_wait_one_queue PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 20)
    set_tests_properties(d3d12_queue_share_one_queue d3d12_queue_wait_one_queue PROPERTIES
                         ENVIRONMENT "LEONIDA_VK_MAX_QUEUES_PER_FAMILY=1;LEONIDA_VK_GRAPHICS_FAMILY_ONLY=1")
    # A misspelt case or count is refused with the usage line (exit code 2) before a device exists, instead of
    # passing with nothing run. No GPU needed.
    set(_share_bad_arguments "case_typo|--case|typo" "iterations_zero|--iterations|0"
        "iterations_word|--iterations|banana" "iterations_too_many|--iterations|99999999999")
    foreach(_bad IN LISTS _share_bad_arguments)
      string(REPLACE "|" ";" _bad "${_bad}")
      list(POP_FRONT _bad _bad_name)
      add_test(NAME d3d12_queue_share_rejects_${_bad_name} COMMAND d3d12_queue_share_test ${_bad})
      set_tests_properties(d3d12_queue_share_rejects_${_bad_name} PROPERTIES TIMEOUT 60
                           PASS_REGULAR_EXPRESSION "\nusage: d3d12_queue_share_test ")
    endforeach()

    # XDXR acceleration structures through BuildRaytracingAccelerationStructure (docs/d3d12_vulkan.md 4.8): the
    # device builds them, tools/xdxr_bvh.py checks them.
    add_executable(d3d12_xdxr_rt_test test/d3d12_xdxr_rt_test.cpp)
    target_link_libraries(d3d12_xdxr_rt_test PRIVATE gta6rt_d3d12_com)
    target_compile_definitions(d3d12_xdxr_rt_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_xdxr_rt_test PRIVATE /W4 /permissive- /utf-8)
    set(LEONIDA_D3D12_XDXR_OUT "${CMAKE_CURRENT_BINARY_DIR}/d3d12_xdxr_cases")
    file(MAKE_DIRECTORY "${LEONIDA_D3D12_XDXR_OUT}")
    add_test(NAME d3d12_xdxr_rt COMMAND d3d12_xdxr_rt_test "${LEONIDA_D3D12_XDXR_OUT}")
    set_tests_properties(d3d12_xdxr_rt PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60 FIXTURES_SETUP d3d12_xdxr_cases)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_FOUND)
      add_test(NAME d3d12_xdxr_rt_check COMMAND "${Python3_EXECUTABLE}" -I
               "${CMAKE_CURRENT_SOURCE_DIR}/../tools/xdxr_bvh.py" check-build "${LEONIDA_D3D12_XDXR_OUT}" --rays 1000)
      set_tests_properties(d3d12_xdxr_rt_check PROPERTIES FIXTURES_REQUIRED d3d12_xdxr_cases SKIP_RETURN_CODE 77)
    endif()

    # GCL8's QueryInterface and its one method need an allocator and a list, not a Vulkan device.
    add_executable(d3d12_gcl8_test test/d3d12_gcl8_test.cpp)
    target_link_libraries(d3d12_gcl8_test PRIVATE gta6rt_d3d12_com)
    target_compile_definitions(d3d12_gcl8_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_gcl8_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_gcl8 COMMAND d3d12_gcl8_test)
    # Python runs on the host; a cross-compiled fixture runs through the emulator (Wine), which the script prepends.
    # The interpreter is named by path, because ctest would put the emulator in front of an executable target.
    set(_trace_runner)
    foreach(_part IN LISTS CMAKE_CROSSCOMPILING_EMULATOR)
      list(APPEND _trace_runner "--runner=${_part}")  # one word each, even one that starts with "-" (xvfb-run -a)
    endforeach()
    add_test(NAME d3d12_trace COMMAND "${Python3_EXECUTABLE}" -I
             "${CMAKE_CURRENT_SOURCE_DIR}/test/check_d3d12_trace.py" "$<TARGET_FILE:d3d12_gcl8_test>" ${_trace_runner})

    # The Vulkan device behind the Xbox-ABI proxy, end to end.
    add_executable(d3d12_vulkan_device_test test/d3d12_vulkan_device_test.cpp)
    target_link_libraries(d3d12_vulkan_device_test PRIVATE gta6rt_d3d12_com dxguid)  # IID_ID3D12MetaCommand (#1440)
    target_compile_definitions(d3d12_vulkan_device_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_vulkan_device_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_vulkan_device COMMAND d3d12_vulkan_device_test)

    add_executable(d3d12_buffer_tiles_test test/d3d12_buffer_tiles_test.cpp)
    target_link_libraries(d3d12_buffer_tiles_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_buffer_tiles_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_buffer_tiles_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_buffer_tiles COMMAND d3d12_buffer_tiles_test)
    set_tests_properties(d3d12_buffer_tiles PROPERTIES SKIP_RETURN_CODE 77
                         FAIL_REGULAR_EXPRESSION "Validation Error;VUID-")
    add_executable(d3d12_texture_tiles_test test/d3d12_texture_tiles_test.cpp)
    target_link_libraries(d3d12_texture_tiles_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_texture_tiles_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_texture_tiles_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_texture_tiles COMMAND d3d12_texture_tiles_test)
    set_tests_properties(d3d12_texture_tiles PROPERTIES SKIP_RETURN_CODE 77
                         FAIL_REGULAR_EXPRESSION "Validation Error;VUID-")

    # DXR acceleration-structure prebuild and command-list recording validation parity test.
    add_executable(d3d12_dxr_validation_test test/d3d12_dxr_validation_test.cpp)
    target_link_libraries(d3d12_dxr_validation_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_dxr_validation_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_dxr_validation_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_dxr_validation COMMAND d3d12_dxr_validation_test)
    add_test(NAME d3d12_dxr_validation_khr COMMAND d3d12_dxr_validation_test)
    set_tests_properties(d3d12_dxr_validation d3d12_dxr_validation_khr PROPERTIES SKIP_RETURN_CODE 77
                         FAIL_REGULAR_EXPRESSION "Validation Error;VUID-")
    set_tests_properties(d3d12_dxr_validation_khr PROPERTIES ENVIRONMENT "LEONIDA_RT_AS=khr")

    # D3D12 permits UpdateTileMappings on DIRECT/COPY/COMPUTE and a sparse buffer must remain
    # accessible across their selected Vulkan families. The optional API-dump check catches an
    # EXCLUSIVE VkBuffer even on drivers that happen to return correct bytes without ownership barriers.
    add_executable(d3d12_sparse_queue_family_test test/d3d12_sparse_queue_family_test.cpp)
    target_link_libraries(d3d12_sparse_queue_family_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_sparse_queue_family_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_sparse_queue_family_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_sparse_queue_family COMMAND d3d12_sparse_queue_family_test)
    set_tests_properties(d3d12_sparse_queue_family PROPERTIES SKIP_RETURN_CODE 77
                         FAIL_REGULAR_EXPRESSION "Validation Error;VUID-")
    if(Python3_FOUND AND NOT CMAKE_CROSSCOMPILING)
      add_test(NAME d3d12_sparse_queue_family_contract COMMAND "${Python3_EXECUTABLE}" -I
               "${CMAKE_CURRENT_SOURCE_DIR}/test/check_sparse_queue_family.py"
               "$<TARGET_FILE:d3d12_sparse_queue_family_test>")
      set_tests_properties(d3d12_sparse_queue_family_contract PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 65)
    endif()
    set_tests_properties(d3d12_vulkan_device PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(d3d12_texture1d_test test/d3d12_texture1d_test.cpp)
    target_link_libraries(d3d12_texture1d_test PRIVATE gta6rt_d3d12_com dxguid)
    target_compile_definitions(d3d12_texture1d_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_texture1d_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_texture1d COMMAND d3d12_texture1d_test)
    set_tests_properties(d3d12_texture1d PROPERTIES SKIP_RETURN_CODE 77)

    # Placed 1D and 3D textures against the host runtime and WARP, and a copy round trip through them.
    add_executable(d3d12_placed_dimensions_test test/d3d12_placed_dimensions_test.cpp)
    target_link_libraries(d3d12_placed_dimensions_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_placed_dimensions_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_placed_dimensions_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_placed_dimensions COMMAND d3d12_placed_dimensions_test)
    set_tests_properties(d3d12_placed_dimensions PROPERTIES SKIP_RETURN_CODE 77
                         FAIL_REGULAR_EXPRESSION "Validation Error;VUID-")

    # ClearUnorderedAccessViewUint/Float on buffers and textures, read back through D3D12.
    add_executable(d3d12_uav_clear_test test/d3d12_uav_clear_test.cpp)
    target_link_libraries(d3d12_uav_clear_test PRIVATE gta6rt_d3d12_com)
    target_compile_definitions(d3d12_uav_clear_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_uav_clear_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_uav_clear COMMAND d3d12_uav_clear_test)
    set_tests_properties(d3d12_uav_clear PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(d3d12_short_uav_clear_test test/d3d12_short_uav_clear_test.cpp)
    target_link_libraries(d3d12_short_uav_clear_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_short_uav_clear_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_short_uav_clear_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_short_uav_clear COMMAND d3d12_short_uav_clear_test)
    set_tests_properties(d3d12_short_uav_clear PROPERTIES SKIP_RETURN_CODE 77
      ENVIRONMENT "LEONIDA_HOST_ADAPTER=warp")

    # Emulation of R9G9B9E5_SHAREDEXP UAV textures as mutable R32_UINT on GPUs without E5B9G9R9 storage:
    # format support, 1D/2D/3D/reserved creation, SRV/UAV T#s, WARP differential UAV clears, cross-backing copies,
    # and set-0 storage write -> E5B9G9R9 sampled HDR round-trip.
    add_executable(d3d12_d24_support_test test/d3d12_d24_support_test.cpp)
    target_link_libraries(d3d12_d24_support_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_d24_support_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_d24_support_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_d24_support COMMAND d3d12_d24_support_test)
    set_tests_properties(d3d12_d24_support PROPERTIES SKIP_RETURN_CODE 77)
    add_executable(d3d12_r9g9b9e5_uav_test test/d3d12_r9g9b9e5_uav_test.cpp)
    target_link_libraries(d3d12_r9g9b9e5_uav_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_r9g9b9e5_uav_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_r9g9b9e5_uav_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_r9g9b9e5_uav COMMAND d3d12_r9g9b9e5_uav_test)
    set_tests_properties(d3d12_r9g9b9e5_uav PROPERTIES SKIP_RETURN_CODE 77)
    if(Vulkan_GLSLANG_VALIDATOR_EXECUTABLE)
      set(LEONIDA_TEST_CS_R9G9B9E5_UAV ${CMAKE_CURRENT_BINARY_DIR}/generated/cs_r9g9b9e5_uav.spv)
      add_custom_command(
        OUTPUT ${LEONIDA_TEST_CS_R9G9B9E5_UAV}
        COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -V --target-env vulkan1.3 -o ${LEONIDA_TEST_CS_R9G9B9E5_UAV}
                ${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/cs_r9g9b9e5_uav.comp
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/cs_r9g9b9e5_uav.comp
        COMMENT "Compiling cs_r9g9b9e5_uav.comp"
        VERBATIM)
      target_sources(d3d12_r9g9b9e5_uav_test PRIVATE ${LEONIDA_TEST_CS_R9G9B9E5_UAV})
      target_compile_definitions(d3d12_r9g9b9e5_uav_test PRIVATE
        LEONIDA_TEST_CS_R9G9B9E5_UAV_SPV="${LEONIDA_TEST_CS_R9G9B9E5_UAV}")
    endif()

    # Debug markers as VK_EXT_debug_utils labels, and private data on every object (D3D-24).
    add_executable(d3d12_debug_markers_test test/d3d12_debug_markers_test.cpp)
    target_link_libraries(d3d12_debug_markers_test PRIVATE gta6rt_d3d12_com)
    target_compile_definitions(d3d12_debug_markers_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_debug_markers_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_debug_markers COMMAND d3d12_debug_markers_test)
    set_tests_properties(d3d12_debug_markers PROPERTIES SKIP_RETURN_CODE 77)

    # Existing-memory heaps and lifetime trackers: the arguments the host refuses (OpenExistingHeapFromAddress,
    # OpenExistingHeapFromFileMapping, CreateLifetimeTracker).
    add_executable(d3d12_existing_heap_test test/d3d12_existing_heap_test.cpp)
    target_link_libraries(d3d12_existing_heap_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_existing_heap_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_existing_heap_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_existing_heap COMMAND d3d12_existing_heap_test)
    set_tests_properties(d3d12_existing_heap PROPERTIES SKIP_RETURN_CODE 77)

    # Fences and timing (D3D-30): multiple-fence waits, clock calibration, adapter LUID, custom heap properties.
    add_executable(d3d12_fence_timing_test test/d3d12_fence_timing_test.cpp)
    target_link_libraries(d3d12_fence_timing_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_fence_timing_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_fence_timing_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_fence_timing COMMAND d3d12_fence_timing_test)
    set_tests_properties(d3d12_fence_timing PROPERTIES SKIP_RETURN_CODE 77)
    # Fence events the host runtime sets or refuses (released fences, no fences), compared with the host devices.
    add_executable(d3d12_fence_events_test test/d3d12_fence_events_test.cpp)
    target_link_libraries(d3d12_fence_events_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_fence_events_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_fence_events_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_fence_events COMMAND d3d12_fence_events_test)
    set_tests_properties(d3d12_fence_events PROPERTIES SKIP_RETURN_CODE 77)
    add_executable(d3d12_fence_device_lost_counter_test test/d3d12_fence_device_lost_counter_test.cpp)
    target_link_libraries(d3d12_fence_device_lost_counter_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_fence_device_lost_counter_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_fence_device_lost_counter_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_fence_device_lost_counter COMMAND d3d12_fence_device_lost_counter_test)
    set_tests_properties(d3d12_fence_device_lost_counter PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 20)
    # Xbox-ABI queue and fence proxies behind the device proxy, on the host runtime (WARP, a hardware adapter) and
    # on the Vulkan device: forwarding, refusal past the public chain, fence unwrapping. 77: no device at all.
    add_executable(queue_proxy_test test/queue_proxy_test.cpp)
    target_link_libraries(queue_proxy_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(queue_proxy_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(queue_proxy_test PRIVATE /W4 /permissive- /utf-8 /EHsc)
    add_test(NAME queue_proxy COMMAND queue_proxy_test)
    set_tests_properties(queue_proxy PROPERTIES SKIP_RETURN_CODE 77)
    # Xbox-ABI command allocator proxies (device slot 9) on the Vulkan device, and the unwrapped PC allocator on the
    # host runtime: forwarding, refused tail, guard region, lists and GPU copies on the proxy. 77: no device at all.
    add_executable(allocator_proxy_test test/allocator_proxy_test.cpp)
    target_link_libraries(allocator_proxy_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(allocator_proxy_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(allocator_proxy_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME allocator_proxy COMMAND allocator_proxy_test)
    set_tests_properties(allocator_proxy PROPERTIES SKIP_RETURN_CODE 77)
    # IDXGIDevice1 on the device proxy (the GDKX boot step: QueryInterface, GetAdapter, EnumOutputs), on the host
    # runtime and on the Vulkan device: COM identity with the device, the adapter's LUID. 77: no device at all.
    add_executable(dxgi_device_test test/dxgi_device_test.cpp)
    target_link_libraries(dxgi_device_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(dxgi_device_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(dxgi_device_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME dxgi_device COMMAND dxgi_device_test)
    set_tests_properties(dxgi_device PROPERTIES SKIP_RETURN_CODE 77)
    # The countdown behind those event-backed waits, with no GPU: the full unsigned count, no wrap past zero.
    add_executable(d3d12_fence_waiter_test test/d3d12_fence_waiter_test.cpp)
    target_compile_definitions(d3d12_fence_waiter_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_fence_waiter_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_fence_waiter COMMAND d3d12_fence_waiter_test)
    # The 1/2/3 method variants (D3D-28) against the methods they wrap, on the Vulkan device directly.
    add_executable(d3d12_method_variants_test test/d3d12_method_variants_test.cpp)
    target_link_libraries(d3d12_method_variants_test PRIVATE gta6rt_d3d12_com)
    target_compile_definitions(d3d12_method_variants_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_method_variants_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_method_variants COMMAND d3d12_method_variants_test)
    # ID3D12Device9::CreateShaderCacheSession, ID3D12ShaderCacheSession and ShaderCacheControl (D3D-34).
    add_executable(d3d12_shader_cache_test test/d3d12_shader_cache_test.cpp)
    target_link_libraries(d3d12_shader_cache_test PRIVATE gta6rt_d3d12_com)
    target_compile_definitions(d3d12_shader_cache_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_shader_cache_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_shader_cache COMMAND d3d12_shader_cache_test)
    set_tests_properties(d3d12_shader_cache PROPERTIES SKIP_RETURN_CODE 77)
    set_tests_properties(d3d12_method_variants PROPERTIES SKIP_RETURN_CODE 77)
    # ID3D12Object private data (GetPrivateData, SetPrivateData, SetPrivateDataInterface, SetName) compared with
    # the host runtime's answers on the device and four child objects.
    add_executable(d3d12_private_data_test test/d3d12_private_data_test.cpp)
    target_link_libraries(d3d12_private_data_test PRIVATE gta6rt_d3d12_com)
    target_compile_definitions(d3d12_private_data_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_private_data_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_private_data COMMAND d3d12_private_data_test)
    set_tests_properties(d3d12_private_data PROPERTIES SKIP_RETURN_CODE 77)

    # CreateSampler2 (D3D-28) against CreateSampler, and its refusals against the host's PC runtime.
    add_executable(d3d12_sampler2_test test/d3d12_sampler2_test.cpp)
    target_link_libraries(d3d12_sampler2_test PRIVATE gta6rt_d3d12_com)
    target_compile_definitions(d3d12_sampler2_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_sampler2_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_sampler2 COMMAND d3d12_sampler2_test)
    set_tests_properties(d3d12_sampler2 PROPERTIES SKIP_RETURN_CODE 77)

    # Descriptors through the proxy: copies between heaps, what texture views and samplers write into their slots,
    # and (where glslangValidator is found) a compute shader the test runs on the device's bindless set, which
    # finds a T# and an S# through the hash table and samples. The shader is GLSL, compiled here.
    add_executable(d3d12_descriptor_test test/d3d12_descriptor_test.cpp)
    target_link_libraries(d3d12_descriptor_test PRIVATE gta6rt_d3d12_com)
    target_compile_definitions(d3d12_descriptor_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_descriptor_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_descriptor COMMAND d3d12_descriptor_test)
    set_tests_properties(d3d12_descriptor PROPERTIES SKIP_RETURN_CODE 77)
    if(Vulkan_GLSLANG_VALIDATOR_EXECUTABLE)
      function(descriptor_test_shader shader definition)
        set(source ${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/${shader})
        set(binary ${CMAKE_CURRENT_BINARY_DIR}/generated/${definition}.spv)
        add_custom_command(OUTPUT ${binary}
          COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -V --target-env vulkan1.3 ${ARGN} -o ${binary} ${source}
          DEPENDS ${source} COMMENT "Compiling descriptor test shader ${shader}" VERBATIM)
        target_sources(d3d12_descriptor_test PRIVATE ${binary})
        target_compile_definitions(d3d12_descriptor_test PRIVATE ${definition}="${binary}")
      endfunction()
      descriptor_test_shader(cs_sample.comp LEONIDA_TEST_CS_SAMPLE_SPV)
      descriptor_test_shader(cs_sample.comp LEONIDA_TEST_GATHER_ALPHA_SPV -DTEST_GATHER_ALPHA=1)
      descriptor_test_shader(cs_sample.comp LEONIDA_TEST_MS_FLOAT_SPV -DTEST_MS_FLOAT=1)
      descriptor_test_shader(cs_sample.comp LEONIDA_TEST_MS_UINT_SPV -DTEST_MS_UINT=1)
      descriptor_test_shader(cs_sample.comp LEONIDA_TEST_MS_SINT_SPV -DTEST_MS_SINT=1)
      descriptor_test_shader(msaa_seed.vert LEONIDA_TEST_MSAA_VERT_SPV)
      descriptor_test_shader(msaa_seed.frag LEONIDA_TEST_MSAA_FRAG_SPV)
    else()
      message(STATUS "glslangValidator not found: d3d12_descriptor_test skips its shader")
    endif()

    add_executable(d3d12_predicated_copy_test test/d3d12_predicated_copy_test.cpp)
    target_link_libraries(d3d12_predicated_copy_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_predicated_copy_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_predicated_copy_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_predicated_copy COMMAND d3d12_predicated_copy_test)
    set_tests_properties(d3d12_predicated_copy PROPERTIES SKIP_RETURN_CODE 77)
    # A DISPATCH-only ExecuteIndirect signature with a GPU count buffer: native Windows WARP versus Vulkan GPU,
    # including a GPU-written count that changes between two calls in the same command list.
    if(Vulkan_GLSLANG_VALIDATOR_EXECUTABLE)
      set(_indirect_count_spv ${CMAKE_CURRENT_BINARY_DIR}/generated/cs_indirect_dispatch_count.spv)
      add_custom_command(OUTPUT ${_indirect_count_spv}
        COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -V --target-env vulkan1.2
          -o ${_indirect_count_spv} ${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/cs_indirect_dispatch_count.comp
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/cs_indirect_dispatch_count.comp
        COMMENT "Compiling the counted indirect dispatch test shader" VERBATIM)
      add_executable(d3d12_indirect_dispatch_count_test test/d3d12_indirect_dispatch_count_test.cpp
        ${_indirect_count_spv})
      target_link_libraries(d3d12_indirect_dispatch_count_test PRIVATE gta6rt_d3d12_com d3d12 dxgi version)
      target_compile_definitions(d3d12_indirect_dispatch_count_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX
        LEONIDA_TEST_INDIRECT_COUNT_SPV="${_indirect_count_spv}")
      target_compile_options(d3d12_indirect_dispatch_count_test PRIVATE /W4 /permissive- /utf-8)
      add_test(NAME d3d12_indirect_dispatch_count COMMAND d3d12_indirect_dispatch_count_test)
      set_tests_properties(d3d12_indirect_dispatch_count PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    # What Close returns after one recorded call, against WARP; today's differences are listed in the test.
    add_executable(d3d12_close_parity_test test/d3d12_close_parity_test.cpp)
    target_link_libraries(d3d12_close_parity_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_close_parity_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_close_parity_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_close_parity COMMAND d3d12_close_parity_test)
    set_tests_properties(d3d12_close_parity PROPERTIES SKIP_RETURN_CODE 77)

    # ExecuteIndirect recording-time Close() parity against live WARP and hardware D3D12, plus GPU dispatch sanity.
    add_executable(d3d12_execute_indirect_validation_test test/d3d12_execute_indirect_validation_test.cpp)
    target_link_libraries(d3d12_execute_indirect_validation_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_execute_indirect_validation_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_execute_indirect_validation_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_execute_indirect_validation COMMAND d3d12_execute_indirect_validation_test)
    set_tests_properties(d3d12_execute_indirect_validation PROPERTIES
      SKIP_RETURN_CODE 77
      FAIL_REGULAR_EXPRESSION "Validation Error;VUID-")

    # ClearState predication reset (GPU readback) + SetPredication / AtomicCopyBuffer Close() parity against WARP & HW.
    add_executable(d3d12_predication_atomic_copy_test test/d3d12_predication_atomic_copy_test.cpp)
    target_link_libraries(d3d12_predication_atomic_copy_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_predication_atomic_copy_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_predication_atomic_copy_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_predication_atomic_copy COMMAND d3d12_predication_atomic_copy_test)
    set_tests_properties(d3d12_predication_atomic_copy PROPERTIES
      SKIP_RETURN_CODE 77
      FAIL_REGULAR_EXPRESSION "Validation Error;VUID-"
    )

    # Differential regression: run explicitly (requires Vulkan hardware and Windows WARP).
    add_executable(d3d12_clear_test test/d3d12_clear_test.cpp)
    target_link_libraries(d3d12_clear_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_clear_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_clear_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_rgb9e5_rtv COMMAND d3d12_clear_test --rgb9e5)
    set_tests_properties(d3d12_rgb9e5_rtv PROPERTIES SKIP_RETURN_CODE 77)
    add_test(NAME d3d12_clear COMMAND d3d12_clear_test)
    set_tests_properties(d3d12_clear PROPERTIES SKIP_RETURN_CODE 77)

    # Synthetic typed-view readbacks against Windows WARP; run explicitly on a Vulkan GPU.
    add_executable(d3d12_typeless_rtv_test test/d3d12_typeless_rtv_test.cpp)
    target_link_libraries(d3d12_typeless_rtv_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_typeless_rtv_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_typeless_rtv_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_typeless_rtv COMMAND d3d12_typeless_rtv_test)
    # 1D render targets and depth buffers: creation against the host and WARP, clears read back against WARP.
    add_executable(d3d12_texture1d_target_test test/d3d12_texture1d_target_test.cpp)
    target_link_libraries(d3d12_texture1d_target_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_texture1d_target_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_texture1d_target_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_texture1d_target COMMAND d3d12_texture1d_target_test)
    set_tests_properties(d3d12_texture1d_target PROPERTIES SKIP_RETURN_CODE 77)  # no Vulkan GPU, or Wine

    # ArraySize -1 in array RTVs and DSVs (every slice from the first on), against the host on the same adapter
    # and WARP.
    add_executable(d3d12_view_slices_test test/d3d12_view_slices_test.cpp)
    target_link_libraries(d3d12_view_slices_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_view_slices_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_view_slices_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_view_slices COMMAND d3d12_view_slices_test)
    set_tests_properties(d3d12_view_slices PROPERTIES SKIP_RETURN_CODE 77)

    # Typed RTV clears and draws of every TYPELESS color family against the host on the same adapter and WARP
    # (D3D-275). The draws need glslangValidator for their GLSL stand-ins of translated shaders.
    add_executable(d3d12_typeless_families_test test/d3d12_typeless_families_test.cpp)
    target_link_libraries(d3d12_typeless_families_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_typeless_families_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_typeless_families_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_typeless_families COMMAND d3d12_typeless_families_test)
    set_tests_properties(d3d12_typeless_families PROPERTIES SKIP_RETURN_CODE 77)
    if(Vulkan_GLSLANG_VALIDATOR_EXECUTABLE)
      foreach(pair "vs_fullscreen.vert;LEONIDA_TYPELESS_VS_SPV;typeless_vs.spv"
                   "ps_typeless_float.frag;LEONIDA_TYPELESS_PS_FLOAT_SPV;typeless_ps_float.spv"
                   "ps_typeless_uint.frag;LEONIDA_TYPELESS_PS_UINT_SPV;typeless_ps_uint.spv")
        list(GET pair 0 shader)
        list(GET pair 1 definition)
        list(GET pair 2 name)
        set(binary ${CMAKE_CURRENT_BINARY_DIR}/generated/${name})
        add_custom_command(OUTPUT ${binary}
          COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -V --target-env vulkan1.3 -o ${binary}
                  ${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/${shader}
          DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/${shader}
          COMMENT "Compiling the typeless draw test shader ${shader}" VERBATIM)
        target_sources(d3d12_typeless_families_test PRIVATE ${binary})
        target_compile_definitions(d3d12_typeless_families_test PRIVATE ${definition}="${binary}")
      endforeach()
    else()
      message(STATUS "glslangValidator not found: d3d12_typeless_families_test runs its clears only")
    endif()

    # Fixed-function state (blend, depth, stencil) swept against the host D3D12 device on the same adapter. Needs
    # glslangValidator for the GLSL stand-ins of translated shaders.
    if(Vulkan_GLSLANG_VALIDATOR_EXECUTABLE)
      add_executable(d3d12_state_sweep_test test/d3d12_state_sweep_test.cpp)
      target_link_libraries(d3d12_state_sweep_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
      target_compile_definitions(d3d12_state_sweep_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
      target_compile_options(d3d12_state_sweep_test PRIVATE /W4 /permissive- /utf-8)
      foreach(pair "vs_fullscreen.vert;LEONIDA_SWEEP_VS_SPV;sweep_vs.spv"
                   "ps_sweep.frag;LEONIDA_SWEEP_PS_SPV;sweep_ps.spv")
        list(GET pair 0 shader)
        list(GET pair 1 definition)
        list(GET pair 2 name)
        set(binary ${CMAKE_CURRENT_BINARY_DIR}/generated/${name})
        add_custom_command(OUTPUT ${binary}
          COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -V --target-env vulkan1.3 -o ${binary}
                  ${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/${shader}
          DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/${shader}
          COMMENT "Compiling the state sweep test shader ${shader}" VERBATIM)
        target_sources(d3d12_state_sweep_test PRIVATE ${binary})
        target_compile_definitions(d3d12_state_sweep_test PRIVATE ${definition}="${binary}")
      endforeach()
      add_test(NAME d3d12_state_sweep COMMAND d3d12_state_sweep_test)
      set_tests_properties(d3d12_state_sweep PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    # 96-bit R32G32B32 textures (D3D-62): footprints and every copy path against Windows WARP.
    add_executable(d3d12_rgb96_test test/d3d12_rgb96_test.cpp)
    target_link_libraries(d3d12_rgb96_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_rgb96_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_rgb96_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_rgb96 COMMAND d3d12_rgb96_test)
    set_tests_properties(d3d12_rgb96 PROPERTIES SKIP_RETURN_CODE 77)  # no Vulkan GPU or no WARP adapter
    set_tests_properties(d3d12_typeless_rtv PROPERTIES SKIP_RETURN_CODE 77)

    # Depth/stencil ResolveSubresourceRegion against the host runtime and WARP; run explicitly on a Vulkan GPU.
    add_executable(d3d12_depth_resolve_test test/d3d12_depth_resolve_test.cpp)
    target_link_libraries(d3d12_depth_resolve_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_depth_resolve_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_depth_resolve_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_depth_resolve COMMAND d3d12_depth_resolve_test)
    set_tests_properties(d3d12_depth_resolve PROPERTIES SKIP_RETURN_CODE 77)
    # The edge cases (TEST-848) draw with a GLSL stand-in for a translated vertex shader; without glslangValidator
    # only the clear cases run.
    if(Vulkan_GLSLANG_VALIDATOR_EXECUTABLE)
      set(binary ${CMAKE_CURRENT_BINARY_DIR}/generated/depth_edge_vs.spv)
      add_custom_command(OUTPUT ${binary}
        COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -V --target-env vulkan1.3 -o ${binary}
                ${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/vs_depth_edge.vert
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/vs_depth_edge.vert
        COMMENT "Compiling the depth resolve test shader vs_depth_edge.vert" VERBATIM)
      target_sources(d3d12_depth_resolve_test PRIVATE ${binary})
      target_compile_definitions(d3d12_depth_resolve_test PRIVATE LEONIDA_DEPTH_EDGE_VS_SPV="${binary}")
    endif()

    # ID3D12Resource::GetDesc against the host runtime and WARP; run explicitly on a Vulkan GPU.
    add_executable(d3d12_resource_desc_test test/d3d12_resource_desc_test.cpp)
    target_link_libraries(d3d12_resource_desc_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_resource_desc_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_resource_desc_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_resource_desc COMMAND d3d12_resource_desc_test)
    set_tests_properties(d3d12_resource_desc PROPERTIES SKIP_RETURN_CODE 77)

    # Typeless depth-stencil textures without ALLOW_DEPTH_STENCIL, copied from a depth buffer, against WARP.
    add_executable(d3d12_typeless_depth_test test/d3d12_typeless_depth_test.cpp)
    target_link_libraries(d3d12_typeless_depth_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_typeless_depth_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_typeless_depth_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_typeless_depth COMMAND d3d12_typeless_depth_test)
    set_tests_properties(d3d12_typeless_depth PROPERTIES SKIP_RETURN_CODE 77)

    # WriteToSubresource/ReadFromSubresource arguments and bytes, compared with the host runtime.
    add_executable(d3d12_subresource_copy_test test/d3d12_subresource_copy_test.cpp)
    target_link_libraries(d3d12_subresource_copy_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_subresource_copy_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_subresource_copy_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_subresource_copy COMMAND d3d12_subresource_copy_test)
    set_tests_properties(d3d12_subresource_copy PROPERTIES SKIP_RETURN_CODE 77)

    # MakeResident, Evict, SetResidencyPriority and EnqueueMakeResident, compared with the host runtime.
    add_executable(d3d12_residency_test test/d3d12_residency_test.cpp)
    target_link_libraries(d3d12_residency_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_residency_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_residency_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_residency COMMAND d3d12_residency_test)
    set_tests_properties(d3d12_residency PROPERTIES SKIP_RETURN_CODE 77)

    # ID3D12Resource::GetHeapProperties for committed, placed and reserved resources, compared with the host runtime.
    add_executable(d3d12_heap_properties_test test/d3d12_heap_properties_test.cpp)
    target_link_libraries(d3d12_heap_properties_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_heap_properties_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_heap_properties_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_heap_properties COMMAND d3d12_heap_properties_test)
    set_tests_properties(d3d12_heap_properties PROPERTIES SKIP_RETURN_CODE 77)

    # CreatePipelineLibrary and ID3D12PipelineLibrary1 arguments, compared with the host runtime.
    add_executable(d3d12_pipeline_library_host_test test/d3d12_pipeline_library_host_test.cpp)
    target_link_libraries(d3d12_pipeline_library_host_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_pipeline_library_host_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_pipeline_library_host_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_pipeline_library_host COMMAND d3d12_pipeline_library_host_test)
    set_tests_properties(d3d12_pipeline_library_host PROPERTIES SKIP_RETURN_CODE 77)

    # Milestone M0: a window, 60 cleared frames presented through the proxy on Vulkan.
    add_executable(d3d12_present_test test/d3d12_present_test.cpp)
    target_link_libraries(d3d12_present_test PRIVATE gta6rt_d3d12_com user32 gdi32 dxgi)
    target_compile_definitions(d3d12_present_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_present_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_present COMMAND d3d12_present_test)
    set_tests_properties(d3d12_present PROPERTIES SKIP_RETURN_CODE 77)

    # Per-pipeline --f16-interp-fallback linking (SHADER-378): 16-param VS + colliding PS rendered on Vulkan.
    add_executable(d3d12_f16_halves_test test/d3d12_f16_halves_test.cpp shims/d3d12_xs_device.cpp)
    target_link_libraries(d3d12_f16_halves_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_f16_halves_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_f16_halves_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_f16_halves COMMAND d3d12_f16_halves_test)
    set_tests_properties(d3d12_f16_halves PROPERTIES SKIP_RETURN_CODE 77)

    # PresentX and the frame events on the Vulkan device (shims/xbox_present.h), checked on swapchain pixels.
    add_executable(xbox_present_test test/xbox_present_test.cpp)
    target_link_libraries(xbox_present_test PRIVATE gta6rt_d3d12_com user32)
    target_compile_definitions(xbox_present_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(xbox_present_test PRIVATE /W4 /permissive- /utf-8)
    # Needs a Vulkan GPU and a window; exits 77 (skipped) without a Vulkan device.
    add_test(NAME xbox_present COMMAND xbox_present_test)
    set_tests_properties(xbox_present PROPERTIES SKIP_RETURN_CODE 77)

    # TEST-528: a real boot-movie frame (the package's Bink2Scarlett.dll) uploaded through the Xbox-ABI proxy and
    # presented with PresentX, checked against the decoded BGRA. Package-gated like bink_decode_test: skips (77)
    # unless LEONIDA_GAME_DIR and LEONIDA_MOVIE are set; no game data is written into the repo.
    add_executable(d3d12_boot_movie_test
      test/d3d12_boot_movie_test.cpp
      src/pe_loader.cpp
      shims/registry.cpp
      shims/xmem.cpp
    )
    target_link_libraries(d3d12_boot_movie_test PRIVATE gta6rt_d3d12_com user32 psapi)
    target_compile_definitions(d3d12_boot_movie_test PRIVATE
      WIN32_LEAN_AND_MEAN NOMINMAX UNICODE _UNICODE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(d3d12_boot_movie_test PRIVATE /W4 /permissive- /utf-8 /EHsc)
    target_link_options(d3d12_boot_movie_test PRIVATE /OPT:NOREF)
    add_dependencies(d3d12_boot_movie_test gta6rt_tls_host)
    add_test(NAME d3d12_boot_movie COMMAND d3d12_boot_movie_test)
    set_tests_properties(d3d12_boot_movie PROPERTIES SKIP_RETURN_CODE 77)

    # First light through D3D12 (docs/first_light.md): the real model and translated VS/PS that tools/first_light.py
    # wrote to LEONIDA_FIRST_LIGHT_DIR, drawn through the device proxy. Skips (77) unless that variable is set.
    add_executable(d3d12_first_light_test test/d3d12_first_light_test.cpp shims/d3d12_xs_device.cpp)
    target_link_libraries(d3d12_first_light_test PRIVATE gta6rt_d3d12_com)
    target_compile_definitions(d3d12_first_light_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX _CRT_SECURE_NO_WARNINGS)
    target_compile_options(d3d12_first_light_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_first_light COMMAND d3d12_first_light_test)
    set_tests_properties(d3d12_first_light PROPERTIES SKIP_RETURN_CODE 77)

    # M1.5 (D3D-53): a draw whose VS and PS are translator output. test/shaders/vs_m15.asm and ps_m15.asm are assembled
    # with llvm-mc and translated by tools/rdna2spirv.py through test/shaders/build.py, which also writes the FSSM
    # programs their index entries are keyed by. Needs llvm-mc with the AMDGPU target (LLVM_MC in the environment, as
    # for the other shader tests, or llvm-mc on PATH or in C:/msys64/ucrt64/bin) and the Vulkan SDK's spirv-as and
    # spirv-val; without them the test is not built.
    if(NOT LEONIDA_LLVM_MC)
      if(DEFINED ENV{LLVM_MC})
        set(LEONIDA_LLVM_MC "$ENV{LLVM_MC}")
      else()
        find_program(LEONIDA_LLVM_MC llvm-mc HINTS "C:/msys64/ucrt64/bin")
      endif()
    endif()
    find_program(LEONIDA_SPIRV_AS spirv-as HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
    find_program(LEONIDA_SPIRV_VAL spirv-val HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
    if(LEONIDA_LLVM_MC AND LEONIDA_SPIRV_AS AND LEONIDA_SPIRV_VAL)
      set(LEONIDA_M15_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated/m15_shaders)
      get_filename_component(LEONIDA_SPIRV_TOOLS_DIR ${LEONIDA_SPIRV_AS} DIRECTORY)
      file(TO_NATIVE_PATH "${LEONIDA_SPIRV_TOOLS_DIR}" LEONIDA_SPIRV_TOOLS_NATIVE)
      set(LEONIDA_M15_FILES ${LEONIDA_M15_DIR}/spirv/index.json ${LEONIDA_M15_DIR}/fssm/vs_m15.fssm
                            ${LEONIDA_M15_DIR}/fssm/ps_m15.fssm)
      add_custom_command(
        OUTPUT ${LEONIDA_M15_FILES}
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${LEONIDA_M15_DIR}
        COMMAND ${CMAKE_COMMAND} -E env "LLVM_MC=${LEONIDA_LLVM_MC}" "LEONIDA_SPIRV_TOOLS=${LEONIDA_SPIRV_TOOLS_NATIVE}"
                $<TARGET_FILE:Python3::Interpreter> -I ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/build.py
                ${LEONIDA_M15_DIR} vs_m15 ps_m15
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/build.py
                ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/compute_fixture.py
                ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/vs_m15.asm
                ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/ps_m15.asm
                ${CMAKE_CURRENT_SOURCE_DIR}/../tools/rdna2spirv.py
        COMMENT "Translating vs_m15 and ps_m15 for the translated draw test"
        VERBATIM)
      add_executable(d3d12_translated_draw_test test/d3d12_translated_draw_test.cpp shims/d3d12_xs_device.cpp
                     ${LEONIDA_M15_FILES})
      target_link_libraries(d3d12_translated_draw_test PRIVATE gta6rt_d3d12_com)
      target_compile_definitions(d3d12_translated_draw_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX
                                 LEONIDA_TEST_M15_DIR="${LEONIDA_M15_DIR}")
      target_compile_options(d3d12_translated_draw_test PRIVATE /W4 /permissive- /utf-8)
      add_test(NAME d3d12_translated_draw COMMAND d3d12_translated_draw_test)
      set_tests_properties(d3d12_translated_draw PROPERTIES SKIP_RETURN_CODE 77)

      # vk_bindless_test (the bindless set and its table at the Vulkan level, no D3D12) takes the translated vs_ids,
      # ps_tex and ps_imgatomic; it was only run by scripts/run_gpu_tests.py and check.py until D3D-314 needed it.
      set(LEONIDA_BINDLESS_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated/bindless_shaders)
      set(LEONIDA_BINDLESS_FILES ${LEONIDA_BINDLESS_DIR}/spirv/vs_ids.spv ${LEONIDA_BINDLESS_DIR}/spirv/ps_tex.spv
                                 ${LEONIDA_BINDLESS_DIR}/spirv/ps_imgatomic.spv)
      add_custom_command(
        OUTPUT ${LEONIDA_BINDLESS_FILES}
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${LEONIDA_BINDLESS_DIR}
        COMMAND ${CMAKE_COMMAND} -E env "LLVM_MC=${LEONIDA_LLVM_MC}" "LEONIDA_SPIRV_TOOLS=${LEONIDA_SPIRV_TOOLS_NATIVE}"
                $<TARGET_FILE:Python3::Interpreter> -I ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/build.py
                ${LEONIDA_BINDLESS_DIR} vs_ids ps_tex ps_imgatomic
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/build.py
                ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/vs_ids.asm
                ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/ps_tex.asm
                ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/ps_imgatomic.asm
                ${CMAKE_CURRENT_SOURCE_DIR}/../tools/rdna2spirv.py
        COMMENT "Translating vs_ids, ps_tex and ps_imgatomic for vk_bindless_test"
        VERBATIM)
      add_custom_target(vk_bindless_shaders ALL DEPENDS ${LEONIDA_BINDLESS_FILES})
      add_dependencies(vk_bindless_test vk_bindless_shaders)
      add_test(NAME vk_bindless_ps_tex COMMAND vk_bindless_test ${LEONIDA_BINDLESS_DIR}/spirv/vs_ids.spv
               ${LEONIDA_BINDLESS_DIR}/spirv/ps_tex.spv)
      add_test(NAME vk_bindless_ps_imgatomic COMMAND vk_bindless_test ${LEONIDA_BINDLESS_DIR}/spirv/vs_ids.spv
               ${LEONIDA_BINDLESS_DIR}/spirv/ps_imgatomic.spv)
      set_tests_properties(vk_bindless_ps_tex vk_bindless_ps_imgatomic PROPERTIES SKIP_RETURN_CODE 77)

      # M2: a textured quad. vs_m2.asm and ps_m2.asm (the PS samples a T#/S# it loads from two descriptor tables) go
      # through the same build; the test creates the heaps, the SRV, the sampler and the root signature through the
      # D3D12 API only.
      set(LEONIDA_M2_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated/m2_shaders)
      set(LEONIDA_M2_FILES ${LEONIDA_M2_DIR}/spirv/index.json ${LEONIDA_M2_DIR}/fssm/vs_m2.fssm
                           ${LEONIDA_M2_DIR}/fssm/ps_m2.fssm ${LEONIDA_M2_DIR}/fssm/ps_m2_uoff.fssm
                           ${LEONIDA_M2_DIR}/fssm/ps_m2_literal.fssm)
      add_custom_command(
        OUTPUT ${LEONIDA_M2_FILES}
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${LEONIDA_M2_DIR}
        COMMAND ${CMAKE_COMMAND} -E env "LLVM_MC=${LEONIDA_LLVM_MC}" "LEONIDA_SPIRV_TOOLS=${LEONIDA_SPIRV_TOOLS_NATIVE}"
                $<TARGET_FILE:Python3::Interpreter> -I ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/build.py
                ${LEONIDA_M2_DIR} vs_m2 ps_m2 ps_m2_uoff ps_m2_literal
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/build.py
                ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/compute_fixture.py
                ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/vs_m2.asm
                ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/ps_m2.asm
                ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/ps_m2_uoff.asm
                ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/ps_m2_literal.asm
                ${CMAKE_CURRENT_SOURCE_DIR}/../tools/rdna2spirv.py
        COMMENT "Translating vs_m2 and ps_m2 for the textured draw test"
        VERBATIM)
      add_executable(d3d12_m2_textured_test test/d3d12_m2_textured_test.cpp shims/d3d12_xs_device.cpp
                     ${LEONIDA_M2_FILES})
      target_link_libraries(d3d12_m2_textured_test PRIVATE gta6rt_d3d12_com)
      target_compile_definitions(d3d12_m2_textured_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX
                                 LEONIDA_TEST_M2_DIR="${LEONIDA_M2_DIR}")
      target_compile_options(d3d12_m2_textured_test PRIVATE /W4 /permissive- /utf-8)
      add_test(NAME d3d12_m2_textured COMMAND d3d12_m2_textured_test)
      set_tests_properties(d3d12_m2_textured PROPERTIES SKIP_RETURN_CODE 77)

      # M2, formats: BC1 and BC7 textures with mip chains and an R8G8B8A8 2D array through the same descriptor path.
      # vs_m2.asm with ps_m2.asm (2D) and ps_m2_array.asm (2D array, slice 1.0 of the view).
      set(LEONIDA_M2F_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated/m2f_shaders)
      set(LEONIDA_M2F_FILES ${LEONIDA_M2F_DIR}/spirv/index.json ${LEONIDA_M2F_DIR}/fssm/vs_m2.fssm
                            ${LEONIDA_M2F_DIR}/fssm/ps_m2.fssm ${LEONIDA_M2F_DIR}/fssm/ps_m2_array.fssm)
      add_custom_command(
        OUTPUT ${LEONIDA_M2F_FILES}
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${LEONIDA_M2F_DIR}
        COMMAND ${CMAKE_COMMAND} -E env "LLVM_MC=${LEONIDA_LLVM_MC}" "LEONIDA_SPIRV_TOOLS=${LEONIDA_SPIRV_TOOLS_NATIVE}"
                $<TARGET_FILE:Python3::Interpreter> -I ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/build.py
                ${LEONIDA_M2F_DIR} vs_m2 ps_m2 ps_m2_array
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/build.py
                ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/compute_fixture.py
                ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/vs_m2.asm
                ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/ps_m2.asm
                ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/ps_m2_array.asm
                ${CMAKE_CURRENT_SOURCE_DIR}/../tools/rdna2spirv.py
        COMMENT "Translating vs_m2, ps_m2 and ps_m2_array for the M2 formats test"
        VERBATIM)
      add_executable(d3d12_m2_formats_test test/d3d12_m2_formats_test.cpp shims/d3d12_xs_device.cpp
                     ${LEONIDA_M2F_FILES})
      target_link_libraries(d3d12_m2_formats_test PRIVATE gta6rt_d3d12_com)
      target_compile_definitions(d3d12_m2_formats_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX
                                 LEONIDA_TEST_M2F_DIR="${LEONIDA_M2F_DIR}")
      target_compile_options(d3d12_m2_formats_test PRIVATE /W4 /permissive- /utf-8)
      add_test(NAME d3d12_m2_formats COMMAND d3d12_m2_formats_test)
      set_tests_properties(d3d12_m2_formats PROPERTIES SKIP_RETURN_CODE 77)

      # M3 (synthetic part): frames with two render targets and a depth/stencil buffer, an occlusion query, resource
      # barriers and a second pass that samples the first one's target, three frames in flight with a fence each.
      # Pass 1's two-output PS and its single-output occlusion probe go through the same translator build.
      set(LEONIDA_M3_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated/m3_shaders)
      set(LEONIDA_M3_FILES ${LEONIDA_M3_DIR}/spirv/index.json ${LEONIDA_M3_DIR}/fssm/vs_m3.fssm
                           ${LEONIDA_M3_DIR}/fssm/ps_m3.fssm ${LEONIDA_M3_DIR}/fssm/vs_m2.fssm
                           ${LEONIDA_M3_DIR}/fssm/ps_m3_post.fssm ${LEONIDA_M3_DIR}/fssm/ps_m3_query.fssm)
      add_custom_command(
        OUTPUT ${LEONIDA_M3_FILES}
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${LEONIDA_M3_DIR}
        COMMAND ${CMAKE_COMMAND} -E env "LLVM_MC=${LEONIDA_LLVM_MC}" "LEONIDA_SPIRV_TOOLS=${LEONIDA_SPIRV_TOOLS_NATIVE}"
                $<TARGET_FILE:Python3::Interpreter> -I ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/build.py
                ${LEONIDA_M3_DIR} vs_m3 ps_m3 vs_m2 ps_m3_post ps_m3_query
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/build.py
                ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/compute_fixture.py
                ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/vs_m3.asm
                ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/ps_m3.asm
                ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/vs_m2.asm
                ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/ps_m3_post.asm
                ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/ps_m3_query.asm
                ${CMAKE_CURRENT_SOURCE_DIR}/../tools/rdna2spirv.py
        COMMENT "Translating M3 frame and single-output occlusion-probe shaders"
        VERBATIM)
      add_executable(d3d12_m3_frame_test test/d3d12_m3_frame_test.cpp shims/d3d12_xs_device.cpp ${LEONIDA_M3_FILES})
      target_link_libraries(d3d12_m3_frame_test PRIVATE gta6rt_d3d12_com)
      target_compile_definitions(d3d12_m3_frame_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX
                                 LEONIDA_TEST_M3_DIR="${LEONIDA_M3_DIR}")
      target_compile_options(d3d12_m3_frame_test PRIVATE /W4 /permissive- /utf-8)
      add_test(NAME d3d12_m3_frame COMMAND d3d12_m3_frame_test)
      add_test(NAME d3d12_m3_first_pass COMMAND d3d12_m3_frame_test --first-pass-only)
      set_tests_properties(d3d12_m3_frame d3d12_m3_first_pass PROPERTIES SKIP_RETURN_CODE 77)

      # M4 (synthetic part): compute PSOs, a UAV table (structured and raw buffers, an R32_UINT storage image), an indirect
      # dispatch from arguments a first dispatch wrote, and a copy queue that streams the input in and reads the results
      # back across queue fences. cs_m4_fill.asm and cs_m4_indirect.asm go through the same build; build.py publishes
      # their synthetic FSSM programs (the compute header and the words their "fssm-data" comments give), which the
      # test wraps in shader containers.
      set(LEONIDA_M4_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated/m4_shaders)
      set(LEONIDA_M4_FILES ${LEONIDA_M4_DIR}/spirv/index.json ${LEONIDA_M4_DIR}/fssm/cs_m4_fill.fssm
                           ${LEONIDA_M4_DIR}/fssm/cs_m4_indirect.fssm)
      add_custom_command(
        OUTPUT ${LEONIDA_M4_FILES}
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${LEONIDA_M4_DIR}
        # LEONIDA_USER_DATA_SHIFT=4: the fixtures read their user data from s0; a D3D12 compute PSO gets its root
        # arguments from s4 (rt::rootsig::pack_stage, D3D-448), so they are built with every SGPR 4 higher.
        COMMAND ${CMAKE_COMMAND} -E env "LLVM_MC=${LEONIDA_LLVM_MC}" "LEONIDA_SPIRV_TOOLS=${LEONIDA_SPIRV_TOOLS_NATIVE}"
                LEONIDA_USER_DATA_SHIFT=4
                $<TARGET_FILE:Python3::Interpreter> -I ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/build.py
                ${LEONIDA_M4_DIR} cs_m4_fill cs_m4_indirect
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/build.py
                ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/compute_fixture.py
                ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/cs_m4_fill.asm
                ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/cs_m4_indirect.asm
                ${CMAKE_CURRENT_SOURCE_DIR}/../tools/rdna2spirv.py
        COMMENT "Translating cs_m4_fill and cs_m4_indirect for the compute test"
        VERBATIM)
      add_executable(d3d12_m4_compute_test test/d3d12_m4_compute_test.cpp shims/d3d12_xs_device.cpp ${LEONIDA_M4_FILES})
      target_link_libraries(d3d12_m4_compute_test PRIVATE gta6rt_d3d12_com)
      target_compile_definitions(d3d12_m4_compute_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX
                                 LEONIDA_TEST_M4_DIR="${LEONIDA_M4_DIR}")
      target_compile_options(d3d12_m4_compute_test PRIVATE /W4 /permissive- /utf-8)
      add_test(NAME d3d12_m4_compute COMMAND d3d12_m4_compute_test)
      set_tests_properties(d3d12_m4_compute PROPERTIES SKIP_RETURN_CODE 77)

      # Typed buffer UAVs and UAV counters: cs_buf_typed.asm writes typed views through V#s with a FORMAT, and
      # cs_buf_counter.asm appends to and consumes from a structured view whose counter is the second V# of its slot;
      # the test also compares typed-buffer and counter clears with the host runtime on the layer's adapter.
      set(LEONIDA_BUFFER_UAV_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated/buffer_uav_shaders)
      set(LEONIDA_BUFFER_UAV_FILES ${LEONIDA_BUFFER_UAV_DIR}/spirv/index.json
                                   ${LEONIDA_BUFFER_UAV_DIR}/fssm/cs_buf_typed.fssm
                                   ${LEONIDA_BUFFER_UAV_DIR}/fssm/cs_buf_counter.fssm)
      add_custom_command(
        OUTPUT ${LEONIDA_BUFFER_UAV_FILES}
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${LEONIDA_BUFFER_UAV_DIR}
        # LEONIDA_USER_DATA_SHIFT=4: root arguments from s4 in a D3D12 compute PSO (see the M4 shaders above).
        COMMAND ${CMAKE_COMMAND} -E env "LLVM_MC=${LEONIDA_LLVM_MC}" "LEONIDA_SPIRV_TOOLS=${LEONIDA_SPIRV_TOOLS_NATIVE}"
                LEONIDA_USER_DATA_SHIFT=4
                $<TARGET_FILE:Python3::Interpreter> -I ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/build.py
                ${LEONIDA_BUFFER_UAV_DIR} cs_buf_typed cs_buf_counter
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/build.py
                ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/compute_fixture.py
                ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/cs_buf_typed.asm
                ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/cs_buf_counter.asm
                ${CMAKE_CURRENT_SOURCE_DIR}/../tools/rdna2spirv.py
        COMMENT "Translating cs_buf_typed and cs_buf_counter for the buffer UAV test"
        VERBATIM)
      add_executable(d3d12_buffer_uav_test test/d3d12_buffer_uav_test.cpp shims/d3d12_xs_device.cpp ${LEONIDA_BUFFER_UAV_FILES})
      target_link_libraries(d3d12_buffer_uav_test PRIVATE gta6rt_d3d12_com)
      target_compile_definitions(d3d12_buffer_uav_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX
                                 LEONIDA_TEST_BUFFER_UAV_DIR="${LEONIDA_BUFFER_UAV_DIR}")
      target_compile_options(d3d12_buffer_uav_test PRIVATE /W4 /permissive- /utf-8)
      add_test(NAME d3d12_buffer_uav COMMAND d3d12_buffer_uav_test)
      set_tests_properties(d3d12_buffer_uav PROPERTIES SKIP_RETURN_CODE 77)
    else()
      message(STATUS "llvm-mc (set LLVM_MC) or the SDK's spirv-as/spirv-val not found: d3d12_translated_draw_test, d3d12_m2_textured_test, d3d12_m2_formats_test, d3d12_m3_frame_test, d3d12_m4_compute_test and d3d12_buffer_uav_test are not built")
    endif()
    # ID3D12Resource::GetHeapProperties and ID3D12DescriptorHeap::GetDesc (node masks, resolved heap flags) against the
    # host runtime and WARP; run explicitly on a Vulkan GPU.
    add_executable(d3d12_node_mask_test test/d3d12_node_mask_test.cpp)
    target_link_libraries(d3d12_node_mask_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_node_mask_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_node_mask_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_node_mask COMMAND d3d12_node_mask_test)
    add_executable(d3d12_indirect_zero_test test/d3d12_indirect_zero_test.cpp)
    target_link_libraries(d3d12_indirect_zero_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_indirect_zero_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_indirect_zero_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_indirect_zero COMMAND d3d12_indirect_zero_test)
    set_tests_properties(d3d12_indirect_zero PROPERTIES SKIP_RETURN_CODE 77)
    set_tests_properties(d3d12_node_mask PROPERTIES SKIP_RETURN_CODE 77)
    set_tests_properties(d3d12_indirect_zero PROPERTIES SKIP_RETURN_CODE 77)

    # CopyResource / CopyTextureRegion between BC and its uncompressed equivalent (and depth <-> colour) against the
    # host runtime and WARP; run explicitly on a Vulkan GPU.
    add_executable(d3d12_copy_xfmt_test test/d3d12_copy_xfmt_test.cpp)
    target_link_libraries(d3d12_copy_xfmt_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_copy_xfmt_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_copy_xfmt_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_copy_xfmt COMMAND d3d12_copy_xfmt_test)
    set_tests_properties(d3d12_copy_xfmt PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(d3d12_stencil_copy_test test/d3d12_stencil_copy_test.cpp)
    target_link_libraries(d3d12_stencil_copy_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_stencil_copy_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_stencil_copy_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_stencil_copy COMMAND d3d12_stencil_copy_test)
    set_tests_properties(d3d12_stencil_copy PROPERTIES SKIP_RETURN_CODE 77
      ENVIRONMENT "LEONIDA_HOST_ADAPTER=warp")

    # Query heaps and query calls by list type (Close results, CreateQueryHeap node masks, timestamp heap kinds)
    # against the host runtime and WARP; run explicitly on a Vulkan GPU.
    add_executable(d3d12_query_rules_test test/d3d12_query_rules_test.cpp)
    target_link_libraries(d3d12_query_rules_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_query_rules_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_query_rules_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_query_rules COMMAND d3d12_query_rules_test)
    set_tests_properties(d3d12_query_rules PROPERTIES SKIP_RETURN_CODE 77)

    # Vulkan-backed IDXGIFactory4 / IDXGISwapChain3 presentation over a Win32 HWND.
    add_executable(dxgi_swapchain_test test/dxgi_swapchain_test.cpp)
    target_link_libraries(dxgi_swapchain_test PRIVATE gta6rt_d3d12_com)
    target_compile_definitions(dxgi_swapchain_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(dxgi_swapchain_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME dxgi_swapchain COMMAND dxgi_swapchain_test)
    set_tests_properties(dxgi_swapchain PROPERTIES SKIP_RETURN_CODE 77)

    # Synthetic (no GPU): VulkanSwapchain's DXGI BufferCount contract when Vulkan returns more images than the
    # app asked for, or the surface minimum exceeds the request (D3D-389).
    add_executable(dxgi_swapchain_count_test test/dxgi_swapchain_count_test.cpp)
    target_link_libraries(dxgi_swapchain_count_test PRIVATE Vulkan::Headers)
    target_compile_definitions(dxgi_swapchain_count_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(dxgi_swapchain_count_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME dxgi_swapchain_count COMMAND dxgi_swapchain_count_test)

    # ExecuteIndirect with CountBuffer (vkCmdDrawIndirectCount) + optional BLAS/TLAS/RayQueryKHR GPU test.
    add_executable(d3d12_indirect_rt_test test/d3d12_indirect_rt_test.cpp)
    target_link_libraries(d3d12_indirect_rt_test PRIVATE gta6rt_d3d12_com)
    target_compile_definitions(d3d12_indirect_rt_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_indirect_rt_test PRIVATE /W4 /permissive- /utf-8)
    if(Vulkan_GLSLANG_VALIDATOR_EXECUTABLE)
      set(LEONIDA_TEST_VS_TRI ${CMAKE_CURRENT_BINARY_DIR}/generated/vs_tri.spv)
      set(LEONIDA_TEST_FS_SOLID ${CMAKE_CURRENT_BINARY_DIR}/generated/fs_solid.spv)
      set(LEONIDA_TEST_CS_RAYQUERY ${CMAKE_CURRENT_BINARY_DIR}/generated/cs_rayquery.spv)
      add_custom_command(
        OUTPUT ${LEONIDA_TEST_VS_TRI}
        COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -V --target-env vulkan1.3 -o ${LEONIDA_TEST_VS_TRI}
                ${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/vs_tri.vert
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/vs_tri.vert
        COMMENT "Compiling vs_tri.vert"
        VERBATIM)
      add_custom_command(
        OUTPUT ${LEONIDA_TEST_FS_SOLID}
        COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -V --target-env vulkan1.3 -o ${LEONIDA_TEST_FS_SOLID}
                ${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/fs_solid.frag
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/fs_solid.frag
        COMMENT "Compiling fs_solid.frag"
        VERBATIM)
      add_custom_command(
        OUTPUT ${LEONIDA_TEST_CS_RAYQUERY}
        COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -V --target-env vulkan1.3 -o ${LEONIDA_TEST_CS_RAYQUERY}
                ${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/cs_rayquery.comp
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/cs_rayquery.comp
        COMMENT "Compiling cs_rayquery.comp"
        VERBATIM)
      target_sources(d3d12_indirect_rt_test PRIVATE
                     ${LEONIDA_TEST_VS_TRI} ${LEONIDA_TEST_FS_SOLID} ${LEONIDA_TEST_CS_RAYQUERY})
      target_compile_definitions(d3d12_indirect_rt_test PRIVATE
        LEONIDA_TEST_VS_TRI_SPV="${LEONIDA_TEST_VS_TRI}"
        LEONIDA_TEST_FS_SOLID_SPV="${LEONIDA_TEST_FS_SOLID}"
        LEONIDA_TEST_CS_RAYQUERY_SPV="${LEONIDA_TEST_CS_RAYQUERY}")
    endif()
    add_test(NAME d3d12_indirect_rt COMMAND d3d12_indirect_rt_test)
    set_tests_properties(d3d12_indirect_rt PROPERTIES SKIP_RETURN_CODE 77)

    # The compute path: root signature, compute PSO from a translator-style index, root arguments, dispatch. Its
    # shader is GLSL written to the translator's push-constant/address contract, compiled here.
    if(NOT Vulkan_GLSLANG_VALIDATOR_EXECUTABLE)
      find_program(Vulkan_GLSLANG_VALIDATOR_EXECUTABLE glslangValidator HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
    endif()
    if(Vulkan_GLSLANG_VALIDATOR_EXECUTABLE)
      set(LEONIDA_TEST_CS ${CMAKE_CURRENT_BINARY_DIR}/generated/cs_user_data.spv)
      add_custom_command(
        OUTPUT ${LEONIDA_TEST_CS}
        COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -V --target-env vulkan1.3 -o ${LEONIDA_TEST_CS}
                ${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/cs_user_data.comp
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/cs_user_data.comp
        COMMENT "Compiling the compute test shader"
        VERBATIM)
      add_executable(d3d12_compute_test test/d3d12_compute_test.cpp shims/d3d12_xs_device.cpp ${LEONIDA_TEST_CS})
      target_link_libraries(d3d12_compute_test PRIVATE gta6rt_d3d12_com)
      target_compile_definitions(d3d12_compute_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX
                                 LEONIDA_TEST_CS_SPV="${LEONIDA_TEST_CS}")
      target_compile_options(d3d12_compute_test PRIVATE /W4 /permissive- /utf-8)
      add_test(NAME d3d12_compute COMMAND d3d12_compute_test)
      set_tests_properties(d3d12_compute PROPERTIES SKIP_RETURN_CODE 77)
      # Compute PSOs from FSPX programs (docs/fspx.md): an ACCC container at slot 62 and a bare kind-6 program.
      add_executable(d3d12_fspx_compute_test test/d3d12_fspx_compute_test.cpp shims/d3d12_xs_device.cpp ${LEONIDA_TEST_CS})
      target_link_libraries(d3d12_fspx_compute_test PRIVATE gta6rt_d3d12_com)
      target_compile_definitions(d3d12_fspx_compute_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX
                                 LEONIDA_TEST_CS_SPV="${LEONIDA_TEST_CS}")
      target_compile_options(d3d12_fspx_compute_test PRIVATE /W4 /permissive- /utf-8)
      add_test(NAME d3d12_fspx_compute COMMAND d3d12_fspx_compute_test)
      set_tests_properties(d3d12_fspx_compute PROPERTIES SKIP_RETURN_CODE 77)
      # Milestone M1: a VS+PS pipeline and indexed draws, shaders written to the same contract.
      set(LEONIDA_TEST_VS ${CMAKE_CURRENT_BINARY_DIR}/generated/vs_draw.spv)
      set(LEONIDA_TEST_PS ${CMAKE_CURRENT_BINARY_DIR}/generated/ps_draw.spv)
      set(LEONIDA_TEST_PS_UINT ${CMAKE_CURRENT_BINARY_DIR}/generated/ps_draw_uint.spv)
      set(LEONIDA_TEST_VS_FULLSCREEN ${CMAKE_CURRENT_BINARY_DIR}/generated/vs_fullscreen.spv)
      set(LEONIDA_TEST_VS_INT ${CMAKE_CURRENT_BINARY_DIR}/generated/vs_draw_int.spv)
      set(LEONIDA_TEST_GS_DOUBLE ${CMAKE_CURRENT_BINARY_DIR}/generated/gs_double.spv)
      foreach(pair "vs_draw.vert;${LEONIDA_TEST_VS}" "ps_draw.frag;${LEONIDA_TEST_PS}"
                   "ps_draw_uint.frag;${LEONIDA_TEST_PS_UINT}" "vs_fullscreen.vert;${LEONIDA_TEST_VS_FULLSCREEN}" "vs_draw_int.vert;${LEONIDA_TEST_VS_INT}"
                   "gs_double.geom;${LEONIDA_TEST_GS_DOUBLE}")
        list(GET pair 0 source)
        list(GET pair 1 output)
        add_custom_command(
          OUTPUT ${output}
          COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -V --target-env vulkan1.3 -o ${output}
                  ${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/${source}
          DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/${source}
          COMMENT "Compiling the draw test shader ${source}"
          VERBATIM)
      endforeach()
      add_executable(d3d12_draw_test test/d3d12_draw_test.cpp shims/d3d12_xs_device.cpp ${LEONIDA_TEST_VS}
                     ${LEONIDA_TEST_PS} ${LEONIDA_TEST_PS_UINT} ${LEONIDA_TEST_VS_FULLSCREEN} ${LEONIDA_TEST_VS_INT})
      target_link_libraries(d3d12_draw_test PRIVATE gta6rt_d3d12_com)
      target_compile_definitions(d3d12_draw_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX
                                 LEONIDA_TEST_VS_SPV="${LEONIDA_TEST_VS}" LEONIDA_TEST_PS_SPV="${LEONIDA_TEST_PS}"
                                 LEONIDA_TEST_PS_UINT_SPV="${LEONIDA_TEST_PS_UINT}"
                                 LEONIDA_TEST_VS_FULLSCREEN_SPV="${LEONIDA_TEST_VS_FULLSCREEN}"
                                 LEONIDA_TEST_VS_INT_SPV="${LEONIDA_TEST_VS_INT}")
      target_compile_options(d3d12_draw_test PRIVATE /W4 /permissive- /utf-8)
      add_test(NAME d3d12_draw COMMAND d3d12_draw_test)
      set_tests_properties(d3d12_draw PROPERTIES SKIP_RETURN_CODE 77)
      # Graphics & compute PSO descriptor validation parity against live Windows D3D12 (WARP and HW).
      add_executable(d3d12_pso_validation_test test/d3d12_pso_validation_test.cpp ${LEONIDA_TEST_VS}
                     ${LEONIDA_TEST_PS} ${LEONIDA_TEST_PS_UINT} ${LEONIDA_TEST_VS_FULLSCREEN} ${LEONIDA_TEST_CS})
      target_link_libraries(d3d12_pso_validation_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
      target_compile_definitions(d3d12_pso_validation_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX
                                 LEONIDA_TEST_VS_SPV="${LEONIDA_TEST_VS}" LEONIDA_TEST_PS_SPV="${LEONIDA_TEST_PS}"
                                 LEONIDA_TEST_PS_UINT_SPV="${LEONIDA_TEST_PS_UINT}"
                                 LEONIDA_TEST_VS_FULLSCREEN_SPV="${LEONIDA_TEST_VS_FULLSCREEN}"
                                 LEONIDA_TEST_CS_SPV="${LEONIDA_TEST_CS}")
      target_compile_options(d3d12_pso_validation_test PRIVATE /W4 /permissive- /utf-8)
      add_test(NAME d3d12_pso_validation COMMAND d3d12_pso_validation_test)
      set_tests_properties(d3d12_pso_validation PROPERTIES SKIP_RETURN_CODE 77
                           FAIL_REGULAR_EXPRESSION "Validation Error;VUID-")
      # Conservative raster Tier 1, with geometric coverage bounds and host D3D12 references.
      set(LEONIDA_CONSERVATIVE_VS ${CMAKE_CURRENT_BINARY_DIR}/generated/vs_conservative.spv)
      set(LEONIDA_CONSERVATIVE_PS ${CMAKE_CURRENT_BINARY_DIR}/generated/ps_conservative.spv)
      foreach(pair "vs_conservative.vert;${LEONIDA_CONSERVATIVE_VS}" "ps_conservative.frag;${LEONIDA_CONSERVATIVE_PS}")
        list(GET pair 0 source)
        list(GET pair 1 output)
        add_custom_command(OUTPUT ${output}
          COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -V --target-env vulkan1.3 -o ${output}
                  ${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/${source}
          DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/${source} VERBATIM)
      endforeach()
      add_executable(d3d12_conservative_raster_test test/d3d12_conservative_raster_test.cpp
        ${LEONIDA_CONSERVATIVE_VS} ${LEONIDA_CONSERVATIVE_PS})
      target_link_libraries(d3d12_conservative_raster_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
      target_compile_definitions(d3d12_conservative_raster_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX
        LEONIDA_CONSERVATIVE_VS="${LEONIDA_CONSERVATIVE_VS}" LEONIDA_CONSERVATIVE_PS="${LEONIDA_CONSERVATIVE_PS}")
      target_compile_options(d3d12_conservative_raster_test PRIVATE /W4 /permissive- /utf-8)
      add_test(NAME d3d12_conservative_raster COMMAND d3d12_conservative_raster_test)
      set_tests_properties(d3d12_conservative_raster PROPERTIES SKIP_RETURN_CODE 77
        ENVIRONMENT "LEONIDA_HOST_ADAPTER=warp")
      set(LEONIDA_CONSERVATIVE_TIER2_VS ${CMAKE_CURRENT_BINARY_DIR}/generated/vs_conservative_tier2.spv)
      set(LEONIDA_CONSERVATIVE_TIER2_PS ${CMAKE_CURRENT_BINARY_DIR}/generated/ps_conservative_tier2.spv)
      foreach(pair "vs_conservative_tier2.vert;${LEONIDA_CONSERVATIVE_TIER2_VS}" "ps_conservative_tier2.frag;${LEONIDA_CONSERVATIVE_TIER2_PS}")
        list(GET pair 0 source)
        list(GET pair 1 output)
        add_custom_command(OUTPUT ${output}
          COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -V --target-env vulkan1.3 -o ${output}
                  ${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/${source}
          DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/${source} VERBATIM)
      endforeach()
      add_executable(d3d12_conservative_tier2_test test/d3d12_conservative_tier2_test.cpp
        ${LEONIDA_CONSERVATIVE_TIER2_VS} ${LEONIDA_CONSERVATIVE_TIER2_PS})
      target_link_libraries(d3d12_conservative_tier2_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
      target_compile_definitions(d3d12_conservative_tier2_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX
        LEONIDA_CONSERVATIVE_TIER2_VS="${LEONIDA_CONSERVATIVE_TIER2_VS}" LEONIDA_CONSERVATIVE_TIER2_PS="${LEONIDA_CONSERVATIVE_TIER2_PS}")
      target_compile_options(d3d12_conservative_tier2_test PRIVATE /W4 /permissive- /utf-8)
      add_test(NAME d3d12_conservative_tier2 COMMAND d3d12_conservative_tier2_test)
      set_tests_properties(d3d12_conservative_tier2 PROPERTIES SKIP_RETURN_CODE 77 ENVIRONMENT "LEONIDA_HOST_ADAPTER=warp")
      set(LEONIDA_CONSERVATIVE_TIER3_PS ${CMAKE_CURRENT_BINARY_DIR}/generated/ps_conservative_tier3.spv)
      add_custom_command(OUTPUT ${LEONIDA_CONSERVATIVE_TIER3_PS}
        COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -V --target-env vulkan1.3 -o ${LEONIDA_CONSERVATIVE_TIER3_PS}
                ${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/ps_conservative_tier3.frag
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/ps_conservative_tier3.frag VERBATIM)
      add_executable(d3d12_conservative_tier3_test test/d3d12_conservative_tier3_test.cpp
        ${LEONIDA_CONSERVATIVE_VS} ${LEONIDA_CONSERVATIVE_TIER3_PS})
      target_link_libraries(d3d12_conservative_tier3_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
      target_compile_definitions(d3d12_conservative_tier3_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX
        LEONIDA_CONSERVATIVE_VS="${LEONIDA_CONSERVATIVE_VS}"
        LEONIDA_CONSERVATIVE_TIER3_PS="${LEONIDA_CONSERVATIVE_TIER3_PS}")
      target_compile_options(d3d12_conservative_tier3_test PRIVATE /W4 /permissive- /utf-8)
      add_test(NAME inner_coverage_oracle COMMAND d3d12_conservative_tier3_test --oracle-only)
      add_test(NAME d3d12_inner_coverage_guards COMMAND d3d12_conservative_tier3_test --guards-only)
      add_test(NAME d3d12_inner_coverage_native COMMAND d3d12_conservative_tier3_test --native-only)
      add_test(NAME d3d12_conservative_tier3 COMMAND d3d12_conservative_tier3_test)
      set_tests_properties(d3d12_inner_coverage_guards d3d12_inner_coverage_native d3d12_conservative_tier3
        PROPERTIES SKIP_RETURN_CODE 77)
      add_executable(d3d12_msaa_draw_test test/d3d12_msaa_draw_test.cpp shims/d3d12_xs_device.cpp
                     ${LEONIDA_TEST_VS} ${LEONIDA_TEST_PS})
      target_link_libraries(d3d12_msaa_draw_test PRIVATE gta6rt_d3d12_com)
      target_compile_definitions(d3d12_msaa_draw_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX
                                 LEONIDA_TEST_VS_SPV="${LEONIDA_TEST_VS}" LEONIDA_TEST_PS_SPV="${LEONIDA_TEST_PS}")
      target_compile_options(d3d12_msaa_draw_test PRIVATE /W4 /permissive- /utf-8)
      add_test(NAME d3d12_msaa_draw COMMAND d3d12_msaa_draw_test)
      set_tests_properties(d3d12_msaa_draw PROPERTIES SKIP_RETURN_CODE 77)
      # Geometry shaders: a VS+GS+PS pipeline whose GS emits each triangle twice.
      add_executable(d3d12_geometry_test test/d3d12_geometry_test.cpp shims/d3d12_xs_device.cpp
                     ${LEONIDA_TEST_VS} ${LEONIDA_TEST_GS_DOUBLE} ${LEONIDA_TEST_PS})
      target_link_libraries(d3d12_geometry_test PRIVATE gta6rt_d3d12_com)
      target_compile_definitions(d3d12_geometry_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX
                                 LEONIDA_TEST_VS_SPV="${LEONIDA_TEST_VS}" LEONIDA_TEST_GS_SPV="${LEONIDA_TEST_GS_DOUBLE}"
                                 LEONIDA_TEST_PS_SPV="${LEONIDA_TEST_PS}")
      target_compile_options(d3d12_geometry_test PRIVATE /W4 /permissive- /utf-8)
      add_test(NAME d3d12_geometry COMMAND d3d12_geometry_test)
      set_tests_properties(d3d12_geometry PROPERTIES SKIP_RETURN_CODE 77)
      # D3D-309: D24 depth bias and 24-bit depth in draws, with and without a pixel shader and with a geometry shader
      # (src/depth24_fragment.h).
      set(LEONIDA_TEST_VS_DEPTH ${CMAKE_CURRENT_BINARY_DIR}/generated/vs_depth.spv)
      set(LEONIDA_TEST_PS_DEPTH_EMPTY ${CMAKE_CURRENT_BINARY_DIR}/generated/ps_depth_empty.spv)
      set(LEONIDA_TEST_PS_DEPTH_WRITE ${CMAKE_CURRENT_BINARY_DIR}/generated/ps_depth_write.spv)
      set(LEONIDA_TEST_GS_DEPTH ${CMAKE_CURRENT_BINARY_DIR}/generated/gs_depth.spv)
      foreach(pair "vs_depth.vert;${LEONIDA_TEST_VS_DEPTH}" "ps_depth_empty.frag;${LEONIDA_TEST_PS_DEPTH_EMPTY}"
                   "ps_depth_write.frag;${LEONIDA_TEST_PS_DEPTH_WRITE}" "gs_depth.geom;${LEONIDA_TEST_GS_DEPTH}")
        list(GET pair 0 source)
        list(GET pair 1 output)
        add_custom_command(
          OUTPUT ${output}
          COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -V --target-env vulkan1.3 -o ${output}
                  ${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/${source}
          DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/${source}
          COMMENT "Compiling the D24 draw test shader ${source}"
          VERBATIM)
      endforeach()
      add_executable(d3d12_depth24_draw_test test/d3d12_depth24_draw_test.cpp shims/d3d12_xs_device.cpp
                     ${LEONIDA_TEST_VS_DEPTH} ${LEONIDA_TEST_PS_DEPTH_EMPTY} ${LEONIDA_TEST_PS_DEPTH_WRITE}
                     ${LEONIDA_TEST_GS_DEPTH})
      target_link_libraries(d3d12_depth24_draw_test PRIVATE gta6rt_d3d12_com)
      target_compile_definitions(d3d12_depth24_draw_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX
                                 LEONIDA_TEST_VS_DEPTH_SPV="${LEONIDA_TEST_VS_DEPTH}"
                                 LEONIDA_TEST_PS_DEPTH_EMPTY_SPV="${LEONIDA_TEST_PS_DEPTH_EMPTY}"
                                 LEONIDA_TEST_PS_DEPTH_WRITE_SPV="${LEONIDA_TEST_PS_DEPTH_WRITE}"
                                 LEONIDA_TEST_GS_DEPTH_SPV="${LEONIDA_TEST_GS_DEPTH}")
      target_compile_options(d3d12_depth24_draw_test PRIVATE /W4 /permissive- /utf-8)
      add_test(NAME d3d12_depth24_draw COMMAND d3d12_depth24_draw_test)
      set_tests_properties(d3d12_depth24_draw PROPERTIES SKIP_RETURN_CODE 77)
      # Set 0 from the compute bind point: a compute PSO whose shader declares set-0 bindings samples through the
      # device's bindless set, which nothing but the layer binds.
      set(LEONIDA_TEST_CS_SET0_SAMPLE ${CMAKE_CURRENT_BINARY_DIR}/generated/cs_set0_sample.spv)
      add_custom_command(
        OUTPUT ${LEONIDA_TEST_CS_SET0_SAMPLE}
        COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -V --target-env vulkan1.3 -o ${LEONIDA_TEST_CS_SET0_SAMPLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/cs_set0_sample.comp
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/cs_set0_sample.comp
        COMMENT "Compiling the set-0 compute test shader"
        VERBATIM)
      add_executable(d3d12_set0_compute_test test/d3d12_set0_compute_test.cpp shims/d3d12_xs_device.cpp
                     ${LEONIDA_TEST_CS_SET0_SAMPLE})
      target_link_libraries(d3d12_set0_compute_test PRIVATE gta6rt_d3d12_com)
      target_compile_definitions(d3d12_set0_compute_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX
                                 LEONIDA_TEST_CS_SET0_SAMPLE_SPV="${LEONIDA_TEST_CS_SET0_SAMPLE}")
      target_compile_options(d3d12_set0_compute_test PRIVATE /W4 /permissive- /utf-8)
      add_test(NAME d3d12_set0_compute COMMAND d3d12_set0_compute_test)
      set_tests_properties(d3d12_set0_compute PROPERTIES SKIP_RETURN_CODE 77)
      # Bindless table churn on a direct and a compute queue: the table grows and live keys move without a
      # device-idle wait (D3D-314); the same shader, run twice (deferred retirement, and the old wait for timing).
      add_executable(d3d12_bindless_churn_test test/d3d12_bindless_churn_test.cpp shims/d3d12_xs_device.cpp
                     ${LEONIDA_TEST_CS_SET0_SAMPLE})
      target_link_libraries(d3d12_bindless_churn_test PRIVATE gta6rt_d3d12_com)
      target_compile_definitions(d3d12_bindless_churn_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX
                                 LEONIDA_TEST_CS_SET0_SAMPLE_SPV="${LEONIDA_TEST_CS_SET0_SAMPLE}")
      target_compile_options(d3d12_bindless_churn_test PRIVATE /W4 /permissive- /utf-8)
      add_test(NAME d3d12_bindless_churn COMMAND d3d12_bindless_churn_test)
      set_tests_properties(d3d12_bindless_churn PROPERTIES SKIP_RETURN_CODE 77)
      add_test(NAME d3d12_bindless_churn_validated COMMAND d3d12_bindless_churn_test --validated)
      set_tests_properties(d3d12_bindless_churn_validated PROPERTIES SKIP_RETURN_CODE 77)
      add_executable(d3d12_pipeline_library_test test/d3d12_pipeline_library_test.cpp
                     ${LEONIDA_TEST_CS} ${LEONIDA_TEST_VS} ${LEONIDA_TEST_PS})
      target_link_libraries(d3d12_pipeline_library_test PRIVATE gta6rt_d3d12_com)
      target_compile_definitions(d3d12_pipeline_library_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX
                                 LEONIDA_TEST_CS_SPV="${LEONIDA_TEST_CS}"
                                 LEONIDA_TEST_VS_SPV="${LEONIDA_TEST_VS}"
                                 LEONIDA_TEST_PS_SPV="${LEONIDA_TEST_PS}")
      target_compile_options(d3d12_pipeline_library_test PRIVATE /W4 /permissive- /utf-8)
      add_test(NAME d3d12_pipeline_library COMMAND d3d12_pipeline_library_test)
      set_tests_properties(d3d12_pipeline_library PROPERTIES SKIP_RETURN_CODE 77)
      # ExecuteBundle: what a bundle's draws inherit and hand back, against the host runtime and WARP.
      add_executable(d3d12_bundle_test test/d3d12_bundle_test.cpp ${LEONIDA_TEST_VS} ${LEONIDA_TEST_PS})
      target_link_libraries(d3d12_bundle_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
      target_compile_definitions(d3d12_bundle_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX _CRT_SECURE_NO_WARNINGS
                                 LEONIDA_TEST_VS_SPV="${LEONIDA_TEST_VS}" LEONIDA_TEST_PS_SPV="${LEONIDA_TEST_PS}")
      target_compile_options(d3d12_bundle_test PRIVATE /W4 /permissive- /utf-8)
      add_test(NAME d3d12_bundle COMMAND d3d12_bundle_test)
      set_tests_properties(d3d12_bundle PROPERTIES SKIP_RETURN_CODE 77)
      # Calls a bundle may not make, Close of the bundle and of its parent, against the host runtime and WARP.
      add_executable(d3d12_bundle_forbidden_test test/d3d12_bundle_forbidden_test.cpp)
      target_link_libraries(d3d12_bundle_forbidden_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
      target_compile_definitions(d3d12_bundle_forbidden_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
      target_compile_options(d3d12_bundle_forbidden_test PRIVATE /W4 /permissive- /utf-8)
      add_test(NAME d3d12_bundle_forbidden COMMAND d3d12_bundle_forbidden_test)
      set_tests_properties(d3d12_bundle_forbidden PROPERTIES SKIP_RETURN_CODE 77)
      # A pixel shader with a writable storage image (an image atomic) needs fragmentStoresAndAtomics: refused on a
      # device without it, created on the runtime's own device, under the validation layer where it is installed.
      set(LEONIDA_TEST_PS_ATOMIC ${CMAKE_CURRENT_BINARY_DIR}/generated/ps_image_atomic.spv)
      add_custom_command(
        OUTPUT ${LEONIDA_TEST_PS_ATOMIC}
        COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -V --target-env vulkan1.3 -o ${LEONIDA_TEST_PS_ATOMIC}
                ${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/ps_image_atomic.frag
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/ps_image_atomic.frag
        COMMENT "Compiling the stores-and-atomics test shader ps_image_atomic.frag"
        VERBATIM)
      add_executable(d3d12_stores_atomics_test test/d3d12_stores_atomics_test.cpp ${LEONIDA_TEST_VS_FULLSCREEN}
                     ${LEONIDA_TEST_PS} ${LEONIDA_TEST_PS_ATOMIC})
      target_link_libraries(d3d12_stores_atomics_test PRIVATE gta6rt_d3d12_com)
      target_compile_definitions(d3d12_stores_atomics_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX
                                 LEONIDA_TEST_VS_FULLSCREEN_SPV="${LEONIDA_TEST_VS_FULLSCREEN}"
                                 LEONIDA_TEST_PS_SPV="${LEONIDA_TEST_PS}"
                                 LEONIDA_TEST_PS_ATOMIC_SPV="${LEONIDA_TEST_PS_ATOMIC}")
      target_compile_options(d3d12_stores_atomics_test PRIVATE /W4 /permissive- /utf-8)
      add_test(NAME d3d12_stores_atomics COMMAND d3d12_stores_atomics_test)
      set_tests_properties(d3d12_stores_atomics PROPERTIES SKIP_RETURN_CODE 77)

      # D3D-29: mesh-shader PSOs from pipeline state streams and DispatchMesh (skips without VK_EXT_mesh_shader).
      set(LEONIDA_TEST_MS ${CMAKE_CURRENT_BINARY_DIR}/generated/ms_draw.spv)
      add_custom_command(
        OUTPUT ${LEONIDA_TEST_MS}
        # Vulkan 1.2 (SPIR-V 1.5), the translator's target: 1.3 would emit LocalSizeId, which needs maintenance4.
        COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -V --target-env vulkan1.2 -o ${LEONIDA_TEST_MS}
                ${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/ms_draw.mesh
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/ms_draw.mesh
        COMMENT "Compiling the mesh test shader ms_draw.mesh"
        VERBATIM)
      add_executable(d3d12_mesh_test test/d3d12_mesh_test.cpp shims/d3d12_xs_device.cpp ${LEONIDA_TEST_MS}
                     ${LEONIDA_TEST_PS} ${LEONIDA_TEST_VS_FULLSCREEN})
      target_link_libraries(d3d12_mesh_test PRIVATE gta6rt_d3d12_com)
      target_compile_definitions(d3d12_mesh_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX
                                 LEONIDA_TEST_MS_SPV="${LEONIDA_TEST_MS}" LEONIDA_TEST_PS_SPV="${LEONIDA_TEST_PS}"
                                 LEONIDA_TEST_VS_FULLSCREEN_SPV="${LEONIDA_TEST_VS_FULLSCREEN}")
      target_compile_options(d3d12_mesh_test PRIVATE /W4 /permissive- /utf-8)
      add_test(NAME d3d12_mesh COMMAND d3d12_mesh_test)
      set_tests_properties(d3d12_mesh PROPERTIES SKIP_RETURN_CODE 77)
    else()
      message(STATUS "glslangValidator not found: d3d12_compute_test, d3d12_draw_test, d3d12_set0_compute_test and d3d12_mesh_test are not built")
    endif()

    # Native D3D12 SV_InnerCoverage reference; its HLSL is compiled at runtime, so glslang is not needed.
    add_executable(d3d12_conservative_tier3_reference_test test/d3d12_conservative_tier3_reference_test.cpp)
    target_link_libraries(d3d12_conservative_tier3_reference_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_conservative_tier3_reference_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_conservative_tier3_reference_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_conservative_tier3_reference COMMAND d3d12_conservative_tier3_reference_test)
    set_tests_properties(d3d12_conservative_tier3_reference PROPERTIES SKIP_RETURN_CODE 77)

    # Subgroup size control in PSOs, end to end: runtime/test/shaders/cs_w64x_readlane.asm translated natively and with
    # --wave64-emulate, and cs_wqm_w32.asm, built here with tools/rdna2spirv.py through test/shaders/build.py. Needs
    # llvm-mc with the AMDGPU target (LLVM_MC in the environment, as for the other shader tests, or llvm-mc on PATH)
    # and the Vulkan SDK's spirv tools; without them the test is not built.
    if(DEFINED ENV{LLVM_MC})
      set(LEONIDA_LLVM_MC "$ENV{LLVM_MC}")
    else()
      find_program(LEONIDA_LLVM_MC llvm-mc HINTS "C:/msys64/ucrt64/bin")
    endif()
    if(LEONIDA_LLVM_MC AND Vulkan_GLSLANG_VALIDATOR_EXECUTABLE)
      set(LEONIDA_SG_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated/subgroup_shaders)
      set(LEONIDA_SG_INDEXES "")
      foreach(case "w64_native;cs_w64x_readlane;0" "w64_emulated;cs_w64x_readlane;1" "w32_native;cs_wqm_w32;0"
                       "ps_w64_native;ps_w64x_readlane;0")
        list(GET case 0 folder)
        list(GET case 1 shader)
        list(GET case 2 emulate)
        set(index ${LEONIDA_SG_DIR}/${folder}/spirv/index.json)
        add_custom_command(
          OUTPUT ${index}
          # The standalone harness runs these with the V# at s[0:3]; d3d12_subgroup_test passes it as four root
          # constants, which the D3D12 compute layout puts at s[4:7] (LEONIDA_USER_DATA_SHIFT=4, D3D-448).
          COMMAND ${CMAKE_COMMAND} -E env LLVM_MC=${LEONIDA_LLVM_MC} LEONIDA_W64_EMULATE=${emulate}
                  LEONIDA_USER_DATA_SHIFT=4
                  $<TARGET_FILE:Python3::Interpreter> -I ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/build.py
                  ${LEONIDA_SG_DIR}/${folder} ${shader}
          DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/build.py
                  ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/compute_fixture.py
                  ${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/${shader}.asm
                  ${CMAKE_CURRENT_SOURCE_DIR}/../tools/rdna2spirv.py
          COMMENT "Translating ${shader} for the subgroup size test (${folder})"
          VERBATIM)
        list(APPEND LEONIDA_SG_INDEXES ${index})
      endforeach()
      # The probe shaders that store the subgroup size the driver ran them with, and a one-triangle vertex shader.
      foreach(source cs_sg_probe.comp ps_sg_probe.frag vs_sg_fullscreen.vert)
        set(output ${LEONIDA_SG_DIR}/glsl/${source}.spv)
        add_custom_command(
          OUTPUT ${output}
          COMMAND ${CMAKE_COMMAND} -E make_directory ${LEONIDA_SG_DIR}/glsl
          COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -V --target-env vulkan1.3 -o ${output}
                  ${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/${source}
          DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test/glsl/${source}
          COMMENT "Compiling ${source}"
          VERBATIM)
        list(APPEND LEONIDA_SG_INDEXES ${output})
      endforeach()
      add_executable(d3d12_subgroup_test test/d3d12_subgroup_test.cpp shims/d3d12_xs_device.cpp ${LEONIDA_SG_INDEXES})
      target_link_libraries(d3d12_subgroup_test PRIVATE gta6rt_d3d12_com)
      target_compile_definitions(d3d12_subgroup_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX
                                 LEONIDA_TEST_SUBGROUP_SHADERS="${LEONIDA_SG_DIR}")
      target_compile_options(d3d12_subgroup_test PRIVATE /W4 /permissive- /utf-8)
      add_test(NAME d3d12_subgroup COMMAND d3d12_subgroup_test)
      set_tests_properties(d3d12_subgroup PROPERTIES SKIP_RETURN_CODE 77)
    else()
      message(STATUS "llvm-mc (set LLVM_MC) or glslangValidator not found: d3d12_subgroup_test is not built")
    endif()

    # Clears on every format class against WARP (needs a Vulkan GPU; exit 77 without one).
    add_executable(d3d12_clear_formats_test test/d3d12_clear_formats_test.cpp)
    target_link_libraries(d3d12_clear_formats_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_clear_formats_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_clear_formats_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_clear_formats COMMAND d3d12_clear_formats_test)
    set_tests_properties(d3d12_clear_formats PROPERTIES SKIP_RETURN_CODE 77)

    # SetDescriptorHeaps validation, Reset/ClearState/bundle parity, lifetime pinning, direct-index flags.
    add_executable(d3d12_set_descriptor_heaps_test test/d3d12_set_descriptor_heaps_test.cpp)
    target_link_libraries(d3d12_set_descriptor_heaps_test PRIVATE gta6rt_d3d12_com d3d12 dxgi)
    target_compile_definitions(d3d12_set_descriptor_heaps_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(d3d12_set_descriptor_heaps_test PRIVATE /W4 /permissive- /utf-8)
    add_test(NAME d3d12_set_descriptor_heaps COMMAND d3d12_set_descriptor_heaps_test)
    set_tests_properties(d3d12_set_descriptor_heaps PROPERTIES SKIP_RETURN_CODE 77
                         FAIL_REGULAR_EXPRESSION "Validation Error;VUID-")

    # The loader can create this device instead of the host's (D3D12XboxCreateDevice, LEONIDA_D3D12=vulkan).
    if(TARGET gta6rt)
      target_link_libraries(gta6rt PRIVATE gta6rt_d3d12_com)
      target_compile_definitions(gta6rt PRIVATE LEONIDA_VULKAN_DEVICE)

      # A stand-in title that imports d3d12_xs.dll!D3D12XboxCreateDevice and runs a clear and a readback on the
      # device it gets, mapped and run by `gta6rt --run` with LEONIDA_D3D12=vulkan: the loader, the game thread, the
      # proxies and the Vulkan device in one process. check_tiny_frame.py reads the verdict, the binding and the
      # call order from outside; d3d12_com_test is its probe for "no usable Vulkan device" (exit code 77).
      add_library(d3d12_xs_stub SHARED test/fixtures/d3d12_xs_stub.cpp test/fixtures/d3d12_xs_stub.def)
      target_compile_definitions(d3d12_xs_stub PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
      target_compile_options(d3d12_xs_stub PRIVATE /W4 /permissive- /utf-8)
      set_target_properties(d3d12_xs_stub PROPERTIES
        OUTPUT_NAME d3d12_xs
        PREFIX ""
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fixtures/stubs/$<CONFIG>"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fixtures/stubs/$<CONFIG>"
      )
      add_executable(tiny_frame test/fixtures/tiny_frame.cpp)
      target_link_libraries(tiny_frame PRIVATE d3d12_xs_stub)
      target_compile_definitions(tiny_frame PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
      target_compile_options(tiny_frame PRIVATE /W4 /permissive- /utf-8)
      set_target_properties(tiny_frame PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fixtures/game/$<CONFIG>"
      )
      if(Python3_FOUND)
        add_test(NAME run_tiny_frame COMMAND "${Python3_EXECUTABLE}" -I
                 "${CMAKE_CURRENT_SOURCE_DIR}/test/check_tiny_frame.py" "$<TARGET_FILE:gta6rt>"
                 "$<TARGET_FILE:tiny_frame>" "--probe=$<TARGET_FILE:d3d12_com_test>" ${_trace_runner})
        set_tests_properties(run_tiny_frame PROPERTIES SKIP_RETURN_CODE 77)
        # The queue test settings are for the tests only: left set, they must not shrink the game's device.
        add_test(NAME run_tiny_frame_queue_test_settings COMMAND "${Python3_EXECUTABLE}" -I
                 "${CMAKE_CURRENT_SOURCE_DIR}/test/check_tiny_frame.py" "$<TARGET_FILE:gta6rt>"
                 "$<TARGET_FILE:tiny_frame>" "--probe=$<TARGET_FILE:d3d12_com_test>" --queue-test-settings
                 ${_trace_runner})
        set_tests_properties(run_tiny_frame_queue_test_settings PROPERTIES SKIP_RETURN_CODE 77)
      endif()
    endif()

    # M1.5 as a `gta6rt --run` guest: test/d3d12_translated_draw_test.cpp built again with LEONIDA_M15_GUEST as
    # tiny_guest_draw.exe, whose only device is the D3D12XboxCreateDevice it imports from d3d12_xs.dll. The import
    # library comes from a failing stub built into fixtures/stubs, away from the guest, so only gta6rt's shim registry
    # can answer it. The guest links nothing from src/ or shims/ and can be
    # copied with its ordinary CRT dependencies, given the m15 folder as its argument. test/run_guest_draw.cmake checks the exit code and the marker line.
    # Opt-in: it needs a Vulkan GPU and the guest cannot tell a missing one from a bug, so default runs are unchanged.
    option(LEONIDA_TEST_GUEST_DRAW
           "Build tiny_guest_draw.exe and run the M1.5 draw through gta6rt --run (needs a Vulkan GPU)" OFF)
    option(LEONIDA_GUEST_SCENE "Build the headless 600-frame synthetic guest test (requires static SDL3)" OFF)
    if(LEONIDA_TEST_GUEST_DRAW OR LEONIDA_GUEST_SCENE)
      # Share the import stub with the tiny_frame clear/readback guest from PR #3488.
      if(NOT TARGET d3d12_xs_stub)
        add_library(d3d12_xs_stub SHARED test/fixtures/d3d12_xs_stub.cpp test/fixtures/d3d12_xs_stub.def)
        target_compile_definitions(d3d12_xs_stub PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
        target_compile_options(d3d12_xs_stub PRIVATE /W4 /permissive- /utf-8)
        set_target_properties(d3d12_xs_stub PROPERTIES
          OUTPUT_NAME d3d12_xs
          PREFIX ""
          RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fixtures/stubs/$<CONFIG>"
          ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fixtures/stubs/$<CONFIG>"
        )
      endif()
      add_executable(tiny_guest_draw test/d3d12_translated_draw_test.cpp)
      target_link_libraries(tiny_guest_draw PRIVATE d3d12_xs_stub Vulkan::Headers)
      target_compile_definitions(tiny_guest_draw PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX LEONIDA_M15_GUEST)
      target_compile_options(tiny_guest_draw PRIVATE /W4 /permissive- /utf-8)
      set_target_properties(tiny_guest_draw PROPERTIES
        OUTPUT_NAME tiny_guest_draw
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fixtures/guest_draw/$<CONFIG>"
      )
      if(LEONIDA_GUEST_SCENE)
        if(NOT TARGET gta6rt OR NOT TARGET SDL3::SDL3-static OR NOT LEONIDA_SDL3)
          message(FATAL_ERROR "LEONIDA_GUEST_SCENE requires static SDL3; configure SDL3_DIR or LEONIDA_SDL3_FETCH")
        endif()
        target_compile_definitions(tiny_guest_draw PRIVATE LEONIDA_GUEST_SCENE)
        target_sources(gta6rt PRIVATE test/scene_input_driver.cpp)
      endif()
      # This guest can use a synthetic fixture produced on another machine. Register the automated test
      # if the tools to generate the fixture are available locally.
      if(CMAKE_HOST_WIN32 AND NOT CMAKE_CROSSCOMPILING AND TARGET d3d12_translated_draw_test AND TARGET gta6rt)
        add_dependencies(tiny_guest_draw d3d12_translated_draw_test gta6rt)
        add_test(NAME d3d12_translated_draw_guest
                 COMMAND ${CMAKE_COMMAND} -DGTA6RT=$<TARGET_FILE:gta6rt> -DGUEST=$<TARGET_FILE:tiny_guest_draw>
                         -DFIXTURE=${LEONIDA_M15_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/test/run_guest_draw.cmake)
        set_tests_properties(d3d12_translated_draw_guest PROPERTIES TIMEOUT 180)
        if(LEONIDA_GUEST_SCENE AND Python3_Interpreter_FOUND)
          add_test(NAME d3d12_guest_scene
                   COMMAND "${Python3_EXECUTABLE}" -I "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/run_guest_scene.py"
                           --loader $<TARGET_FILE:gta6rt> --guest $<TARGET_FILE:tiny_guest_draw>
                           --shaders "${LEONIDA_M15_DIR}" --negative-controls)
          set_tests_properties(d3d12_guest_scene PROPERTIES TIMEOUT 900)
        endif()
      else()
        message(STATUS "tiny_guest_draw will build. For an external synthetic M1.5 fixture or Wine run see runtime/README.md.")
      endif()
    endif()
    # bvh_builder_test can put the package's XboxBvhBuilder on this device too (LEONIDA_BVH_DEVICE=vulkan).
    if(TARGET bvh_builder_test)
      target_link_libraries(bvh_builder_test PRIVATE gta6rt_d3d12_com)
      target_compile_definitions(bvh_builder_test PRIVATE LEONIDA_VULKAN_DEVICE)
    endif()
    # fsr_dispatch_test can put the package's FidelityFX DLL on this device too (LEONIDA_FSR_DEVICE=vulkan).
    if(TARGET fsr_dispatch_test)
      target_link_libraries(fsr_dispatch_test PRIVATE gta6rt_d3d12_com)
      target_compile_definitions(fsr_dispatch_test PRIVATE LEONIDA_VULKAN_DEVICE)
    endif()
  elseif(TARGET gta6rt)
    message(STATUS "Python 3 or d3d12.h missing: the D3D12 COM objects and d3d12_com_test are not built")
  endif()

  # Needs only the Vulkan headers, not a GPU.
  if(BUILD_TESTING)
    add_executable(typeless_rtv_test test/typeless_rtv_test.cpp)
    target_link_libraries(typeless_rtv_test PRIVATE Vulkan::Headers)
    if(WIN32)
      target_compile_definitions(typeless_rtv_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    endif()
    add_test(NAME typeless_rtv COMMAND typeless_rtv_test)
    add_executable(pipeline_cache_snapshot_test test/pipeline_cache_snapshot_test.cpp)
    target_link_libraries(pipeline_cache_snapshot_test PRIVATE Vulkan::Headers)
    add_test(NAME pipeline_cache_snapshot COMMAND pipeline_cache_snapshot_test)
    add_executable(predicated_copy_test test/predicated_copy_test.cpp)
    target_link_libraries(predicated_copy_test PRIVATE Vulkan::Headers)
    if(WIN32)
      target_compile_definitions(predicated_copy_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    endif()
    add_test(NAME predicated_copy COMMAND predicated_copy_test)
    # CopyResource (buffer-to-buffer) and CopyTexture (including depth24_as_float) apply ResourceBinding::buffer_offset (no GPU).
    add_executable(replay_copy_offset_test test/replay_copy_offset_test.cpp)
    target_link_libraries(replay_copy_offset_test PRIVATE Vulkan::Headers)
    if(WIN32)
      target_compile_definitions(replay_copy_offset_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    endif()
    add_test(NAME replay_copy_offset COMMAND replay_copy_offset_test)
    # PreparedList::wrote_readback and readback_src_stages match the actual producer stage (depth24, rgb96, UAV clear; no GPU).
    add_executable(replay_readback_stages_test test/replay_readback_stages_test.cpp)
    target_link_libraries(replay_readback_stages_test PRIVATE Vulkan::Headers)
    if(WIN32)
      target_compile_definitions(replay_readback_stages_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    endif()
    add_test(NAME replay_readback_stages COMMAND replay_readback_stages_test)
    # emit of an enhanced texture barrier: no heap allocation for one layout piece, one barrier for each piece (no GPU).
    add_executable(replay_texture_barrier_test test/replay_texture_barrier_test.cpp)
    target_link_libraries(replay_texture_barrier_test PRIVATE Vulkan::Headers)
    if(WIN32)
      target_compile_definitions(replay_texture_barrier_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    endif()
    add_test(NAME replay_texture_barrier COMMAND replay_texture_barrier_test)
    add_executable(clear_color_test test/clear_color_test.cpp)
    target_link_libraries(clear_color_test PRIVATE Vulkan::Headers)
    if(WIN32)
      target_compile_definitions(clear_color_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    endif()
    add_test(NAME clear_color COMMAND clear_color_test)
    # FramePacer of the Xbox presentation path on a fake clock (src/frame_pacer.h); no GPU, no Vulkan.
    add_executable(frame_pacer_test test/frame_pacer_test.cpp)
    find_package(Threads REQUIRED)
    target_link_libraries(frame_pacer_test PRIVATE Threads::Threads)
    if(MSVC)
      target_compile_options(frame_pacer_test PRIVATE /W4 /permissive- /utf-8)
    endif()
    add_test(NAME frame_pacer COMMAND frame_pacer_test)

    # Replay refuses a list whose whole-view or implicit zero-clear needs an unset vkCmdClear*Image (no GPU).
    add_executable(replay_clear_gate_test test/replay_clear_gate_test.cpp)
    target_link_libraries(replay_clear_gate_test PRIVATE Vulkan::Headers)
    if(WIN32)
      target_compile_definitions(replay_clear_gate_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    endif()
    add_test(NAME replay_clear_gate COMMAND replay_clear_gate_test)
    # WriteBufferImmediate suspends conditional rendering around vkCmdUpdateBuffer so Intel drivers do not drop it.
    add_executable(replay_write_immediate_predication_test test/replay_write_immediate_predication_test.cpp)
    target_link_libraries(replay_write_immediate_predication_test PRIVATE Vulkan::Headers)
    if(WIN32)
      target_compile_definitions(replay_write_immediate_predication_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    endif()
    add_test(NAME replay_write_immediate_predication COMMAND replay_write_immediate_predication_test)
    add_executable(d3d12_state_map_test test/d3d12_state_map_test.cpp)
    target_link_libraries(d3d12_state_map_test PRIVATE Vulkan::Headers)
    if(WIN32)
      target_compile_definitions(d3d12_state_map_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    endif()
    add_test(NAME d3d12_state_map COMMAND d3d12_state_map_test)
    # A buffer UAV clear writes where its resource starts in a shared VkBuffer (no GPU).
    add_executable(replay_uav_clear_offset_test test/replay_uav_clear_offset_test.cpp)
    target_link_libraries(replay_uav_clear_offset_test PRIVATE Vulkan::Headers)
    add_test(NAME replay_uav_clear_offset COMMAND replay_uav_clear_offset_test)
    # ResolveQueryData suspends conditional rendering around un-ended vkCmdFillBuffer runs as well as dispatches (D3D-148).
    add_executable(replay_query_predication_test test/replay_query_predication_test.cpp)
    target_link_libraries(replay_query_predication_test PRIVATE Vulkan::Headers)
    if(WIN32)
      target_compile_definitions(replay_query_predication_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    endif()
    add_test(NAME replay_query_predication COMMAND replay_query_predication_test)
    # Query edge cases and queue ownership, with CPU mocks (no GPU or game data).
    add_executable(replay_query_edges_test test/replay_query_edges_test.cpp)
    target_link_libraries(replay_query_edges_test PRIVATE Vulkan::Headers)
    if(WIN32)
      target_compile_definitions(replay_query_edges_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    endif()
    add_test(NAME replay_query_edges COMMAND replay_query_edges_test)
    # GFX10 T#/S#/V# descriptor encoding and its Vulkan mappings (src/gfx10_descriptors.h).
    add_executable(gfx10_descriptors_test test/gfx10_descriptors_test.cpp)
    target_link_libraries(gfx10_descriptors_test PRIVATE Vulkan::Headers)
    add_test(NAME gfx10_descriptors COMMAND gfx10_descriptors_test)
    # CPU cost of the D3D12 -> Vulkan replay per command (record, prepare, emit) with no-op Vulkan entry points:
    # no GPU or driver. ctest runs its quick self-test; run it without arguments for the numbers.
    add_executable(replay_cpu_bench test/replay_cpu_bench.cpp)
    target_link_libraries(replay_cpu_bench PRIVATE Vulkan::Headers)
    if(WIN32)
      target_compile_definitions(replay_cpu_bench PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    endif()
    add_test(NAME replay_cpu_bench_selftest COMMAND replay_cpu_bench --self-test)
    add_test(NAME vk_descriptor_bench_selftest COMMAND vk_descriptor_bench --self-test)
    # mods.h: mod.json manifests and archetype swaps (docs/modding.md)
    add_executable(mods_test test/mods_test.cpp)
    target_link_libraries(mods_test PRIVATE Vulkan::Headers)
    add_test(NAME mods COMMAND mods_test)
    if(WIN32)
      # code_hooks.h: the remapping thunk and direct-call redirection the archetype hook uses (x64 Windows)
      add_executable(code_hooks_test test/code_hooks_test.cpp)
      target_link_libraries(code_hooks_test PRIVATE Vulkan::Headers)
      target_compile_definitions(code_hooks_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
      add_test(NAME code_hooks COMMAND code_hooks_test)
    endif()
    # The subgroup size a pipeline stage asks for, from synthetic index entries and device properties.
    add_executable(subgroup_size_test test/subgroup_size_test.cpp)
    target_link_libraries(subgroup_size_test PRIVATE Vulkan::Headers)
    add_test(NAME subgroup_size COMMAND subgroup_size_test)
    add_executable(conservative_raster_test test/conservative_raster_test.cpp)
    add_test(NAME conservative_raster COMMAND conservative_raster_test)
    # The GPU address arena's size from LEONIDA_ARENA_MB and synthetic device buffer limits.
    add_executable(arena_size_test test/arena_size_test.cpp)
    target_link_libraries(arena_size_test PRIVATE Vulkan::Headers)
    add_test(NAME arena_size COMMAND arena_size_test)
  endif()
else()
  include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/vulkan_headers_missing.cmake")
  leonida_vulkan_headers_missing()
endif()

# Timed condition-variable waits on Wine 9 (#3796): every Windows executable and DLL built against the dynamic CRT
# (/MD, the default) gets src/crt_timed_wait.cpp, which keeps msvcprt.lib's statically linked timed wait out of the
# link so that a wait and the mutex it unlocks always use msvcp140's own layout. A target that links the CRT
# statically (MSVC_RUNTIME_LIBRARY MultiThreaded, MultiThreadedDebug) already agrees with itself and is left alone,
# as are the fixtures linked with /NODEFAULTLIB, which have no CRT.
if(WIN32 AND MSVC)
  add_library(gta6rt_crt_timed_wait OBJECT src/crt_timed_wait.cpp)
  target_compile_definitions(gta6rt_crt_timed_wait PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
  target_compile_options(gta6rt_crt_timed_wait PRIVATE /W4 /permissive- /utf-8)
  # A timed wait is woken by a thread that takes its mutex; shared_mutex and sleep_for still work.
  add_executable(crt_timed_wait_test test/crt_timed_wait_test.cpp)
  target_compile_options(crt_timed_wait_test PRIVATE /W4 /permissive- /EHsc /utf-8)
  add_test(NAME crt_timed_wait COMMAND crt_timed_wait_test)
  set_tests_properties(crt_timed_wait PROPERTIES TIMEOUT 60)
  get_property(_gta6rt_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
  foreach(_target IN LISTS _gta6rt_targets)
    get_target_property(_type ${_target} TYPE)
    if(NOT _type MATCHES "^(EXECUTABLE|SHARED_LIBRARY|MODULE_LIBRARY)$")
      continue()
    endif()
    get_target_property(_crt ${_target} MSVC_RUNTIME_LIBRARY)
    get_target_property(_link_options ${_target} LINK_OPTIONS)
    if((_crt AND NOT _crt MATCHES "DLL") OR "/NODEFAULTLIB" IN_LIST _link_options)
      continue()  # static CRT, or no CRT at all (the CRT-free fixtures)
    endif()
    target_sources(${_target} PRIVATE $<TARGET_OBJECTS:gta6rt_crt_timed_wait>)
  endforeach()
endif()
