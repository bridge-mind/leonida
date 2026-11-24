# Cross-compile the Windows runtime from Linux with clang-cl and lld-link against an MSVC CRT and Windows SDK laid
# out by xwin (https://github.com/Jake-Shadle/xwin) in the /winsysroot style:
#
#   xwin --accept-license --sdk-version 10.0.26100 splat --use-winsysroot-style --preserve-ms-arch-notation \
#        --output ~/.local/share/xwin-sdk
#   cmake -S runtime -B runtime/build/windows --toolchain "$PWD/runtime/cmake/clang-cl-xwin.cmake" \
#         -DXWIN_DIR="$HOME/.local/share/xwin-sdk"
#   cmake --build runtime/build/windows --parallel
#   ctest --test-dir runtime/build/windows        # through Wine; see docs/testing.md for Proton
#
# The result is the same set of PE files the Windows build makes. A relative --toolchain path is resolved against
# the build folder, hence $PWD. -DCMAKE_CXX_FLAGS=... replaces the flags set here (--target, /winsysroot), so add
# flags with CMAKE_CXX_FLAGS_RELWITHDEBINFO or target options instead.
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR AMD64)

if(NOT XWIN_DIR)
  set(XWIN_DIR "$ENV{XWIN_DIR}")
endif()
if(NOT XWIN_DIR OR NOT EXISTS "${XWIN_DIR}/VC/Tools/MSVC")
  message(FATAL_ERROR "Set XWIN_DIR (or -DXWIN_DIR=) to an xwin splat made with --use-winsysroot-style")
endif()
set(XWIN_DIR "${XWIN_DIR}" CACHE PATH "xwin /winsysroot directory (VC/ and Windows Kits/)")
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES XWIN_DIR)

# The LLVM tools, by their unversioned names unless given (-DCMAKE_CXX_COMPILER=clang-cl-19 -DCMAKE_LINKER=lld-link-19
# ...). The MSVC STL that xwin lays out needs Clang 19 or newer.
foreach(_tool C_COMPILER:clang-cl CXX_COMPILER:clang-cl LINKER:lld-link AR:llvm-lib RC_COMPILER:llvm-rc MT:llvm-mt)
  string(REPLACE ":" ";" _tool "${_tool}")
  list(GET _tool 0 _var)
  list(GET _tool 1 _name)
  if(NOT CMAKE_${_var})
    set(CMAKE_${_var} ${_name})
  endif()
endforeach()

set(_xwin_compile "--target=x86_64-pc-windows-msvc /winsysroot \"${XWIN_DIR}\"")
set(CMAKE_C_FLAGS_INIT "${_xwin_compile}")
set(CMAKE_CXX_FLAGS_INIT "${_xwin_compile}")
foreach(_kind EXE SHARED MODULE)
  set(CMAKE_${_kind}_LINKER_FLAGS_INIT "/winsysroot:\"${XWIN_DIR}\"")
endforeach()

# xwin leaves out the debug CRT (msvcrtd, ucrtd: not redistributable), so only Release and RelWithDebInfo link. CMake's
# MSVC platform file defaults the build type to Debug, so default it here, and CMake's own checks build in Release.
if(NOT DEFINED CACHE{CMAKE_BUILD_TYPE})
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Release or RelWithDebInfo (xwin has no debug CRT)")
endif()
file(GLOB _xwin_debug_crt "${XWIN_DIR}/VC/Tools/MSVC/*/lib/x64/msvcrtd.lib")
if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT _xwin_debug_crt)
  message(FATAL_ERROR "CMAKE_BUILD_TYPE=Debug needs the debug CRT: splat again with xwin's --include-debug-libs, "
                      "or use RelWithDebInfo")
endif()
set(CMAKE_TRY_COMPILE_CONFIGURATION Release)

# The Vulkan headers are the same on every platform, but /usr/include also holds glibc's, which must not shadow the
# MSVC CRT. Point Vulkan_INCLUDE_DIR at a folder that holds only vulkan/ and vk_video/ (links to the host copies),
# unless another folder was given (-DVulkan_INCLUDE_DIR=<Vulkan-Headers>/include). The links are (re)made on every
# configure, so deleting the folder doesn't silently drop the Vulkan targets.
if(CMAKE_BINARY_DIR)
  set(_xwin_vulkan "${CMAKE_BINARY_DIR}/xwin-vulkan-include")
  if(NOT Vulkan_INCLUDE_DIR OR Vulkan_INCLUDE_DIR STREQUAL _xwin_vulkan)
    foreach(_host_include /usr/include /usr/local/include)
      if(EXISTS "${_host_include}/vulkan/vulkan.h")
        file(MAKE_DIRECTORY "${_xwin_vulkan}")
        foreach(_sub vulkan vk_video)
          if(EXISTS "${_host_include}/${_sub}" AND NOT EXISTS "${_xwin_vulkan}/${_sub}")
            file(CREATE_LINK "${_host_include}/${_sub}" "${_xwin_vulkan}/${_sub}" SYMBOLIC)
          endif()
        endforeach()
        set(Vulkan_INCLUDE_DIR "${_xwin_vulkan}" CACHE PATH "Vulkan headers without the host's libc headers")
        break()
      endif()
    endforeach()
  endif()
endif()

# find_library and find_path look in the newest SDK's um folders: d3d12.lib and dxgi.lib for the host-comparison
# tests, d3d12.h for tools/gen_d3d12_stubs.py.
file(GLOB _xwin_um_libs LIST_DIRECTORIES true "${XWIN_DIR}/Windows Kits/10/Lib/*/um/x64")
if(NOT _xwin_um_libs)
  message(FATAL_ERROR "No \"Windows Kits/10/Lib/<version>/um/x64\" under ${XWIN_DIR}: make the splat with "
                      "--use-winsysroot-style --preserve-ms-arch-notation and an SDK (--sdk-version 10.0.26100)")
endif()
list(SORT _xwin_um_libs COMPARE NATURAL ORDER DESCENDING)
list(GET _xwin_um_libs 0 _xwin_um_lib)
list(APPEND CMAKE_LIBRARY_PATH "${_xwin_um_lib}")
get_filename_component(_xwin_sdk_version "${_xwin_um_lib}/../.." ABSOLUTE)
get_filename_component(_xwin_sdk_version "${_xwin_sdk_version}" NAME)
list(APPEND CMAKE_INCLUDE_PATH "${XWIN_DIR}/Windows Kits/10/Include/${_xwin_sdk_version}/um")
# Programs (Python, glslangValidator, Wine) come from the host; libraries and headers from the SDK or the paths above;
# CMake packages only from the SDK root, so a host package (an ELF SDL3, say) never ends up in a Windows link.
set(CMAKE_FIND_ROOT_PATH "${XWIN_DIR}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# LLVM's MASM stands in for ml64, so unimplemented_policy_test keeps its register-preservation check (abi_probe.asm).
# runtime/CMakeLists.txt keeps a LEONIDA_ML64 that is already set.
find_program(LEONIDA_ML64 NAMES ml64 llvm-ml64)

# ctest runs the PE tests through Wine, or Proton's wine (-DCMAKE_CROSSCOMPILING_EMULATOR=<Proton>/files/bin/wine).
if(NOT CMAKE_CROSSCOMPILING_EMULATOR)
  find_program(XWIN_WINE NAMES wine wine64)
  if(XWIN_WINE)
    set(CMAKE_CROSSCOMPILING_EMULATOR "${XWIN_WINE}")
  elseif(CMAKE_BINARY_DIR)
    message(STATUS "Wine not found: ctest can't run the Windows tests (set CMAKE_CROSSCOMPILING_EMULATOR)")
  endif()
endif()
