cmake_minimum_required(VERSION 3.21)
project(conform_native LANGUAGES CXX)

# The scene contract and the HLSL are the shared spec; this project only reads them.
set(CONFORM_SPEC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../spec" CACHE PATH
    "Directory holding scenes.h and shaders/")
set(DXC_EXECUTABLE
    "C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/dxc.exe"
    CACHE FILEPATH "DirectX Shader Compiler")

if(NOT EXISTS "${CONFORM_SPEC_DIR}/scenes.h")
  message(FATAL_ERROR
    "scenes.h not found at ${CONFORM_SPEC_DIR}/scenes.h. Point -DCONFORM_SPEC_DIR at the "
    "directory that holds scenes.h and shaders/.")
endif()
if(NOT EXISTS "${DXC_EXECUTABLE}")
  message(FATAL_ERROR
    "dxc.exe not found at ${DXC_EXECUTABLE}. Install the Windows SDK or pass "
    "-DDXC_EXECUTABLE=<full path to dxc.exe>.")
endif()

if(NOT MSVC)
  message(FATAL_ERROR "This project targets the Visual Studio toolchain; use -G \"Visual Studio 17 2022\" -A x64.")
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY "${SHADER_OUTPUT_DIR}")

set(FULLSCREEN_HLSL "${CONFORM_SPEC_DIR}/shaders/fullscreen_color.hlsl")
set(VERTEX_COLOR_HLSL "${CONFORM_SPEC_DIR}/shaders/vertex_color.hlsl")

set(FULLSCREEN_VS_CSO "${SHADER_OUTPUT_DIR}/fullscreen_color.vs.cso")
set(FULLSCREEN_PS_CSO "${SHADER_OUTPUT_DIR}/fullscreen_color.ps.cso")
set(VERTEX_COLOR_VS_CSO "${SHADER_OUTPUT_DIR}/vertex_color.vs.cso")
set(VERTEX_COLOR_PS_CSO "${SHADER_OUTPUT_DIR}/vertex_color.ps.cso")

# -Qstrip_debug and -Qstrip_reflect keep the .cso small and deterministic: nothing in the
# file depends on the build directory or the machine. -WX makes a shader warning a build
# failure rather than a silent behaviour change in the reference images.
set(DXC_COMMON_ARGS -nologo -O3 -WX -Qstrip_debug -Qstrip_reflect)

add_custom_command(
  OUTPUT "${FULLSCREEN_VS_CSO}"
  COMMAND "${DXC_EXECUTABLE}" -T vs_6_6 -E VSMain ${DXC_COMMON_ARGS}
          "-Fo${FULLSCREEN_VS_CSO}" "${FULLSCREEN_HLSL}"
  DEPENDS "${FULLSCREEN_HLSL}" "${DXC_EXECUTABLE}"
  COMMENT "Compiling fullscreen_color.hlsl VSMain as vs_6_6"
  VERBATIM)

add_custom_command(
  OUTPUT "${FULLSCREEN_PS_CSO}"
  COMMAND "${DXC_EXECUTABLE}" -T ps_6_6 -E PSMain ${DXC_COMMON_ARGS}
          "-Fo${FULLSCREEN_PS_CSO}" "${FULLSCREEN_HLSL}"
  DEPENDS "${FULLSCREEN_HLSL}" "${DXC_EXECUTABLE}"
  COMMENT "Compiling fullscreen_color.hlsl PSMain as ps_6_6"
  VERBATIM)

add_custom_command(
  OUTPUT "${VERTEX_COLOR_VS_CSO}"
  COMMAND "${DXC_EXECUTABLE}" -T vs_6_6 -E VSMain ${DXC_COMMON_ARGS}
          "-Fo${VERTEX_COLOR_VS_CSO}" "${VERTEX_COLOR_HLSL}"
  DEPENDS "${VERTEX_COLOR_HLSL}" "${DXC_EXECUTABLE}"
  COMMENT "Compiling vertex_color.hlsl VSMain as vs_6_6"
  VERBATIM)

add_custom_command(
  OUTPUT "${VERTEX_COLOR_PS_CSO}"
  COMMAND "${DXC_EXECUTABLE}" -T ps_6_6 -E PSMain ${DXC_COMMON_ARGS}
          "-Fo${VERTEX_COLOR_PS_CSO}" "${VERTEX_COLOR_HLSL}"
  DEPENDS "${VERTEX_COLOR_HLSL}" "${DXC_EXECUTABLE}"
  COMMENT "Compiling vertex_color.hlsl PSMain as ps_6_6"
  VERBATIM)

add_custom_target(conform_shaders DEPENDS
  "${FULLSCREEN_VS_CSO}" "${FULLSCREEN_PS_CSO}"
  "${VERTEX_COLOR_VS_CSO}" "${VERTEX_COLOR_PS_CSO}")

add_executable(conform_native
  src/main.cpp
  src/diagnostics.cpp
  src/gpu.cpp
  src/renderer.cpp
  src/scene_plan.cpp
  src/png_io.cpp
  src/verify.cpp
  src/manifest.cpp)

target_include_directories(conform_native PRIVATE "${CONFORM_SPEC_DIR}")
target_compile_definitions(conform_native PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
# /we4062 is separate from /W4 because C4062 (enumerator not handled in a switch that has
# no default) is off by default, so /W4 /WX would let a new SceneKind render as a clear.
target_compile_options(conform_native PRIVATE /W4 /WX /we4062 /permissive- /utf-8 /EHsc)
target_link_libraries(conform_native PRIVATE d3d12 dxgi windowscodecs ole32)
add_dependencies(conform_native conform_shaders)

add_custom_command(TARGET conform_native POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E make_directory "$<TARGET_FILE_DIR:conform_native>/shaders"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
          "${FULLSCREEN_VS_CSO}" "${FULLSCREEN_PS_CSO}"
          "${VERTEX_COLOR_VS_CSO}" "${VERTEX_COLOR_PS_CSO}"
          "$<TARGET_FILE_DIR:conform_native>/shaders/"
  COMMENT "Copying compiled shaders next to conform_native.exe"
  VERBATIM)

set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" PROPERTY VS_STARTUP_PROJECT conform_native)
