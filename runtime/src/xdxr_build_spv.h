#pragma once
#include <cstdint>

namespace leonida {

// xdxr_build.comp compiled with: glslc --target-env=vulkan1.2 -O -mfmt=c xdxr_build.comp -o xdxr_build.inc
inline constexpr std::uint32_t xdxr_build_spirv[] =
#include "xdxr_build.inc"
;

} // namespace leonida
