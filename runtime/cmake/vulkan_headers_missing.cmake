# Reports a configure without Vulkan headers. Everything under if(TARGET Vulkan::Headers) in runtime/CMakeLists.txt
# is then left out: not only the GPU test tools but the D3D12 COM objects and the Vulkan D3D12 device in gta6rt
# (LEONIDA_D3D12=vulkan), and with them most of ctest. A STATUS line was easy to miss: ctest then reports
# "100% tests passed" on a fraction of the suite. -DLEONIDA_REQUIRE_VULKAN=ON makes it a configure error (CI sets it).
# Kept in its own file so tests/test_cmake_vulkan_missing.py can run it with `cmake -P`.
function(leonida_vulkan_headers_missing)
  set(text
      "Vulkan headers not found, so this build leaves out the Vulkan D3D12 device in gta6rt (LEONIDA_D3D12=vulkan "
      "is unavailable), the D3D12 COM objects and their tests, the D3D12 layer and GPU tests, the GPU test tools "
      "(run_gpu_tests.py needs them), vk_pipeline_check, d3d12_state_map_test and gfx10_descriptors_test. ctest "
      "then runs only the CPU tests and can still report 100% passed. Set VULKAN_SDK in the environment that runs "
      "CMake (then delete CMakeCache.txt, which keeps the failed lookup), install the Vulkan headers, or pass "
      "-DLEONIDA_VULKAN_HEADERS_FETCH=ON to download the pinned Khronos Vulkan-Headers release. "
      "-DLEONIDA_REQUIRE_VULKAN=ON turns this warning into an error.")
  string(CONCAT text ${text})
  if(LEONIDA_REQUIRE_VULKAN)
    message(FATAL_ERROR "${text}")
  else()
    message(WARNING "${text}")
  endif()
endfunction()
