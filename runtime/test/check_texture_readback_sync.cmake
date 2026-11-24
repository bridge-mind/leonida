# Run with -DTOOL=<vk_texture_test> -DSPIRV=<directory built by the GPU suite>
# -DOUTPUT=<readback file> -P runtime/test/check_texture_readback_sync.cmake.
# For a cross build, pass the configured emulator as "-DRUNNER=<emulator;args>".
# Requires the Khronos validation layer and a device that supports the texture harness.
if(NOT DEFINED OUTPUT OR OUTPUT STREQUAL "")
  message(FATAL_ERROR "Specify the readback file with -DOUTPUT")
endif()
get_filename_component(spirv_dir "${SPIRV}" ABSOLUTE)
# Relative output paths are resolved by the harness from its working directory.
get_filename_component(readback "${OUTPUT}" ABSOLUTE BASE_DIR "${spirv_dir}")
if(IS_DIRECTORY "${readback}")
  message(FATAL_ERROR "Readback path is a directory: ${readback}")
endif()
file(REMOVE "${readback}")
if(EXISTS "${readback}" OR IS_SYMLINK "${readback}")
  message(FATAL_ERROR "Could not remove the previous readback: ${readback}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
          VK_LOADER_DEBUG=layer VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT
          ${RUNNER} "${TOOL}" cube "${OUTPUT}"
  WORKING_DIRECTORY "${spirv_dir}" TIMEOUT 90
  RESULT_VARIABLE code OUTPUT_VARIABLE log ERROR_VARIABLE log)
if(NOT code STREQUAL "0" OR NOT log MATCHES "vk_texture_test cube: PASSED"
    OR NOT log MATCHES "Insert instance layer \"VK_LAYER_KHRONOS_validation\""
    OR log MATCHES "SYNC-HAZARD|Validation Error")
  message(FATAL_ERROR "Texture draw with synchronization validation returned ${code}\n${log}")
endif()
if(NOT EXISTS "${readback}" OR IS_DIRECTORY "${readback}")
  message(FATAL_ERROR "Texture draw did not write a readback: ${readback}\n${log}")
endif()
# The cube harness writes a 64x64 VK_FORMAT_R32G32B32A32_SFLOAT frame.
math(EXPR expected_size "64 * 64 * 16")
file(SIZE "${readback}" actual_size)
if(NOT actual_size EQUAL expected_size)
  message(FATAL_ERROR "Texture readback has ${actual_size} bytes, expected ${expected_size}: ${readback}\n${log}")
endif()
