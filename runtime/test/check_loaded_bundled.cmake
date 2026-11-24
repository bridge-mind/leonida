# Run the run-time bundled-DLL load fixture (tiny_bundled_load) under the loader and check its verdict from outside,
# as check_loaded_gdk.cmake does.
get_filename_component(guest_dir "${GUEST}" DIRECTORY)
get_filename_component(guest_name "${GUEST}" NAME)
execute_process(
  COMMAND ${RUNNER} "${LOADER}" --run "${guest_name}"
  WORKING_DIRECTORY "${guest_dir}"
  TIMEOUT 90
  RESULT_VARIABLE code OUTPUT_VARIABLE output ERROR_VARIABLE output)
string(REPLACE "\r\n" "\n" output "${output}")
if(NOT code STREQUAL "0" OR NOT output MATCHES "(^|\n)tiny_bundled_load: PASS(\n|$)"
    OR NOT output MATCHES "(^|\n)game exited with code 0(\n|$)")
  message(FATAL_ERROR "Loaded bundled-DLL run-time fixture returned ${code}\n${output}")
endif()
