# Observe the guest verdict and the loader from outside the process. Under Wine an
# exit status alone can look successful even when the mapped program reported failure.
get_filename_component(guest_dir "${GUEST}" DIRECTORY)
get_filename_component(guest_name "${GUEST}" NAME)
execute_process(
  COMMAND ${RUNNER} "${LOADER}" --run "${guest_name}"
  WORKING_DIRECTORY "${guest_dir}"
  TIMEOUT 90
  RESULT_VARIABLE code OUTPUT_VARIABLE output ERROR_VARIABLE output)
string(REPLACE "\r\n" "\n" output "${output}")
if(NOT code STREQUAL "0" OR NOT output MATCHES "(^|\n)tiny_async: PASS(\n|$)"
    OR NOT output MATCHES "(^|\n)game exited with code 0(\n|$)")
  message(FATAL_ERROR "Loaded async fixture returned ${code}\n${output}")
endif()
