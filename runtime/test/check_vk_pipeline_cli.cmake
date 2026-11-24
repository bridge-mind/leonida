file(MAKE_DIRECTORY "${TEST_DIR}")
file(WRITE "${TEST_DIR}/index.json" "not JSON\n")

# ${RUNNER} is CMAKE_CROSSCOMPILING_EMULATOR: CTest does not apply it to a script
# test (the command is the host CMake), so the script runs the tool through it itself.
function(check_invalid option)
  execute_process(COMMAND ${RUNNER} "${TOOL}" "${TEST_DIR}" "${option}" ${ARGN}
                  RESULT_VARIABLE code OUTPUT_VARIABLE log ERROR_VARIABLE log TIMEOUT 5)
  if(NOT code STREQUAL "2" OR NOT log MATCHES "${option} requires a valid value")
    message(FATAL_ERROR "${option} ${ARGN}: expected usage error, got ${code}\n${log}")
  endif()
endfunction()

check_invalid(--count)
check_invalid(--count --compute-only)
check_invalid(--count 18446744073709551616)
check_invalid(--jobs 4294967296)
check_invalid(--start -1)
check_invalid(--max-bytes 1MiB)
check_invalid(--large-module-bytes -1)
check_invalid(--slow nonsense)
check_invalid(--slow nan)
check_invalid(--slow -0.5)
check_invalid(--slow 1.2.3)  # strtod must consume the whole token, not just its numeric prefix
check_invalid(--stage --graphics-only)
check_invalid(--stats)

function(check_protected_input name)
  execute_process(COMMAND ${RUNNER} "${TOOL}" "${TEST_DIR}" ${ARGN}
                  RESULT_VARIABLE code OUTPUT_VARIABLE log ERROR_VARIABLE log TIMEOUT 5)
  string(FIND "${log}" "must not overwrite ${name}" found)
  if(NOT code STREQUAL "2" OR found EQUAL -1)
    message(FATAL_ERROR "${ARGN}: expected input/output path conflict, got ${code}\n${log}")
  endif()
endfunction()

file(WRITE "${TEST_DIR}/skip.txt" "example\n")
file(SHA256 "${TEST_DIR}/index.json" index_before)
file(SHA256 "${TEST_DIR}/skip.txt" skip_before)
check_protected_input("index.json" "${TEST_DIR}/index.json")
check_protected_input("index.json" --stats "${TEST_DIR}/./index.json")
check_protected_input("--skip-file" --skip-file "${TEST_DIR}/skip.txt" --inflight-file "${TEST_DIR}/skip.txt")

# A hard link has a distinct path even after normalization. Some filesystems cannot create one.
file(CREATE_LINK "${TEST_DIR}/skip.txt" "${TEST_DIR}/skip-link.txt" RESULT hard_link)
if(hard_link STREQUAL "0")
  check_protected_input("--skip-file" --skip-file "${TEST_DIR}/skip.txt" --inflight-file "${TEST_DIR}/skip-link.txt")
  file(REMOVE "${TEST_DIR}/skip-link.txt")
endif()

file(SHA256 "${TEST_DIR}/index.json" index_after)
file(SHA256 "${TEST_DIR}/skip.txt" skip_after)
if(NOT index_before STREQUAL index_after OR NOT skip_before STREQUAL skip_after)
  message(FATAL_ERROR "Input/output conflicts modified the index or the skip file")
endif()

execute_process(COMMAND ${RUNNER} "${TOOL}" "${TEST_DIR}"
                --jobs 2 --start 3 --count 0 --max-bytes 0 --large-module-bytes 0
                --slow 0.5 --stage cs --stats report.csv
                RESULT_VARIABLE code OUTPUT_VARIABLE log ERROR_VARIABLE log TIMEOUT 5)
# The index parser names the file and where it stopped (#1847: "index.json: expected '{' at byte 0").
if(NOT code STREQUAL "2" OR NOT log MATCHES "index.json: ")
  message(FATAL_ERROR "Expected valid arguments to reach the malformed index, got ${code}\n${log}")
endif()

# A malformed byte count must not reach Vulkan or wrap to an in-range value.
foreach(bytes -1 1.5 4294967296)
  file(WRITE "${TEST_DIR}/index.json" "{\"example\":{\"stage\":\"cs\",\"lds_bytes\":${bytes}}}\n")
  execute_process(COMMAND ${RUNNER} "${TOOL}" "${TEST_DIR}"
                  RESULT_VARIABLE code OUTPUT_VARIABLE log ERROR_VARIABLE log TIMEOUT 5)
  # The index parser says where it stopped (#1847: "expected unsigned 32-bit lds_bytes at byte 37").
  if(NOT code STREQUAL "2" OR NOT log MATCHES "at byte [0-9]+")
    message(FATAL_ERROR "lds_bytes=${bytes}: expected a malformed index, got ${code}\n${log}")
  endif()

  file(WRITE "${TEST_DIR}/index.json" "{\"example\":{\"stage\":\"cs\",\"push_constant_bytes\":${bytes}}}\n")
  execute_process(COMMAND ${RUNNER} "${TOOL}" "${TEST_DIR}"
                  RESULT_VARIABLE code OUTPUT_VARIABLE log ERROR_VARIABLE log TIMEOUT 5)
  if(NOT code STREQUAL "2" OR NOT log MATCHES "at byte [0-9]+")
    message(FATAL_ERROR "push_constant_bytes=${bytes}: expected a malformed index, got ${code}\n${log}")
  endif()

  file(WRITE "${TEST_DIR}/index.json" "{\"example\":{\"stage\":\"cs\",\"pc_tail\":{\"set\":1,\"binding\":0,\"bytes\":${bytes}}}}\n")
  execute_process(COMMAND ${RUNNER} "${TOOL}" "${TEST_DIR}"
                  RESULT_VARIABLE code OUTPUT_VARIABLE log ERROR_VARIABLE log TIMEOUT 5)
  if(NOT code STREQUAL "2" OR NOT log MATCHES "at byte [0-9]+")
    message(FATAL_ERROR "pc_tail.bytes=${bytes}: expected a malformed index, got ${code}\n${log}")
  endif()
endforeach()
