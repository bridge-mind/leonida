# atexit handlers, static destructors and every way out: tiny_atexit run through `gta6rt --run` must print the
# sequence written below and end with its exit code. That is what the program does natively on Windows, and the
# native run is held to the same sequence, so the two cannot agree on a wrong answer. Under Wine both runs go through
# RUNNER, and Wine's own CRT differs in places (it drops a handler registered while handlers run, and its _exit does
# not wait for the exit lock), so there a native difference is reported without failing; --run is still held to the
# Windows sequence.
get_filename_component(guest_dir "${GUEST}" DIRECTORY)
get_filename_component(guest_name "${GUEST}" NAME)

set(ctors "ctor first;ctor second;main")
set(handlers "atexit two;atexit registers late;atexit late;atexit one;dtor second;dtor first")
set(expect_return "${ctors};${handlers}")
set(code_return 3)
set(expect_exit "${ctors};${handlers}")
set(code_exit 7)
set(expect__exit "${ctors}")
set(code__exit 8)
set(expect_quick_exit "${ctors}")
set(code_quick_exit 9)
set(expect_ExitProcess "${ctors}")
set(code_ExitProcess 10)
set(expect_cexit "${ctors};${handlers};after _cexit")
set(code_cexit 11)
set(expect_c_exit "${ctors};after _c_exit;${handlers}")
set(code_c_exit 12)
# exit(7) is running a slow handler when a second thread calls _exit(7): the exit lock holds it back.
set(expect_concurrent_exit "${ctors};slow handler started;slow handler done;${handlers}")
set(code_concurrent_exit 7)

# The program's own lines, in order, without the prefix; everything gta6rt prints is dropped.
function(guest_lines output result)
  string(REPLACE "\r\n" "\n" output "${output}")
  string(REPLACE ";" "," output "${output}")
  string(REPLACE "\n" ";" output "${output}")
  set(lines)
  foreach(line IN LISTS output)
    if(line MATCHES "^tiny_atexit: (.*)$")
      list(APPEND lines "${CMAKE_MATCH_1}")
    endif()
  endforeach()
  set(${result} "${lines}" PARENT_SCOPE)
endfunction()

set(failures "")
foreach(mode return exit _exit quick_exit ExitProcess cexit c_exit concurrent_exit)
  execute_process(COMMAND ${RUNNER} "${GUEST}" ${mode}
    WORKING_DIRECTORY "${guest_dir}" TIMEOUT 60
    RESULT_VARIABLE native_code OUTPUT_VARIABLE native_out ERROR_VARIABLE native_err)
  execute_process(COMMAND ${RUNNER} "${LOADER}" --run "${guest_name}" -- ${mode}
    WORKING_DIRECTORY "${guest_dir}" TIMEOUT 60
    RESULT_VARIABLE run_code OUTPUT_VARIABLE run_out ERROR_VARIABLE run_err)
  guest_lines("${native_out}" native_lines)
  guest_lines("${run_out}" run_lines)
  if(NOT native_code STREQUAL code_${mode} OR NOT native_lines STREQUAL expect_${mode})
    string(CONCAT native_diff "[${mode}] native run differs from the expected sequence"
      "\n  expected (${code_${mode}}): ${expect_${mode}}\n  native   (${native_code}): ${native_lines}")
    if(RUNNER)
      message(STATUS "${native_diff}\n  (the host CRT differs from Windows; not a gta6rt failure)")
    else()
      string(APPEND failures "\n${native_diff}\n  (the fixture or the host changed)")
    endif()
  endif()
  if(NOT run_code STREQUAL code_${mode} OR NOT run_lines STREQUAL expect_${mode})
    string(APPEND failures "\n[${mode}] gta6rt --run differs from the expected sequence"
      "\n  expected (${code_${mode}}): ${expect_${mode}}\n  --run    (${run_code}): ${run_lines}\n${run_err}")
  endif()
endforeach()
if(failures)
  message(FATAL_ERROR "atexit/termination differences:${failures}")
endif()
message(STATUS "tiny_atexit: 8 termination paths match the Windows sequence")
