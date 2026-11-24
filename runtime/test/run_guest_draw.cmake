# Runs tiny_guest_draw.exe under `gta6rt --run` (test d3d12_translated_draw_guest in CMakeLists.txt). It passes only if
# gta6rt exits 0 AND the guest printed its marker line, which it does after every pixel check: a loader that returns 0
# without running the guest to the end, or a guest that prints and then fails, both fail here. CTest's
# PASS_REGULAR_EXPRESSION would ignore the exit code, which is why the checks are not test properties.
#   cmake -DGTA6RT=<gta6rt.exe> -DGUEST=<tiny_guest_draw.exe> -DFIXTURE=<build.py's m15 folder> [-DTIMEOUT=<s>]
#         -P run_guest_draw.cmake
cmake_minimum_required(VERSION 3.24)

foreach(var GTA6RT GUEST FIXTURE)
  if(NOT DEFINED ${var} OR "${${var}}" STREQUAL "")
    message(FATAL_ERROR "run_guest_draw.cmake: -D${var}=... is required")
  endif()
endforeach()
if(NOT DEFINED TIMEOUT)
  set(TIMEOUT 120)
endif()
foreach(path "${GTA6RT}" "${GUEST}" "${FIXTURE}/spirv/index.json" "${FIXTURE}/fssm/vs_m15.fssm"
             "${FIXTURE}/fssm/ps_m15.fssm")
  if(NOT EXISTS "${path}")
    message(FATAL_ERROR "run_guest_draw.cmake: ${path} does not exist")
  endif()
endforeach()

# The line test/d3d12_translated_draw_test.cpp prints under LEONIDA_M15_GUEST once every check has passed.
set(marker "M1.5 guest: PASS, 4064 of 4096 pixels exact, 32 edge pixels the clear colour")

file(TO_NATIVE_PATH "${GUEST}" guest)
file(TO_NATIVE_PATH "${FIXTURE}" fixture)
file(TO_NATIVE_PATH "${FIXTURE}/spirv" shaders)
get_filename_component(guest_dir "${GUEST}" DIRECTORY)
# Set here rather than as the test's ENVIRONMENT, before starting gta6rt on native Windows.
set(ENV{LEONIDA_D3D12} vulkan)
set(ENV{LEONIDA_SHADERS} "${shaders}")

execute_process(
  COMMAND "${GTA6RT}" --run "${guest}" -- "${fixture}"
  WORKING_DIRECTORY "${guest_dir}"
  TIMEOUT ${TIMEOUT}
  RESULT_VARIABLE result
  OUTPUT_VARIABLE out
  ERROR_VARIABLE err)
message("--- gta6rt stdout ---\n${out}\n--- gta6rt stderr ---\n${err}\n--- exit: ${result}")

string(REPLACE "\r\n" "\n" out "\n${out}\n")
string(REPLACE "\r\n" "\n" err "${err}")
set(failures)
if(NOT "${result}" STREQUAL "0")
  list(APPEND failures "exit is '${result}', not 0")
endif()
string(FIND "${out}" "\n${marker}\n" first)
string(FIND "${out}" "\n${marker}\n" last REVERSE)
if(first EQUAL -1)
  list(APPEND failures "stdout has no line '${marker}'")
elseif(NOT first EQUAL last)
  list(APPEND failures "stdout has the line '${marker}' more than once")
endif()
# The device must be the shim's Vulkan device (shims/d3d12_xs.cpp logs it), not the host's D3D12 device.
set(pointer "(0[xX])?0*[1-9A-Fa-f][0-9A-Fa-f]*")
set(factory_line "\\[rt\\] d3d12_xs: D3D12XboxCreateDevice -> proxy ${pointer} \\(Vulkan device ${pointer}\\)")
if(NOT "\n${err}\n" MATCHES "\n${factory_line}\n")
  list(APPEND failures "stderr does not show D3D12XboxCreateDevice returning the Vulkan device")
endif()
list(LENGTH failures failure_count)
if(failure_count GREATER 0)
  list(JOIN failures "; " failures)
  message(FATAL_ERROR "tiny_guest_draw under gta6rt --run failed: ${failures}")
endif()
message("tiny_guest_draw under gta6rt --run: exit 0 and '${marker}'")
