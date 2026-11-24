# cmake -DTOOL=<PE file> -DWANT=<bytes> -P check_stack_reserve.cmake
#
# Fails unless the PE header of TOOL reserves at least WANT bytes of stack (SizeOfStackReserve). On Windows that
# reserve is the stack of the main thread and of every thread created without a size of its own, std::thread
# included, so it is what a driver's compiler runs on in vk_pipeline_check, and in gta6rt on every game thread but
# the entry thread (BUILD-1457). Only the header is read.

# A little-endian unsigned integer of `bytes` bytes at `offset` in `file`.
function(read_le file offset bytes out)
  file(READ "${file}" hex OFFSET ${offset} LIMIT ${bytes} HEX)
  string(LENGTH "${hex}" length)
  math(EXPR expected "${bytes} * 2")
  if(NOT length EQUAL expected)
    message(FATAL_ERROR "${file}: cannot read ${bytes} bytes at offset ${offset}")
  endif()
  set(big_endian "")
  math(EXPR last "${bytes} - 1")
  foreach(i RANGE ${last})
    math(EXPR at "(${last} - ${i}) * 2")
    string(SUBSTRING "${hex}" ${at} 2 byte)
    string(APPEND big_endian "${byte}")
  endforeach()
  math(EXPR value "0x${big_endian}" OUTPUT_FORMAT DECIMAL)
  set(${out} "${value}" PARENT_SCOPE)
endfunction()

if(NOT EXISTS "${TOOL}")
  message(FATAL_ERROR "no such file: ${TOOL}")
endif()
# if(<number> LESS <not a number>) is false, so a missing or misspelt WANT would pass every file.
if(NOT "${WANT}" MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "WANT must be a number of bytes greater than 0, got \"${WANT}\"")
endif()
read_le("${TOOL}" 0 2 dos_magic)
if(NOT dos_magic EQUAL 23117)        # "MZ"
  message(FATAL_ERROR "${TOOL} is not a PE file (no MZ header)")
endif()
read_le("${TOOL}" 60 4 pe_offset)    # e_lfanew
read_le("${TOOL}" ${pe_offset} 4 pe_signature)
if(NOT pe_signature EQUAL 17744)     # "PE\0\0"
  message(FATAL_ERROR "${TOOL} is not a PE file (no PE signature at ${pe_offset})")
endif()
math(EXPR optional_header "${pe_offset} + 24")
read_le("${TOOL}" ${optional_header} 2 optional_magic)
if(optional_magic EQUAL 523)         # PE32+: SizeOfStackReserve is 8 bytes
  set(width 8)
elseif(optional_magic EQUAL 267)     # PE32: 4 bytes
  set(width 4)
else()
  message(FATAL_ERROR "${TOOL}: unknown optional header magic ${optional_magic}")
endif()
math(EXPR reserve_offset "${optional_header} + 72")   # the same offset in PE32 and PE32+
read_le("${TOOL}" ${reserve_offset} ${width} reserve)

if(reserve LESS WANT)
  message(FATAL_ERROR "${TOOL} reserves ${reserve} bytes of stack, expected at least ${WANT}: the link option "
                      "from runtime/CMakeLists.txt (BUILD-1457) did not reach the linker")
endif()
message(STATUS "stack reserve: ${reserve} bytes (at least ${WANT} wanted)")
