# Run the stand-in game from its own folder by bare name, as a player would (`gta6rt --run game.exe`), and again
# with `--game-dir .`. In both, GAME: has to name that folder as an absolute path: the guest moves its current
# directory away and then reads GAME:\tiny_game_probe.txt (argument game-file). Before, a bare name left GAME: with
# an empty root (`console paths: GAME: -> , writes -> overlay`) and the open failed with ERROR_PATH_NOT_FOUND.
# The verdict is checked from the output, as check_loaded_gdk.cmake does, because under Wine an exit status alone
# can look successful.
get_filename_component(guest_dir "${GUEST}" DIRECTORY)
get_filename_component(guest_name "${GUEST}" NAME)
file(WRITE "${guest_dir}/tiny_game_probe.txt" "game")
set(failures "")
foreach(variant "bare" "dot")
  if(variant STREQUAL "dot")
    set(dir_args --game-dir .)
  else()
    set(dir_args "")
  endif()
  execute_process(
    COMMAND ${RUNNER} "${LOADER}" ${dir_args} --run "${guest_name}" -- hello 42 game-file
    WORKING_DIRECTORY "${guest_dir}"
    TIMEOUT 90
    RESULT_VARIABLE code OUTPUT_VARIABLE output ERROR_VARIABLE output)
  string(REPLACE "\r\n" "\n" output "${output}")
  # An absolute GAME: root: a drive letter (Windows, and Wine's Z:) or a leading slash, and an overlay inside it.
  if(NOT code STREQUAL "0" OR NOT output MATCHES "(^|\n)tiny_game: PASS GAME:"
      OR NOT output MATCHES "(^|\n)tiny_game: OK(\n|$)"
      OR NOT output MATCHES "(^|\n)console paths: GAME: -> ([A-Za-z]:|[\\/])[^\n]*, writes -> ([A-Za-z]:|[\\/])[^\n]*overlay(\n|$)"
      OR NOT output MATCHES "(^|\n)game exited with code 0(\n|$)")
    string(APPEND failures "--- ${variant}: returned ${code}\n${output}\n")
  endif()
endforeach()
file(REMOVE "${guest_dir}/tiny_game_probe.txt")
if(failures)
  message(FATAL_ERROR "GAME: did not resolve to the game folder\n${failures}")
endif()
