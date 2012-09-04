# Andre Anjos <andre.anjos@idiap.ch>
# Sat  1 Sep 22:04:04 2012 CEST

# Support settings for external code built on Bob-cxx

get_filename_component(SELF_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
include(${SELF_DIR}/bob-targets.cmake)
get_filename_component(bob_INCLUDE_DIRS "${SELF_DIR}/../../include" ABSOLUTE)
get_filename_component(bob_LIBRARY_DIRS "${SELF_DIR}/../../lib" ABSOLUTE)
