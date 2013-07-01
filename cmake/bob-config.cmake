# Andre Anjos <andre.anjos@idiap.ch>
# Sat  1 Sep 22:04:04 2012 CEST
# <patch> Flavio Tarsetti <Tarsetti.Flavio@gmail.com>
# Mon  1 Jul 10:00:02 2013 CEST

# Support settings for external code built on Bob-cxx

get_filename_component(SELF_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
include(${SELF_DIR}/bob-targets.cmake)
get_filename_component(bob_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}../../../include" ABSOLUTE)
get_filename_component(bob_LIBRARY_DIRS "${CMAKE_CURRENT_LIST_DIR}../../../lib" ABSOLUTE)
