# Configures a remote bob installation
# Andre Anjos - 13.august.2010

# This file is installed on every bob build so *external* code can compile in
# a few easy steps. If you want to change the way bob itself is compiled,
# this is *not* the place.

# Locates and loads all bob exported dependencies
find_file(bob_BUILD_INFO bob.cmake)
include(${bob_BUILD_INFO})

# Defines the includes
get_filename_component(bob_CMAKE_DIR ${bob_BUILD_INFO} PATH)
get_filename_component(bob_SHARE_DIR ${bob_CMAKE_DIR} PATH)
get_filename_component(bob_PREFIX ${bob_SHARE_DIR} PATH)

# Loads all externals
find_file(bob_DEPENDENCIES_FILE bob-external.cmake)
include("${bob_DEPENDENCIES_FILE}")

set(bob_INCLUDE_DIRS ${bob_PREFIX}/include/bob)
set(bob_LIBRARY_DIRS ${bob_PREFIX}/lib)
foreach(dep ${bob_DEPENDENCIES})
  find_package(${dep})
  set(bob_INCLUDE_DIRS "${bob_INCLUDE_DIRS};${${dep}_INCLUDE_DIRS}")
  set(bob_LIBRARY_DIRS "${bob_LIBRARY_DIRS};${${dep}_LIBRARY_DIRS}")
endforeach(dep ${bob_DEPENDENCIES})
list(REMOVE_DUPLICATES bob_INCLUDE_DIRS)
list(REMOVE_DUPLICATES bob_LIBRARY_DIRS)

# This macro helps users to build bob-based executables
macro(bob_add_executable name sources dependencies)
  include_directories(${bob_INCLUDE_DIRS})
  link_directories(${bob_LIBRARY_DIRS})
  add_executable(${name} ${sources})
  foreach(dep ${dependencies})
    target_link_libraries(${name} bob_${dep})
  endforeach(dep ${dependencies})
endmacro(bob_add_executable name sources dependencies)
