# Configures a remote torch installation
# Andre Anjos - 13.august.2010

# This file is installed on every torch build so *external* code can compile in
# a few easy steps. If you want to change the way torch itself is compiled,
# this is *not* the place.

# Locates and loads all torch exported dependencies
find_file(torch_BUILD_INFO torch.cmake)
include(${torch_BUILD_INFO})

# Defines the includes
get_filename_component(torch_CMAKE_DIR ${torch_BUILD_INFO} PATH)
get_filename_component(torch_SHARE_DIR ${torch_CMAKE_DIR} PATH)
get_filename_component(torch_PREFIX ${torch_SHARE_DIR} PATH)

# Loads all externals
find_file(torch_DEPENDENCIES_FILE torch-external.cmake)
include("${torch_DEPENDENCIES_FILE}")

set(torch_INCLUDE_DIRS ${torch_PREFIX}/include/torch)
set(torch_LIBRARY_DIRS ${torch_PREFIX}/lib)
foreach(dep ${torch_DEPENDENCIES})
  find_package(${dep})
  set(torch_INCLUDE_DIRS "${torch_INCLUDE_DIRS};${${dep}_INCLUDE_DIRS}")
  set(torch_LIBRARY_DIRS "${torch_LIBRARY_DIRS};${${dep}_LIBRARY_DIRS}")
endforeach(dep ${torch_DEPENDENCIES})
list(REMOVE_DUPLICATES torch_INCLUDE_DIRS)
list(REMOVE_DUPLICATES torch_LIBRARY_DIRS)

# This macro helps users to build torch-based executables
macro(torch_add_executable name sources dependencies)
  include_directories(${torch_INCLUDE_DIRS})
  link_directories(${torch_LIBRARY_DIRS})
  add_executable(${name} ${sources})
  foreach(dep ${dependencies})
    target_link_libraries(${name} torch_${dep})
  endforeach(dep ${dependencies})
endmacro(torch_add_executable name sources dependencies)
