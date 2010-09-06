# Configures and builds torch
# Andre Anjos - 13.august.2010

project(torch5spro)
cmake_minimum_required(VERSION 2.6)

# Set the compiler flags and build type(release or debug)
set(COMMON_FLAGS "-D__STDC_CONSTANT_MACROS -pthread -g -Wall")

# Cmake does not support include_directories(SYSTEM) correctly on OSX, so the
# compiler issues warnings/errors on Python and ffmpeg includes that do not 
# respect a strict standard. To make it compile, we do not use "-pedantic" on
# OSX.
if(NOT (${CMAKE_SYSTEM_NAME} MATCHES "Darwin"))
  set(COMMON_FLAGS "${COMMON_FLAGS} -pedantic")
endif(NOT (${CMAKE_SYSTEM_NAME} MATCHES "Darwin"))

set(CMAKE_CXX_FLAGS_RELEASE "${COMMON_FLAGS} -O3 -march=native")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_DEBUG "${COMMON_FLAGS}")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")

# This is our RPATH policy
# use, i.e. don't skip the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
# when building, don't use the install RPATH already
# (but later on when installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 
# the RPATH to be used when installing
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# This is where we install cmake files for external projects
set(cmakedir share/cmake)

# Sets up externals and install
include(cmake/torch-external.cmake)
foreach(ext ${torch_DEPENDENCIES}) 
  find_package(${ext} REQUIRED PATHS external)
  install(FILES external/${ext}Config.cmake DESTINATION ${cmakedir})
endforeach(ext ${torch_DEPENDENCIES}) 
install(FILES cmake/torch-external.cmake DESTINATION ${cmakedir})

# Enables the testing framework
enable_testing()

# This makes sure we can reach all external headers
if(NOT ("$ENV{CMAKE_PREFIX_PATH}" STREQUAL ""))
  string(REGEX MATCHALL "[^:]+" prefix_path $ENV{CMAKE_PREFIX_PATH})
  list(REMOVE_DUPLICATES prefix_path)
  foreach(prefix ${prefix_path})
    include_directories(SYSTEM "${prefix}/include")
    link_directories("${prefix}/lib")
  endforeach(prefix ${prefix_path})
endif(NOT ("$ENV{CMAKE_PREFIX_PATH}" STREQUAL ""))

# This makes sure we have all headers within reach before starting to compile
include_directories(${CMAKE_INSTALL_PREFIX}/include/torch)

add_subdirectory(python)
add_subdirectory(oourafft)
add_subdirectory(lbfgs)
add_subdirectory(th)
add_subdirectory(core)
add_subdirectory(sp)
add_subdirectory(ap)
add_subdirectory(ip)
add_subdirectory(measurer)
add_subdirectory(machine)
add_subdirectory(scanning)
add_subdirectory(trainer)

install(EXPORT torch DESTINATION ${cmakedir})
install(FILES cmake/torchConfig.cmake DESTINATION ${cmakedir})
