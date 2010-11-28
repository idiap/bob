# Configures and builds torch
# Andre Anjos - 13.august.2010

project(torch5spro)
cmake_minimum_required(VERSION 2.6)

# Force __LP64__ scheme on OSX
if((${CMAKE_SYSTEM_NAME} MATCHES "Darwin"))
  set(COMMON_FLAGS "-m64")
  # Bogus on OSX?
  set(CMAKE_INCLUDE_SYSTEM_FLAG_C "-isystem ")
  set(CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-isystem ")
endif((${CMAKE_SYSTEM_NAME} MATCHES "Darwin"))

# We allow you to force the compilation to be done using a base 32-bit
# architecture if you call cmake with "-DTORCH_FORCE_32BITS=yes"
if(TORCH_FORCE_32BITS)
  set(COMMON_FLAGS "-m32")
endif(TORCH_FORCE_32BITS)

# Set the compiler flags and build type(release or debug)
set(COMMON_FLAGS "${COMMON_FLAGS} -D__STDC_CONSTANT_MACROS -pthread -g -Wall -pedantic")

set(CMAKE_CXX_FLAGS_RELEASE "${COMMON_FLAGS} -O3 -march=native")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -funroll-loops")
set(CMAKE_CXX_FLAGS_DEBUG "${COMMON_FLAGS} -DTORCH_DEBUG")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_Fortran_FLAGS_DEBUG "${COMMON_FLAGS}")

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

# Setup BOOST pre-conditions
if ("${TORCH_LINKAGE}" STREQUAL static)
  set(Boost_USE_STATIC_LIBS ON)
endif ("${TORCH_LINKAGE}" STREQUAL static)

# Sets up externals and install the central dependency file
include(cmake/torch-external.cmake)
foreach(ext ${torch_DEPENDENCIES})
  find_package(${ext} REQUIRED PATHS external)
  install(FILES external/${ext}Config.cmake DESTINATION ${cmakedir})
endforeach(ext ${torch_DEPENDENCIES})
foreach(ext ${torch_OPTIONALS})
  find_package(${ext} PATHS external)
  install(FILES external/${ext}Config.cmake DESTINATION ${cmakedir})
endforeach(ext ${torch_OPTIONALS})
install(FILES cmake/torch-external.cmake DESTINATION ${cmakedir})

# Enables the testing framework
enable_testing()

# This makes sure we have all headers within reach before starting to compile
include_directories(${CMAKE_INSTALL_PREFIX}/include/torch)

add_subdirectory(python)
add_subdirectory(dfftpack41)
add_subdirectory(lbfgs)
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
