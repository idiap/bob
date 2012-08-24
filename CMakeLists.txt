# Configures and builds bob
# Andre Anjos - 13.august.2010

project(bob)
cmake_minimum_required(VERSION 2.6)

# -----------------
# START of options.
# -----------------

if (NOT CMAKE_BUILD_TYPE)
  message(FATAL_ERROR "error: choose either Release or Debug as build type by specifying -DCMAKE_BUILD_TYPE=Release or -DCMAKE_BUILD_TYPE=Debug on your cmake command line.")
endif ()

set(BOB_SYSTEM_PROCESSOR ${CMAKE_SYSTEM_PROCESSOR})
set(BOB_SYSTEM_NAME ${CMAKE_SYSTEM_NAME})
if(APPLE)
  set(BOB_SYSTEM_PROCESSOR "x86_64") #forced, see __LP64__ comment bellow
  set(BOB_SYSTEM_NAME "macosx")
endif()

string(TOLOWER "${BOB_SYSTEM_NAME}-${BOB_SYSTEM_PROCESSOR}-${CMAKE_BUILD_TYPE}" BOB_PLATFORM_STR)
set(BOB_PLATFORM "${BOB_PLATFORM_STR}" CACHE STRING "The name of the platform Bob is being built to")

if(BOB_BUILD_COUNTER)
  execute_process(COMMAND ${CMAKE_SOURCE_DIR}/bin/version.py --counter=${BOB_BUILD_COUNTER} OUTPUT_VARIABLE BOB_DISCOVERED_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
else()
  execute_process(COMMAND ${CMAKE_SOURCE_DIR}/bin/version.py OUTPUT_VARIABLE BOB_DISCOVERED_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
set(BOB_VERSION_INTERNAL "${BOB_DISCOVERED_VERSION}" CACHE INTERNAL "The version of Bob that is currently being built")
if(NOT BOB_VERSION)
  set(BOB_VERSION ${BOB_VERSION_INTERNAL})
endif()

set(BOB_SOVERSION_INTERNAL "OFF" CACHE INTERNAL "If we need to use a SOVERSION
on libraries, this should set it")
if(NOT BOB_SOVERSION)
  set(BOB_SOVERSION ${BOB_SOVERSION_INTERNAL})
endif()

option(SET_PUBLIC_LIBRARY_PATH "Use `ld -rpath' (Linux) or `install_name_tool' (Apple) when linking public libraries, executables. This option does not cover python bindings to C++ installed at the Python directories." ON)

option(BOB_BUILD_CXX "Builds the C++ infrastructure." ON)

option(BOB_INSTALL_PYTHON_INTERPRETER "When enabled, installs a version of the python interpreter preset to prepend Bob's environment." OFF)

# Note you cannot build the Python stuff without the C++. This will work
# though, if you build the whole project and then just wants to rebuild the C++
# bit.
option(BOB_BUILD_PYTHON "Builds the python bindings." ON)

# ---------------
# END of options.
# ---------------

message(STATUS "Bob version '${BOB_VERSION}' (${BOB_PLATFORM_STR})")

# Flag setting

# For both C and C++
set(COMMON_FLAGS "-pedantic -Wall -pthread")
if (NOT CMAKE_COMPILER_IS_GNUCC)
  # Then, it must be clang/clang++
  set(COMMON_FLAGS "-Qunused-arguments")
endif ()

# Force __LP64__ scheme on Mac OSX
if(APPLE)
  set(COMMON_FLAGS "${COMMON_FLAGS} -m64")
  # Bogus on OSX?
  set(CMAKE_INCLUDE_SYSTEM_FLAG_C "-isystem ")
  set(CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-isystem ")
endif(APPLE)

# For both RELEASE and DEBUG builds
set(COMMON_CXX_FLAGS "-std=c++0x")
set(COMMON_C_FLAGS "-std=c99")

# For specific builds
set(COMMON_RELEASE_FLAGS "-O2 -mtune=generic -DNDEBUG")
set(COMMON_DEBUG_FLAGS "-g -DBOB_DEBUG -DBZ_DEBUG")

# Cache into CMake
set(CMAKE_CXX_FLAGS_RELEASE ${COMMON_RELEASE_FLAGS} CACHE STRING "Flags used by the compiler during release builds" FORCE)
set(CMAKE_C_FLAGS_RELEASE ${COMMON_RELEASE_FLAGS} CACHE STRING "Flags used by the compiler during release builds" FORCE)
set(CMAKE_CXX_FLAGS_DEBUG ${COMMON_DEBUG_FLAGS} CACHE STRING "Flags used by the compiler during debug builds." FORCE)
set(CMAKE_C_FLAGS_DEBUG ${COMMON_DEBUG_FLAGS} CACHE STRING "Flags used by the compiler during debug builds." FORCE)
set(CMAKE_CXX_FLAGS "${COMMON_CXX_FLAGS} ${COMMON_FLAGS}" CACHE STRING "Flags used by the compiler during all build types." FORCE)
set(CMAKE_C_FLAGS "${COMMON_C_FLAGS} ${COMMON_FLAGS}" CACHE STRING "Flags used by the compiler during all build types." FORCE)

if(SET_PUBLIC_LIBRARY_PATH)
  if (APPLE)
    set(CMAKE_INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib")
  else()
    # This is our RPATH policy under Linux
    set(CMAKE_SKIP_BUILD_RPATH  FALSE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  endif()
endif()

# This will make sure we consider pkg-config directories that may have been
# added when the user set CMAKE_PREFIX_PATH.
if(CMAKE_PREFIX_PATH)
  string(REPLACE ":" ";" PKG_CONFIG_CHECK_PATH ${CMAKE_PREFIX_PATH})
  foreach (ext ${PKG_CONFIG_CHECK_PATH})
    if(EXISTS ${ext}/lib/pkgconfig)
      set(ENV{PKG_CONFIG_PATH} "${ext}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
    endif()
  endforeach()
endif()

# Sets up externals and install the central dependency file
include(cmake/bob-external.cmake)

foreach (ext ${bob_DEPENDENCIES})
  find_package(${ext} REQUIRED PATHS external)
endforeach ()

foreach (ext ${bob_OPTIONALS})
  find_package(${ext} PATHS external)
endforeach ()

# This is where we install cmake files for external projects
set(cmakedir share/cmake/bob)
install(DIRECTORY external/ DESTINATION ${cmakedir} PATTERN "*.cmake")
install (FILES cmake/bob-external.cmake DESTINATION ${cmakedir})

# Enables the testing framework
enable_testing()

# Some common definitions
add_definitions("-DHAVE_LBFGS=1")

include(cmake/macros.cmake)

set(BUILD_SHARED_LIBS "ON" CACHE BOOL "Build shared libs")

if (BOB_BUILD_CXX)
  add_subdirectory(cxx)
endif ()

if (BOB_BUILD_PYTHON)
  add_subdirectory(python)
endif ()

install(EXPORT bob DESTINATION ${cmakedir})
install(FILES cmake/bobConfig.cmake DESTINATION ${cmakedir})

# Adds Doxygen and Sphinx generation
include(cmake/documentation.cmake)
