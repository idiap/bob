# Configures and builds bob
# Andre Anjos - 13.august.2010

project(bob)
cmake_minimum_required(VERSION 2.6)

# -----------------
# START of options.
# -----------------

if (NOT CMAKE_BUILD_TYPE)
  message(FATAL_ERROR "error: choose either RELEASE or DEBUG as build type by specifying -DCMAKE_BUILD_TYPE=RELEASE or -DCMAKE_BUILD_TYPE=DEBUG on your cmake command line.")
endif ()

set(BOB_SYSTEM_PROCESSOR ${CMAKE_SYSTEM_PROCESSOR})
set(BOB_SYSTEM_NAME ${CMAKE_SYSTEM_NAME})
if(APPLE)
  set(BOB_SYSTEM_PROCESSOR "x86_64") #forced, see __LP64__ comment bellow
  set(BOB_SYSTEM_NAME "macosx")
endif()

string(TOLOWER "${BOB_SYSTEM_NAME}-${BOB_SYSTEM_PROCESSOR}-${CMAKE_BUILD_TYPE}" BOB_PLATFORM_STR)
set(BOB_PLATFORM "${BOB_PLATFORM_STR}" CACHE ON "The name of the platform Bob is being built to")

set(BOB_VERSION "unknown" CACHE ON "The version of Bob that is currently being built")

# TODO: This needs to be correctly set to the version supported by this build!
set(BOB_DATABASE_VERSION "nightlies/last" CACHE ON "Version of the databases to download")

option(BOB_NON_ROOT_INSTALL "Generates python scripts keeping in mind the installation tree will not be sitting on a place automatically searched by python." OFF)

option(BOB_BUILD_CXX "Builds the C++ infrastructure." ON)

# Note you cannot build the Python stuff without the C++. This will work
# though, if you build the whole project and then just wants to rebuild the C++
# bit.
option(BOB_BUILD_PYTHON "Builds the python bindings." ON)

option(BOB_CREATE_DATABASES "Builds databases from scratch. If not set, the default, will download the databases from Bob servers instead." OFF)

set(BOB_DB_INSTALL_PREFIX "" CACHE ON "Installation directory for the databases, if they are to be copied somewhere during the install target.")

set(BOB_DATABASE_URL "http://www.idiap.ch/software/bob/databases/" CACHE ON "Where to download databases from.")

# ---------------
# END of options.
# ---------------

# Force __LP64__ scheme on Mac OSX
if(APPLE)
  set(COMMON_FLAGS "-m64")
  # Bogus on OSX?
  set(CMAKE_INCLUDE_SYSTEM_FLAG_C "-isystem ")
  set(CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-isystem ")
endif(APPLE)

set(COMMON_FLAGS "${COMMON_FLAGS} -pedantic -Wno-variadic-macros -Wno-long-long -Wall")

if (CMAKE_COMPILER_IS_GNUCC)

  # Set the compiler flags and build type(release or debug)
  set(COMMON_FLAGS "${COMMON_FLAGS} -pthread")

  set(CMAKE_CXX_FLAGS_RELEASE "-O2 -mtune=generic -DNDEBUG")

  set(CMAKE_CXX_FLAGS_DEBUG "-g -DBOB_DEBUG -DBZ_DEBUG")

else (CMAKE_COMPILER_IS_GNUCC)

  # Then, it must be clang/clang++
  set(CMAKE_CXX_FLAGS_RELEASE "-O2 -mtune=generic -DNDEBUG")

  set(CMAKE_CXX_FLAGS_DEBUG "-g -DBOB_DEBUG -DBZ_DEBUG")

endif (CMAKE_COMPILER_IS_GNUCC)

set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE} CACHE STRING "Flags used by the compiler during release builds" FORCE)
set(CMAKE_C_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE} CACHE STRING "Flags used by the compiler during release builds" FORCE)
set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG} CACHE STRING "Flags used by the compiler during debug builds." FORCE)
set(CMAKE_C_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG} CACHE STRING "Flags used by the compiler during debug builds." FORCE)
set(CMAKE_CXX_FLAGS ${COMMON_FLAGS} CACHE STRING "Flags used by the compiler during all build types." FORCE)
set(CMAKE_C_FLAGS ${COMMON_FLAGS} CACHE STRING "Flags used by the compiler during all build types." FORCE)

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

# Sets up externals and install the central dependency file
include(cmake/bob-external.cmake)

foreach (ext ${bob_DEPENDENCIES})
  find_package(${ext} REQUIRED PATHS external)
  install(FILES external/${ext}Config.cmake DESTINATION ${cmakedir})
endforeach ()

foreach (ext ${bob_OPTIONALS})
  find_package(${ext} PATHS external)
  install(FILES external/${ext}Config.cmake DESTINATION ${cmakedir})
endforeach ()

install (FILES cmake/bob-external.cmake DESTINATION ${cmakedir})

# Enables the testing framework
enable_testing()

# Some common definitions
add_definitions("-DHAVE_LBFGS=1")

include(cmake/macros.cmake)

set(BUILD_SHARED_LIBS "ON" CACHE BOOL "Build shared libs")

if (BOB_BUILD_CXX)
  if (QT4_FOUND)
    add_library(bob_visioner SHARED IMPORTED)
  endif ()

  add_subdirectory(cxx)
endif ()

if (BOB_BUILD_PYTHON)
  add_subdirectory(python)
endif ()

install(EXPORT bob DESTINATION ${cmakedir})
install(FILES cmake/bobConfig.cmake DESTINATION ${cmakedir})
