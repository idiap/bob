# Configures and builds bob
# Andre Anjos - 13.august.2010

project(bob)
cmake_minimum_required(VERSION 2.6)

# All options for building bob
string(TOLOWER 
  "${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}-${CMAKE_BUILD_TYPE}" 
  BOB_PLATFORM_TMP)
option(BOB_PLATFORM 
  "The name of the platform Bob is being built to" ${BOB_PLATFORM_TMP})

option(BOB_VERSION 
  "The version of Bob that is currently being built" "unknown")

option(BUILD_CXX "Builds the C++ infrastructure." ON)

# Note you cannot build the Python stuff without the C++. This will work
# though, if you build the whole project and then just wants to rebuild the C++
# bit.
option(BUILD_PYTHON "Builds the python bindings." ON)

option(CREATE_DATABASES "Builds databases from scratch. If not set, the default, will download the databases from Bob servers instead." OFF)

option(DB_INSTALL_PREFIX "Installation directory for the databases, if they are to be copied somewhere during the install target." OFF)

option(DATABASE_URL "Where to download databases from." "http://www.idiap.ch/software/bob/databases/")

# TODO: This needs to be correctly set to the version supported by this build!
option(DATABASE_VERSION "Version of the databases to download" "nightlies/last")

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
foreach(ext ${bob_DEPENDENCIES})
  find_package(${ext} REQUIRED PATHS external)
  install(FILES external/${ext}Config.cmake DESTINATION ${cmakedir})
endforeach(ext ${bob_DEPENDENCIES})
foreach(ext ${bob_OPTIONALS})
  find_package(${ext} PATHS external)
  install(FILES external/${ext}Config.cmake DESTINATION ${cmakedir})
endforeach(ext ${bob_OPTIONALS})
install(FILES cmake/bob-external.cmake DESTINATION ${cmakedir})

# Enables the testing framework
enable_testing()

# Some common definitions
add_definitions("-DHAVE_LBFGS=1")

include(cmake/macros.cmake)

set(BUILD_SHARED_LIBS "ON" CACHE BOOL "Build shared libs")

if (BUILD_CXX)
  if (QT4_FOUND)
    add_library(bob_visioner SHARED IMPORTED)
  endif ()

  add_subdirectory(cxx)
endif ()

if (BUILD_PYTON)
  add_subdirectory(python)
endif ()

install(EXPORT bob DESTINATION ${cmakedir})
install(FILES cmake/bobConfig.cmake DESTINATION ${cmakedir})
