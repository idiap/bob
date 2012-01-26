# Configures and builds bob
# Andre Anjos - 13.august.2010

project(bob)
cmake_minimum_required(VERSION 2.6)

# The central database version and location
set(BOB_DATABASE_URL "http://www.idiap.ch/software/bob/chrome/site/databases/")
set(BOB_DATABASE_VERSION "nightlies/last")

# Force __LP64__ scheme on Mac OSX
if(APPLE)
  set(COMMON_FLAGS "-m64")
  # Bogus on OSX?
  set(CMAKE_INCLUDE_SYSTEM_FLAG_C "-isystem ")
  set(CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-isystem ")
endif(APPLE)

set(COMMON_FLAGS "${COMMON_FLAGS} -D__STDC_CONSTANT_MACROS -g -Wall -pedantic")

if (CMAKE_COMPILER_IS_GNUCC)

  # Set the compiler flags and build type(release or debug)
  set(COMMON_FLAGS "${COMMON_FLAGS} -pthread")

  set(CMAKE_CXX_FLAGS_RELEASE "${COMMON_FLAGS} -Wno-variadic-macros -Wno-long-long -O2 -mtune=generic")

  set(CMAKE_CXX_FLAGS_DEBUG "${COMMON_FLAGS} -Wno-variadic-macros -Wno-long-long -DBOB_DEBUG -DBZ_DEBUG")

  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
  set(CMAKE_C_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")

else (CMAKE_COMPILER_IS_GNUCC)

  # Then, it must be clang/clang++
  set(CMAKE_CXX_FLAGS_RELEASE "${COMMON_FLAGS} -Wno-variadic-macros -Wno-long-long -O2 -mtune=generic")

  set(CMAKE_CXX_FLAGS_DEBUG "${COMMON_FLAGS} -Wno-variadic-macros -Wno-long-long -DBOB_DEBUG -DBZ_DEBUG")

  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
  set(CMAKE_C_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")

endif (CMAKE_COMPILER_IS_GNUCC)

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
if ("${BOB_LINKAGE}" STREQUAL static)
  set(Boost_USE_STATIC_LIBS ON)
endif ("${BOB_LINKAGE}" STREQUAL static)

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

# This makes sure we have all headers within reach before starting to compile
include_directories(${CMAKE_INSTALL_PREFIX}/include/bob)

# Some common definitions
add_definitions("-DHAVE_LBFGS=1")

if (BOB_CXX)
  add_subdirectory(cxx)
endif (BOB_CXX)
if (BOB_PYTHON)
  add_subdirectory(python)
endif (BOB_PYTHON)

install(EXPORT bob DESTINATION ${cmakedir})
install(FILES cmake/bobConfig.cmake DESTINATION ${cmakedir})
