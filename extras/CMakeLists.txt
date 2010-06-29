# Configures all extras in one shot. 
# Andre Anjos - 28.june.2010

# Sets up all extras in one go. 
project(extras)
cmake_minimum_required(VERSION 2.8)

# The type of build we aim for, options we use are RELEASE/DEBUG, but you 
# can have a look here:
# http://www.cmake.org/cmake/help/cmake-2-8-docs.html#variable:CMAKE_BUILD_TYPE
# to check for more.
set(CMAKE_BUILD_TYPE RELEASE)

# These are the C flags we like to use with the CMake projects we have here.
set(CMAKE_C_FLAGS_RELEASE "-Wall -g -O3 -ffast-math -fomit-frame-pointer -msse2")
# For 32-bits generation on a 64-bit architecture, uncomment this:
#set(CMAKE_C_FLAGS_RELEASE "-m32 -Wall -g -O3 -ffast-math -fomit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
set(CMAKE_C_FLAGS_DEBUG "-Wall -g")
set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})

# These will define some standard installation paths
set(PREFIX ${extras_SOURCE_DIR})
execute_process(COMMAND uname -s OUTPUT_VARIABLE unames)
execute_process(COMMAND uname -m OUTPUT_VARIABLE unamem)
set(PLATFORM "${unames}-${unamem}-${CMAKE_BUILD_TYPE}")
string(REGEX REPLACE "\n" "" PLATFORM ${PLATFORM})
string(TOLOWER ${PLATFORM} PLATFORM)
set(INSTALL_DIR "${PREFIX}/${PLATFORM}")
set(INCLUDE_DIR "${PREFIX}/include")

########################################################
# Make it verbose

message("---------------------------------------------")
message("\"extras\" configuration")
message("Build type: ${PLATFORM}")
message("C flags: ${CMAKE_C_FLAGS_${CMAKE_BUILD_TYPE}}")
message("C++ flags: ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}")
message("Install Prefix: ${PREFIX}")
message("Install Directory: ${INSTALL_DIR}")
message("Include Directory: ${INCLUDE_DIR}")
message("---------------------------------------------")

########################################################
# These are projects that can be constructed individually alright
add_subdirectory(jpeg)
add_subdirectory(oourafft)
add_subdirectory(lbfgs)

# This will handle the build of ffmpeg as a external project
macro(build_ffmpeg version branch)
  include(ExternalProject)
  if (${CMAKE_BUILD_TYPE} EQUAL "DEBUG")
    set(FFMPEG_DEBUG "--disable-optimizations")
  endif (${CMAKE_BUILD_TYPE} EQUAL "DEBUG")
  ExternalProject_Add(
    ffmpeg
    PREFIX "ffmpeg-${version}"
    SVN_REPOSITORY svn://svn.mplayerhq.hu/ffmpeg/${branch}
    UPDATE_COMMAND svn update .
    CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=<INSTALL_DIR> --incdir=${INCLUDE_DIR} --disable-zlib --disable-bzlib --disable-ffmpeg --disable-ffserver --disable-ffplay --enable-gpl --enable-shared ${FFMPEG_DEBUG}
    INSTALL_DIR ${INSTALL_DIR}
  )
endmacro(build_ffmpeg)

build_ffmpeg("0.6" "branches/0.6")
