# Configures all extras in one shot. 
# Andre Anjos - 28.june.2010

# Sets up all extras in one go. 
project(extras)
cmake_minimum_required(VERSION 2.8)

# These are the C flags we like to use with the CMake projects we have here.
set(CMAKE_C_FLAGS_RELEASE "-Wall -g -O3 -ffast-math -fomit-frame-pointer -msse2")
# For 32-bits generation on a 64-bit architecture, uncomment this:
#set(CMAKE_C_FLAGS_RELEASE "-m32 -Wall -g -O3 -ffast-math -fomit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
set(CMAKE_C_FLAGS_DEBUG "-Wall -g")
set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})

########################################################
# Make it verbose

message("---------------------------------------------")
message("\"extras\" configuration")
message("C flags: ${CMAKE_C_FLAGS_${CMAKE_BUILD_TYPE}}")
message("C++ flags: ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}")
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
