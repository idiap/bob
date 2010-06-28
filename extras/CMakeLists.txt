# AA 28.06.2010

# Sets up all extras in one go. 

cmake_minimum_required(VERSION 2.6)

# These are projects that can be constructed individually alright
add_subdirectory(jpeg)
add_subdirectory(oourafft)
add_subdirectory(lbfgs)

# The next lines will handle the ffmpeg dependency in a nice way.
# Please note that for find_package(xyz) succeed, you need to have the file
# "xyzConfig.cmake" available in the current directory or in CMake's module
# path.

# This is a summary of what we do:
# 1. If you define FFMPEG_TRY_LOCAL_FIRST, I try to find the pkg-config
# files for that package, in the standard system directories. If this search is # successful, this is stopped and the ffmpeg on the system is used.
# 2. If you don't define FFMPEG_TRY_LOCAL_FIRST, I try to download and build
# the ffmpeg stable branch and then require a pkg-config search on the newly
# created installation. That has to succeed or is a fatal cmake error.
# 3. If you define FFMPEG_TRY_LOCAL_FIRST and I don't find a local installation
# then what was written in (2) also applies.
include(Directories.cmake)
if(DEFINED FFMPEG_TRY_LOCAL_FIRST)
  find_package(ffmpeg REQUIRED) # searches for ffmpeg flags
  if(ffmpeg_FOUND)
    message("Pre-installed ffmpeg found. Using it!")
  else(ffmpeg_FOUND)
    add_subdirectory(ffmpeg) # triggers download/build of ffmpeg
    set(ENV{PKG_CONFIG_PATH} "${PKG_CONFIG_PATH}:${INSTALL_DIR}/lib/pkgconfig")
    find_package(ffmpeg REQUIRED) # searches for ffmpeg flags
  endif(ffmpeg_FOUND)
else(DEFINED FFMPEG_TRY_LOCAL_FIRST)
  add_subdirectory(ffmpeg) # triggers download/build of ffmpeg
  set(ENV{PKG_CONFIG_PATH} "${PKG_CONFIG_PATH}:${INSTALL_DIR}/lib/pkgconfig")
  find_package(ffmpeg REQUIRED) # searches for ffmpeg flags
endif(DEFINED FFMPEG_TRY_LOCAL_FIRST)

# Gives out a summary of the ffmpeg library versions we will be using.
message("(ffmpeg) libavformat: ${ffmpeg_libavformat_VERSION}")
message("(ffmpeg) libswscale: ${ffmpeg_libswscale_VERSION}")
message("(ffmpeg) libavutil: ${ffmpeg_libavutil_VERSION}")
