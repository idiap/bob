project(io)

# HDF5
set(HDF5_FIND_COMPONENTS C CXX)
include(FindHDF5)

# ImageMagick
include(FindPkgConfig)
pkg_check_modules(ImageMagick REQUIRED ImageMagick++>=6.5)

# Matio
pkg_check_modules(matio matio)
if(matio_FOUND)
  set(HAVE_MATIO ON CACHE BOOL "Has matio installed")

  include(CheckCSourceCompiles)
  set(CMAKE_REQUIRED_INCLUDES "${matio_INCLUDEDIR}")
  set(CMAKE_REQUIRED_FLAGS "-L${matio_LIBDIR}")
  set(CMAKE_REQUIRED_LIBRARIES "${matio_LIBRARIES}")
  CHECK_C_SOURCE_COMPILES("#include <matio.h> 
    int main() { struct ComplexSplit s; }" HAVE_MATIO_OLD_COMPLEXSPLIT)
  set(CMAKE_REQUIRED_LIBRARIES)
  set(CMAKE_REQUIRED_FLAGS)
  set(CMAKE_REQUIRED_INCLUDES)
endif()

# FFMpeg
# These are the versions of the several libraries shipped with ffmpeg. You may
# have to tune your requirements according to you needs.
#
# ffmpeg | avformat | avcodec | avutil  | swscale | old style | swscale GPL?
# =======+==========+=========+=========+=========+===========+==============
# 0.5    | 52.31.0  | 52.20.0 | 49.15.0 | 0.7.1   | yes       | yes
# 0.5.1  | 52.31.0  | 52.20.1 | 49.15.0 | 0.7.1   | yes       | yes
# 0.5.2  | 52.31.0  | 52.20.1 | 49.15.0 | 0.7.1   | yes       | yes
# 0.5.3  | 52.31.0  | 52.20.1 | 49.15.0 | 0.7.1   | yes       | yes
# 0.6    | 52.64.2  | 52.72.2 | 50.15.1 | 0.11.0  | no        | no
# 0.6.1  | 52.64.2  | 52.72.2 | 50.15.1 | 0.11.0  | no        | no
# 0.7.1  | 52.122.0 | 52.110.0| 50.43.0 | 0.14.1  | no        | no
# trunk  | 53.4.0   | 53.7.0  | 51.10.0 | 2.0.0   | no        | no

# Our base build requires ffmpeg >= 0.5. This is available on most platforms,
# but please note that if you link to anything <= 0.6, your code will become
# GPL'd. See table above for details.
pkg_check_modules(FFMPEG libavformat>=52.31.0 libavcodec>=52.20.0 libavutil>=49.15.0 libswscale>=0.7.1)

if(FFMPEG_FOUND)
  set(HAVE_FFMPEG ON CACHE BOOL "Has ffmpeg installed")

  find_program(FFMPEG_BINARY ffmpeg)

  if(FFMPEG_BINARY)
    # Setup the FFMPEG "official version"
    execute_process(COMMAND ${CMAKE_SOURCE_DIR}/bin/ffmpeg-version.sh ${FFMPEG_BINARY} ${FFMPEG_LIBRARY_DIRS} OUTPUT_VARIABLE FFMPEG_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
  else()
    set(FFMPEG_VERSION "unknown")
  endif()

  find_package_message(FFMPEG "Found FFmpeg ${FFMPEG_VERSION}" "[${FFMPEG_LIBRARY_DIRS}][${FFMPEG_VERSION}]")

  #set(FFMPEG_VERSION ${FFMPEG_VERSION} CACHE STRING "FFmpeg version")
  set(FFMPEG_VERSION "${FFMPEG_VERSION}" CACHE INTERNAL "FFmpeg version")

endif(FFMPEG_FOUND)

add_subdirectory(cxx)
add_subdirectory(python)
 
set(ENABLED_PACKAGES "${ENABLED_PACKAGES}" PARENT_SCOPE)
