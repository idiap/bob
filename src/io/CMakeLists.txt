project(io)

# HDF5
set(HDF5_FIND_COMPONENTS C CXX)
include(FindHDF5)

# netpbm
include(netpbm.cmake)

# libjpeg
include(FindJPEG)
if(JPEG_FOUND)
  set(HAVE_LIBJPEG ON CACHE BOOL "Has libjpeg installed")
else(JPEG_FOUND)
  message(FATAL_ERROR "libjpeg has not been found.")
endif(JPEG_FOUND)

# libpng
include(FindPNG)
if(PNG_FOUND)
  set(HAVE_LIBPNG ON CACHE BOOL "Has libpng installed")
else(PNG_FOUND)
  message(FATAL_ERROR "libpng has not been found.")
endif(PNG_FOUND)

#libtiff
include(FindTIFF)
if(TIFF_FOUND)
  set(HAVE_LIBTIFF ON CACHE BOOL "Has libtiff installed")
else(TIFF_FOUND)
  message(FATAL_ERROR "libtiff has not been found.")
endif(TIFF_FOUND) 

#giflib
include(FindGIF)
if(GIF_FOUND)
  set(HAVE_GIFLIB ON CACHE BOOL "Has giflib installed")
else(GIF_FOUND)
  message(FATAL_ERROR "giflib has not been found.")
endif(GIF_FOUND) 

# Matio
include(FindPkgConfig)
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

  # Resolve matio libraries to a precise path
  set(matio_RESOLVED_LIBRARIES "")
  foreach(lib ${matio_LIBRARIES})
    resolve_library(${lib} ${matio_LIBDIR} matio_RESOLVED_LIBRARIES)
  endforeach()
  set(matio_RESOLVED_LIBRARIES ${matio_RESOLVED_LIBRARIES} CACHE INTERNAL "Resolved matio libraries")
endif()

# FFMpeg -- Our base build requires ffmpeg >= 0.5. This is available on most
# platforms, but please note that if you link to anything <= 0.6, your code
# will become GPL'd. See table above for details.
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

  # Resolve FFMPEG libraries to a precise path
  set(FFMPEG_RESOLVED_LIBRARIES "")
  foreach(lib ${FFMPEG_LIBRARIES})
    resolve_library(${lib} "${FFMPEG_LIBRARY_DIRS}" FFMPEG_RESOLVED_LIBRARIES)
  endforeach()
  set(FFMPEG_RESOLVED_LIBRARIES ${FFMPEG_RESOLVED_LIBRARIES} CACHE INTERNAL "Resolved FFMPEG libraries")

  include(CheckCSourceCompiles)
  set(CMAKE_REQUIRED_INCLUDES "${FFMPEG_INCLUDE_DIRS}")
  set(CMAKE_REQUIRED_FLAGS "-L${FFMPEG_LIBRARY_DIRS}")
  set(CMAKE_REQUIRED_LIBRARIES "${FFMPEG_LIBRARIES}")
  CHECK_C_SOURCE_COMPILES("#include <libavformat/avformat.h>
  int main() { int res = avformat_alloc_output_context2(0, 0, 0, 0); }" HAVE_FFMPEG_AVFORMAT_ALLOC_OUTPUT_CONTEXT2)
  CHECK_C_SOURCE_COMPILES("#include <libavcodec/avcodec.h>
  int main() { enum AVCodecID test; }" HAVE_FFMPEG_AVCODEC_AVCODECID)
  set(CMAKE_REQUIRED_LIBRARIES)
  set(CMAKE_REQUIRED_FLAGS)
  set(CMAKE_REQUIRED_INCLUDES)

endif(FFMPEG_FOUND)

add_subdirectory(cxx)
add_subdirectory(python)

set(ENABLED_PACKAGES "${ENABLED_PACKAGES}" PARENT_SCOPE)
