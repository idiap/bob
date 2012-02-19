# Tries to find a local version of ffmpeg/developement installed
# Andre Anjos - 25.june.2010

include(FindPkgConfig)

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
  link_directories(${FFMPEG_LIBRARY_DIRS})
  add_definitions("-DHAVE_FFMPEG=1")

  find_program(FFMPEG_BINARY ffmpeg)

  if(FFMPEG_BINARY)
    # Setup the FFMPEG "official version"
    execute_process(COMMAND ${CMAKE_SOURCE_DIR}/bin/ffmpeg-version.sh ${FFMPEG_BINARY} ${FFMPEG_LIBRARY_DIRS} OUTPUT_VARIABLE FFMPEG_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
  else()
    set(FFMPEG_VERSION "unknown-version")
  endif()

  add_definitions("-DFFMPEG_VERSION=\"${FFMPEG_VERSION}\"")

  find_package_message(FFMPEG "Found FFmpeg ${FFMPEG_VERSION}" "[${FFMPEG_LIBRARY_DIRS}][${FFMPEG_VERSION}]")

endif(FFMPEG_FOUND)
