#
# Top level project
#
PROJECT(EXAMPLE)

CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

# Set the include directories and linking libraries
SET(TORCH5SPRO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")
SET(FFMPEG_DIR "/usr/")

INCLUDE_DIRECTORIES("${TORCH5SPRO_DIR}")

LINK_LIBRARIES(	"${TORCH5SPRO_DIR}/build_${TORCH_OS}_${TORCH_ARCH}/libtorch5spro.a"
		"${TORCH5SPRO_DIR}/extras/jpeg/build_${TORCH_OS}_${TORCH_ARCH}/libjpeg.a"
		"${FFMPEG_DIR}/lib/libavutil.so"
		"${FFMPEG_DIR}/lib/libavformat.so"
		"${FFMPEG_DIR}/lib/libavcodec.so"
		"${FFMPEG_DIR}/lib/libswscale.so")

# Set the compiler flags and build type (release or debug)
SET(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
SET(CMAKE_CXX_FLAGS_DEBUG "-g")
#SET(CMAKE_BUILD_TYPE RELEASE)
SET(CMAKE_BUILD_TYPE DEBUG)

MESSAGE("---------------------------------------------" "")
MESSAGE("CXX RELEASE FLAGS: " "${CMAKE_CXX_FLAGS_RELEASE}")
MESSAGE("CXX DEBUG FLAGS:   " "${CMAKE_CXX_FLAGS_DEBUG}")
MESSAGE("BUILD TYPE:        " "${CMAKE_BUILD_TYPE}")
MESSAGE("---------------------------------------------" "")

########################################################
# For each source file build an executable
########################################################

SET(EXAMPLE_SRC
	testio.cc)

FOREACH(file ${EXAMPLE_SRC})
	STRING(REGEX REPLACE "([a-z;A-Z;0-9]+).*" "\\1" file_exe "${file}" )
	ADD_EXECUTABLE(${file_exe} ${file})
ENDFOREACH(file)

########################################################
