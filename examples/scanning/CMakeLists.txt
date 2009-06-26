#
# Top level project
#
PROJECT(EXAMPLE)

CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

# Set the include directories and linking libraries
SET(TORCH5SPRO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")
INCLUDE_DIRECTORIES("${TORCH5SPRO_DIR}")
LINK_LIBRARIES(	"${TORCH5SPRO_DIR}/build_${TORCH_OS}_${TORCH_ARCH}/libtorch5spro.a"
                "${TORCH5SPRO_DIR}/extras/jpeg/build_${TORCH_OS}_${TORCH_ARCH}/libjpeg.a")

# Set the compiler flags and build type (release or debug)
SET(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3 -ffast-math -fomit-frame-pointer -msse2")
#SET(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3 -ffast-math -fomit-frame-pointer -m32")
#SET(CMAKE_CXX_FLAGS_DEBUG "-g")
SET(CMAKE_BUILD_TYPE RELEASE)
#SET(CMAKE_BUILD_TYPE DEBUG)

MESSAGE("---------------------------------------------" "")
MESSAGE("CXX RELEASE FLAGS: " "${CMAKE_CXX_FLAGS_RELEASE}")
MESSAGE("CXX DEBUG FLAGS:   " "${CMAKE_CXX_FLAGS_DEBUG}")
MESSAGE("BUILD TYPE:        " "${CMAKE_BUILD_TYPE}")
MESSAGE("---------------------------------------------" "")

########################################################
# For each source file build an executable
########################################################

SET(EXAMPLE_SRC
	testPatternList.cc
	scanning.cc)

FOREACH(file ${EXAMPLE_SRC})
	STRING(REGEX REPLACE "([a-z;A-Z;0-9]+).*" "\\1" file_exe "${file}" )
	ADD_EXECUTABLE(${file_exe} ${file})
ENDFOREACH(file)

########################################################
