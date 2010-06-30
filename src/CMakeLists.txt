#
# Top level project
#
PROJECT(TORCH5SPRO)

CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

# Set the compiler flags and build type(release or debug)
SET(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3 -ffast-math -fomit-frame-pointer ")
SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
SET(CMAKE_CXX_FLAGS_DEBUG "-g")
SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")

# Set the source files and includes to be added to the library
SET(TORCH5SPRO_SRC "" CACHE INTERNAL "sources")
SET(TORCH5SPRO_INCLUDE "" CACHE INTERNAL "included")

########################################################
# Additional libraries
########################################################

# CBLAS
ADD_DEFINITIONS(-DUSE_CBLAS)
INCLUDE_DIRECTORIES("/usr/include")

# JPEG
ADD_DEFINITIONS(-DHAVE_JPEG)
INCLUDE_DIRECTORIES("${INCLUDE_DIR}/jpeg")

# OOURA FFT
ADD_DEFINITIONS(-DHAVE_OOURAFFT)
INCLUDE_DIRECTORIES("${INCLUDE_DIR}/oourafft")

# LBFGS
ADD_DEFINITIONS(-DHAVE_LBFGS)
ADD_DEFINITIONS(-DUSE_SSE)
INCLUDE_DIRECTORIES("${INCLUDE_DIR}/lbfgs")

# TIFF
#ADD_DEFINITIONS(-DHAVE_TIFF)

# FFMPEG
# EDIT LES: Remove ffmpeg due to compilation error
INCLUDE_DIRECTORIES("${INCLUDE_DIR}")
INCLUDE_DIRECTORIES("${INCLUDE_DIR}/libavutil")
INCLUDE_DIRECTORIES("${INCLUDE_DIR}/libavformat")
INCLUDE_DIRECTORIES("${INCLUDE_DIR}/libavcodec")
INCLUDE_DIRECTORIES("${INCLUDE_DIR}/libswscale")
ADD_DEFINITIONS(-DHAVE_FFMPEG)
# End EDIT

########################################################
# Sub-directories with source code
########################################################

SET(TORCH5SPRO_SUBDIRS
	th
	core
  ip
	sp
	ap
	scanning
	machine
	trainer
	boosting
	measurer
	geometry)

SET(TORCH5SPRO_INCLUDE_DIR "${INCLUDE_DIR}/torch5spro")

FOREACH(subdir ${TORCH5SPRO_SUBDIRS})
	ADD_SUBDIRECTORY(${subdir})

	# Add the includes from the subdirectoris too
	SET(TORCH5SPRO_INCLUDE "${TORCH5SPRO_INCLUDE};${${subdir}_INCLUDE}")

	# Add the includes from the subdirectoris too
	SET(TORCH5SPRO_SRC "${TORCH5SPRO_SRC};${${subdir}_SRC}")

ENDFOREACH(subdir)


########################################################
# Build the library

# This is the library with the source files gathered
#	from the sub-directories specified above
INCLUDE_DIRECTORIES(${TORCH5SPRO_INCLUDE})
ADD_LIBRARY(torch5spro SHARED ${TORCH5SPRO_SRC})
ADD_LIBRARY(torch5spro-static STATIC ${TORCH5SPRO_SRC})
SET_TARGET_PROPERTIES(torch5spro-static PROPERTIES OUTPUT_NAME "torch5spro")
SET_TARGET_PROPERTIES(torch5spro-static PROPERTIES PREFIX "lib")
INSTALL(TARGETS torch5spro LIBRARY DESTINATION ${INSTALL_DIR}/lib)
INSTALL(TARGETS torch5spro-static ARCHIVE DESTINATION ${INSTALL_DIR}/lib)

# Prepare the main header
EXECUTE_PROCESS(COMMAND "../makeinclude.sh" ${INCLUDE_DIR}/torch5spro
			"TensorGen.h THTensorGen.h THStorageGen.h")

########################################################
# Make it verbose

MESSAGE("---------------------------------------------" "")
MESSAGE("\"Torch5\" configuration
MESSAGE("C flags:   " "${CMAKE_C_FLAGS_${CMAKE_BUILD_TYPE}}")
MESSAGE("C++ flags:   " "${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}")
MESSAGE("---------------------------------------------" "")

########################################################
