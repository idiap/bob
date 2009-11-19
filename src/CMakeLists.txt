#
# Top level project
#
PROJECT(TORCH5SPRO)

CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

# Set the compiler flags and build type(release or debug)
SET(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3 -ffast-math -fomit-frame-pointer ")
SET(CMAKE_CXX_FLAGS_DEBUG "-g")
#SET(CMAKE_BUILD_TYPE RELEASE)
SET(CMAKE_BUILD_TYPE DEBUG)

# Set the source files and includes to be added to the library
SET(TORCH5SPRO_SRC "" CACHE INTERNAL "sources")
SET(TORCH5SPRO_INCLUDE "" CACHE INTERNAL "included")

########################################################
# Additional libraries
########################################################

# JPEG
ADD_DEFINITIONS(-DHAVE_JPEG)
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/../extras/jpeg")

# OOURA FFT
ADD_DEFINITIONS(-DHAVE_OOURAFFT)
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/../extras/oourafft")

# LBFGS
ADD_DEFINITIONS(-DHAVE_LBFGS)
ADD_DEFINITIONS(-DUSE_SSE)
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/../extras/lbfgs")

# TIFF
#ADD_DEFINITIONS(-DHAVE_TIFF)

# FFMPEG
ADD_DEFINITIONS(-DHAVE_FFMPEG)
#TODO

########################################################
# Sub-directories with source code
########################################################

SET(TORCH5SPRO_SUBDIRS
	th
	core
	sp
	ap
	ip
	scanning
	machine
	trainer
	boosting
	measurer
	geometry)

FOREACH(subdir ${TORCH5SPRO_SUBDIRS})
	ADD_SUBDIRECTORY(${subdir})

	# Add the sources from the subdirectories
	SET(TORCH5SPRO_SRC "${TORCH5SPRO_SRC};${${subdir}_SRC}")

	# Add the includes from the subdirectoris too
	SET(TORCH5SPRO_INCLUDE "${TORCH5SPRO_INCLUDE};${${subdir}_INCLUDE}")
ENDFOREACH(subdir)

########################################################
# Build the library

# This is the library with the source files gathered
#	from the sub-directories specified above
ADD_LIBRARY(torch5spro ${TORCH5SPRO_SRC})

INCLUDE_DIRECTORIES(${TORCH5SPRO_INCLUDE})

# Prepare the main header
EXECUTE_PROCESS(COMMAND "../makeincludeh.csh"
			"${TORCH5SPRO_INCLUDE}"
			"${CMAKE_CURRENT_SOURCE_DIR}/../include"
			"../torch5spro.h"
			"include"
			"TensorGen.h THTensorGen.h THStorageGen.h")

########################################################
# Make it verbose

MESSAGE("---------------------------------------------" "")
MESSAGE("SOURCES: " "${TORCH5SPRO_SRC}")
MESSAGE("---------------------------------------------" "")
MESSAGE("INCLUDES:" "${TORCH5SPRO_INCLUDE}")
MESSAGE("---------------------------------------------" "")
MESSAGE("CXX RELEASE FLAGS: " "${CMAKE_CXX_FLAGS_RELEASE}")
MESSAGE("CXX DEBUG FLAGS:   " "${CMAKE_CXX_FLAGS_DEBUG}")
MESSAGE("C RELEASE FLAGS: " "${CMAKE_C_FLAGS_RELEASE}")
MESSAGE("C DEBUG FLAGS:   " "${CMAKE_C_FLAGS_DEBUG}")
MESSAGE("BUILD TYPE:        " "${CMAKE_BUILD_TYPE}")
MESSAGE("---------------------------------------------" "")

########################################################
