#
# Top level project
#
project(torch)

cmake_minimum_required(VERSION 2.4)

########################################################
# Additional libraries
########################################################

# JPEG
ADD_DEFINITIONS(-DHAVE_JPEG)
INCLUDE_DIRECTORIES("${INCLUDE_DIR}/jpeg")

# LBFGS
ADD_DEFINITIONS(-DHAVE_LBFGS)
ADD_DEFINITIONS(-DUSE_SSE)
INCLUDE_DIRECTORIES("${INCLUDE_DIR}/lbfgs")

# TIFF
#ADD_DEFINITIONS(-DHAVE_TIFF)

# FFMPEG
IF(ffmpeg_FOUND)
  ADD_DEFINITIONS(-DHAVE_FFMPEG)
  INCLUDE_DIRECTORIES(${ffmpeg_INCLUDE_DIRS})
  LINK_DIRECTORIES(${ffmpeg_LIBRARY_DIRS})
ENDIF(ffmpeg_FOUND)

# Avoids warning during linkage configuration see: cmake --help-policy CMP0003
# for details.
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

# This makes sure we have all headers within reach before starting to compile
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
link_directories(${INSTALL_DIR}/lib)

# to-do:
#	ip
#	scanning
#	machine
#	trainer
#	boosting
#	measurer
#	geometry
foreach(subdir th core sp ap ip)
	add_subdirectory(${subdir})
endforeach(subdir)
