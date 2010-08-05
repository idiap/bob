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

# This makes sure we have all headers within reach before starting to compile
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
link_directories(${INSTALL_DIR}/lib)

# to-do:
#	scanning
#	machine
#	trainer
#	boosting
#	geometry
foreach(subdir th core sp ap ip measurer)
	add_subdirectory(${subdir})
endforeach(subdir)
