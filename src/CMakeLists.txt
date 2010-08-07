#
# Top level project
#
project(torch)

cmake_minimum_required(VERSION 2.4)

########################################################
# Additional libraries
########################################################

# LBFGS
ADD_DEFINITIONS(-DHAVE_LBFGS)
ADD_DEFINITIONS(-DUSE_SSE)
INCLUDE_DIRECTORIES("${INCLUDE_DIR}/lbfgs")

# This makes sure we have all headers within reach before starting to compile
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
link_directories(${INSTALL_DIR}/lib)

# to-do:
#	trainer
foreach(subdir th core sp ap ip measurer machine scanning)
	add_subdirectory(${subdir})
endforeach(subdir)
