#
# Top level project
#
project(torch)

cmake_minimum_required(VERSION 2.4)

# This makes sure we have all headers within reach before starting to compile
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
link_directories(${INSTALL_DIR}/lib)

#foreach(subdir oourafft lbfgs th core sp ap ip measurer machine scanning trainer)
foreach(subdir th)
	add_subdirectory(${subdir})
endforeach(subdir)
