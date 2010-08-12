project(src)

cmake_minimum_required(VERSION 2.6)

# This makes sure we have all headers within reach before starting to compile
include_directories(${CMAKE_INSTALL_PREFIX}/include/torch)

add_subdirectory(python)

foreach(proj oourafft lbfgs th core sp ap ip measurer machine scanning trainer)
	add_subdirectory(${proj})
endforeach(proj)

set(cmakedir share/cmake)
install(EXPORT torch DESTINATION ${cmakedir})
install(FILES torchConfig.cmake DESTINATION ${cmakedir})
