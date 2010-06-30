# Configures all extras in one shot. 
# Andre Anjos - 28.june.2010

# Sets up all extras in one go. 
project(torch5spro)
cmake_minimum_required(VERSION 2.8)

# Set the compiler flags and build type(release or debug)
SET(CMAKE_CXX_FLAGS_RELEASE "-pthread -O3 -ffast-math -fomit-frame-pointer")
SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
SET(CMAKE_CXX_FLAGS_DEBUG "-pthread -g")
SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")

add_subdirectory(extras)
add_subdirectory(src)

# To compile the examples, we need to make the include directories absolute
set(tmp)
foreach(dir ${TORCH5SPRO_INCLUDE})
  set(tmp ${tmp};${CMAKE_CURRENT_SOURCE_DIR}/src/${dir})
endforeach(dir ${TORCH5SPRO_INCLUDE})
# I also need to include to top level include directory into this
set(tmp ${tmp};${INCLUDE_DIR}/torch5spro)
set(TORCH5SPRO_INCLUDE ${tmp} CACHE INTERNAL "included")

# If you cannot compile the examples
add_subdirectory(examples EXCLUDE_FROM_ALL)
