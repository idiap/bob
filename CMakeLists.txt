# Configures all (extras, torch and examples) in one shot.
# Andre Anjos - 28.june.2010

# Sets up all extras in one go. 
project(torch5spro)
cmake_minimum_required(VERSION 2.8)

# Set the compiler flags and build type(release or debug)
set(CMAKE_CXX_FLAGS_RELEASE "-pthread -O3 -ffast-math -fomit-frame-pointer")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_DEBUG "-pthread -g")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")

# This will autoconfigure certain system libraries we need for both torch and
# the examples
set(cblas_DIR extras)
find_package(cblas) # configures lapack, cblas and atlas

set(SystemBoost_DIR extras)
find_package(SystemBoost) # configures boost found on the current system

add_subdirectory(extras)
add_subdirectory(src)

# To compile the examples, we need to make the include directories absolute
set(tmp)
foreach(dir ${TORCH5SPRO_INCLUDE})
  set(tmp ${tmp};${CMAKE_CURRENT_SOURCE_DIR}/src/${dir})
endforeach(dir ${TORCH5SPRO_INCLUDE})
# I also need to include to top level include directory into this and the
# top-level torch "src" directory, since the header files will not be in the
# install area yet, just after "make install". To get the examples compiling we
# need the "src" directory to be there then.
set(tmp ${tmp};${INCLUDE_DIR}/torch5spro;${CMAKE_CURRENT_SOURCE_DIR}/src)
set(TORCH5SPRO_INCLUDE ${tmp} CACHE INTERNAL "included")

# If you call our build scripts (release.sh or debug.sh), the examples are
# built at the end, after extras and torch5 as the examples need a header file
# to be computed
add_subdirectory(examples EXCLUDE_FROM_ALL)
