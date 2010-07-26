# Configures all (extras, torch and examples) in one shot.
# Andre Anjos - 28.june.2010

# Sets up all extras in one go. 
project(torch5spro)
cmake_minimum_required(VERSION 2.8)

# Enables the testing framework
enable_testing()

# Set the compiler flags and build type(release or debug)
set(CMAKE_CXX_FLAGS_RELEASE "-pthread -O3 -ffast-math -fomit-frame-pointer")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_DEBUG "-pthread -g")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")

# This will autoconfigure certain system libraries we need for both torch and
# the examples
set(cblas_DIR extras)
find_package(cblas) # configures lapack, cblas and atlas

add_subdirectory(extras)

# This will bring in the ffmpeg dependency. Please note that if you comment out
# the ffmpeg building in extras/CMakeLists.txt, we will use that instead of the
# system's version.
set(ffmpeg_DIR extras)
find_package(ffmpeg)

add_subdirectory(src)

# To compile the examples and, possibly, the pythonic bindings, we need
# to include the place for the unified header
set(TORCH5SPRO_INCLUDE ${TORCH5SPRO_INCLUDE};${INCLUDE_DIR}/torch5spro CACHE INTERNAL "included")

# This will bring in the Python bindings, if python was found

set(SystemBoost_DIR extras)
find_package(SystemBoost) # configures boost found on the current system

set(python_DIR extras)
find_package(python) # configures pytohn found on the current system

if (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)
  add_subdirectory(python)
else (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)
  message("Boost::Python bindings are DISABLED")
endif (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)

# If you call our build scripts (release.sh or debug.sh), the examples are
# built at the end, after extras and torch5 as the examples need a header file
# to be computed
add_subdirectory(examples EXCLUDE_FROM_ALL)

# This adds a target to produce both vim and emacs tag files for the whole
# project. To learn how to use tag files with your editor do "man ctags" on
# your prompt
execute_process(COMMAND ${DIST_ROOT}/gentags.sh)
