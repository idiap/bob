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

# This will bring in the ffmpeg dependency. 
set(ffmpeg_DIR extras)
find_package(ffmpeg)

add_subdirectory(src)

set(SystemBoost_DIR extras)
find_package(SystemBoost) # configures boost found on the current system

#add_subdirectory(test)
#add_subdirectory(benchmark)

# This will bring in the Python bindings, if python was found
set(python_DIR extras)
find_package(python) # configures pytohn found on the current system

#if (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)
#  add_subdirectory(python)
#else (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)
#  message("Boost::Python bindings are DISABLED")
#endif (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)
