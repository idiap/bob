# Configures all (extras, torch and examples) in one shot.
# Andre Anjos - 28.june.2010

# Sets up all extras in one go. 
project(torch5spro)
cmake_minimum_required(VERSION 2.6)

# Enables the testing framework
enable_testing()

# Set the compiler flags and build type(release or debug)
set(CMAKE_CXX_FLAGS_RELEASE "-pthread -O3 -g -Wall -ffast-math -fomit-frame-pointer -msse2")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_DEBUG "-pthread -g -Wall")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")

# This will bring in the cblas/lapack/atlas dependence. 
set(cblas_DIR extras)
find_package(cblas REQUIRED)

# This will bring in the jpeg dependence. 
set(jpeg_DIR extras)
find_package(jpeg REQUIRED)

# This will bring in the ffmpeg dependence. 
set(ffmpeg_DIR extras)
find_package(ffmpeg REQUIRED)

add_subdirectory(src)

# configures boost found on the current system
set(SystemBoost_DIR extras)
find_package(SystemBoost REQUIRED)

#add_subdirectory(benchmark)

# This will bring in the Python bindings, if python was found
set(python_DIR extras)
find_package(python REQUIRED)

#if (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)
#  add_subdirectory(python)
#else (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)
#  message("Boost::Python bindings are DISABLED")
#endif (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)
