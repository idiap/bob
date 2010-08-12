# Configures and builds torch
# Andre Anjos - 28.june.2010

project(torch)
cmake_minimum_required(VERSION 2.6)

# Set the compiler flags and build type(release or debug)
set(CMAKE_CXX_FLAGS_RELEASE "-pthread -O3 -g -Wall -ffast-math -fomit-frame-pointer -msse2")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_DEBUG "-pthread -g -Wall")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")

# Sets up externals and install
foreach(ext cblas jpeg ffmpeg boost python)
  find_package(${ext} REQUIRED PATHS external)
  install(FILES external/${ext}Config.cmake DESTINATION share/cmake)
endforeach(ext cblas jpeg ffmpeg boost python)

# Enables the testing framework
enable_testing()

# Builds the whole project
add_subdirectory(src)
