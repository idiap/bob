# Configures and builds torch
# Andre Anjos - 28.june.2010

project(torch)
cmake_minimum_required(VERSION 2.6)

# Set the compiler flags and build type(release or debug)
set(CMAKE_CXX_FLAGS_RELEASE "-pthread -O3 -g -Wall -ffast-math -fomit-frame-pointer -msse2")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_DEBUG "-pthread -g -Wall")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")

# Sets up externals
find_package(cblas REQUIRED PATHS external) #cblas, lapack and atlas
find_package(jpeg REQUIRED PATHS external)
find_package(ffmpeg REQUIRED PATHS external)
find_package(SystemBoost REQUIRED PATHS external)
find_package(python REQUIRED PATHS external)

# Enables the testing framework
enable_testing()

# Builds the whole project
add_subdirectory(src)
