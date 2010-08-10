# Builds all python code in one shot
# Andre Anjos - 12.july.2010

project(torch_test)
cmake_minimum_required(VERSION 2.6)

# This makes sure we have all headers within reach before starting to compile
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)

add_subdirectory(core)
