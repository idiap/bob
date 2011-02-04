# Configures and builds the C++ bit of torch
# Andre Anjos - 18.january.2011

project(torch-cxx)
cmake_minimum_required(VERSION 2.6)

add_subdirectory(dfftpack41)
add_subdirectory(lbfgs)
add_subdirectory(core)
add_subdirectory(database)
add_subdirectory(sp)
add_subdirectory(ap)
add_subdirectory(ip)
add_subdirectory(measurer)
add_subdirectory(machine)
add_subdirectory(scanning)
add_subdirectory(trainer)
