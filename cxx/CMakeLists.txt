# Configures and builds the C++ bit of bob
# Andre Anjos - 18.january.2011

project(bob-cxx)
cmake_minimum_required(VERSION 2.6)

add_subdirectory(lbfgs)
add_subdirectory(core)
add_subdirectory(io)
add_subdirectory(measure)
add_subdirectory(sp)
add_subdirectory(math)
add_subdirectory(ip)
add_subdirectory(machine)
add_subdirectory(trainer)

if(QT4_FOUND)
  add_subdirectory(visioner)
  add_subdirectory(daq)
endif()
