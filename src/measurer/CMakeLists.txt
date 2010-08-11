project(measurer)
cmake_minimum_required(VERSION 2.4)

# This defines the dependencies of this package
set(deps core) #other torch subprojects

set(src "src/measurer.cc")

include(../macros.cmake)
torch_library(measurer "${src}" "${deps}" "${shared}")
