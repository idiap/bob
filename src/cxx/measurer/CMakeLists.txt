project(measurer-cxx)
cmake_minimum_required(VERSION 2.6)

# This defines the dependencies of this package
set(deps core) #other torch subprojects

set(src "src/measurer.cc")

include(../../cmake/macros.cmake)
torch_library(measurer "${src}" "${deps}" "${shared}")
torch_header_install(measurer "torch::measurer")
