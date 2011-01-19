PROJECT(ap-cxx)
cmake_minimum_required(VERSION 2.6)

# This defines the dependencies of this package
set(deps core sp) #other torch subprojects

# This defines the list of source files inside this package.
SET(src 
   "src/apCore.cc"
   "src/apMFCC.cc"
   "src/apLFCC.cc")

include(../../cmake/macros.cmake)
torch_library(ap "${src}" "${deps}" "${shared}")
torch_header_install(ap "torch::ap")
