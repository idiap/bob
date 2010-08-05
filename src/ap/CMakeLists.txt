PROJECT(ap)
cmake_minimum_required(VERSION 2.4)

# This defines the dependencies of this package
set(deps core) #other torch subprojects

# This defines the list of source files inside this package.
SET(src "apMFCC.cc" "apLFCC.cc")

include(../macros.cmake)
torch_package(ap "${src}" "${deps}" "${shared}")
