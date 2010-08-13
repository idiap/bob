PROJECT(ap)
cmake_minimum_required(VERSION 2.6)

# This defines the dependencies of this package
set(deps core sp) #other torch subprojects

# Avoids warning during linkage configuration see: cmake --help-policy CMP0003
# for details.
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

# This defines the list of source files inside this package.
SET(src 
   "src/apCore.cc"
   "src/apMFCC.cc"
   "src/apLFCC.cc")

include(../cmake/macros.cmake)
torch_library(ap "${src}" "${deps}" "${shared}")
