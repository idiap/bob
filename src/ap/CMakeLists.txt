PROJECT(ap)
cmake_minimum_required(VERSION 2.4)

# This defines the dependencies of this package
set(deps core) #other torch subprojects

# Avoids warning during linkage configuration see: cmake --help-policy CMP0003
# for details.
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

# This defines the list of source files inside this package.
SET(src "apMFCC.cc" "apLFCC.cc")

include(../macros.cmake)
torch_package(ap "${src}" "${deps}" "${shared}")
