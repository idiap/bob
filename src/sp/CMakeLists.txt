project(sp)
cmake_minimum_required(VERSION 2.4)

# This defines the dependencies of this package
set(deps core) #other torch subprojects

# Avoids warning during linkage configuration see: cmake --help-policy CMP0003
# for details.
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

add_definitions(-DHAVE_OOURAFFT)
include_directories("${INCLUDE_DIR}/oourafft")
set(shared "-loourafft")

# This defines the list of source files inside this package.
set(src "spFFT.cc" "spDCT.cc" "spDeltaOne.cc" "ooura.cc")

include(../macros.cmake)
torch_package(sp "${src}" "${deps}" "${shared}")
