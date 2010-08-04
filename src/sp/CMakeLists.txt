project(sp)
cmake_minimum_required(VERSION 2.4)

# This defines the dependencies of this package
set(deps core) #other torch subprojects

add_definitions(-DHAVE_OOURAFFT)
include_directories("${INCLUDE_DIR}/oourafft")
set(shared "-loourafft")
set(archive "")

# This defines the list of source files inside this package.
set(src "spFFT.cc" "spDCT.cc" "spDeltaOne.cc" "ooura.cc")

include(../macros.cmake)
torch_package(sp "${src}" "${deps}" "${shared}" "${archive}")
