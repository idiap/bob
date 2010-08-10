project(sp)
cmake_minimum_required(VERSION 2.4)

# This defines the dependencies of this package
set(deps oourafft core) #other torch subprojects

add_definitions(-DHAVE_OOURAFFT)

# This defines the list of source files inside this package.
set(src 
#        "spCore.cc"
#        "spCoreChain.cc"
        "spFFT.cc"
        "spDCT.cc"
        "spDeltaOne.cc"
        "ooura.cc")

include(../macros.cmake)
torch_package(sp "${src}" "${deps}" "${shared}")
