project(sp)
cmake_minimum_required(VERSION 2.6)

# This defines the dependencies of this package
set(deps oourafft core) #other torch subprojects

# This defines the list of source files inside this package.
set(src 
    "src/spCore.cc"
    "src/spCoreChain.cc"
    "src/spFFT.cc"
    "src/spDCT.cc"
    "src/spDeltaOne.cc")

add_definitions(-DHAVE_OOURAFFT)

include(../macros.cmake)
torch_library(sp "${src}" "${deps}" "")

# Python bindings
set(pysrc
   "python/src/spcore.cc"
   "python/src/transforms.cc"
   "python/src/main.cc"
   )
 
torch_python_bindings(sp "${pysrc}")
torch_python_install(sp)
