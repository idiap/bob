project(sp-cxx)
cmake_minimum_required(VERSION 2.6)

# This defines the dependencies of this package
set(deps dfftpack41 core) #other torch subprojects

# This defines the list of source files inside this package.
set(src 
    "src/spCore.cc"
    "src/spCoreChain.cc"
    "src/spDFT.cc"
    "src/spFFT.cc"
    "src/spDCT.cc"
    "src/spFCT.cc"
    "src/spDeltaOne.cc")

include(../../cmake/macros.cmake)
torch_library(sp "${src}" "${deps}" "")
torch_header_install(sp "torch::sp")

# Defines tests for this package
torch_test(sp dftdct_naive test/dftdctnaive.cc)
torch_test(sp fftfct_fftpack test/fftfctfftpack.cc)
