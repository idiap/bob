project(sp-cxx)
cmake_minimum_required(VERSION 2.6)

# This defines the dependencies of this package
set(deps core) #other bob subprojects
set(shared "${FFTW3_LIBRARY}")

# This defines the list of source files inside this package.
set(src 
    "src/spCore.cc"
    "src/spCoreChain.cc"
    "src/FFT1D.cc"
    "src/FFT1DNaive.cc"
    "src/FFT2D.cc"
    "src/FFT2DNaive.cc"
    "src/DCT1D.cc"
    "src/DCT1DNaive.cc"
    "src/DCT2D.cc"
    "src/DCT2DNaive.cc")

include(../../cmake/macros.cmake)
bob_header_install(sp-cxx sp "bob::sp")
bob_library(sp "${src}" "${deps}" "${shared}")
add_dependencies(bob_sp sp-cxx_header_install)

# Defines tests for this package
#bob_test(sp dftdct_naive test/dftdctnaive.cc)
#bob_test(sp fftfct_fftpack test/fftfctfftpack.cc)
bob_test(sp convolution test/convolution.cc)
#bob_test(sp extrapolation test/extrapolation.cc)
bob_test(sp fft_fct test/fft_fct.cc)
