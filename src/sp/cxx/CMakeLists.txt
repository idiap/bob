project(bob_sp)

# This defines the dependencies of this package
set(bob_deps "bob_core")
set(shared "${bob_deps};${FFTW3_LIBRARY}")
set(incdir ${cxx_incdir};${FFT3_INCLUDE_DIR})

# This defines the list of source files inside this package.
set(src
    "FFT1D.cc"
    "FFT1DNaive.cc"
    "FFT1DKiss.cc"
    "FFT2D.cc"
    "FFT2DNaive.cc"
    "FFT2DKiss.cc"
    "DCT1D.cc"
    "DCT1DNaive.cc"
    "DCT1DKiss.cc"
    "DCT2D.cc"
    "DCT2DNaive.cc"
    "DCT2DKiss.cc"
    "Quantization.cc"
    )

# Define the library, compilation and linkage options
bob_sort_headers(incdir)
foreach(inc ${incdir})
  include_directories(SYSTEM ${inc})
endforeach()
bob_add_library(${PROJECT_NAME} "${src}")
target_link_libraries(${PROJECT_NAME} ${shared})

# Defines tests for this package
bob_add_test(${PROJECT_NAME} convolution test/conv.cc)
bob_add_test(${PROJECT_NAME} fft_fct test/fft_fct.cc)
bob_add_test(${PROJECT_NAME} fftkiss test/fftkiss.cc)

bob_add_benchmark(${PROJECT_NAME} fft_fct benchmark/fft_fct.cc)

# Pkg-Config generator
bob_pkgconfig(${PROJECT_NAME} "${bob_deps}")
