PROJECT(math-cxx)
cmake_minimum_required(VERSION 2.6)

# This defines the dependencies of this package
set(deps "core") #other torch subprojects
set(shared "")

# This defines the list of source files inside this package.
set(src_blas
   ""
)

set(src_lapack
  "src/linsolve.cc"
  "src/svd.cc"
)

set(src_optimization
  "src/interiorpointLP.cc"
)

set(src 
   "${src_blas}" 
   "${src_lapack}"
   "${src_optimization}")

include(../../cmake/macros.cmake)
torch_header_install(math-cxx math "torch::math")
torch_library(math "${src}" "${deps}" "${shared}")
add_dependencies(torch_math math-cxx_header_install)

# Defines tests for this package
torch_test(math linsolve test/linsolve.cc)
torch_test(math svd test/svd.cc)
torch_test(math interiorpointLP test/interiorpointLP.cc)
