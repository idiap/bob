project(lbfgs-cxx)

cmake_minimum_required(VERSION 2.6)

set(src src/lbfgs.c)

include(../../cmake/macros.cmake)
torch_header_install(lbfgs-cxx lbfgs "torch::lbfgs")
torch_library(lbfgs "${src}" "" "")
add_dependencies(torch_lbfgs lbfgs-cxx_header_install)
