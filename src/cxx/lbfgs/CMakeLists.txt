project(lbfgs-cxx)

cmake_minimum_required(VERSION 2.6)

set(src src/lbfgs.c)

include(../../cmake/macros.cmake)
bob_header_install(lbfgs-cxx lbfgs "bob::lbfgs")
bob_library(lbfgs "${src}" "" "")
add_dependencies(bob_lbfgs lbfgs-cxx_header_install)
