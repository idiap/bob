project(error-cxx)
cmake_minimum_required(VERSION 2.6)

# This defines the dependencies of this package
set(deps "core") #other torch subprojects
set(shared "") #external libraries

# This defines the list of source files inside this package.
set(src
    "src/eval.cc"
    )

include(../../cmake/macros.cmake)
torch_header_install(error-cxx error "torch::error")
torch_library(error "${src}" "${deps}" "${shared}")
add_dependencies(torch_error error-cxx_header_install)
