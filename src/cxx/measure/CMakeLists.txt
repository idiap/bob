project(measure-cxx)
cmake_minimum_required(VERSION 2.6)

# This defines the dependencies of this package
set(deps "core") #other bob subprojects
set(shared "") #external libraries

# This defines the list of source files inside this package.
set(src
    "src/error.cc"
    )

include(../../cmake/macros.cmake)
bob_header_install(measure-cxx measure "bob::measure")
bob_library(measure "${src}" "${deps}" "${shared}")
add_dependencies(bob_measure measure-cxx_header_install)
