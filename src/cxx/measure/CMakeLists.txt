project(measure-cxx)
cmake_minimum_required(VERSION 2.6)

# This defines the dependencies of this package
set(deps "core") #other bob subprojects
set(shared "") #external libraries

# This defines the list of source files inside this package.
set(src
    "src/error.cc"
    )

bob_library(measure "${src}" "${deps}" "${shared}")
