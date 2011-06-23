PROJECT(machine-cxx)
cmake_minimum_required(VERSION 2.6)

# This defines the dependencies of this package
set(deps core io math) #other torch subprojects
set(shared "")

# This defines the list of source files inside this package.
set(src
    "src/Exception.cc"
    "src/KMeansMachine.cc"
    "src/Gaussian.cc"
    "src/GMMMachine.cc"
    "src/GMMStats.cc"
    "src/EigenMachineException.cc"
    "src/TwoDPCAMachine.cc"
    "src/LinearMachine.cc"
)

include(../../cmake/macros.cmake)
torch_header_install(machine-cxx machine "torch::machine")
torch_library(machine "${src}" "${deps}" "${shared}")
add_dependencies(torch_machine machine-cxx_header_install trainer-cxx_header_install)

# Defines tests for this package
torch_test(machine linear test/linear.cc)
set_property(TEST cxx-machine-linear APPEND PROPERTY ENVIRONMENT
  "TORCH_MACHINE_TESTDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/../../python/machine/test/data")
