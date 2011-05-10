PROJECT(machine-cxx)
cmake_minimum_required(VERSION 2.6)

# This defines the dependencies of this package
set(deps core database machine) #other torch subprojects
set(shared "")

# This defines the list of source files inside this package.
set(src
    "src/KMeansMachine.cc"
    "src/FrameSample.cc"
    "src/Gaussian.cc"
    "src/GMMMachine.cc"
    "src/GMMStats.cc"
)

include(../../cmake/macros.cmake)
torch_header_install(machine-cxx machine "torch::machine")
torch_library(machine "${src}" "${deps}" "${shared}")
add_dependencies(torch_machine machine-cxx_header_install)

# Defines tests for this package
#torch_test(machine block test/block.cc)

# Defines environment variable used to localize data for the tests
#set_property(TEST cxx-machine-gabor APPEND PROPERTY ENVIRONMENT
#  "TORCH_IP_TESTDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/test/data")