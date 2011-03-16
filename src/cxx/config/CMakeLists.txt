project(config-cxx)
cmake_minimum_required(VERSION 2.6)

# This defines the dependencies of this package
set(deps "database") #other torch subprojects
set(shared "${Boost_PYTHON_LIBRARY_RELEASE};${PYTHON_LIBRARIES}")

# This defines the list of source files inside this package.
set(src
    "src/Python.cc"
    "src/Exception.cc"
    "src/Configuration.cc"
    )

include(../../cmake/macros.cmake)
torch_header_install(config-cxx config "torch::config")
torch_library(config "${src}" "${deps}" "${shared}")

# Defines tests for this package
torch_test(config basic test/basic.cc)

# Defines environment variable used to localize data for the tests
set_property(TEST cxx-config-basic APPEND PROPERTY ENVIRONMENT
  "TORCH_TESTDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/test/data")
