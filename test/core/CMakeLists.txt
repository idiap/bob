# Andre Anjos <andre.anjos@idiap.ch>
# 29/July/2010

project(test_core)
cmake_minimum_required(VERSION 2.6)

set(THIS test_core)

# Creates and installs the binary component for the python library
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
link_directories(${INSTALL_DIR}/lib)

# Please note we don't install test executables
add_executable(${THIS}_tensor tensor.cc)
target_link_libraries(${THIS}_tensor torch_core)
target_link_libraries(${THIS}_tensor ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
add_test(cxx-core-tensor ${THIS}_tensor)
