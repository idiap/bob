# Andre Anjos <andre.anjos@idiap.ch>
# 29/July/2010

project(benchmark_core)
cmake_minimum_required(VERSION 2.6)

set(THIS benchmark_core)

# Creates and installs the binary component for the python library
include_directories(${TORCH5SPRO_INCLUDE})
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${INSTALL_DIR}/lib)

# Please note we don't install benchmark executables
add_executable(${THIS}_tensor tensor.cc)
target_link_libraries(${THIS}_tensor torch5spro)
install(TARGETS ${THIS}_tensor RUNTIME DESTINATION ${INSTALL_DIR}/bin) 
