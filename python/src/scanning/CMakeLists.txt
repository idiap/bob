# Andre Anjos <andre.anjos@idiap.ch>
# 9/July/2010

project(pytorch_scanning)
cmake_minimum_required(VERSION 2.6)

set(THIS pytorch_scanning)

# Creates and installs the binary component for the python library
include_directories(${TORCH5SPRO_INCLUDE})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
link_directories(${PYTHON_LIBRARY_DIRS})
link_directories(${INSTALL_DIR}/lib)

add_library(${THIS} SHARED 
  pattern.cc
  scanner.cc
  facefinder.cc
  main.cc
  )
set_target_properties(${THIS} PROPERTIES SUFFIX ".so")
target_link_libraries(${THIS} torch5spro)
target_link_libraries(${THIS} ${Boost_PYTHON_LIBRARY})
target_link_libraries(${THIS} ${PYTHON_LIBRARIES})

install(TARGETS ${THIS} LIBRARY DESTINATION ${INSTALL_DIR}/lib)
