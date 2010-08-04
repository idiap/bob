# Andre Anjos <andre.anjos@idiap.ch>
# 9/July/2010

project(pytorch_core)
cmake_minimum_required(VERSION 2.6)

set(THIS pytorch_core)



# Creates and installs the binary component for the python library
include_directories(${TORCH5SPRO_INCLUDE})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
link_directories(${PYTHON_LIBRARY_DIRS})
link_directories(${INSTALL_DIR}/lib)
IF(ffmpeg_FOUND)
  ADD_DEFINITIONS(-DHAVE_FFMPEG)
  include_directories(${ffmpeg_INCLUDE_DIRS})
  link_directories(${ffmpeg_LIBRARY_DIRS})
ENDIF(ffmpeg_FOUND)

add_library(${THIS} SHARED 
  object.cc
  file.cc
  tensor.cc
  tensorfile.cc
  spcore.cc
  ipcore.cc
  main.cc
  )
set_target_properties(${THIS} PROPERTIES SUFFIX ".so")
target_link_libraries(${THIS} torch5spro)
target_link_libraries(${THIS} ${Boost_PYTHON_LIBRARY})
target_link_libraries(${THIS} ${PYTHON_LIBRARIES})

install(TARGETS ${THIS} LIBRARY DESTINATION ${INSTALL_DIR}/lib)
