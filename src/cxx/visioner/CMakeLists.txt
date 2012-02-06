project(visioner-cxx)
cmake_minimum_required(VERSION 2.6)

include("ExternalProject")

# Accesses the git command
find_program(GIT_EXECUTABLE NAMES git PATH_SUFFIXES Git/cmd Git/bin
  DOC "git command line client"
)

if(NOT GIT_EXECUTABLE)
  message(FATAL_ERROR "error: could not find git for cloning visioner")
endif(NOT GIT_EXECUTABLE)

ExternalProject_Add(visioner
  DOWNLOAD_COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR}/../../.. && ${GIT_EXECUTABLE} submodule init -- src/cxx/visioner/project && ${GIT_EXECUTABLE} submodule update && cd ${CMAKE_CURRENT_BINARY_DIR} && rsync -avz ${CMAKE_CURRENT_SOURCE_DIR}/project/ ${CMAKE_CURRENT_BINARY_DIR}/visioner/
  CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} -DLBFGS_INCLUDE_DIR=${CMAKE_INSTALL_PREFIX}/include/bob -DLBFGS_LIBRARY_DIR=${CMAKE_BINARY_DIR}/cxx/lbfgs
  )

add_dependencies(visioner bob_lbfgs)

ExternalProject_Get_Property(visioner BINARY_DIR)
ExternalProject_Get_Property(visioner SOURCE_DIR)

if(APPLE)
  set_target_properties(bob_visioner PROPERTIES IMPORTED_LOCATION ${BINARY_DIR}/src/libbob_visioner.dylib)
else()
  set_target_properties(bob_visioner PROPERTIES IMPORTED_LOCATION ${BINARY_DIR}/src/libbob_visioner.so)
endif()

set(VISIONER_INCLUDE_DIRS
  "${SOURCE_DIR}/src/ml"
  "${SOURCE_DIR}/src/proc"
  "${SOURCE_DIR}/src/util"
  "${SOURCE_DIR}/src/vision"
  CACHE INTERNAL "Visioner include directories"
  )
