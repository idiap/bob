project(visioner-cxx)
cmake_minimum_required(VERSION 2.6)

include("ExternalProject")

# These are the basic variables to setup the visioner checkout from the proper
# place and with the supported tag. It also provides basis in case we cannot
# reach the directory ${VISIONER_REPO} directly.
set(VISIONER_REPO "git@github.com:idiap/visioner.git")

# Accesses the git command
find_program(GIT_EXECUTABLE NAMES git PATH_SUFFIXES Git/cmd Git/bin
  DOC "git command line client"
)

if(NOT GIT_EXECUTABLE)
  message(FATAL_ERROR "error: could not find git for cloning visioner")
endif(NOT GIT_EXECUTABLE)

ExternalProject_Add(visioner
  DOWNLOAD_COMMAND rm -rf visioner && ${GIT_EXECUTABLE} clone ${VISIONER_REPO}
  UPDATE_COMMAND git pull origin +HEAD
  PATCH_COMMAND ""
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
