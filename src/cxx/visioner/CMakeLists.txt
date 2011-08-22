project(visioner-cxx)
cmake_minimum_required(VERSION 2.6)

include("ExternalProject")

# These are the basic variables to setup the visioner checkout from the proper
# place and with the supported tag. It also provides basis in case we cannot
# reach the directory ${VISIONER_REPO} directly.
set(VISIONER_REPO "http://www.idiap.ch/software/torch5spro/git/visioner.git/")

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
  CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} -DTORCH_INCLUDE_DIR=${CMAKE_INSTALL_PREFIX}/include/torch -DTORCH_LIBRARY_DIR=${CMAKE_BINARY_DIR}/cxx/lbfgs
  )
add_dependencies(visioner torch_lbfgs)
