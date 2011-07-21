project(visioner-cxx)
cmake_minimum_required(VERSION 2.6)

include("ExternalProject")

# These are the basic variables to setup the visioner checkout from the proper
# place and with the supported tag. It also provides basis in case we cannot
# reach the directory ${VISIONER_REPO} directly.
set(VISIONER_REPO "/idiap/group/torch5spro/sandboxes/visioner.git")
set(VISIONER_BRANCH "v1.5.2-analyzer-tmax-faster")
set(IDIAP_LOGIN "$ENV{USER}@dynamix08.idiap.ch")

# Here we test to see if the directory is visible from the build spot
if(IS_DIRECTORY ${VISIONER_REPO})
  set(USE_VISIONER_REPO "${VISIONER_REPO}")
else(IS_DIRECTORY ${VISIONER_REPO})
  set(USE_VISIONER_REPO "${IDIAP_LOGIN}:${VISIONER_REPO}")
endif(IS_DIRECTORY ${VISIONER_REPO})

# Accesses the git command
find_program(GIT_EXECUTABLE NAMES git PATH_SUFFIXES Git/cmd Git/bin
  DOC "git command line client"
)

if(NOT GIT_EXECUTABLE)
  message(FATAL_ERROR "error: could not find git for cloning visioner")
endif(NOT GIT_EXECUTABLE)

ExternalProject_Add(visioner
  DOWNLOAD_COMMAND rm -rf visioner && ${GIT_EXECUTABLE} clone ${USE_VISIONER_REPO}
  UPDATE_COMMAND git pull origin +HEAD
  PATCH_COMMAND ""
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} -DTORCH_INCLUDE_DIR=${CMAKE_INSTALL_PREFIX}/include/torch -DTORCH_LIBRARY_DIR=${CMAKE_BINARY_DIR}/cxx/lbfgs
  )
add_dependencies(visioner torch_lbfgs)
