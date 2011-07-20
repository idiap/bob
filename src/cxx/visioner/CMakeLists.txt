project(visioner-cxx)
cmake_minimum_required(VERSION 2.6)

# This defines the dependencies of this package
include("ExternalProject")

ExternalProject_Add(visioner
  DOWNLOAD_DIR Visioner
  GIT_REPOSITORY aanjos@idiap:/idiap/user/catana/Visioner
  #GIT_REPOSITORY /Users/andre/work/Visioner
  GIT_TAG v1.5.2-analyzer-tmax-faster
  PATCH_COMMAND patch -p1 -t -N < ${CMAKE_CURRENT_SOURCE_DIR}/patch0
  UPDATE_COMMAND ""
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} -DTORCH_INCLUDE_DIR=${CMAKE_INSTALL_PREFIX}/include/torch -DTORCH_LIBRARY_DIR=${CMAKE_INSTALL_PREFIX}/lib
  )
add_dependencies(visioner torch_lbfgs)
