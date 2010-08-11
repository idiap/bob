project(th)
cmake_minimum_required(VERSION 2.6)

# This defines the dependencies of this package to other torch subprojects
set(deps "") #other torch subprojects
set(shared "") #shared externals to link against (link)
if(cblas_FOUND)
  set(shared "${cblas_LIBRARIES}")
  add_definitions(-DUSE_CBLAS)
  include_directories(${cblas_INCLUDE};${lapack_INCLUDE})
  link_directories(${cblas_LIBDIR})
endif(cblas_FOUND)

# This defines the list of source files inside this package.
set(src
		"src/THBlas.c"
		"src/THGeneral.c"
		"src/THLogAdd.c"
		"src/THRandom.c"
		"src/THStorage.c"
		"src/THTensor.c"
		"src/THTensorMath.c")

include(../macros.cmake)
torch_library(th "${src}" "${deps}" "${shared}")
