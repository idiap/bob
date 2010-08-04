project(th)
cmake_minimum_required(VERSION 2.4)

# This defines the dependencies of this package to other torch subprojects
set(deps "") #other torch subprojects
set(shared "") #shared externals to link against (link)
set(archive "") #external archives to link against (merge)
if(cblas_FOUND)
  set(shared "${cblas_LIBRARIES}")
  set(archive "${cblas_ARCHIVES}")
  add_definitions(-DUSE_CBLAS)
  include_directories(${cblas_INCLUDE};${lapack_INCLUDE})
endif(cblas_FOUND)

# This defines the list of source files inside this package.
set(src
		"THBlas.c"
		"THGeneral.c"
		"THLogAdd.c"
		"THRandom.c"
		"THStorage.c"
		"THTensor.c"
		"THTensorMath.c")

include(../macros.cmake)
torch_package(th "${src}" "${deps}" "${shared}" "${archive}")
