project(th)
cmake_minimum_required(VERSION 2.4)

set(this th)

set(src
		"THBlas.c"
		"THGeneral.c"
		"THLogAdd.c"
		"THRandom.c"
		"THStorage.c"
		"THTensor.c"
		"THTensorMath.c")

# Note: This package depends on an external build of BLAS/CBLAS
if(cblas_FOUND)
  add_definitions(-DUSE_CBLAS)
  include_directories(${cblas_INCLUDE};${lapack_INCLUDE})
endif(cblas_FOUND)

# Shared library build
set(libname torch_${this})
add_library(${libname} SHARED ${src})
if(cblas_FOUND)
  target_link_libraries(${libname} ${cblas_LIBRARIES})
endif(cblas_FOUND)

# Static library build
add_library(${libname}-static STATIC ${src})
set_target_properties(${libname}-static PROPERTIES OUTPUT_NAME "${libname}")
set_target_properties(${libname}-static PROPERTIES PREFIX "lib")
if(cblas_FOUND)
  target_link_libraries(${libname}-static ${cblas_ARCHIVES})
endif(cblas_FOUND)

# This installs the package
install(TARGETS ${libname} LIBRARY DESTINATION ${INSTALL_DIR}/lib)
install(TARGETS ${libname}-static ARCHIVE DESTINATION ${INSTALL_DIR}/lib)
file(COPY ./ DESTINATION ${INCLUDE_DIR}/torch/${this} 
  FILES_MATCHING PATTERN "*.h")
