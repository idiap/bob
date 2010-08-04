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

set(this core)
set(deps torch_th)

# Note: This package can be built an external build of BLAS/CBLAS
if(cblas_FOUND)
  add_definitions(-DUSE_CBLAS)
  include_directories(${cblas_INCLUDE};${lapack_INCLUDE})
  torch_shlib(torch_${this} ${src} ${deps} ${cblas_LIBRARIES} ${INSTALL_DIR}/lib)
  torch_shlib(torch_${this} ${src} ${deps} ${cblas_ARCHIVES} ${INSTALL_DIR}/lib)
else(cblas_FOUND)
  torch_archive(torch_${this} ${src} ${deps} ${cblas_LIBRARIES} ${INSTALL_DIR}/lib)
  torch_archive(torch_${this} ${src} ${deps} ${cblas_ARCHIVES} ${INSTALL_DIR}/lib)
endif(cblas_FOUND)

file(COPY ./ DESTINATION ${INCLUDE_DIR}/torch/${this} FILES_MATCHING PATTERN "*.h")
