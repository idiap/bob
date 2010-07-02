#
# Top level project
#
PROJECT(EXAMPLE)

CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

SET(EXAMPLE_SRC
	iimage.cc
	crop.cc
	scale.cc
	lbp.cc
	histo.cc
	smoothGaussian.cc
	sobel.cc
	ipflipper2d.cc
	ipflipper3d.cc
	shift.cc
	rotate.cc
	blocks.cc
	geomNorm.cc
	memtest.cc
	histoEqual.cc
	multiscaleRetinex.cc
	selfQuotientImage.cc
	relaxation.cc
  #diffusionVcycle.cc
	tanTriggs.cc
	histoEqualTensor.cc
	multiscaleRetinexTensor.cc
	selfQuotientImageTensor.cc
	relaxationTensor.cc
  #diffusionVcycleTensor.cc
	tanTriggsTensor.cc)

LINK_DIRECTORIES(${INSTALL_DIR}/lib)
INCLUDE_DIRECTORIES(${TORCH5SPRO_INCLUDE})

# Avoids warning during linkage configuration see: cmake --help-policy CMP0003
# for details.
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

FOREACH(file ${EXAMPLE_SRC})
	STRING(REGEX REPLACE "([a-z;A-Z;0-9]+).*" "\\1" file_exe "${file}" )
	ADD_EXECUTABLE(${file_exe} ${file})
  ADD_DEPENDENCIES(${file_exe} torch5spro-uniheader) #has to be ready
  if (${TORCH_LINKAGE} STREQUAL dynamic)
    TARGET_LINK_LIBRARIES(${file_exe} torch5spro)
  else (${TORCH_LINKAGE} STREQUAL dynamic)
    TARGET_LINK_LIBRARIES(${file_exe} torch5spro-static)
  endif (${TORCH_LINKAGE} STREQUAL dynamic)
  
  ADD_CUSTOM_TARGET(install-${file_exe} COMMAND cp -f ${CMAKE_CURRENT_BINARY_DIR}/${file_exe} ${INSTALL_DIR}/bin DEPENDS ${file_exe})
  ADD_DEPENDENCIES(all-examples ${file_exe})
  ADD_DEPENDENCIES(install-all-examples install-${file_exe})
ENDFOREACH(file)
