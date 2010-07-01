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
	diffusionVcycle.cc
	tanTriggs.cc
	histoEqualTensor.cc
	multiscaleRetinexTensor.cc
	selfQuotientImageTensor.cc
	relaxationTensor.cc
	diffusionVcycleTensor.cc
	tanTriggsTensor.cc)

LINK_DIRECTORIES(${INSTALL_DIR}/lib;/usr/lib/atlas)
INCLUDE_DIRECTORIES(${TORCH5SPRO_INCLUDE})

SET(ATLAS_DIR "/usr/lib/atlas") #maybe we need this?
FOREACH(file ${EXAMPLE_SRC})
	STRING(REGEX REPLACE "([a-z;A-Z;0-9]+).*" "\\1" file_exe "${file}" )
	ADD_EXECUTABLE(${file_exe} ${file})
  ADD_DEPENDENCIES(${file_exe} jpeg) #has to compile before
  ADD_DEPENDENCIES(${file_exe} oourafft) #has to compile before
  ADD_DEPENDENCIES(${file_exe} torch5spro-uniheader) #has to be ready
  if (${TORCH_LINKAGE} STREQUAL dynamic)
    TARGET_LINK_LIBRARIES(${file_exe} torch5spro)
    TARGET_LINK_LIBRARIES(${file_exe} libjpeg.so)
    TARGET_LINK_LIBRARIES(${file_exe} liboourafft.so)
    TARGET_LINK_LIBRARIES(${file_exe} libblas.so)
  else (${TORCH_LINKAGE} STREQUAL dynamic)
    TARGET_LINK_LIBRARIES(${file_exe} torch5spro-static)
    TARGET_LINK_LIBRARIES(${file_exe} libjpeg.a)
    TARGET_LINK_LIBRARIES(${file_exe} liboourafft.a)
    TARGET_LINK_LIBRARIES(${file_exe} libblas.a)
    TARGET_LINK_LIBRARIES(${file_exe} liblapack.a)
    TARGET_LINK_LIBRARIES(${file_exe} libatlas.a)
  endif (${TORCH_LINKAGE} STREQUAL dynamic)
  
  ADD_CUSTOM_TARGET(install-${file_exe} COMMAND cp -f ${CMAKE_CURRENT_BINARY_DIR}/${file_exe} ${INSTALL_DIR}/bin DEPENDS ${file_exe})
  ADD_DEPENDENCIES(all-examples ${file_exe})
  ADD_DEPENDENCIES(install-all-examples install-${file_exe})
ENDFOREACH(file)
