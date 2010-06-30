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
  TARGET_LINK_LIBRARIES(${file_exe} torch5spro)
  TARGET_LINK_LIBRARIES(${file_exe} libjpeg.so)
  TARGET_LINK_LIBRARIES(${file_exe} liboourafft.so)
  TARGET_LINK_LIBRARIES(${file_exe} libblas.so)
  TARGET_LINK_LIBRARIES(${file_exe} liblapack.so)
  ADD_CUSTOM_TARGET(install-${file_exe} COMMAND cp -f ${CMAKE_CURRENT_BINARY_DIR}/${file_exe} ${INSTALL_DIR}/bin)

  ADD_DEPENDENCIES(build-examples ${file_exe})
  ADD_DEPENDENCIES(install-examples install-${file_exe})
ENDFOREACH(file)
