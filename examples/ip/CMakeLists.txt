#
# Top level project
#
PROJECT(EXAMPLE)

CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

# Set the include directories and linking libraries
SET(TORCH5SPRO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")
SET(ATLAS_DIR "/usr/lib/atlas")
INCLUDE_DIRECTORIES("${TORCH5SPRO_DIR}")
LINK_LIBRARIES(	"${TORCH5SPRO_DIR}/build_${TORCH_OS}_${TORCH_ARCH}/libtorch5spro.a"
		"${TORCH5SPRO_DIR}/extras/jpeg/build_${TORCH_OS}_${TORCH_ARCH}/libjpeg.a"
		"${ATLAS_DIR}/libblas.a"
		"${ATLAS_DIR}/libblas.so"
		"${ATLAS_DIR}/liblapack.a"
		"${ATLAS_DIR}/liblapack.so"
		"${TORCH5SPRO_DIR}/extras/oourafft/build_${TORCH_OS}_${TORCH_ARCH}/liboourafft.a")

# Set the compiler flags and build type (release or debug)
SET(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
SET(CMAKE_CXX_FLAGS_DEBUG "-g")
#SET(CMAKE_BUILD_TYPE RELEASE)
SET(CMAKE_BUILD_TYPE DEBUG)

MESSAGE("---------------------------------------------" "")
MESSAGE("CXX RELEASE FLAGS: " "${CMAKE_CXX_FLAGS_RELEASE}")
MESSAGE("CXX DEBUG FLAGS:   " "${CMAKE_CXX_FLAGS_DEBUG}")
MESSAGE("BUILD TYPE:        " "${CMAKE_BUILD_TYPE}")
MESSAGE("---------------------------------------------" "")

########################################################
# For each source file build an executable
########################################################

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

FOREACH(file ${EXAMPLE_SRC})
	STRING(REGEX REPLACE "([a-z;A-Z;0-9]+).*" "\\1" file_exe "${file}" )
	ADD_EXECUTABLE(${file_exe} ${file})
ENDFOREACH(file)

########################################################
