#
# Top level project
#
PROJECT(TORCH5SPRO)

CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

# Set the source files and includes to be added to the library
SET(TORCH5SPRO_SRC "" CACHE INTERNAL "sources")
SET(TORCH5SPRO_INCLUDE "" CACHE INTERNAL "included")

########################################################
# Additional libraries
########################################################

# CBLAS
ADD_DEFINITIONS(-DUSE_CBLAS)
INCLUDE_DIRECTORIES("/usr/include")

# JPEG
ADD_DEFINITIONS(-DHAVE_JPEG)
INCLUDE_DIRECTORIES("${INCLUDE_DIR}/jpeg")

# OOURA FFT
ADD_DEFINITIONS(-DHAVE_OOURAFFT)
INCLUDE_DIRECTORIES("${INCLUDE_DIR}/oourafft")

# LBFGS
ADD_DEFINITIONS(-DHAVE_LBFGS)
ADD_DEFINITIONS(-DUSE_SSE)
INCLUDE_DIRECTORIES("${INCLUDE_DIR}/lbfgs")

# TIFF
#ADD_DEFINITIONS(-DHAVE_TIFF)

# FFMPEG
# EDIT LES: Remove ffmpeg due to compilation error
INCLUDE_DIRECTORIES("${INCLUDE_DIR}")
INCLUDE_DIRECTORIES("${INCLUDE_DIR}/libavutil")
INCLUDE_DIRECTORIES("${INCLUDE_DIR}/libavformat")
INCLUDE_DIRECTORIES("${INCLUDE_DIR}/libavcodec")
INCLUDE_DIRECTORIES("${INCLUDE_DIR}/libswscale")
ADD_DEFINITIONS(-DHAVE_FFMPEG)
# End EDIT

########################################################
# Sub-directories with source code
########################################################

SET(TORCH5SPRO_SUBDIRS
	th
	core
  ip
	sp
	ap
	scanning
	machine
	trainer
	boosting
	measurer
	geometry)

FOREACH(subdir ${TORCH5SPRO_SUBDIRS})
	ADD_SUBDIRECTORY(${subdir})

	# Add the includes from the subdirectoris too
	SET(TORCH5SPRO_INCLUDE "${TORCH5SPRO_INCLUDE};${${subdir}_INCLUDE}" CACHE
    INTERNAL "included")

	# Add the includes from the subdirectoris too
	SET(TORCH5SPRO_SRC "${TORCH5SPRO_SRC};${${subdir}_SRC}")

ENDFOREACH(subdir)

########################################################
# Build the library

# This is the library with the source files gathered
#	from the sub-directories specified above
INCLUDE_DIRECTORIES(${TORCH5SPRO_INCLUDE})
LINK_DIRECTORIES(${INSTALL_DIR}/lib;/usr/lib/atlas)
ADD_LIBRARY(torch5spro SHARED ${TORCH5SPRO_SRC})
ADD_LIBRARY(torch5spro-static STATIC ${TORCH5SPRO_SRC})
SET_TARGET_PROPERTIES(torch5spro-static PROPERTIES OUTPUT_NAME "torch5spro")
SET_TARGET_PROPERTIES(torch5spro-static PROPERTIES PREFIX "lib")
INSTALL(TARGETS torch5spro LIBRARY DESTINATION ${INSTALL_DIR}/lib)
INSTALL(TARGETS torch5spro-static ARCHIVE DESTINATION ${INSTALL_DIR}/lib)
INSTALL(DIRECTORY ./ DESTINATION ${INCLUDE_DIR}/torch5spro
        FILES_MATCHING PATTERN "*.h")

# There is a dependence between Torch5 and all the extras
# for ffmpeg, specifically, we need to make sure to download it before we
# compile Torch5.
ADD_DEPENDENCIES(torch5spro ffmpeg jpeg lbfgs oourafft) 
ADD_DEPENDENCIES(torch5spro-static ffmpeg jpeg lbfgs oourafft) 

# The linkage which is required for torch5spro
TARGET_LINK_LIBRARIES(torch5spro libjpeg.so)
TARGET_LINK_LIBRARIES(torch5spro liblbfgs.so)
TARGET_LINK_LIBRARIES(torch5spro liboourafft.so)
TARGET_LINK_LIBRARIES(torch5spro libavutil.so)
TARGET_LINK_LIBRARIES(torch5spro libavcodec.so)
TARGET_LINK_LIBRARIES(torch5spro libavformat.so)
TARGET_LINK_LIBRARIES(torch5spro libswscale.so)
TARGET_LINK_LIBRARIES(torch5spro libcblas.so)
TARGET_LINK_LIBRARIES(torch5spro liblapack.so)

########################################################
# Make it verbose

MESSAGE("---------------------------------------------")
MESSAGE("\"Torch5\" configuration")
MESSAGE("C flags: ${CMAKE_C_FLAGS_${CMAKE_BUILD_TYPE}}")
MESSAGE("C++ flags: ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}")
MESSAGE("---------------------------------------------")

########################################################
