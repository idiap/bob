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
INCLUDE_DIRECTORIES(${cblas_INCLUDE};${lapack_INCLUDE})
LINK_DIRECTORIES(${INSTALL_DIR}/lib)

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
IF(ffmpeg_FOUND)
  ADD_DEFINITIONS(-DHAVE_FFMPEG)
  INCLUDE_DIRECTORIES(${ffmpeg_INCLUDE_DIRS})
  LINK_DIRECTORIES(${ffmpeg_LIBRARY_DIRS})
ENDIF(ffmpeg_FOUND)

########################################################
# Sub-directories with source code
########################################################

SET(TORCH5SPRO_SUBDIRS
	th)
#	core
#	ip
#	sp
#	ap
#	scanning
#	machine
#	trainer
#	boosting
#	measurer
#	geometry)

FOREACH(subdir ${TORCH5SPRO_SUBDIRS})
	ADD_SUBDIRECTORY(${subdir})
ENDFOREACH(subdir)

########################################################
# Build the library

# Avoids warning during linkage configuration see: cmake --help-policy CMP0003
# for details.
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

# This is the library with the source files gathered
#	from the sub-directories specified above
INCLUDE_DIRECTORIES(${TORCH5SPRO_INCLUDE})
ADD_LIBRARY(torch5spro SHARED ${TORCH5SPRO_SRC})
ADD_LIBRARY(torch5spro-static STATIC ${TORCH5SPRO_SRC})
SET_TARGET_PROPERTIES(torch5spro-static PROPERTIES OUTPUT_NAME "torch5spro")
SET_TARGET_PROPERTIES(torch5spro-static PROPERTIES PREFIX "lib")
INSTALL(TARGETS torch5spro LIBRARY DESTINATION ${INSTALL_DIR}/lib)
INSTALL(TARGETS torch5spro-static ARCHIVE DESTINATION ${INSTALL_DIR}/lib)

# There is a dependence between Torch5 and all the extras
ADD_DEPENDENCIES(torch5spro jpeg lbfgs oourafft) 
ADD_DEPENDENCIES(torch5spro-static jpeg lbfgs oourafft) 

# The linkage which is required for torch5spro
TARGET_LINK_LIBRARIES(torch5spro-static libjpeg.a)
TARGET_LINK_LIBRARIES(torch5spro-static liblbfgs.a)
TARGET_LINK_LIBRARIES(torch5spro-static liboourafft.a)
TARGET_LINK_LIBRARIES(torch5spro-static ${cblas_ARCHIVES})
TARGET_LINK_LIBRARIES(torch5spro -ljpeg)
TARGET_LINK_LIBRARIES(torch5spro -llbfgs)
TARGET_LINK_LIBRARIES(torch5spro -loourafft)
TARGET_LINK_LIBRARIES(torch5spro ${cblas_LIBRARIES})

# The ffmpeg addons
IF(ffmpeg_FOUND)
  TARGET_LINK_LIBRARIES(torch5spro ${ffmpeg_LIBRARIES})
  FOREACH(ffmod ${ffmpeg_STATIC_LIBRARIES})
    TARGET_LINK_LIBRARIES(torch5spro-static lib${ffmod}.a)
  ENDFOREACH(ffmod ${ffmpeg_STATIC_LIBRARIES})
ENDIF(ffmpeg_FOUND)

# This will trigger cmake to immediately copy Torch headers into the install
# location
FILE(COPY ./ DESTINATION ${INCLUDE_DIR}/torch5spro
     FILES_MATCHING PATTERN "*.h")
# This will build the unified header file, in the end, after the shared library
# has been built
ADD_CUSTOM_TARGET(torch5spro-uniheader ALL COMMAND ${DIST_ROOT}/makeinclude.py 
                  ${CMAKE_CURRENT_SOURCE_DIR}
                  ${INCLUDE_DIR}/torch5spro/torch5spro.h 
                  TensorGen.h THTensorGen.h THStorageGen.h
                  DEPENDS torch5spro
                  COMMENT "Building Torch unified header")

########################################################
# Make it verbose

MESSAGE("---------------------------------------------")
MESSAGE("\"Torch5\" configuration")
MESSAGE("C flags: ${CMAKE_C_FLAGS_${CMAKE_BUILD_TYPE}}")
MESSAGE("C++ flags: ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}")
MESSAGE("---------------------------------------------")

########################################################
