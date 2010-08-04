#
# Top level project
#
PROJECT(TORCH5SPRO)

CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

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
  #	core)
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

# Avoids warning during linkage configuration see: cmake --help-policy CMP0003
# for details.
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)
