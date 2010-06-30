#
# Top level project
#
PROJECT(JPEG)

CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

# These are the C flags we like to use with the CMake projects we have here.
set(CMAKE_C_FLAGS_RELEASE "-Wall -g -O3 -ffast-math -fomit-frame-pointer -msse2")
# For 32-bits generation on a 64-bit architecture, uncomment this:
#set(CMAKE_C_FLAGS_RELEASE "-m32 -Wall -g -O3 -ffast-math -fomit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
set(CMAKE_C_FLAGS_DEBUG "-Wall -g")
set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})

########################################################
# Make it verbose

message("---------------------------------------------")
message("\"jpeg\" configuration")
message("C flags: ${CMAKE_C_FLAGS_${CMAKE_BUILD_TYPE}}")
message("C++ flags: ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}")
message("---------------------------------------------")

########################################################
# Build the library

SET(JPEG_SRC
	jcapimin.c
	jcapistd.c
	jccoefct.c
	jccolor.c
	jcdctmgr.c
	jchuff.c
	jcinit.c
	jcmainct.c
	jcmarker.c
	jcmaster.c
	jcomapi.c
	jcparam.c
	jcphuff.c
	jcprepct.c
	jcsample.c
	jctrans.c
	jdapimin.c
	jdapistd.c
	jdatadst.c
	jdatasrc.c
	jdcoefct.c
	jdcolor.c
	jddctmgr.c
	jdhuff.c
	jdinput.c
	jdmainct.c
	jdmarker.c
	jdmaster.c
	jdmerge.c
	jdphuff.c
	jdpostct.c
	jdsample.c
	jdtrans.c
	jerror.c
	jfdctflt.c
	jfdctfst.c
	jfdctint.c
	jidctflt.c
	jidctfst.c
	jidctint.c
	jidctred.c
	jmemmgr.c
	jmemnobs.c
	jquant1.c
	jquant2.c
	jutils.c)

ADD_LIBRARY(jpeg-static STATIC ${JPEG_SRC})
SET_TARGET_PROPERTIES(jpeg-static PROPERTIES OUTPUT_NAME "jpeg")
SET_TARGET_PROPERTIES(jpeg-static PROPERTIES PREFIX "lib")
ADD_LIBRARY(jpeg SHARED ${JPEG_SRC})

INSTALL(DIRECTORY . DESTINATION ${INCLUDE_DIR}/jpeg
          FILES_MATCHING PATTERN "*.h")
INSTALL(TARGETS jpeg LIBRARY DESTINATION ${INSTALL_DIR}/lib)
INSTALL(TARGETS jpeg-static ARCHIVE DESTINATION ${INSTALL_DIR}/lib)
