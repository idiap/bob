#
# Top level project
#
PROJECT(JPEG)

CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

# Set the compiler flags and build type(release or debug)
SET(CMAKE_C_FLAGS_RELEASE "-Wall -O3 -ffast-math -fomit-frame-pointer -msse2")
# For 64-bit archi forcing to emulate 32-bit
#SET(CMAKE_C_FLAGS_RELEASE "-Wall -O3 -ffast-math -fomit-frame-pointer -m32")

SET(CMAKE_BUILD_TYPE RELEASE)

#SET(CMAKE_C_FLAGS_DEBUG "-g")
#SET(CMAKE_BUILD_TYPE DEBUG)

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

ADD_LIBRARY(jpeg
		${JPEG_SRC})

########################################################
# Make it verbose

MESSAGE("---------------------------------------------" "")
MESSAGE("Building jpeg library" "")
MESSAGE("C RELEASE FLAGS: " "${CMAKE_C_FLAGS_RELEASE}")
MESSAGE("C DEBUG FLAGS:   " "${CMAKE_C_FLAGS_DEBUG}")
MESSAGE("BUILD TYPE:        " "${CMAKE_BUILD_TYPE}")
MESSAGE("---------------------------------------------" "")

########################################################
