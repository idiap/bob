#
# Top level project
#
PROJECT(OOURAFFT)

CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

# Set the compiler flags and build type(release or debug)
#SET(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG")
SET(CMAKE_C_FLAGS_RELEASE "-Wall -O3 -ffast-math -fomit-frame-pointer -msse2")
# For 64-bit archi forcing to emulate 32-bit
#SET(CMAKE_C_FLAGS_RELEASE "-Wall -O3 -ffast-math -fomit-frame-pointer -m32")
SET(CMAKE_BUILD_TYPE RELEASE)

#SET(CMAKE_C_FLAGS_DEBUG "-g")
#SET(CMAKE_BUILD_TYPE DEBUG)

########################################################
# Build the library

SET(OOURAFFT_SRC
	alloc.c
	fftsg.c
	fftsg2d.c
	shrtdct.c)

ADD_LIBRARY(oourafft
		${OOURAFFT_SRC})

########################################################
# Make it verbose

MESSAGE("---------------------------------------------" "")
MESSAGE("Building ooura FFT library" "")
MESSAGE("C RELEASE FLAGS: " "${CMAKE_C_FLAGS_RELEASE}")
MESSAGE("C DEBUG FLAGS:   " "${CMAKE_C_FLAGS_DEBUG}")
MESSAGE("BUILD TYPE:        " "${CMAKE_BUILD_TYPE}")
MESSAGE("---------------------------------------------" "")

########################################################
