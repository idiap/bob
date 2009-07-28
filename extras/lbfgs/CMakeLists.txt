#
# Top level project
#
PROJECT(LBFGS)

CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

# Set the compiler flags and build type(release or debug)
#SET(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
SET(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3 -ffast-math -fomit-frame-pointer -msse2")
# For 64-bit archi forcing to emulate 32-bit
#SET(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3 -ffast-math -fomit-frame-pointer")
SET(CMAKE_BUILD_TYPE RELEASE)

#SET(CMAKE_CXX_FLAGS_DEBUG "-g")
#SET(CMAKE_BUILD_TYPE DEBUG)

########################################################
# Build the library

SET(LBFGS_SRC
	lbfgs.c)

ADD_LIBRARY(lbfgs
		${LBFGS_SRC})

########################################################
# Make it verbose

MESSAGE("---------------------------------------------" "")
MESSAGE("Building L-BFGS library" "")
MESSAGE("C RELEASE FLAGS: " "${CMAKE_C_FLAGS_RELEASE}")
MESSAGE("C DEBUG FLAGS:   " "${CMAKE_C_FLAGS_DEBUG}")
MESSAGE("BUILD TYPE:        " "${CMAKE_BUILD_TYPE}")
MESSAGE("---------------------------------------------" "")

########################################################
