#
# Top level project
#
PROJECT(OOURAFFT)

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
message("\"oourafft\" configuration")
message("C flags: ${CMAKE_C_FLAGS_${CMAKE_BUILD_TYPE}}")
message("C++ flags: ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}")
message("---------------------------------------------")

########################################################
# Build the library

SET(OOURAFFT_SRC alloc.c fftsg.c fftsg2d.c shrtdct.c)

ADD_LIBRARY(oourafft-static STATIC ${OOURAFFT_SRC})
SET_TARGET_PROPERTIES(oourafft-static PROPERTIES OUTPUT_NAME "oourafft")
SET_TARGET_PROPERTIES(oourafft-static PROPERTIES PREFIX "lib")
ADD_LIBRARY(oourafft SHARED ${OOURAFFT_SRC})

INSTALL(DIRECTORY . DESTINATION ${INCLUDE_DIR}/oourafft
          FILES_MATCHING PATTERN "*.h")
INSTALL(TARGETS oourafft LIBRARY DESTINATION ${INSTALL_DIR}/lib)
INSTALL(TARGETS oourafft-static ARCHIVE DESTINATION ${INSTALL_DIR}/lib)
