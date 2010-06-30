#
# Top level project
#
PROJECT(LBFGS)

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
message("\"lbfgs\" configuration")
message("C flags: ${CMAKE_C_FLAGS_${CMAKE_BUILD_TYPE}}")
message("C++ flags: ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}")
message("---------------------------------------------")

########################################################
# Build the library

SET(LBFGS_SRC lbfgs.c)

ADD_LIBRARY(lbfgs-static STATIC ${LBFGS_SRC})
SET_TARGET_PROPERTIES(lbfgs-static PROPERTIES OUTPUT_NAME "lbfgs")
SET_TARGET_PROPERTIES(lbfgs-static PROPERTIES PREFIX "lib")
ADD_LIBRARY(lbfgs SHARED ${LBFGS_SRC})

INSTALL(DIRECTORY . DESTINATION ${INCLUDE_DIR}/lbfgs
          FILES_MATCHING PATTERN "*.h")
INSTALL(TARGETS lbfgs LIBRARY DESTINATION ${INSTALL_DIR}/lib)
INSTALL(TARGETS lbfgs-static ARCHIVE DESTINATION ${INSTALL_DIR}/lib)
