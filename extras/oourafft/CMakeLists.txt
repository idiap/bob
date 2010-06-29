#
# Top level project
#
PROJECT(OOURAFFT)

CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

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
