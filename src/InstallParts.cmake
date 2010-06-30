# Creates and install libraries for individual subprojects
# Andre Anjos - 28.june.2010

# This will compile and install the libraries for this project.
# WARNING: This does not work because of the interdependence between all torch5
# projects!!
# set(libname "t5${THIS}")
# ADD_LIBRARY(${libname} SHARED ${${THIS}_SRC})
# ADD_LIBRARY(${libname}-static STATIC ${${THIS}_SRC})
# SET_TARGET_PROPERTIES(${libname}-static PROPERTIES OUTPUT_NAME ${libname})
# SET_TARGET_PROPERTIES(${libname}-static PROPERTIES PREFIX "lib")
# INSTALL(TARGETS ${libname} LIBRARY DESTINATION ${INSTALL_DIR}/lib)
# INSTALL(TARGETS ${libname}-static ARCHIVE DESTINATION ${INSTALL_DIR}/lib)

# This handles the installation of the header files
INSTALL(DIRECTORY . DESTINATION ${TORCH5SPRO_INCLUDE_DIR}
        FILES_MATCHING PATTERN "*.h")
