# -- Try to find netpbm
#
# libnetpbm is a C programming library for reading, writing, and manipulating Netpbm images.
# The library is available at http://netpbm.sourceforge.net.
#
#  -- This module defines :
#
#  NETPBM_FOUND - the system has libnetpbm
#  NETPBM_INCLUDE_DIR - where to find pam.h
#  NETPBM_INCLUDE_DIRS libnetpbm includes
#  NETPBM_LIBRARY - where to find the libnetpbm library
#  NETPBM_LIBRARIES - aditional libraries
#
# Laurent El Shafey <laurent.el-shafey@idiap.ch> 2012

if(NOT NETPBM_INCLUDE_DIR OR NOT NETPBM_INCLUDE_DIRS)
  find_path(NETPBM_INCLUDE_DIR 
    NAMES pam.h
    PATH_SUFFIXES netpbm
    DOC "LibNETPBM include directory"
  )
  set(NETPBM_INCLUDE_DIRS "${NETPBM_INCLUDE_DIR}")
endif()

if(NOT NETPBM_LIBRARY OR NOT NETPBM_LIBRARIES)
  find_library(NETPBM_LIBRARY
    NAMES netpbm
    PATH_SUFFIXES netpbm
    DOC "LibNETPBM library directory"
  )
  set(NETPBM_LIBRARIES "${NETPBM_LIBRARY}")
endif()

if(NETPBM_INCLUDE_DIR AND NETPBM_LIBRARY)
  set(NETPBM_FOUND "YES")
  find_package_message(MyNetpbm "Found libnetpbm: ${NETPBM_INCLUDE_DIR} - ${NETPBM_LIBRARY}" "[${NETPBM_LIBRARY}][${NETPBM_INCLUDE_DIR}]")
endif()

# handle the QUIETLY and REQUIRED arguments and set NETPBM_FOUND to TRUE if 
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
set(NETPBM_FIND_REQUIRED ON)
find_package_handle_standard_args(NETPBM DEFAULT_MSG NETPBM_LIBRARY NETPBM_INCLUDE_DIR)

mark_as_advanced(
  NETPBM_LIBRARY
  NETPBM_LIBRARIES
  NETPBM_INCLUDE_DIR
  NETPBM_INCLUDE_DIRS
)

if(NETPBM_FOUND)
  set(HAVE_NETPBM ON CACHE BOOL "Has libnetpbm installed")
endif()
