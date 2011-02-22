# - Find matio library
# Find the native matio includes and library
# This module defines
#  matio_INCLUDE_DIR, where to find tiff.h, etc.
#  matio_LIBRARIES, libraries to link against to use Matio.
#  MATIO_FOUND, If false, do not try to use Matio.
# also defined, but not for general use are
#  matio_LIBRARY, where to find the Matio library.

find_path(matio_INCLUDE_DIR matio.h)

find_library(matio_LIBRARY NAMES matio)

# handle the QUIETLY and REQUIRED arguments and set MATIO_FOUND to TRUE if 
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(matio DEFAULT_MSG matio_LIBRARY matio_INCLUDE_DIR)

if(MATIO_FOUND)
  SET(matio_LIBRARIES ${matio_LIBRARY})
  message( STATUS "Matio FOUND: Compiling add-on modules...")
else(MATIO_FOUND)
  message( STATUS "Matio NOT FOUND: Disabling...")
endif(MATIO_FOUND)

mark_as_advanced(matio_INCLUDE_DIR matio_LIBRARY)
