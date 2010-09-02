# - Find Blitz library
# Find the native Blitz includes and library
# This module defines
#  Blitz_INCLUDE_DIR, where to find tiff.h, etc.
#  Blitz_LIBRARIES, libraries to link against to use Blitz.
#  Blitz_FOUND, If false, do not try to use Blitz.
# also defined, but not for general use are
#  Blitz_LIBRARY, where to find the Blitz library.

find_path(Blitz_INCLUDE_DIR blitz/blitz.h)

find_library(Blitz_LIBRARY NAMES blitz)

# handle the QUIETLY and REQUIRED arguments and set Blitz_FOUND to TRUE if 
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Blitz DEFAULT_MSG Blitz_LIBRARY Blitz_INCLUDE_DIR)

if(Blitz_FOUND)
  SET(Blitz_LIBRARIES ${Blitz_LIBRARY})
endif(Blitz_FOUND)

mark_as_advanced(Blitz_INCLUDE_DIR Blitz_LIBRARY)
