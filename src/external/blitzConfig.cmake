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

if(BLITZ_FOUND)
  SET(Blitz_LIBRARIES ${Blitz_LIBRARY})

  # and we try to determine if the the found library supports 64-bits array
  # positions.
  include(CheckTypeSize)
  set(CMAKE_REQUIRED_DEFINITIONS "-xc++")
  set(CMAKE_REQUIRED_FLAGS "-lstdc++")
  set(CMAKE_EXTRA_INCLUDE_FILES "${Blitz_INCLUDE_DIR}/blitz/blitz.h")
  CHECK_TYPE_SIZE("blitz::sizeType" BLITZ_SIZETYPE)
  CHECK_TYPE_SIZE("blitz::diffType" BLITZ_DIFFTYPE)
  set(CMAKE_EXTRA_INCLUDE_FILES)
  set(CMAKE_REQUIRED_FLAGS)
  set(CMAKE_REQUIRED_DEFINITIONS)
  if(HAVE_BLITZ_SIZETYPE)
    add_definitions(-DHAVE_BLITZ_SIZETYPE=1)
  else(HAVE_BLITZ_SIZETYPE)
  endif(HAVE_BLITZ_SIZETYPE)
  if(HAVE_BLITZ_DIFFTYPE)
    add_definitions(-DHAVE_BLITZ_DIFFTYPE=1)
  endif(HAVE_BLITZ_DIFFTYPE)
  if(HAVE_BLITZ_SIZETYPE AND HAVE_BLITZ_DIFFTYPE)
    message(STATUS "Blitz has \"sizeType\" and \"diffType\" defined -- you can allocate arrays with more than 2G-pointees")
  else(HAVE_BLITZ_SIZETYPE AND HAVE_BLITZ_DIFFTYPE)
    message(STATUS "Older version of Blitz detected -- please note the 2G-pointee limit for arrays!")
  endif(HAVE_BLITZ_SIZETYPE AND HAVE_BLITZ_DIFFTYPE)

  # and has blitz/tinyvec2.h and not blitz/tinyvec-et.h
  find_file(TINYVEC2_FOUND "blitz/tinyvec2.h" ${Blitz_INCLUDE_DIR})
  if(TINYVEC2_FOUND)
    add_definitions(-DHAVE_BLITZ_TINYVEC2_H=1)
  endif(TINYVEC2_FOUND)

endif(BLITZ_FOUND)

mark_as_advanced(Blitz_INCLUDE_DIR Blitz_LIBRARY)
