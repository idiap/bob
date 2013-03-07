# - Find Blitz library
# Find the native Blitz includes and library
# This module defines
#  Blitz_INCLUDE_DIR, where to find tiff.h, etc.
#  Blitz_LIBRARIES, libraries to link against to use Blitz.
#  Blitz_FOUND, If false, do not try to use Blitz.
# also defined, but not for general use are
#  Blitz_LIBRARY, where to find the Blitz library.

include(FindPkgConfig)

execute_process(COMMAND ${PKG_CONFIG_EXECUTABLE} blitz --silence-errors --modversion OUTPUT_VARIABLE PKG_CONFIG_blitz_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)

if(PKG_CONFIG_blitz_VERSION)
  #use pkg-config to find blitz
  if(CMAKE_VERSION VERSION_LESS "2.8.2")
    pkg_check_modules(Blitz REQUIRED blitz)
  else()
    #starting at cmake-2.8.2, the QUIET option can be used
    pkg_check_modules(Blitz REQUIRED QUIET blitz)
  endif()
  
  # Resolve Blitz library to a precise path
  set(Blitz_INCLUDE_DIR ${Blitz_INCLUDE_DIRS})
  set(Blitz_RESOLVED_LIBRARY "")
  resolve_library(${Blitz_LIBRARIES} "${Blitz_LIBRARY_DIRS}" Blitz_RESOLVED_LIBRARY)
  set(Blitz_RESOLVED_LIBRARY ${Blitz_RESOLVED_LIBRARY} CACHE INTERNAL "Resolved Blitz library")
else(PKG_CONFIG_blitz_VERSION)
  find_path(Blitz_INCLUDE_DIR blitz/blitz.h)

  find_library(Blitz_LIBRARY NAMES blitz)

  # handle the QUIETLY and REQUIRED arguments and set Blitz_FOUND to TRUE if 
  # all listed variables are TRUE
  include(FindPackageHandleStandardArgs)
  set(Blitz_FIND_REQUIRED ON)
  find_package_handle_standard_args(Blitz DEFAULT_MSG Blitz_LIBRARY Blitz_INCLUDE_DIR)

  set(Blitz_RESOLVED_LIBRARY ${Blitz_LIBRARY} CACHE INTERNAL "Resolved Blitz library")
  set(Blitz_INCLUDE_DIRS ${Blitz_INCLUDE_DIR})
endif(PKG_CONFIG_blitz_VERSION)

if(Blitz_FOUND)
  # and we try to determine if the the found library supports 64-bits array
  # positions.
  include(CheckCXXSourceCompiles)
  set(CMAKE_REQUIRED_INCLUDES "${Blitz_INCLUDE_DIR}")
  CHECK_CXX_SOURCE_COMPILES("#include <blitz/blitz.h>
    int main() { blitz::sizeType s; blitz::diffType d; }" HAVE_BLITZ_SPECIAL_TYPES)
  set(CMAKE_REQUIRED_INCLUDES)

  # and has blitz/tinyvec2.h and not blitz/tinyvec-et.h
  find_file(HAVE_BLITZ_TINYVEC2_H "blitz/tinyvec2.h" ${Blitz_INCLUDE_DIR})

  include(FindPackageHandleStandardArgs)
  find_package_message(Blitz "Found Blitz++: ${Blitz_LIBRARIES} (>2G-pointees: ${HAVE_BLITZ_SPECIAL_TYPES}; New: ${HAVE_BLITZ_TINYVEC2_H})" "[${Blitz_LIBRARIES}][${Blitz_INCLUDE_DIR}]")

else(Blitz_FOUND)

  message(FATAL_ERROR "Blitz++ was not found. Bob requires it for building.")

endif(Blitz_FOUND)

mark_as_advanced(Blitz_INCLUDE_DIR Blitz_LIBRARY)
