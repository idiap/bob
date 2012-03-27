find_path(VLFEAT_INCLUDE_DIR vl/sift.h)
find_library(VLFEAT_LIBRARY NAMES vl)

set(VLFEAT_FOUND FALSE)
if(VLFEAT_INCLUDE_DIR AND VLFEAT_LIBRARY)
  add_definitions("-D HAVE_VLFEAT=1")
  set(VLFEAT_FOUND TRUE)
  find_package_message(VLFEAT "Found VLFeat: ${VLFEAT_LIBRARY}" "[${VLFEAT_LIBRARY}][${VLFEAT_INCLUDE_DIR}]")
else()
  if(NOT VLFEAT_INCLUDE_DIR)
    message( STATUS "VLFeat include file <vl/sift.h> not found!")
  endif()
  if(NOT VLFEAT_LIBRARY)
    message( STATUS "VLFeat library not found!")
  endif()
endif()

MARK_AS_ADVANCED(
  VLFEAT_INCLUDE_DIR
  VLFEAT_LIBRARY
  VLFEAT_FOUND
  )
