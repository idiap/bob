find_path(VLFEAT_INCLUDE_DIR vl/sift.h)
find_library(VLFEAT_LIBRARY NAMES vl)

set(VLFEAT_FOUND FALSE)
if(VLFEAT_INCLUDE_DIR AND VLFEAT_LIBRARY)
  add_definitions("-D HAVE_VLFEAT=1")
  set(VLFEAT_FOUND TRUE)
  message( STATUS "VLFeat FOUND: Compiling add-on modules...")
else()
  message( STATUS "VLFeat NOT FOUND: Disabling...")
endif()

MARK_AS_ADVANCED(
  VLFEAT_INCLUDE_DIR
  VLFEAT_LIBRARY
  VLFEAT_FOUND
  )
