project(ip)

find_path(VLFEAT_INCLUDE_DIR vl/sift.h)
find_library(VLFEAT_LIBRARY NAMES vl)

set(VLFEAT_FOUND FALSE)
if(VLFEAT_INCLUDE_DIR AND VLFEAT_LIBRARY)
  set(VLFEAT_FOUND ON)
  set(HAVE_VLFEAT ON CACHE BOOL "Has the vlfeat library installed")
  find_package_message(VLFEAT "Found VLFEAT: ${VLFEAT_LIBRARY}" "[${VLFEAT_LIBRARY}][${VLFEAT_INCLUDE_DIR}]")
endif()

add_subdirectory(cxx)
add_subdirectory(python)
 
set(ENABLED_PACKAGES "${ENABLED_PACKAGES}" PARENT_SCOPE)
