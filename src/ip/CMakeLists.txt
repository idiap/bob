project(ip)

find_path(VLFEAT_INCLUDE_DIR vl/sift.h)
find_library(VLFEAT_LIBRARY NAMES vl)

if(WITH_VLFEAT)
  if(VLFEAT_INCLUDE_DIR AND VLFEAT_LIBRARY)
    find_package_message(VLFEAT "Found VLFEAT: ${VLFEAT_LIBRARY}" "[${VLFEAT_LIBRARY}][${VLFEAT_INCLUDE_DIR}]")
  else()
    message(FATAL_ERROR "VLFeat has not been found. use '-DWITH_VLFEAT=OFF' to disable this check.")
  endif()
endif()

add_subdirectory(cxx)
add_subdirectory(python)
 
set(ENABLED_PACKAGES "${ENABLED_PACKAGES}" PARENT_SCOPE)
