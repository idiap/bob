project(daq)

# Private dependencies for this package
include(cv.cmake)

find_path(V4L2_INCLUDE_DIRS linux/videodev2.h)

if (V4L2_INCLUDE_DIRS)
  set(V4L2_FOUND TRUE)
else ()
  set(V4L2_FOUND FALSE)
endif ()

mark_as_advanced(V4L2_FOUND)
add_subdirectory(cxx)
add_subdirectory(python)
 
set(ENABLED_PACKAGES "${ENABLED_PACKAGES}" PARENT_SCOPE)
