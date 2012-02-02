# Try to find if Video for Linux 2 is available

find_path(V4L2_INCLUDE_DIRS linux/videodev2.h)

if (V4L2_INCLUDE_DIRS)
  set(V4L2_FOUND TRUE)
else ()
  set(V4L2_FOUND FALSE)
endif ()

mark_as_advanced(V4L2_FOUND)