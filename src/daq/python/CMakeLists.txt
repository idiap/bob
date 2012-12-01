project(bob_daq_py${PYVER})

# This defines the dependencies of this package
set(bob_deps "bob_daq;bob_python")
set(shared "${bob_deps}")
set(incdir ${py_incdir};${JPEG_INCLUDE_DIRS};${FFMPEG_INCLUDE_DIRS})

# This is required for FFMPEG compilation
add_definitions("-D__STDC_CONSTANT_MACROS")

set(src
   "version.cc"
   "all.cc"
   "main.cc"
   )

if (OPENCV_FOUND)
  list(APPEND incdir ${OpenCV_INCLUDE_DIRS})
endif()

include(${QT_USE_FILE})
list(APPEND incdir ${QT_INCLUDE_DIR})

if (V4L2_FOUND)
  add_definitions(-DV4L2)
endif () 

# Define the library, compilation and linkage options
bob_sort_headers(incdir)
foreach(inc ${incdir})
  include_directories(SYSTEM ${inc})
endforeach()
bob_add_library(${PROJECT_NAME} "${src}")
target_link_libraries(${PROJECT_NAME} ${shared})

# Pkg-Config generator
bob_pkgconfig(${PROJECT_NAME} "${bob_deps}")
