project(bob_daq_py${PYVER})

# This defines the dependencies of this package
set(bob_deps "bob_daq;bob_python")
set(shared "${bob_deps}")
set(incdir ${py_incdir};${ImageMagick_INCLUDE_DIRS})

set(src
   "version.cc"
   "main.cc"
   )

if (FFMPEG_FOUND)
  list(APPEND src
    "all.cc"
    )
  list(APPEND incdir "${FFMPEG_INCLUDE_DIRS}")
  add_definitions("-D__STDC_CONSTANT_MACROS")
endif()

if (OPENCV_FOUND)
  list(APPEND incdir ${OpenCV_INCLUDE_DIRS})
endif()

include(${QT_USE_FILE})
list(APPEND incdir ${QT_INCLUDE_DIR})

# Define the library, compilation and linkage options
bob_sort_headers(incdir)
foreach(inc ${incdir})
  include_directories(SYSTEM ${inc})
endforeach()
bob_add_library(${PROJECT_NAME} "${src}")
target_link_libraries(${PROJECT_NAME} ${shared})

# Pkg-Config generator
bob_pkgconfig(${PROJECT_NAME} "${bob_deps}")
