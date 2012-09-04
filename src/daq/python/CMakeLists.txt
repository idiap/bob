project(bob_daq_py)

# This defines the dependencies of this package
set(bob_deps "bob_daq;bob_python")
set(shared "${bob_deps}")
set(incdir "${python_INCLUDE_DIRS};${ImageMagick_INCLUDE_DIRS}")

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

list(REMOVE_DUPLICATES incdir)
foreach(inc ${incdir})
  include_directories(SYSTEM ${inc})
endforeach()
bob_add_library(${PROJECT_NAME} "${src}")

# Pkg-Config generator
bob_pkgconfig(${PROJECT_NAME} "${bob_deps}")
