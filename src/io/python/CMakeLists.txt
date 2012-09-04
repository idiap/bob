project(bob_io_py)

# This defines the dependencies of this package
set(bob_deps "bob_io;bob_python")
set(incdir "${python_INCLUDE_DIRS};${HDF5_CXX_INCLUDE_DIR}")

# This defines the list of source files inside this package.
set(src
   "version.cc"
   "exception.cc"
   "file.cc"
   "array.cc"
   "arrayset.cc"
   "hdf5_extras.cc"
   "hdf5.cc"
   "datetime.cc"
   "main.cc"
   )
 
if(FFMPEG_FOUND)
  set(ccsrc "${ccsrc};src/video.cc")
  list(APPEND incdir "${FFMPEG_INCLUDE_DIRS}")
  add_definitions("-D__STDC_CONSTANT_MACROS")
endif(FFMPEG_FOUND)

# Define the library, compilation and linkage options
list(REMOVE_DUPLICATES incdir)
foreach(inc ${incdir})
  include_directories(SYSTEM ${inc})
endforeach()
bob_add_library(${PROJECT_NAME} "${src}")
target_link_libraries(${PROJECT_NAME} ${shared})

# Pkg-Config generator
bob_pkgconfig(${PROJECT_NAME} "${bob_deps}")
