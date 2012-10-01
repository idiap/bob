project(bob_core_py${PYVER})

add_subdirectory(random)

# This defines the dependencies of this package
set(bob_deps "bob_core;bob_python")
set(shared "${bob_deps}")
set(incdir ${py_incdir})

# Google Perftools detection and linkage
find_path(GOOGLE_PERFTOOLS_INCLUDE_1 NAMES google/profiler.h)
find_path(GOOGLE_PERFTOOLS_INCLUDE_2 NAMES google/tcmalloc.h)
find_library(GOOGLE_PERFTOOLS_LIBRARY NAMES profiler)

if(GOOGLE_PERFTOOLS_INCLUDE_1 AND GOOGLE_PERFTOOLS_INCLUDE_2 AND 
   GOOGLE_PERFTOOLS_LIBRARY)
  list(APPEND incdir "${GOOGLE_PERFTOOLS_INCLUDE_1}")
  list(APPEND incdir "${GOOGLE_PERFTOOLS_INCLUDE_2}")
  list(REMOVE_DUPLICATES incdir)
  list(APPEND shared "${GOOGLE_PERFTOOLS_LIBRARY}")
  find_package_message(GOOGLE_PERFTOOLS "Found Google PerfTools: ${GOOGLE_PERFTOOLS_INCLUDE_DIRS}" "[${GOOGLE_PERFTOOLS_INCLUDE_DIRS}][${GOOGLE_PERFTOOLS_LIBRARY}]")
  set(HAVE_GOOGLE_PERFTOOLS ON CACHE BOOL "Has Google's Perftools installed")
endif()
  
# Python bindings
set(src
   "exception.cc"
   "logging.cc"
   "profile.cc"
   "convert.cc"
   "tinyvector.cc"
   "typeinfo.cc"
   "version.cc"
   "blitz_numpy.cc"
   "ndarray_numpy.cc"
   "numpy_scalars.cc"
   "main.cc"
   )

# Define the library, compilation and linkage options
bob_sort_headers(incdir)
foreach(inc ${incdir})
  include_directories(SYSTEM ${inc})
endforeach()
bob_add_library(${PROJECT_NAME} "${src}")
target_link_libraries(${PROJECT_NAME} ${shared})

# Pkg-Config generator
bob_pkgconfig(${PROJECT_NAME} "${bob_deps}")
