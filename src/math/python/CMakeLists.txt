project(bob_math_py${PYVER})

# This defines the dependencies of this package
set(bob_deps "bob_math;bob_python")
set(shared "${bob_deps}")
set(incdir ${py_incdir})

# Python bindings
set(src
   "interiorpointLP.cc"
   "linsolve.cc"
   "norminv.cc"
   "stats.cc"
   "histogram.cc"
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
