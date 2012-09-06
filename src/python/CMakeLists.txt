project(bob_python)

# This defines the dependencies of this package
set(bob_deps "bob_core")
set(shared "${bob_deps};${Boost_PYTHON_LIBRARY_RELEASE};${PYTHON_LIBRARIES}")
set(incdir ${py_incdir})

# Python bindings
set(src
   "ndarray.cc"
   )

# Define the library, compilation and linkage options
bob_sort_headers(incdir)
foreach(inc ${incdir})
  include_directories(SYSTEM ${inc})
endforeach()
bob_add_library(${PROJECT_NAME} "${src}")
target_link_libraries(${PROJECT_NAME} ${shared})
set_target_properties(bob_python PROPERTIES COMPILE_FLAGS "-Wno-long-long -Wno-unused-function")

# Pkg-Config generator
bob_pkgconfig(${PROJECT_NAME} "${bob_deps}")
