project(bob_measure_py)

# This defines the dependencies of this package
set(bob_deps "bob_measure;bob_python")
set(shared "${bob_deps}")
set(incdir ${python_INCLUDE_DIRS})

# Python bindings
set(src
   "error.cc"
   "main.cc"
   )

foreach(inc ${incdir})
  include_directories(SYSTEM ${inc})
endforeach()
bob_add_library(${PROJECT_NAME} "${src}")
target_link_libraries(${PROJECT_NAME} ${shared})

# Pkg-Config generator
bob_pkgconfig(${PROJECT_NAME} "${bob_deps}")
