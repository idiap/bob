project(bob_sp_py)

# This defines the dependencies of this package
set(bob_deps "bob_core;bob_python")
set(shared "${bob_deps};${Boost_PYTHON_LIBRARY_RELEASE}")
set(incdir ${python_INCLUDE_DIRS})

# Python bindings
set(src
   "version.cc"
   "extrapolate.cc"
   "dct.cc"
   "fft.cc"
   "conv.cc"
   "main.cc"
   )

foreach(inc ${incdir})
  include_directories(SYSTEM ${inc})
endforeach()
bob_add_library(${PROJECT_NAME} "${src}")
target_link_libraries(${PROJECT_NAME} ${shared})

# Pkg-Config generator
bob_pkgconfig(${PROJECT_NAME} "${bob_deps}")
