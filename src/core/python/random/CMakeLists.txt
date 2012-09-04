project(core_random_py)

# This defines the dependencies of this package
set(bob_deps "bob_core;bob_python")
set(shared "${bob_deps}")
set(incdir ${python_INCLUDE_DIRS};${boost_INCLUDE_DIRS};${Blitz_INCLUDE_DIR})

# Python bindings
set(src
   "random.cc"
   "main.cc"
   )

# Define the library, compilation and linkage options
list(REMOVE_DUPLICATES incdir)
foreach(inc ${incdir})
  include_directories(SYSTEM ${inc})
endforeach()
bob_add_library(${PROJECT_NAME} "${src}")
target_link_libraries(${PROJECT_NAME} ${shared})

# Pkg-Config generator
bob_pkgconfig(${PROJECT_NAME} "${bob_deps}")
