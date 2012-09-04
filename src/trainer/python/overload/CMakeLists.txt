project(bob_trainer_overload_py)

# This defines the dependencies of this package
set(bob_deps "bob_trainer;bob_python")
set(shared "${bob_deps}")
set(incdir ${python_INCLUDE_DIRS})

set(src
   "kmeans_wrappers.cc"
   "gmm_wrappers.cc"
   "main.cc"
   )

# Define the library, compilation and linkage options
foreach(inc ${incdir})
  include_directories(SYSTEM ${inc})
endforeach()
bob_add_library(${PROJECT_NAME} "${src}")
target_link_libraries(${PROJECT_NAME} ${shared})

# Pkg-Config generator
bob_pkgconfig(${PROJECT_NAME} "${bob_deps}")
