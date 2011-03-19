project(math-python)
cmake_minimum_required(VERSION 2.6)

set(pysrc
   "src/linsolve.cc"
   "src/svd.cc"
   "src/main.cc"
   )
 
include(../../cmake/macros.cmake)

torch_python_package_bindings(math "${pysrc}")
torch_python_install(math)

# Our test units
