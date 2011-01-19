project(sp-python)
cmake_minimum_required(VERSION 2.6)

set(pysrc
   "src/spcore.cc"
   "src/transforms.cc"
   "src/main.cc"
   )
 
include(../../cmake/macros.cmake)

torch_python_bindings(sp "${pysrc}")
torch_python_install(sp)
