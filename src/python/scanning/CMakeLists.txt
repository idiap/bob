project(scanning)
cmake_minimum_required(VERSION 2.6)

set(pysrc
   "src/explorer.cc"
   "src/facefinder.cc"
   "src/pattern.cc"
   "src/scanner.cc"
   "src/ipgeomnorm.cc"
   "src/gtfile.cc"
   "src/main.cc"
   )
 
include(../../cmake/macros.cmake)

torch_python_package_bindings(scanning "${pysrc}")
torch_python_install(scanning)
