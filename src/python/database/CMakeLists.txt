project(database-python)
cmake_minimum_required(VERSION 2.6)

# Python bindings
set(pysrc
   "src/array.cc"
   "src/arrayset.cc"
   "src/dataset.cc"
   "src/xmlparser.cc"
   "src/member.cc"
   "src/rule.cc"
   "src/main.cc"
   "src/relation.cc"
   "src/relationset.cc"
   )
 
include(../../cmake/macros.cmake)

torch_python_bindings(database "${pysrc}" "core" "")
torch_python_install(database)
