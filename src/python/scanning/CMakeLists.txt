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

# Our test units
#torch_python_add_test(python-scanning-facefinder-tests 
#  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_facefinder.py)
#torch_python_add_test(python-scanning-cropping-tests 
#  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_cropping.py)
