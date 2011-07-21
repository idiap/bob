project(visioner-python)
cmake_minimum_required(VERSION 2.6)

set(pysrc
   "src/localize.cc"
   "src/main.cc"
   )

include(../../cmake/macros.cmake)

# Specially for these bindings, we will have to find libtorch_visioner which
# comes from an external build.
find_library(torch_visioner torch_visioner)
torch_python_bindings(visioner "${pysrc}" "" ${torch_visioner})
add_dependencies(pytorch_visioner core-array-python_header_install visioner)
torch_python_install(visioner)
  
# Our test units
#torch_python_add_test(python-visioner-kmeans-tests 
#  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_visioner.py)
