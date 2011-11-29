project(sp-python)
cmake_minimum_required(VERSION 2.6)

set(pysrc
   "src/version.cc"
   "src/convolution.cc"
   "src/extrapolate.cc"
   "src/fft_dct.cc"
   "src/spcore.cc"
   "src/main.cc"
   )
 
include(../../cmake/macros.cmake)

torch_python_package_bindings(sp "${pysrc}")
add_dependencies(pytorch_sp python-core_header_install)
target_link_libraries(pytorch_sp pytorch_core_ndarray)

torch_python_install(sp)

# Our test units
torch_python_add_test(python-sp-fft_dct-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/fft_dct.py)
torch_python_add_test(python-sp-convolution-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_convolution.py)
torch_python_add_test(python-sp-extrapolation-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/extrapolate.py)
