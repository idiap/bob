project(sp-python)
cmake_minimum_required(VERSION 2.6)

set(pysrc
   "src/convolution.cc"
   "src/fftfct.cc"
   "src/spcore.cc"
   "src/transforms.cc"
   "src/main.cc"
   )
 
include(../../cmake/macros.cmake)

torch_python_package_bindings(sp "${pysrc}")
torch_python_install(sp)

# Our test units
torch_python_add_test(python-sp-naive-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_naive.py)
torch_python_add_test(python-sp-fftpack-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_fftpack.py)
torch_python_add_test(python-sp-fftpack-blitz-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_fftpack_blitz.py)
torch_python_add_test(python-sp-convolution-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_convolution.py)
