project(sp-python)
cmake_minimum_required(VERSION 2.6)

set(pysrc
   "src/version.cc"
   "src/extrapolate.cc"
   "src/fft_dct.cc"
   "src/main.cc"
   )
 
include(../../cmake/macros.cmake)

bob_python_package_bindings(sp "${pysrc}")
add_dependencies(pybob_sp python-core_header_install)
target_link_libraries(pybob_sp pybob_core_ndarray)

bob_python_install(sp)

# Our test units
bob_python_add_test(python-sp-fft_dct-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/fft_dct.py)
bob_python_add_test(python-sp-extrapolation-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/extrapolate.py)
