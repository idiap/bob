project(sp-python)
cmake_minimum_required(VERSION 2.6)

set(ccsrc
   "src/version.cc"
   "src/extrapolate.cc"
   "src/fft_dct.cc"
   "src/main.cc"
   )

set(pysrc
   "lib/__init__.py"
   )

bob_python_package_bindings(sp "${ccsrc}" "${pysrc}" "core_array")

# Our test units
bob_python_add_test(python-sp-fft_dct-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/fft_dct.py)
bob_python_add_test(python-sp-extrapolation-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/extrapolate.py)
