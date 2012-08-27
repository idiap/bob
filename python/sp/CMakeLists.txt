project(sp-python)
cmake_minimum_required(VERSION 2.6)

set(ccsrc
   "src/version.cc"
   "src/extrapolate.cc"
   "src/dct.cc"
   "src/fft.cc"
   "src/conv.cc"
   "src/main.cc"
   )

bob_python_package_bindings(sp "${ccsrc}" "python_integration")

# Our test units
bob_python_add_unittest(sp lib/test/fft_dct.py)
bob_python_add_unittest(sp lib/test/extrapolate.py)
