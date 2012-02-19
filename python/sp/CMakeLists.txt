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
   "lib/test/__init__.py"
   "lib/test/fft_dct.py"
   "lib/test/extrapolate.py"
   )

bob_python_package_bindings(sp "${ccsrc}" "${pysrc}" "core_array")

# Our test units
bob_python_add_unittest(sp lib/test/fft_dct.py main)
bob_python_add_unittest(sp lib/test/extrapolate.py main)
