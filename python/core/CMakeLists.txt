project(core-python)
cmake_minimum_required(VERSION 2.6)

# Python bindings
set(ccsrc
   "src/version.cc"
   "src/exception.cc"
   "src/logging.cc"
   "src/profile.cc"
   "src/main.cc"
   )

set(ccsrc_ndarray
   "src/ndarray.cc"
   "src/blitz_numpy.cc"
   "src/ndarray_numpy.cc"
   "src/tinyvector.cc"
   "src/typeinfo.cc"
   "src/convert.cc"
   #"src/array_wrapper_example.cc"
   "src/main_ndarray.cc"
   )

set(ccsrc_random
   "src/random.cc"
   "src/main_random.cc"
   )

bob_python_package_bindings(core "${ccsrc}" "")
bob_python_subpackage_bindings(core array "${ccsrc_ndarray}" "")
bob_python_subpackage_bindings(core random "${ccsrc_random}" "")

# Our test units
bob_python_add_unittest(core lib/test/exception.py)
bob_python_add_unittest(core lib/test/convert.py)
