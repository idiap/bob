project(core-python)
cmake_minimum_required(VERSION 2.6)

# Python bindings
set(pysrc
   "src/version.cc"
   "src/exception.cc"
   "src/logging.cc"
   "src/profile.cc"
   "src/main.cc"
   )

set(pysrc_ndarray
   "src/ndarray.cc"
   "src/blitz_numpy.cc"
   "src/ndarray_numpy.cc"
   "src/tinyvector.cc"
   "src/typeinfo.cc"
   "src/convert.cc"
   #"src/array_wrapper_example.cc"
   "src/main_ndarray.cc"
   )

set(pysrc_random
   "src/random.cc"
   "src/main_random.cc"
   )

bob_python_package_bindings(core "${pysrc}" "")
bob_python_submodule_bindings(core array "${pysrc_ndarray}" "")
bob_python_submodule_bindings(core random "${pysrc_random}" "")

# Our test units
bob_python_add_test(python-core-exception-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_exception.py)
bob_python_add_test(python-core-conversion-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_convert.py)
bob_python_add_test(python-core-info-table-test 
  ${CMAKE_CURRENT_SOURCE_DIR}/script/info_table.py)

