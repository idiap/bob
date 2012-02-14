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

set(pysrc
  "lib/__init__.py"
  "lib/array/__init__.py"
  "lib/script/config.py"
  "lib/script/__init__.py"
  "lib/random/__init__.py"
  )

bob_python_package_bindings(core "${ccsrc}" "${pysrc}" "")
bob_python_submodule_bindings(core array "${ccsrc_ndarray}" "")
bob_python_submodule_bindings(core random "${ccsrc_random}" "")

# Installs and test our scripts
bob_python_script(core "bob_config.py" "lib/script/config.py" "main")
bob_python_add_test(python-core-config-test bob_config.py)

# Our test units
bob_python_add_test(python-core-exception-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_exception.py)
bob_python_add_test(python-core-conversion-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_convert.py)
