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

include(../../cmake/macros.cmake)

bob_python_package_bindings(core "${pysrc}")

bob_header_install(python-core core "bob::python::core")

bob_python_submodule_bindings(core ndarray "${pysrc_ndarray}")
add_dependencies(pybob_core_ndarray python-core_header_install)

bob_python_submodule_bindings(core random "${pysrc_random}")
add_dependencies(pybob_core_random python-core_header_install)

bob_python_install(core)

# Our test units
bob_python_add_test(python-core-exception-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_exception.py)
bob_python_add_test(python-core-conversion-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_convert.py)
bob_python_add_test(python-core-info-table-test 
  ${CMAKE_CURRENT_SOURCE_DIR}/script/info_table.py)

# Installs all python scripts
file(COPY script/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin FILES_MATCHING PATTERN "*.py")
