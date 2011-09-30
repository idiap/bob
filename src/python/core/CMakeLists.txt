project(core-python)
cmake_minimum_required(VERSION 2.6)

# Python bindings
set(pysrc
   "src/version.cc"
   "src/exception.cc"
   "src/logging.cc"
   "src/object.cc"
   "src/tensor.cc"
   "src/profile.cc"
   "src/main.cc"
   )

set(pysrc_pycore
   "src/pycore.cc"
   )

set(pysrc_array
   "src/blitz_numpy.cc"
   "src/tinyvector.cc"
   "src/array_wrapper_example.cc"
   "src/main_array.cc"
   )

set(pysrc_vector
   "src/vector.cc"
   "src/array_vector_1.cc"
   "src/array_vector_2.cc"
   "src/array_vector_3.cc"
   "src/array_vector_4.cc"
   "src/main_vector.cc"
   )

set(pysrc_mapstring
   "src/mapstring.cc"
   "src/arraymapstring.cc"
   "src/main_mapstring.cc"
   )

set(pysrc_tuple
   "src/tuple.cc"
   "src/main_tuple.cc"
   )

set(pysrc_random
   "src/random.cc"
   "src/main_random.cc"
   )

include(../../cmake/macros.cmake)

torch_python_package_bindings(core "${pysrc}")

torch_header_install(python-core core "torch::python::core")

torch_python_submodule_bindings(core utils "${pysrc_pycore}")
add_dependencies(pytorch_core_utils python-core_header_install)

torch_python_submodule_bindings(core array "${pysrc_array}")
add_dependencies(pytorch_core_array python-core_header_install)
target_link_libraries(pytorch_core_array pytorch_core_utils)

torch_python_submodule_bindings(core vector "${pysrc_vector}")
target_link_libraries(pytorch_core_vector pytorch_core_utils)

torch_python_submodule_bindings(core mapstring "${pysrc_mapstring}")
torch_python_submodule_bindings(core tuple "${pysrc_tuple}")
torch_python_submodule_bindings(core random "${pysrc_random}")
torch_python_install(core)

# Our test units
torch_python_add_test(python-core-exception-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_exception.py)

# Installs all python scripts
file(COPY script/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin FILES_MATCHING PATTERN "*.py")
