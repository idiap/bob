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
   "src/array_check.cc"
   "src/repmat.cc"
   "src/reshape.cc"
   "src/main.cc"
   )
 
set(pysrc_array
   "src/TypeMapper.cc"
   "src/ndarray.cc"
   "src/tinyvector.cc"
   "src/range.cc"
   "src/array_base.cc"
   "src/array_constructors.cc"
   "src/array_indexing.cc"
   "src/array_indexing_1.cc"
   "src/array_indexing_2.cc"
   "src/array_indexing_3.cc"
   "src/array_indexing_4.cc"
   "src/array_info.cc"
   "src/array_reduction_1.cc"
   "src/array_reduction_2.cc"
   "src/array_reduction_3.cc"
   "src/array_reduction_4.cc"
   "src/array_math_1.cc"
   "src/array_math_2.cc"
   "src/array_math_3.cc"
   "src/array_math_4.cc"
   "src/array_memory.cc"
   "src/array_arithmetics_1.cc"
   "src/array_arithmetics_2.cc"
   "src/array_arithmetics_3.cc"
   "src/array_arithmetics_4.cc"
   "src/array_order.cc"
   "src/array_cast.cc"
   "src/array_convert.cc"
   "src/array_storage.cc"
   "src/main_array.cc"
   "src/array_as_row.cc"
   )

set(pysrc_vector
   "src/vector.cc"
   "src/arrayvector.cc"
   "src/main_vector.cc"
   )

set(pysrc_map
   "src/map.cc"
   "src/arraymap.cc"
   "src/main_map.cc"
   )

set(pysrc_random
   "src/random.cc"
   "src/main_random.cc"
   )

include(../../cmake/macros.cmake)

torch_python_package_bindings(core "${pysrc}")
torch_header_install(core-array-python core "torch::python::core")
torch_python_submodule_bindings(core array "${pysrc_array}")
add_dependencies(pytorch_core_array core-array-python_header_install)
torch_python_submodule_bindings(core vector "${pysrc_vector}")
torch_python_submodule_bindings(core map "${pysrc_map}")
torch_python_submodule_bindings(core random "${pysrc_random}")
torch_python_install(core)

# Our test units
torch_python_add_test(python-core-array-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_array.py)
torch_python_add_test(python-core-exception-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_exception.py)

# Installs all python scripts
file(COPY script/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin FILES_MATCHING PATTERN "*.py")
