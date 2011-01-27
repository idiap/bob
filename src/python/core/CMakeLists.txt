project(core-python)
cmake_minimum_required(VERSION 2.6)

# Python bindings
set(pysrc
   "src/exception.cc"
   "src/logging.cc"
   "src/object.cc"
   "src/file.cc"
   "src/tensor.cc"
   "src/tensorfile.cc"
   "src/profile.cc"
   "src/DataSet.cc"
   "src/ListDataSet.cc"
   "src/main.cc"
   )
 
set(pysrc_array
   "src/TypeMapper.cc"
   "src/ndarray.cc"
   "src/tinyvector.cc"
   "src/array.cc"
   "src/array_bool_1.cc"
   "src/array_int8_1.cc"
   "src/array_uint8_1.cc"
   "src/array_int64_1.cc"
   "src/array_uint64_1.cc"
   "src/array_float32_1.cc"
   "src/array_complex64_1.cc"
   "src/array_int16_1.cc"
   "src/array_uint16_1.cc"
   "src/array_float64_1.cc"
   "src/array_int32_1.cc"
   "src/array_uint32_1.cc"
   "src/array_complex128_1.cc"
   "src/array_bool_2.cc"
   "src/array_int8_2.cc"
   "src/array_uint8_2.cc"
   "src/array_int64_2.cc"
   "src/array_uint64_2.cc"
   "src/array_float32_2.cc"
   "src/array_complex64_2.cc"
   "src/array_int16_2.cc"
   "src/array_uint16_2.cc"
   "src/array_float64_2.cc"
   "src/array_int32_2.cc"
   "src/array_uint32_2.cc"
   "src/array_complex128_2.cc"
   "src/array_bool_3.cc"
   "src/array_int8_3.cc"
   "src/array_uint8_3.cc"
   "src/array_int64_3.cc"
   "src/array_uint64_3.cc"
   "src/array_float32_3.cc"
   "src/array_complex64_3.cc"
   "src/array_int16_3.cc"
   "src/array_uint16_3.cc"
   "src/array_float64_3.cc"
   "src/array_int32_3.cc"
   "src/array_uint32_3.cc"
   "src/array_complex128_3.cc"
   "src/array_bool_4.cc"
   "src/array_int8_4.cc"
   "src/array_uint8_4.cc"
   "src/array_int64_4.cc"
   "src/array_uint64_4.cc"
   "src/array_float32_4.cc"
   "src/array_complex64_4.cc"
   "src/array_int16_4.cc"
   "src/array_uint16_4.cc"
   "src/array_float64_4.cc"
   "src/array_int32_4.cc"
   "src/array_uint32_4.cc"
   "src/array_complex128_4.cc"
   "src/main_array.cc"
   )

include(../../cmake/macros.cmake)

torch_header_install(core "torch::python::core")
torch_python_package_bindings(core "${pysrc}")
torch_python_submodule_bindings(core array "${pysrc_array}")
torch_python_install(core)

# Our test units
torch_python_add_test(python-core-array-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_array.py)
torch_python_add_test(python-core-exception-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_exception.py)
torch_python_add_test(python-core-listdataset-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_listdataset.py)
