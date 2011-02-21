project(database-python)
cmake_minimum_required(VERSION 2.6)

# Python bindings
set(pysrc
   "src/array.cc"
   "src/arrayset.cc"
   "src/dataset.cc"
   "src/exception.cc"
   "src/rule.cc"
   "src/relation.cc"
   "src/relationset.cc"
   "src/binfile.cc"
   "src/codec.cc"
   "src/main.cc"
   )
 
include(../../cmake/macros.cmake)

torch_python_bindings(database "${pysrc}" "database" "")
torch_python_install(database)

# Our test units
torch_python_add_test(python-database-dataset-read-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_dataset.py)
torch_python_add_test(python-database-dataset-write-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_writing.py)
torch_python_add_test(python-database-binfile-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_binfile.py)
torch_python_add_test(python-database-array-pycodec-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_codec.py)
torch_python_add_test(python-database-arrayset-pycodec-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_codec_set.py)
torch_python_add_test(python-database-array-cxxcodec-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_arraycodecs.py)
torch_python_add_test(python-database-arrayset-cxxcodec-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_arraysetcodecs.py)
