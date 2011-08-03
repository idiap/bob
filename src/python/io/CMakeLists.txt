project(io-python)
cmake_minimum_required(VERSION 2.6)

# Python bindings
set(pysrc
   "src/array.cc"
   "src/arrayset.cc"
   "src/exception.cc"
   "src/binfile.cc"
   "src/tensorfile.cc"
   "src/hdf5.cc"
   "src/codec.cc"
   "src/transcode.cc"
   "src/datetime.cc"
   "src/video.cc"
   "src/main.cc"
   )
 
include(../../cmake/macros.cmake)

torch_python_bindings(io "${pysrc}" "io" "")
add_dependencies(pytorch_io core-array-python_header_install)
torch_python_install(io)

# Installs all python scripts
file(COPY script/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin FILES_MATCHING PATTERN "*.py")

# Our test units
torch_python_add_test(python-io-arrayset-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_arrayset.py)
torch_python_add_test(python-io-array-pycodec-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_codec.py)
torch_python_add_test(python-io-arrayset-pycodec-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_codec_set.py)
torch_python_add_test(python-io-array-cxxcodec-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_arraycodecs.py)
torch_python_add_test(python-io-arrayset-cxxcodec-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_arraysetcodecs.py)
torch_python_add_test(python-io-video-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_video.py)
torch_python_add_test(python-io-hdf5-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_hdf5.py)

# Our examples (run and install)
torch_python_add_test(python-io-video2frame-example
  ${CMAKE_CURRENT_SOURCE_DIR}/example/video2frame.py --test)
torch_example_install("python" "io" example/video2frame.py)
