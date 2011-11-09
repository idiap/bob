project(io-python)
cmake_minimum_required(VERSION 2.6)

# Python bindings
set(pysrc
   "src/exception.cc"
   "src/file.cc"
   "src/array.cc"
   "src/arrayset.cc"
   "src/hdf5_extras.cc"
   "src/hdf5.cc"
   "src/video.cc"
   "src/datetime.cc"
   "src/main.cc"
   )
 
include(../../cmake/macros.cmake)

torch_python_package_bindings(io "${pysrc}")

add_dependencies(pytorch_io python-core_header_install)
target_link_libraries(pytorch_io pytorch_core_ndarray)

torch_python_install(io)

# Installs all python scripts
file(COPY script/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin FILES_MATCHING PATTERN "*.py")

# Our test units
torch_python_add_test(python-io-arrayset-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_arrayset.py)
torch_python_add_test(python-io-video-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_video.py)
torch_python_add_test(python-io-hdf5-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_hdf5.py)

# Our examples (run and install)
torch_python_add_test(python-io-video2frame-example
  ${CMAKE_CURRENT_SOURCE_DIR}/example/video2frame.py --test)
torch_example_install("python" "io" example/video2frame.py)
