project(io-python)
cmake_minimum_required(VERSION 2.6)

# Python bindings
set(pysrc
   "src/version.cc"
   "src/exception.cc"
   "src/file.cc"
   "src/array.cc"
   "src/arrayset.cc"
   "src/hdf5_extras.cc"
   "src/hdf5.cc"
   "src/datetime.cc"
   "src/main.cc"
   )
 
include(../../cmake/macros.cmake)

if(FFMPEG_FOUND)
  set(pysrc "${pysrc};src/video.cc")
endif(FFMPEG_FOUND)

bob_python_package_bindings(io "${pysrc}")

add_dependencies(pybob_io python-core_header_install)
target_link_libraries(pybob_io pybob_core_ndarray)

bob_python_install(io)

# Installs all python scripts
file(COPY script/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin FILES_MATCHING PATTERN "*.py")

# Our test units
bob_python_add_test(python-io-array-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_array.py)
bob_python_add_test(python-io-arrayset-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_arrayset.py)
bob_python_add_test(python-io-hdf5-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_hdf5.py)
bob_python_add_test(python-io-file-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_file.py)

if(FFMPEG_FOUND)
  bob_python_add_test(python-io-video-tests
    ${CMAKE_CURRENT_SOURCE_DIR}/test/test_video.py)
endif(FFMPEG_FOUND)

if(FFMPEG_FOUND AND ImageMagick_FOUND)
# Our examples (run and install)
bob_python_add_test(python-io-video2frame-example
  ${CMAKE_CURRENT_SOURCE_DIR}/example/video2frame.py --test)
bob_example_install("python" "io" example/video2frame.py)
endif(FFMPEG_FOUND AND ImageMagick_FOUND)
