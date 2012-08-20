project(io-python)
cmake_minimum_required(VERSION 2.6)

# Python bindings
set(ccsrc
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
 
if(FFMPEG_FOUND)
  set(ccsrc "${ccsrc};src/video.cc")
endif(FFMPEG_FOUND)

add_definitions("-D__STDC_CONSTANT_MACROS")

bob_python_package_bindings(io "${ccsrc}" "python_integration")

# Our test units
# TTT bob_python_add_unittest(io lib/test/array.py)
# TTT bob_python_add_unittest(io lib/test/arrayset.py)
# TTT bob_python_add_unittest(io lib/test/hdf5.py)
# TTT bob_python_add_unittest(io lib/test/file.py)

#if(FFMPEG_FOUND)
  # TTT bob_python_add_unittest(io lib/test/video.py)
#endif(FFMPEG_FOUND)

#if(FFMPEG_FOUND AND ImageMagick_FOUND)
  # Our examples (run and install)
  # EEE bob_python_example(io video2frame.py lib/example/video2frame.py)
  # TTT bob_python_add_test(io video2frame.py --test ${CMAKE_CURRENT_SOURCE_DIR}/data/test.mov)
#endif(FFMPEG_FOUND AND ImageMagick_FOUND)
