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

set(pysrc
   "lib/__init__.py"
   "lib/__array__.py"
   "lib/__arrayset__.py"
   "lib/__file__.py"
   "lib/__hdf5__.py"
   "lib/test/__init__.py"
   "lib/test/file.py"
   "lib/test/array.py"
   "lib/test/video.py"
   "lib/test/hdf5.py"
   "lib/test/arrayset.py"
   "lib/example/__init__.py"
   "lib/example/video2frame.py"
   "lib/script/__init__.py"
   "lib/script/filelist.py"
   "lib/script/arrayset_transcode.py"
   "lib/script/array_transcode.py"
   )

bob_python_package_bindings(io "${ccsrc}" "${pysrc}" "core_array")

# Our test units
bob_python_add_unittest(io lib/test/array.py main
  "${CMAKE_CURRENT_SOURCE_DIR}/data")
bob_python_add_unittest(io lib/test/arrayset.py main
  "${CMAKE_CURRENT_SOURCE_DIR}/data")
bob_python_add_unittest(io lib/test/hdf5.py main
  "${CMAKE_CURRENT_SOURCE_DIR}/data")
bob_python_add_unittest(io lib/test/file.py main
  "${CMAKE_CURRENT_SOURCE_DIR}/data")

if(FFMPEG_FOUND)
  bob_python_add_unittest(io lib/test/video.py main
    "${CMAKE_CURRENT_SOURCE_DIR}/data")
endif(FFMPEG_FOUND)

if(FFMPEG_FOUND AND ImageMagick_FOUND)
  # Our examples (run and install)
  bob_python_example(io "video2frame.py" "lib/example/video2frame.py" "main")
  bob_python_add_test(io video2frame.py --test
    ${CMAKE_CURRENT_SOURCE_DIR}/data/test.mov)
endif(FFMPEG_FOUND AND ImageMagick_FOUND)
