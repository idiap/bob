project(ip-python)
cmake_minimum_required(VERSION 2.6)

set(pysrc
   "src/ipcore.cc"
   "src/color.cc"
   "src/oldcolor.cc"
   "src/image.cc"
   "src/lbp.cc"
   "src/filters.cc"
   "src/filters_new.cc"
   "src/video.cc"
   "src/videotensor.cc"
   "src/vision.cc"
   "src/flow.cc"
   "src/TanTriggs.cc"
   "src/main.cc"
   )

include(../../cmake/macros.cmake)

torch_python_package_bindings(ip "${pysrc}")
torch_python_install(ip)
  
# Our test units
torch_python_add_test(python-ip-video-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_video.py)
torch_python_add_test(python-ip-videotensor-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_videotensor.py)
torch_python_add_test(python-ip-lbp-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_lbp.py)
torch_python_add_test(python-ip-filter-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_filters.py)
torch_python_add_test(python-ip-test-optflow
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_flow.py)
