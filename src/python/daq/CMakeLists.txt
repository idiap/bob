project(daq-python)
cmake_minimum_required(VERSION 2.6)

set(pysrc
   "src/main.cc"
   )

include(../../cmake/macros.cmake)

if (V4L2_FOUND)
  add_definitions(-DV4L2)
endif ()

if (OPENCV_FOUND)
  link_directories(${OpenCV_LIB_DIR})
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

include(${QT_USE_FILE})
include_directories(${QT_INCLUDE_DIR})


bob_python_package_bindings(daq "${pysrc}")

add_dependencies(pybob_daq python-core_header_install)

target_link_libraries(pybob_daq pybob_core_ndarray ${QT_LIBRARIES})

bob_python_install(daq)

if (V4L2_FOUND)
  bob_example_install("python" "daq" example/capture.py)
endif()

bob_python_add_test(python-daq-test
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test.py)
