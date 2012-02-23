project(daq-python)
cmake_minimum_required(VERSION 2.6)

set(ccsrc
   "src/version.cc"
   "src/all.cc"
   "src/main.cc"
   )

if(BOB_VISIONER_ENABLED)
  add_definitions(-DVISIONER)
endif()

if (V4L2_FOUND)
  add_definitions(-DV4L2)
endif ()

if (OPENCV_FOUND)
  link_directories(${OpenCV_LIB_DIR})
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

include(${QT_USE_FILE})
include_directories(${QT_INCLUDE_DIR})

bob_python_package_bindings(daq "${ccsrc}" "core_array")

target_link_libraries(pybob_daq ${QT_LIBRARIES})

if (V4L2_FOUND)
  bob_python_example(daq "capture.py" lib/example/capture.py)
  # note: cannot test live capture
endif()

bob_python_add_unittest(daq lib/test/all.py main
  "${CMAKE_CURRENT_SOURCE_DIR}/../io/data")
