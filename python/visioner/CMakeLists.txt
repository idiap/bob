project(visioner-python)
cmake_minimum_required(VERSION 2.6)

set(ccsrc
   "src/version.cc"
   "src/localize.cc"
   "src/main.cc"
   )

set(pysrc
   "lib/__init__.py"
   )

bob_python_package_bindings(visioner "${ccsrc}" "${pysrc}" "core_array")

# Visioner depends on Qt4
include (${QT_USE_FILE})
include_directories(SYSTEM ${QT_INCLUDES})

# Our test units
bob_python_add_test(python-visioner-detection-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_detection.py)
bob_python_add_test(python-visioner-localization-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_localization.py)
bob_python_add_test(python-visioner-facebox-script-1
  ${CMAKE_CURRENT_SOURCE_DIR}/script/visioner_facebox.py dummy --self-test=1)
bob_python_add_test(python-visioner-facebox-script-2
  ${CMAKE_CURRENT_SOURCE_DIR}/script/visioner_facebox.py dummy --self-test=2)
bob_python_add_test(python-visioner-facepoints-script-1
  ${CMAKE_CURRENT_SOURCE_DIR}/script/visioner_facepoints.py dummy --self-test=1)
bob_python_add_test(python-visioner-facepoints-script-2
  ${CMAKE_CURRENT_SOURCE_DIR}/script/visioner_facepoints.py dummy --self-test=2)
