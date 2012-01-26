project(visioner-python)
cmake_minimum_required(VERSION 2.6)

set(pysrc
   "src/version.cc"
   "src/localize.cc"
   "src/main.cc"
   )

include(../../cmake/macros.cmake)

bob_python_bindings(visioner "${pysrc}" "core" "bob_visioner")

# The visioner expects a flat directory structure
include_directories(${VISIONER_INCLUDE_DIRS})

# Visioner depends on Qt4
include (${QT_USE_FILE})
include_directories(SYSTEM ${QT_INCLUDES})

add_dependencies(pybob_visioner visioner python-core_header_install)
target_link_libraries(pybob_visioner pybob_core_ndarray)

bob_python_install(visioner)
  
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

# Installs all python scripts
file(COPY script/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin FILES_MATCHING PATTERN "*.py")
