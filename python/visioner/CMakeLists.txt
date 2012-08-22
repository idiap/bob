project(visioner-python)
cmake_minimum_required(VERSION 2.6)

set(ccsrc
   "src/version.cc"
   "src/localize.cc"
   "src/train.cc"
   "src/main.cc"
   )

bob_python_package_bindings(visioner "${ccsrc}" "python_integration")

# Visioner depends on Qt4
include (${QT_USE_FILE})
include_directories(SYSTEM ${QT_INCLUDES})

# Our test units
bob_python_add_unittest(visioner lib/test/detection.py main "${CMAKE_CURRENT_SOURCE_DIR}/../io/data")
bob_python_add_unittest(visioner lib/test/localization.py main "${CMAKE_CURRENT_SOURCE_DIR}/../io/data")
bob_python_add_unittest(visioner lib/test/scripts.py main "${CMAKE_CURRENT_SOURCE_DIR}/../io/data")
