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
   "src/vision.cc"
   "src/flow.cc"
   "src/DCTFeatures.cc"
   "src/TanTriggs.cc"
   "src/main.cc"
   )

include(../../cmake/macros.cmake)

torch_python_package_bindings(ip "${pysrc}")
add_dependencies(pytorch_ip core-array-python_header_install)
torch_python_install(ip)
  
# Our test units
torch_python_add_test(python-ip-lbp-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_lbp.py)
torch_python_add_test(python-ip-filter-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_filters.py)
torch_python_add_test(python-ip-test-optflow
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_flow.py)
torch_python_add_test(python-ip-test-color
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_color.py)

# Installs all python scripts
file(COPY script/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin FILES_MATCHING PATTERN "*.py")

# Our examples (run and install)
torch_python_add_test(python-ip-optflowHS-example
  ${CMAKE_CURRENT_SOURCE_DIR}/example/optflowHS.py --test)
torch_example_install("python" "ip" example/optflowHS.py)
