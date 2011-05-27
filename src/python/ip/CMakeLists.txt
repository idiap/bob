project(ip-python)
cmake_minimum_required(VERSION 2.6)

set(pysrc
   "src/ipcore.cc"
   "src/color.cc"
   "src/oldlbp.cc"
   "src/filters.cc"
   "src/filters_new.cc"
   "src/flow.cc"
   "src/DCTFeatures.cc"
   "src/gaussian.cc"
   "src/Gabor.cc"
   "src/GeomNorm.cc"
   "src/FaceEyesNorm.cc"
   "src/Rotate.cc"
   "src/TanTriggs.cc"
   "src/exception.cc"
   "src/histo.cc"
   "src/LBP.cc"
   "src/MultiscaleRetinex.cc"
   "src/main.cc"
   )

include(../../cmake/macros.cmake)

torch_python_package_bindings(ip "${pysrc}")
add_dependencies(pytorch_ip core-array-python_header_install)
torch_python_install(ip)
  
# Our test units
torch_python_add_test(python-ip-lbp-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_lbp.py)
torch_python_add_test(python-ip-lbp-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_lbp_new.py)
torch_python_add_test(python-ip-filter_new-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_filters_new.py)
torch_python_add_test(python-ip-test-optflow
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_flow.py)
torch_python_add_test(python-ip-test-color
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_color.py)

torch_python_add_test(python-ip-facenorm
  ${CMAKE_CURRENT_SOURCE_DIR}/script/facenorm.py --self-test)
torch_python_add_test(python-ip-tantriggs
  ${CMAKE_CURRENT_SOURCE_DIR}/script/tantriggs.py --self-test)
torch_python_add_test(python-ip-blockDCT
  ${CMAKE_CURRENT_SOURCE_DIR}/script/blockDCT.py --self-test)

# Installs all python scripts
file(COPY script/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin FILES_MATCHING PATTERN "*.py")

# Our examples (run and install)
torch_python_add_test(python-ip-optflowHS-example
  ${CMAKE_CURRENT_SOURCE_DIR}/example/optflowHS.py --test)
torch_example_install("python" "ip" example/optflowHS.py)
