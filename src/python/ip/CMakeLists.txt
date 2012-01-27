project(ip-python)
cmake_minimum_required(VERSION 2.6)

set(pysrc
   "src/version.cc"
   "src/ipcore.cc"
   "src/color.cc"
   "src/filters.cc"
   "src/block.cc"
   "src/crop_shift.cc"
   "src/extrapolate_mask.cc"
   "src/flipflop.cc"
   "src/gamma_correction.cc"
   "src/generate_with_center.cc"
   "src/integral.cc"
   "src/scale.cc"
   "src/shear.cc"
   "src/zigzag.cc"
   "src/flow.cc"
   "src/DCTFeatures.cc"
   "src/gaussian.cc"
   "src/Median.cc"
   "src/Gabor.cc"
   "src/GeomNorm.cc"
   "src/FaceEyesNorm.cc"
   "src/Rotate.cc"
   "src/TanTriggs.cc"
   "src/histo.cc"
   "src/LBP.cc"
   "src/MultiscaleRetinex.cc"
   "src/drawing.cc"
   "src/SpatioTemporalGradient.cc"
   "src/main.cc"
   )

include(../../cmake/macros.cmake)

# If we have VLFeat installed, enable the compilation of relevant modules
if(VLFEAT_FOUND)
  include_directories(${VLFEAT_INCLUDE_DIR})
  set(pysrc "src/vlsift.cc;src/vldsift.cc;${pysrc}")
endif()

bob_python_package_bindings(ip "${pysrc}")

add_dependencies(pybob_ip python-core_header_install)
target_link_libraries(pybob_ip pybob_core_ndarray)

bob_python_install(ip)
  
# Our test units
bob_python_add_test(python-ip-lbp-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_lbp_new.py)
bob_python_add_test(python-ip-filter_new-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_filters_new.py)
bob_python_add_test(python-ip-optflow-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_flow.py)
bob_python_add_test(python-ip-spatio-temporal-gradient-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_tempgrad.py)
bob_python_add_test(python-ip-color-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_color.py)
bob_python_add_test(python-ip-drawing-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_drawing.py)
bob_python_add_test(python-ip-histo-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_histo.py)
# If we have VLFeat installed, enable the test of relevant modules
if(VLFEAT_FOUND)
bob_python_add_test(python-ip-vlsift-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_vlsift.py)
bob_python_add_test(python-ip-vldsift-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_vldsift.py)
endif()

bob_python_add_test(python-ip-facenorm
  ${CMAKE_CURRENT_SOURCE_DIR}/script/facenorm.py --self-test)
bob_python_add_test(python-ip-tantriggs
  ${CMAKE_CURRENT_SOURCE_DIR}/script/tantriggs.py --self-test)
bob_python_add_test(python-ip-blockDCT
  ${CMAKE_CURRENT_SOURCE_DIR}/script/blockDCT.py --self-test)

# Installs all python scripts
file(COPY script/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin FILES_MATCHING PATTERN "*.py")

# Our examples (run and install)
bob_python_add_test(python-ip-optflowHS-example
  ${CMAKE_CURRENT_SOURCE_DIR}/example/optflowHS.py --test)
bob_example_install("python" "ip" example/optflowHS.py)
