project(ip-python)
cmake_minimum_required(VERSION 2.6)

set(ccsrc
   "src/version.cc"
   "src/color.cc"
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

# If we have VLFeat installed, enable the compilation of relevant modules
if(VLFEAT_FOUND)
  include_directories(${VLFEAT_INCLUDE_DIR})
  set(ccsrc "src/vlsift.cc;src/vldsift.cc;${ccsrc}")
endif()

set(pysrc
   "lib/__init__.py"
   "lib/flowutils.py"
   "lib/test/__init__.py"
   "lib/test/tempgrad.py"
   "lib/test/vldsift.py"
   "lib/test/histo.py"
   "lib/test/vlsift.py"
   "lib/test/color.py"
   "lib/test/drawing.py"
   "lib/test/optflow.py"
   "lib/test/face_extract.py"
   "lib/test/filters_new.py"
   "lib/test/lbp_new.py"
   "lib/script/__init__.py"
   "lib/script/facenorm.py"
   "lib/script/blockDCT.py"
   "lib/script/tantriggs.py"
   "lib/example/__init__.py"
   "lib/example/optflowHS.py"
   )

bob_python_package_bindings(ip "${ccsrc}" "${pysrc}" "core_array")

# Our test units
bob_python_add_unittest(ip lib/test/lbp_new.py main 
  "${CMAKE_CURRENT_SOURCE_DIR}/data")
bob_python_add_unittest(ip lib/test/filters_new.py main 
  "${CMAKE_CURRENT_SOURCE_DIR}/data")
bob_python_add_unittest(ip lib/test/optflow.py main 
  "${CMAKE_CURRENT_SOURCE_DIR}/data")
bob_python_add_unittest(ip lib/test/tempgrad.py main 
  "${CMAKE_CURRENT_SOURCE_DIR}/data")
bob_python_add_unittest(ip lib/test/color.py main 
  "${CMAKE_CURRENT_SOURCE_DIR}/data")
bob_python_add_unittest(ip lib/test/drawing.py main 
  "${CMAKE_CURRENT_SOURCE_DIR}/data")
bob_python_add_unittest(ip lib/test/histo.py main 
  "${CMAKE_CURRENT_SOURCE_DIR}/data")

# If we have VLFeat installed, enable the test of relevant modules
if(VLFEAT_FOUND)
  bob_python_add_unittest(ip lib/test/vlsift.py main
    "${CMAKE_CURRENT_SOURCE_DIR}/data")
  bob_python_add_unittest(ip lib/test/vldsift.py main
    "${CMAKE_CURRENT_SOURCE_DIR}/data")
endif()

bob_python_script(ip "facenorm.py" "lib/script/facenorm.py" "main")
bob_python_add_test(ip facenorm.py --self-test)

bob_python_script(ip "tantriggs.py" "lib/script/tantriggs.py" "main")
bob_python_add_test(ip tantriggs.py --self-test)

bob_python_script(ip "blockDCT.py" "lib/script/blockDCT.py" "main")
bob_python_add_test(ip blockDCT.py --self-test)

# Our examples (run and install)
bob_python_example(ip "optflowHS.py" "lib/example/optflowHS.py" "main")
bob_python_add_test(ip optflowHS.py --test
  ${CMAKE_CURRENT_SOURCE_DIR}/../io/data/test.mov)
