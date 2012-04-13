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
   "src/GaborWaveletTransform.cc"
   "src/GeomNorm.cc"
   "src/FaceEyesNorm.cc"
   "src/rotate.cc"
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

bob_python_package_bindings(ip "${ccsrc}" "python_integration")

# Our test units
bob_python_add_unittest(ip lib/test/lbp_new.py)
bob_python_add_unittest(ip lib/test/filters_new.py)
bob_python_add_unittest(ip lib/test/optflow.py)
bob_python_add_unittest(ip lib/test/tempgrad.py)
bob_python_add_unittest(ip lib/test/color.py)
bob_python_add_unittest(ip lib/test/drawing.py)
bob_python_add_unittest(ip lib/test/histo.py)

# If we have VLFeat installed, enable the test of relevant modules
if(VLFEAT_FOUND)
  bob_python_add_unittest(ip lib/test/vldsift.py)
endif()

# Our examples (run and install)
bob_python_example(ip optflow_hs.py lib/example/optflow_hs.py)
bob_python_add_test(ip optflow_hs.py --test
  ${CMAKE_CURRENT_SOURCE_DIR}/../io/data/test.mov)
