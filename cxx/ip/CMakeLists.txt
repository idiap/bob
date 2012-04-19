PROJECT(ip-cxx)
cmake_minimum_required(VERSION 2.6)

# This defines the dependencies of this package
set(deps core io math sp) #other bob subprojects
set(shared "")

# This defines the list of source files inside this package.
set(src
   "src/block.cc"
   "src/color.cc"
   "src/Exception.cc"
   "src/HornAndSchunckFlow.cc"
   "src/crop.cc"
   "src/shift.cc"
   "src/TanTriggs.cc"
   "src/GeomNorm.cc"
   "src/maxRectInMask.cc"
   "src/FaceEyesNorm.cc"
   "src/GaborWaveletTransform.cc"
   "src/histo.cc"
   "src/HOG.cc"
   "src/LBP.cc"
   "src/LBP4R.cc"
   "src/LBP8R.cc"
   "src/LBP16R.cc"
   "src/LBPTopOperator.cc"
   "src/Sobel.cc"
   "src/Gaussian.cc"
   "src/MultiscaleRetinex.cc"
   "src/SpatioTemporalGradient.cc"
)

# If we have VLFeat installed, enable the compilation of relevant modules
if(VLFEAT_FOUND)
  include_directories(SYSTEM ${VLFEAT_INCLUDE_DIR})
  set(shared "${shared};${VLFEAT_LIBRARY}")
  set(src "${src};src/VLSIFT.cc;src/VLDSIFT.cc")
endif(VLFEAT_FOUND)

bob_library(ip "${src}" "${deps}" "${shared}")

# Defines tests for this package
bob_test(ip block test/block.cc)
bob_test(ip crop test/crop.cc)
bob_test(ip dctfeatures test/dctfeatures.cc)
bob_test(ip extrapolateMask test/extrapolateMask.cc)
bob_test(ip flipflop test/flipflop.cc)
bob_test(ip gwt test/GaborWaveletTransform.cc)
bob_test(ip gaussian test/Gaussian.cc)
bob_test(ip median test/Median.cc)
bob_test(ip gammaCorrection test/gammaCorrection.cc)
#bob_test(ip geomnorm test/geomnorm.cc)
bob_test(ip facenorm test/facenorm.cc)
bob_test(ip integral test/integral.cc)
bob_test(ip lbp test/LBP.cc)
bob_test(ip lbphsfeatures test/lbphsfeatures.cc)
bob_test(ip maxRectInMask test/maxRectInMask.cc)
bob_test(ip msr test/MSR.cc)
bob_test(ip rotate test/rotate.cc)
bob_test(ip scale test/scale.cc)
bob_test(ip shear test/shear.cc)
bob_test(ip shift test/shift.cc)
bob_test(ip generatewithcenter test/generatewithcenter.cc)
bob_test(ip tantriggs test/TanTriggs.cc)
bob_test(ip sobel test/Sobel.cc)
bob_test(ip zigzag test/zigzag.cc)

# Defines environment variable used to localize data for the tests
set_property(TEST cxx-ip-gwt APPEND PROPERTY ENVIRONMENT
  "BOB_IP_TESTDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/test/data")
#set_property(TEST cxx-ip-geomnorm APPEND PROPERTY ENVIRONMENT
#  "BOB_IP_TESTDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/test/data")
set_property(TEST cxx-ip-facenorm APPEND PROPERTY ENVIRONMENT
  "BOB_IP_TESTDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/test/data")
set_property(TEST cxx-ip-gaussian APPEND PROPERTY ENVIRONMENT
  "BOB_IP_TESTDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/test/data")
set_property(TEST cxx-ip-msr APPEND PROPERTY ENVIRONMENT
  "BOB_IP_TESTDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/test/data")
set_property(TEST cxx-ip-rotate APPEND PROPERTY ENVIRONMENT
  "BOB_IP_TESTDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/test/data")
set_property(TEST cxx-ip-scale APPEND PROPERTY ENVIRONMENT
  "BOB_IP_TESTDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/test/data")
set_property(TEST cxx-ip-tantriggs APPEND PROPERTY ENVIRONMENT
  "BOB_IP_TESTDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/test/data")

