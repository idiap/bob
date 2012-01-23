PROJECT(ip-cxx)
cmake_minimum_required(VERSION 2.6)

# This defines the dependencies of this package
set(deps core io math sp) #other bob subprojects
set(shared "")

# This defines the list of source files inside this package.
set(src_ip
   "src/ipCore.cc"
   "src/vision.cc"
   "src/color.cc"
   "src/OldColor.cc"
   "src/Complex.cc"
   "src/Convert.cc"
   "src/Image.cc"
   "src/ipIntegral.cc"
   "src/ipScaleYX.cc"
   "src/ipCrop.cc"
   "src/ipHaarLienhart.cc"
   "src/ipHLBPLienhart.cc"
   "src/ipLBP.cc"
   "src/ipLBP4R.cc"
   "src/ipLBP8R.cc"
   "src/ipLBPBitmap.cc"
   "src/ipSmoothGaussian.cc"
   "src/ipSobel.cc"
   "src/ipFlip.cc"
   "src/ipRotate.cc"
   "src/ipShift.cc"
   "src/ipHistoEqual.cc"
   "src/ipMSRSQIGaussian.cc"
   "src/ipMultiscaleRetinex.cc"
   "src/ipSelfQuotientImage.cc"
   "src/multigrid.cc"
   "src/ipVcycle.cc"
   "src/ipRelaxation.cc"
   "src/ipTanTriggs.cc"
   "src/ipRescaleGray.cc"
   "src/ipLBPTopOperator.cc"
   "src/Exception.cc"
   "src/HornAndSchunckFlow.cc"
   "src/crop.cc"
   "src/shift.cc"
   "src/Rotate.cc"
   "src/TanTriggs.cc"
   "src/GeomNorm.cc"
   "src/maxRectInMask.cc"
   "src/FaceEyesNorm.cc"
   "src/GaborFrequency.cc"
   "src/GaborSpatial.cc"
   "src/GaborBankFrequency.cc"
   "src/GaborBankSpatial.cc"
   "src/histo.cc"
   "src/LBP.cc"
   "src/LBP4R.cc"
   "src/LBP8R.cc"
   "src/LBPTopOperator.cc"
   "src/Sobel.cc"
   "src/Gaussian.cc"
   "src/MultiscaleRetinex.cc"
   "src/SpatioTemporalGradient.cc"
)
#   "src/ipHOG.cc"

set(src_geometry
   "src/Affine2D.cc"
   "src/Circle2D.cc"
   "src/Line2D.cc"
   "src/Matrix2D.cc"
   "src/Octon2D.cc"
   "src/Plane.cc"
   "src/Point2D.cc"
   "src/Point3D.cc"
   "src/Polygon2D.cc"
   "src/Ray2D.cc"
   "src/Rectangle2D.cc"
   "src/RotationMatrix2D.cc"
   "src/Segment2D.cc"
   "src/Triangle2D.cc"
   "src/Vector2D.cc"
   "src/Vector3D.cc"
   "src/trigonometry.cc")

set(src 
   "${src_ip}" 
#   "${src_imagefiles}"
   "${src_geometry}")

# If we have VLFeat installed, enable the compilation of relevant modules
if(VLFEAT_FOUND)
  include_directories(${VLFEAT_INCLUDE_DIR})
  set(shared "${shared};${VLFEAT_LIBRARY}")
  set(src "${src};src/VLSIFT.cc;src/VLDSIFT.cc")
endif(VLFEAT_FOUND)

include(../../cmake/macros.cmake)
bob_header_install(ip-cxx ip "bob::ip")
bob_library(ip "${src}" "${deps}" "${shared}")
add_dependencies(bob_ip ip-cxx_header_install)

# Defines tests for this package
bob_test(ip block test/block.cc)
bob_test(ip crop test/crop.cc)
bob_test(ip dctfeatures test/dctfeatures.cc)
bob_test(ip extrapolateMask test/extrapolateMask.cc)
bob_test(ip flipflop test/flipflop.cc)
bob_test(ip gabor test/Gabor.cc)
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
set_property(TEST cxx-ip-gabor APPEND PROPERTY ENVIRONMENT
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

