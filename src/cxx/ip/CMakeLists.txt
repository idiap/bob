PROJECT(ip-cxx)
cmake_minimum_required(VERSION 2.6)

# This defines the dependencies of this package
set(deps core database sp) #other torch subprojects
set(shared "${JPEG_LIBRARIES};${ffmpeg_LIBRARIES}")

# This defines the list of source files inside this package.
set(src_ip
   "src/ipCore.cc"
   "src/vision.cc"
   "src/OldColor.cc"
   "src/color.cc"
   "src/Complex.cc"
   "src/Convert.cc"
   "src/Image.cc"
   "src/ImageFile.cc"
   "src/ipIntegral.cc"
   "src/xtprobeImageFile.cc"
   "src/ipScaleYX.cc"
   "src/ipCrop.cc"
   "src/ipBlock.cc"
   "src/ipHaarLienhart.cc"
   "src/ipHLBPLienhart.cc"
   "src/ipLBP.cc"
   "src/ipLBP4R.cc"
   "src/ipLBP8R.cc"
   "src/ipLBPBitmap.cc"
   "src/ipHisto.cc"
   "src/ipSmoothGaussian.cc"
   "src/ipSobel.cc"
   "src/ipFlip.cc"
   "src/ipRotate.cc"
   "src/ipShift.cc"
   "src/Video.cc"
   "src/VideoTensor.cc"
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
   "src/ImageScanDataSet.cc"
   "src/Exception.cc"
   "src/HornAndSchunckFlow.cc"
   "src/TanTriggs.cc"
)
#   "src/ipHOG.cc"
#   "src/ipGabor.cc"


set(src_imagefiles
   "src/pgmImageFile.cc"
   "src/ppmImageFile.cc"
   "src/gifImageFile.cc"
   "src/jpegImageFile.cc"
   "src/tiffImageFile.cc")

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
   "${src_imagefiles}"
   "${src_geometry}")

include(../../cmake/macros.cmake)
torch_header_install(ip-cxx ip "torch::ip")
torch_library(ip "${src}" "${deps}" "${shared}")
add_dependencies(torch_ip ip-cxx_header_install)

# Defines tests for this package
torch_test(ip crop test/crop.cc)
torch_test(ip flipflop test/flipflop.cc)
torch_test(ip gammaCorrection test/gammaCorrection.cc)
torch_test(ip rotate test/rotate.cc)
torch_test(ip scale test/scale.cc)
torch_test(ip shear test/shear.cc)
torch_test(ip shift test/shift.cc)

# Defines environment variable used to localize data for the tests
set_property(TEST cxx-ip-rotate APPEND PROPERTY ENVIRONMENT
  "TORCH_IP_TESTDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/test/data")
set_property(TEST cxx-ip-scale APPEND PROPERTY ENVIRONMENT
  "TORCH_IP_TESTDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/test/data")

