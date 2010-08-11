PROJECT(ip)
cmake_minimum_required(VERSION 2.4)

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

# This defines the dependencies of this package
set(deps core sp) #other torch subprojects

# jpeg
if(JPEG_FOUND)
  add_definitions(-DHAVE_JPEG)
  include_directories(${JPEG_INCLUDE_DIR})
  set(shared "${JPEG_LIBRARIES}")
endif(JPEG_FOUND)

# ffmpeg
if(ffmpeg_FOUND)
  add_definitions(-DHAVE_FFMPEG)
  include_directories(${ffmpeg_INCLUDE_DIRS})
  link_directories(${ffmpeg_LIBRARY_DIRS})
  set(shared "${shared};${ffmpeg_LIBRARIES}")
endif(ffmpeg_FOUND)

# This defines the list of source files inside this package.
set(src_ip
   "src/ipCore.cc"
   "src/vision.cc"
   "src/Color.cc"
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

include(../macros.cmake)
torch_library(ip "${src}" "${deps}" "${shared}")

set(pysrc
   "python/src/ipcore.cc"
   "python/src/color.cc"
   "python/src/image.cc"
   "python/src/lbp.cc"
   "python/src/video.cc"
   "python/src/videotensor.cc"
   "python/src/vision.cc"
   "python/src/main.cc"
   )

torch_python_bindings(ip "${pysrc}")
torch_python_install(ip)
