PROJECT(ip)
cmake_minimum_required(VERSION 2.4)

# This defines the dependencies of this package
set(deps core) #other torch subprojects

# Avoids warning during linkage configuration see: cmake --help-policy CMP0003
# for details.
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

add_definitions(-DHAVE_OOURAFFT)
include_directories("${INCLUDE_DIR}/oourafft")
set(shared "-loourafft")

# This defines the list of source files inside this package.
set(src_ip
		"vision.cc"
		"Color.cc"
		"Complex.cc"
		"Convert.cc"
		"Image.cc"
		"ImageFile.cc"
		"ipIntegral.cc"
		"xtprobeImageFile.cc"
		"ipScaleYX.cc"
		"ipCrop.cc"
		"ipBlock.cc"
		"ipHaarLienhart.cc"
		"ipHLBPLienhart.cc"
		"ipLBP.cc"
		"ipLBP4R.cc"
		"ipLBP8R.cc"
		"ipLBPBitmap.cc"
		"ipHisto.cc"
		"ipSmoothGaussian.cc"
		"ipSobel.cc"
		"ipFlip.cc"
		"ipRotate.cc"
		"ipShift.cc"
#		"ipGeomNorm.cc"
		"Video.cc"
		"VideoTensor.cc"
		"ipHistoEqual.cc"
		"ipMSRSQIGaussian.cc"
		"ipMultiscaleRetinex.cc"
		"ipSelfQuotientImage.cc"
		"multigrid.cc"
		"ipVcycle.cc"
		"ipRelaxation.cc"
		"ipTanTriggs.cc"
		"ipRescaleGray.cc"
		"ipLBPTopOperator.cc"
		"ImageScanDataSet.cc"
)
#		"ipHOG.cc"
#		"ipGabor.cc"


set(src_imagefiles
		"pgmImageFile.cc"
		"ppmImageFile.cc"
		"gifImageFile.cc"
		"jpegImageFile.cc"
		"tiffImageFile.cc")

set(src_geometry
		"Affine2D.cc"
		"Circle2D.cc"
		"Line2D.cc"
		"Matrix2D.cc"
		"Octon2D.cc"
		"Plane.cc"
		"Point2D.cc"
		"Point3D.cc"
		"Polygon2D.cc"
		"Ray2D.cc"
		"Rectangle2D.cc"
		"RotationMatrix2D.cc"
		"Segment2D.cc"
		"Triangle2D.cc"
		"Vector2D.cc"
		"Vector3D.cc"
		"trigonometry.cc")

set(src 
		"${src_ip}" 
		"${src_imagefiles}"
		"${src_geometry}")


include(../macros.cmake)
torch_package(ip "${src}" "${deps}" "${shared}")
