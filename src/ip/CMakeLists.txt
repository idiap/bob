PROJECT(ip)
cmake_minimum_required(VERSION 2.4)

# This defines the dependencies of this package
set(deps core) #other torch subprojects

add_definitions(-DHAVE_OOURAFFT)
include_directories("${INCLUDE_DIR}/oourafft")
set(shared "-loourafft")
set(archive "")

# This defines the list of source files inside this package.
set(src
		"vision.cc"
		"Color.cc"
		"Complex.cc"
		"Convert.cc"
		"Image.cc"
		"ImageFile.cc"
		"ipIntegral.cc"
#		"xtprobeImageFile.cc"
		"ipScaleYX.cc"
		"ipCrop.cc"
		"ipBlock.cc"
		"ipHaarLienhart.cc"
		"ipHLBPLienhart.cc"
		"ipLBP.cc"
#		"ipLBP4R.cc"
#		"ipLBP8R.cc"
		"ipLBPBitmap.cc"
		"ipHisto.cc"
		"ipSmoothGaussian.cc"
		"ipSobel.cc"
		"ipFlip.cc"
#		"ipRotate.cc"
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

include(../macros.cmake)
torch_package(ip "${src}" "${deps}" "${shared}" "${archive}")
