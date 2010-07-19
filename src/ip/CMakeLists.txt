#
# Add the source files to the list of files to be added to the library
#
PROJECT(ip)
set(THIS ip)

SET(${THIS}_SRC
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
		"ipGeomNorm.cc"
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
#		"ipHOG.cc"
#		"ipGabor.cc"
	CACHE INTERNAL "sources")

SET(${THIS}_SUBDIRS imagefiles)
include(../SetupPackage.cmake)
