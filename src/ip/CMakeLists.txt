#
# Add the source files to the list of files to be added to the library
#
SET(	ip_SRC
		"${CMAKE_CURRENT_SOURCE_DIR}/vision.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/Color.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/Complex.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/Convert.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/Image.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/ImageFile.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/ipIntegral.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/xtprobeImageFile.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/ipScaleYX.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/ipCrop.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/ipBlock.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/ipHaarLienhart.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/ipHLBPLienhart.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/ipLBP.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/ipLBP4R.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/ipLBP8R.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/ipLBPBitmap.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/ipHisto.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/ipSmoothGaussian.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/ipSobel.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/ipFlip.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/ipRotate.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/ipShift.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/ipGeomNorm.cc"
	CACHE INTERNAL "sources")

SET(	ip_INCLUDE
		${CMAKE_CURRENT_SOURCE_DIR}
	CACHE INTERNAL "includes")

# Add some sub-directories
SET(ip_SUBDIRS
	imagefiles)

FOREACH(subdir ${ip_SUBDIRS})
	ADD_SUBDIRECTORY(${subdir})

	# Add the sources from the subdirectories
	SET(ip_SRC "${ip_SRC};${${subdir}_SRC}" CACHE INTERNAL "sources")

	# Add the includes from the subdirectoris too
	SET(ip_INCLUDE	"${ip_INCLUDE};${${subdir}_INCLUDE}" CACHE INTERNAL "includes")
ENDFOREACH(subdir)
