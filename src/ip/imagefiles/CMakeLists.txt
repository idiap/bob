#
# Add the source files to the list of files to be added to the library
#
PROJECT(imagefiles)
set(THIS imagefiles)

SET(${THIS}_SRC
		"pgmImageFile.cc"
		"ppmImageFile.cc"
		"gifImageFile.cc"
		"jpegImageFile.cc"
		"tiffImageFile.cc"
	CACHE INTERNAL "sources")

include(../../InstallParts.cmake)
include(../../SetupPackage.cmake)
