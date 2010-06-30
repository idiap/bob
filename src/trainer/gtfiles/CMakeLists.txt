#
# Add the source files to the list of files to be added to the library
#
PROJECT(gtfiles)
set(THIS gtfiles)

SET(${THIS}_SRC
		"eyecenterGTFile.cc"
		"eyecornerGTFile.cc"
		"bancaGTFile.cc"
		"cootesGTFile.cc"
		"frontalEyeNoseChinGTFile.cc"
		"halfprofileEyeNoseChinGTFile.cc"
		"profileEyeNoseChinGTFile.cc"
#		"bbx2eye19x19deye10GTFile.cc"
	CACHE INTERNAL "sources")

include(../../SetupPackage.cmake)
