#
# Add the source files to the list of files to be added to the library
#
SET(	gtfiles_SRC
		"${CMAKE_CURRENT_SOURCE_DIR}/eyecenterGTFile.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/eyecornerGTFile.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/bancaGTFile.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/cootesGTFile.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/frontalEyeNoseChinGTFile.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/halfprofileEyeNoseChinGTFile.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/profileEyeNoseChinGTFile.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/bbx2eye19x19deye10GTFile.cc"
	CACHE INTERNAL "sources")

# Include this directory
SET(	gtfiles_INCLUDE
		${CMAKE_CURRENT_SOURCE_DIR}
	CACHE INTERNAL "includes")
