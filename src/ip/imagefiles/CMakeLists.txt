#
# Add the source files to the list of files to be added to the library
#
SET(	imagefiles_SRC
		"${CMAKE_CURRENT_SOURCE_DIR}/pgmImageFile.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/ppmImageFile.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/gifImageFile.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/jpegImageFile.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/tiffImageFile.cc"
	CACHE INTERNAL "sources")

# Include this directory
SET(	imagefiles_INCLUDE
		${CMAKE_CURRENT_SOURCE_DIR}
	CACHE INTERNAL "includes")
