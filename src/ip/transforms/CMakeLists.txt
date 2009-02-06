#
# Add the source files to the list of files to be added to the library
#
SET(	transforms_SRC
		"${CMAKE_CURRENT_SOURCE_DIR}/ipFFT.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/ipDCT.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/ooura.cc"
	CACHE INTERNAL "sources")

# Include this directory
SET(	transforms_INCLUDE
		${CMAKE_CURRENT_SOURCE_DIR}
	CACHE INTERNAL "includes")
