#
# Add the source files to the list of files to be added to the library
#
SET(	sp_SRC
		"${CMAKE_CURRENT_SOURCE_DIR}/spFFT.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/spDCT.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/ooura.cc"
	CACHE INTERNAL "sources")

# Include this directory
SET(	sp_INCLUDE
		${CMAKE_CURRENT_SOURCE_DIR}
	CACHE INTERNAL "includes")
