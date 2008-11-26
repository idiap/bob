#
# Add the source files to the list of files to be added to the library
#
SET(	dirname_SRC
		"${CMAKE_CURRENT_SOURCE_DIR}/File1.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/File2.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/File3.cc"
	CACHE INTERNAL "sources")

SET(	dirname_INCLUDE
		${CMAKE_CURRENT_SOURCE_DIR}
	CACHE INTERNAL "includes")

