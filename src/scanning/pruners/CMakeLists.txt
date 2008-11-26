#
# Add the source files to the list of files to be added to the library
#
SET(	pruners_SRC
		"${CMAKE_CURRENT_SOURCE_DIR}/ipSWDummyPruner.cc"
	CACHE INTERNAL "sources")

SET(	pruners_INCLUDE
		${CMAKE_CURRENT_SOURCE_DIR}
	CACHE INTERNAL "includes")
