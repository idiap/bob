#
# Add the source files to the list of files to be added to the library
#
SET(	selectors_SRC
		"${CMAKE_CURRENT_SOURCE_DIR}/DummySelector.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/OverlapSelector.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/MeanShiftSelector.cc"
	CACHE INTERNAL "sources")

SET(	selectors_INCLUDE
		${CMAKE_CURRENT_SOURCE_DIR}
	CACHE INTERNAL "includes")
