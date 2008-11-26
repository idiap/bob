#
# Add the source files to the list of files to be added to the library
#
SET(	explorers_SRC
		"${CMAKE_CURRENT_SOURCE_DIR}/ExhaustiveScaleExplorer.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/SpiralScaleExplorer.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/RandomScaleExplorer.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/MSExplorer.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/GreedyExplorer.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/PyramidExplorer.cc"
	CACHE INTERNAL "sources")

SET(	explorers_INCLUDE
		${CMAKE_CURRENT_SOURCE_DIR}
	CACHE INTERNAL "includes")
