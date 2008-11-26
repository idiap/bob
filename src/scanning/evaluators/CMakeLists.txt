#
# Add the source files to the list of files to be added to the library
#
SET(	evaluators_SRC
		"${CMAKE_CURRENT_SOURCE_DIR}/ipSWDummyEvaluator.cc"
	CACHE INTERNAL "sources")

SET(	evaluators_INCLUDE
		${CMAKE_CURRENT_SOURCE_DIR}
	CACHE INTERNAL "includes")
