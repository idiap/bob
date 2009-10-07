#
# Add the source files to the list of files to be added to the library
#
SET(	criterions_SRC
		"${CMAKE_CURRENT_SOURCE_DIR}/MSECriterion.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/TwoClassNLLCriterion.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/MVSECriterion.cc"
	CACHE INTERNAL "sources")

SET(	criterions_INCLUDE
		${CMAKE_CURRENT_SOURCE_DIR}
	CACHE INTERNAL "includes")

