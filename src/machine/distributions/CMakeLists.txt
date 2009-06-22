#
# Add the source files to the list of files to be added to the library
#
SET(	distributions_SRC
#		"${CMAKE_CURRENT_SOURCE_DIR}/Multinomial.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/MultiVariateNormalDistribution.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/MultiVariateDiagonalGaussianDistribution.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/MultiVariateMAPDiagonalGaussianDistribution.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/MultiVariateMeansDistribution.cc"
	CACHE INTERNAL "sources")

SET(	distributions_INCLUDE
		${CMAKE_CURRENT_SOURCE_DIR}
	CACHE INTERNAL "includes")

