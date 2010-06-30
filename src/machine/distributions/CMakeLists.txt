#
# Add the source files to the list of files to be added to the library
#
PROJECT(distributions)
set(THIS distributions)

SET(${THIS}_SRC
#		"Multinomial.cc"
		"MultiVariateNormalDistribution.cc"
		"MultiVariateDiagonalGaussianDistribution.cc"
		"MultiVariateMAPDiagonalGaussianDistribution.cc"
		"MultiVariateMeansDistribution.cc"
	CACHE INTERNAL "sources")

include(../../SetupPackage.cmake)
include(../../InstallParts.cmake)
