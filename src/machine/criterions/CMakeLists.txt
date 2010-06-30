#
# Add the source files to the list of files to be added to the library
#
PROJECT(criterions)
set(THIS criterions)

SET(${THIS}_SRC
		"MSECriterion.cc"
		"TwoClassNLLCriterion.cc"
		"MVSECriterion.cc"
	CACHE INTERNAL "sources")

include(../../InstallParts.cmake)
include(../../SetupPackage.cmake)
