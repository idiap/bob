#
# Add the source files to the list of files to be added to the library
#
PROJECT(pruners)
set(THIS pruners)

SET(${THIS}_SRC "ipSWVariancePruner.cc"
	CACHE INTERNAL "sources")

include(../../InstallParts.cmake)
include(../../SetupPackage.cmake)
