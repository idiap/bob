#
# Add the source files to the list of files to be added to the library
#
PROJECT(explorers)
set(THIS explorers)

SET(${THIS}_SRC
		"ExhaustiveScaleExplorer.cc"
		"SpiralScaleExplorer.cc"
		"RandomScaleExplorer.cc"
		"MSExplorer.cc"
		"ContextExplorer.cc"
		"PyramidExplorer.cc"
	CACHE INTERNAL "sources")

include(../../SetupPackage.cmake)
