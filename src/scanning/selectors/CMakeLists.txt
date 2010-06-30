#
# Add the source files to the list of files to be added to the library
#
PROJECT(selectors)
set(THIS selectors)

SET(${THIS}_SRC "DummySelector.cc" "OverlapSelector.cc" "MeanShiftSelector.cc"
	CACHE INTERNAL "sources")

include(../../SetupPackage.cmake)
