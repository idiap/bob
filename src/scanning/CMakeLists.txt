#
# Add the source files to the list of files to be added to the library
#
PROJECT(scanning)
set(THIS scanning)

SET(${THIS}_SRC
		"Pattern.cc"
		"ipSWEvaluator.cc"
		"ipSWPruner.cc"
		"ScaleExplorer.cc"
		"Explorer.cc"
		"Scanner.cc"
		"Selector.cc"
#		"PatternTracker.cc"
		"FaceFinder.cc"
	CACHE INTERNAL "sources")

set(${THIS}_SUBDIRS
	explorers
	pruners
	selectors
#	trackers
	context)
include(../SetupPackage.cmake)
include(../InstallParts.cmake)
