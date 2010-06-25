#
# Add the source files to the list of files to be added to the library
#
SET(	scanning_SRC
		"${CMAKE_CURRENT_SOURCE_DIR}/Pattern.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/ipSWEvaluator.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/ipSWPruner.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/ScaleExplorer.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/Explorer.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/Scanner.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/Selector.cc"
#		"${CMAKE_CURRENT_SOURCE_DIR}/PatternTracker.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/FaceFinder.cc"
	CACHE INTERNAL "sources")

SET(	scanning_INCLUDE
		${CMAKE_CURRENT_SOURCE_DIR}
	CACHE INTERNAL "includes")

# Add some sub-directories
SET(scanning_SUBDIRS
	explorers
	pruners
	selectors
#	trackers
	context)

FOREACH(subdir ${scanning_SUBDIRS})
	ADD_SUBDIRECTORY(${subdir})

	# Add the sources from the subdirectories
	SET(scanning_SRC "${scanning_SRC};${${subdir}_SRC}" CACHE INTERNAL "sources")

	# Add the includes from the subdirectoris too
	SET(scanning_INCLUDE "${scanning_INCLUDE};${${subdir}_INCLUDE}" CACHE INTERNAL "includes")
ENDFOREACH(subdir)
