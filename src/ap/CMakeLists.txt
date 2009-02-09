#
# Add the source files to the list of files to be added to the library
#
SET(	ap_SRC
		"${CMAKE_CURRENT_SOURCE_DIR}/apMFCC.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/apLFCC.cc"
	CACHE INTERNAL "sources")

SET(	ap_INCLUDE
		${CMAKE_CURRENT_SOURCE_DIR}
	CACHE INTERNAL "includes")

# Add some sub-directories
SET(ap_SUBDIRS
	)

FOREACH(subdir ${ap_SUBDIRS})
	ADD_SUBDIRECTORY(${subdir})

	# Add the sources from the subdirectories
	SET(ap_SRC "${ap_SRC};${${subdir}_SRC}" CACHE INTERNAL "sources")

	# Add the includes from the subdirectoris too
	SET(ap_INCLUDE	"${ip_INCLUDE};${${subdir}_INCLUDE}" CACHE INTERNAL "includes")
ENDFOREACH(subdir)
