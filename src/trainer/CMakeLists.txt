#
# Add the source files to the list of files to be added to the library
#
SET(	trainer_SRC
#		"${CMAKE_CURRENT_SOURCE_DIR}/GTFile.cc"
#		"${CMAKE_CURRENT_SOURCE_DIR}/EMTrainer.cc"
	CACHE INTERNAL "sources")

SET(	trainer_INCLUDE
		${CMAKE_CURRENT_SOURCE_DIR}
	CACHE INTERNAL "includes")

# Add some sub-directories
SET(trainer_SUBDIRS
)
#	gtfiles)

FOREACH(subdir ${trainer_SUBDIRS})
	ADD_SUBDIRECTORY(${subdir})

	# Add the sources from the subdirectories
	SET(trainer_SRC "${trainer_SRC};${${subdir}_SRC}" CACHE INTERNAL "sources")

	# Add the includes from the subdirectoris too
	SET(trainer_INCLUDE "${trainer_INCLUDE};${${subdir}_INCLUDE}" CACHE INTERNAL "includes")
ENDFOREACH(subdir)
