#
# Add the source files to the list of files to be added to the library
#
SET(	machine_SRC
                "${CMAKE_CURRENT_SOURCE_DIR}/Classifier.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/CascadeMachine.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/GradientMachine.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/Criterion.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/ProbabilityDistribution.cc"
	CACHE INTERNAL "sources")

SET(	machine_INCLUDE
		${CMAKE_CURRENT_SOURCE_DIR}
	CACHE INTERNAL "includes")

# Add some sub-directories
SET(machine_SUBDIRS
	gradients criterions normalisations distributions)

FOREACH(subdir ${machine_SUBDIRS})
	ADD_SUBDIRECTORY(${subdir})

	# Add the sources from the subdirectories
	SET(machine_SRC "${machine_SRC};${${subdir}_SRC}" CACHE INTERNAL "sources")

	# Add the includes from the subdirectoris too
	SET(machine_INCLUDE "${machine_INCLUDE};${${subdir}_INCLUDE}" CACHE INTERNAL "includes")
ENDFOREACH(subdir)

