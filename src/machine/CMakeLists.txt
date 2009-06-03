#
# Add the source files to the list of files to be added to the library
#
SET(	machine_SRC
                "${CMAKE_CURRENT_SOURCE_DIR}/Classifier.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/CascadeMachine.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/LRMachine.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/FLDAMachine.cc"
	CACHE INTERNAL "sources")

SET(	machine_INCLUDE
		${CMAKE_CURRENT_SOURCE_DIR}
	CACHE INTERNAL "includes")

