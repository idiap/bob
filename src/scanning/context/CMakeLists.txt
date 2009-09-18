#
# Add the source files to the list of files to be added to the library
#
SET(	context_SRC
		"${CMAKE_CURRENT_SOURCE_DIR}/Context.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/ContextDataSet.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/ContextMachine.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/ContextTrainer.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/LRMachine.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/LRTrainer.cc"
	CACHE INTERNAL "sources")

SET(	context_INCLUDE
		${CMAKE_CURRENT_SOURCE_DIR}
	CACHE INTERNAL "includes")
