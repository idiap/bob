#
# Add the source files to the list of files to be added to the library
#
SET(	machine_SRC
                "${CMAKE_CURRENT_SOURCE_DIR}/Machines.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/CascadeMachine.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/LBPMachine.cc"
	CACHE INTERNAL "sources")

SET(	machine_INCLUDE
		${CMAKE_CURRENT_SOURCE_DIR}
	CACHE INTERNAL "includes")

