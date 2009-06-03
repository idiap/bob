#
# Add the source files to the list of files to be added to the library
#
SET(	profile_SRC
		"${CMAKE_CURRENT_SOURCE_DIR}/Profile.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/ProfileDataSet.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/mat.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/ProfileMachine.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/ProfileTrainer.cc"
	CACHE INTERNAL "sources")

SET(	profile_INCLUDE
		${CMAKE_CURRENT_SOURCE_DIR}
	CACHE INTERNAL "includes")
