#
# Add the source files to the list of files to be added to the library
#
SET(	gradients_SRC
		"${CMAKE_CURRENT_SOURCE_DIR}/Exp.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/Log.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/Sigmoid.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/Linear.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/Tanh.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/MLP.cc"
	CACHE INTERNAL "sources")

SET(	gradients_INCLUDE
		${CMAKE_CURRENT_SOURCE_DIR}
	CACHE INTERNAL "includes")

