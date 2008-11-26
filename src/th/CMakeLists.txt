#
# Add the source files to the list of files to be added to the library
#
SET(	th_SRC
		"${CMAKE_CURRENT_SOURCE_DIR}/THBlas.c"
		"${CMAKE_CURRENT_SOURCE_DIR}/THGeneral.c"
		"${CMAKE_CURRENT_SOURCE_DIR}/THLogAdd.c"
		"${CMAKE_CURRENT_SOURCE_DIR}/THRandom.c"
		"${CMAKE_CURRENT_SOURCE_DIR}/THStorage.c"
		"${CMAKE_CURRENT_SOURCE_DIR}/THTensor.c"
		"${CMAKE_CURRENT_SOURCE_DIR}/THTensorMath.c"
	CACHE INTERNAL "sources")

SET(	th_INCLUDE
		${CMAKE_CURRENT_SOURCE_DIR}
	CACHE INTERNAL "includes")

