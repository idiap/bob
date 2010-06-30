#
# Add the source files to the list of files to be added to the library
#
PROJECT(th)
set(THIS th)

SET(${THIS}_SRC
		"THBlas.c"
		"THGeneral.c"
		"THLogAdd.c"
		"THRandom.c"
		"THStorage.c"
		"THTensor.c"
		"THTensorMath.c"
	CACHE INTERNAL "sources")

include(../SetupPackage.cmake)
