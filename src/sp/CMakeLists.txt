#
# Add the source files to the list of files to be added to the library
#
PROJECT(sp)
set(THIS sp)

SET(${THIS}_SRC "spFFT.cc" "spDCT.cc" "spDeltaOne.cc" "ooura.cc"
	CACHE INTERNAL "sources")

include(../SetupPackage.cmake)
