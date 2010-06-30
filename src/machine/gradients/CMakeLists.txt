#
# Add the source files to the list of files to be added to the library
#
PROJECT(gradients)
set(THIS gradients)

SET(${THIS}_SRC
		"Exp.cc"
		"Log.cc"
		"Sigmoid.cc"
		"Linear.cc"
		"Tanh.cc"
		"MLP.cc"
	CACHE INTERNAL "sources")

include(../../SetupPackage.cmake)
