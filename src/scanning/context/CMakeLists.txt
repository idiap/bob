#
# Add the source files to the list of files to be added to the library
#
PROJECT(context)
set(THIS context)

SET(${THIS}_SRC
		"Context.cc"
		"ContextDataSet.cc"
		"ContextMachine.cc"
		"ContextTrainer.cc"
		"LRMachine.cc"
		"LRTrainer.cc"
	CACHE INTERNAL "sources")

include(../../SetupPackage.cmake)
