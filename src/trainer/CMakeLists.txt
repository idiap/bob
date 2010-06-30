#
# Add the source files to the list of files to be added to the library
#
PROJECT(trainer)
set(THIS trainer)

SET(${THIS}_SRC
		"GTFile.cc"
		"GradientTrainer.cc"
		"StochasticGradientTrainer.cc"
		"EMTrainer.cc"
	CACHE INTERNAL "sources")

SET(${THIS}_SUBDIRS gtfiles)
include(../SetupPackage.cmake)
