#
# Add the source files to the list of files to be added to the library
#
PROJECT(machine)
set(THIS machine)

SET(${THIS}_SRC
    "Classifier.cc"
		"CascadeMachine.cc"
		"TreeClassifier.cc"
		"GradientMachine.cc"
		"Criterion.cc"
		"ProbabilityDistribution.cc"
	CACHE INTERNAL "sources")

set(${THIS}_SUBDIRS gradients criterions normalisations distributions)
include(../SetupPackage.cmake)
