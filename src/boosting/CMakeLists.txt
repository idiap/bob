#
# Add the source files to the list of files to be added to the library
#
PROJECT(boosting)
set(THIS boosting)

SET(${THIS}_SRC
	"BoostingTrainer.cc"
	"BoostingRoundLBPTrainer.cc"
	"WeakLearner.cc"
	"StumpMachine.cc"
	"StumpTrainer.cc"
	"LBPRoundTrainer.cc"
	"IntLutMachine.cc"
	"RealLutMachine.cc"
	"FTrainer.cc"
	"CascadeTrainer.cc"
	CACHE INTERNAL "sources")

include(../SetupPackage.cmake)
include(../InstallParts.cmake)
