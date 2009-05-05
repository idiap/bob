#
# Add the source files to the list of files to be added to the library
#
SET(	boosting_SRC
	"${CMAKE_CURRENT_SOURCE_DIR}/BoostingTrainer.cc"
	"${CMAKE_CURRENT_SOURCE_DIR}/BoostingRoundLBPTrainer.cc"
	"${CMAKE_CURRENT_SOURCE_DIR}/WeakLearner.cc"
	"${CMAKE_CURRENT_SOURCE_DIR}/StumpMachine.cc"
	"${CMAKE_CURRENT_SOURCE_DIR}/StumpTrainer.cc"
	"${CMAKE_CURRENT_SOURCE_DIR}/LBPRoundTrainer.cc"
	"${CMAKE_CURRENT_SOURCE_DIR}/IntLutMachine.cc"
	"${CMAKE_CURRENT_SOURCE_DIR}/RealLutMachine.cc"
	"${CMAKE_CURRENT_SOURCE_DIR}/FTrainer.cc"
	"${CMAKE_CURRENT_SOURCE_DIR}/CascadeTrainer.cc"
	CACHE INTERNAL "sources")

SET(	boosting_INCLUDE
		${CMAKE_CURRENT_SOURCE_DIR}
	CACHE INTERNAL "includes")

