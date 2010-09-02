PROJECT(trainer)
cmake_minimum_required(VERSION 2.6)

# This defines the dependencies of this package
set(deps core ip machine lbfgs measurer) #other torch subprojects

SET(src_trainer
   "src/Trainer.cc"
   "src/GradientTrainer.cc"
   "src/StochasticGradientTrainer.cc"
   "src/EMTrainer.cc")

SET(src_boosting
   "src/BoostingRoundLBPTrainer.cc"
   "src/BoostingRoundTrainer.cc"
   "src/BoostingTrainer.cc"
   "src/CascadeTrainer.cc"
   "src/FTrainer.cc"
   "src/LBPRoundTrainer.cc"
   "src/StumpTrainer.cc"
   "src/WeakLearner.cc")

set(src 
	"${src_trainer}"
	"${src_boosting}")

include(../cmake/macros.cmake)
torch_library(trainer "${src}" "${deps}" "${shared}")
