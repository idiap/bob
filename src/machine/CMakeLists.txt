PROJECT(machine)
cmake_minimum_required(VERSION 2.4)

# This defines the dependencies of this package
set(deps core sp) #other torch subprojects

# Avoids warning during linkage configuration see: cmake --help-policy CMP0003
# for details.
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

set(src_machine
   "src/Machine.cc"	
   "src/Classifier.cc"
   "src/CascadeMachine.cc"
   "src/TreeClassifier.cc"
   "src/GradientMachine.cc"
   "src/Criterion.cc"
   "src/ProbabilityDistribution.cc")

set(src_criterions
   "src/MSECriterion.cc"
   "src/TwoClassNLLCriterion.cc"
   "src/MVSECriterion.cc")

set(src_distributions
   "src/MultiVariateNormalDistribution.cc"
   "src/MultiVariateDiagonalGaussianDistribution.cc"
   "src/MultiVariateMAPDiagonalGaussianDistribution.cc"
   "src/MultiVariateMeansDistribution.cc")

set(src_gradients
   "src/Exp.cc"
   "src/Log.cc"
   "src/Sigmoid.cc"
   "src/Linear.cc"
   "src/Tanh.cc"
   "src/MLP.cc")

set(src_normalisations
   "src/MeanVarNorm.cc")

set(src_boosting
   "src/IntLutMachine.cc"
   "src/RealLutMachine.cc"
   "src/StumpMachine.cc")

set(src
	"${src_machine}"
	"${src_criterions}"
	"${src_distributions}"
	"${src_gradients}"
	"${src_normalisations}"
	"${src_boosting}")

include(../macros.cmake)
torch_library(machine "${src}" "${deps}" "${shared}")
