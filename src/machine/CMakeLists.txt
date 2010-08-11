PROJECT(machine)
cmake_minimum_required(VERSION 2.4)

# This defines the dependencies of this package
set(deps core) #other torch subprojects

# Avoids warning during linkage configuration see: cmake --help-policy CMP0003
# for details.
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

set(src_machine
    "Machine.cc"	
	"Classifier.cc"
	"CascadeMachine.cc"
	"TreeClassifier.cc"
	"GradientMachine.cc"
	"Criterion.cc"
	"ProbabilityDistribution.cc")

set(src_criterions
	"MSECriterion.cc"
	"TwoClassNLLCriterion.cc"
	"MVSECriterion.cc")

set(src_distributions
	"MultiVariateNormalDistribution.cc"
	"MultiVariateDiagonalGaussianDistribution.cc"
	"MultiVariateMAPDiagonalGaussianDistribution.cc"
	"MultiVariateMeansDistribution.cc")

set(src_gradients
	"Exp.cc"
	"Log.cc"
	"Sigmoid.cc"
	"Linear.cc"
	"Tanh.cc"
	"MLP.cc")

set(src_normalisations
	"MeanVarNorm.cc")

set(src_boosting
	"IntLutMachine.cc"
	"RealLutMachine.cc"
	"StumpMachine.cc")

set(src
	"${src_machine}"
	"${src_criterions}"
	"${src_distributions}"
	"${src_gradients}"
	"${src_normalisations}"
	"${src_boosting}")

include(../macros.cmake)
torch_package(machine "${src}" "${deps}" "${shared}")
