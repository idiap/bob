PROJECT(machine)
cmake_minimum_required(VERSION 2.4)

set(src_machine	
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

set(src
	"${src_machine}"
	"${src_criterions}"
	"${src_distributions}"
	"${src_gradients}"
	"${src_normalisations}")

include(../macros.cmake)
torch_package(machine "${src}" "${deps}" "${shared}")
