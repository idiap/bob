PROJECT(machine)
cmake_minimum_required(VERSION 2.6)

# This defines the dependencies of this package
set(deps core sp) #other torch subprojects

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

include(../cmake/macros.cmake)
torch_library(machine "${src}" "${deps}" "${shared}")

# Python bindings
set(pysrc
   "python/src/main.cc"
   "python/src/machine.cc"
   "python/src/ProbabilityDistribution.cc"
   "python/src/MultiVariateNormalDistribution.cc"
   "python/src/MultiVariateDiagonalGaussianDistribution.cc"
   )
 
torch_python_bindings(machine "${pysrc}")
torch_python_install(machine)
