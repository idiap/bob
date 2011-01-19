project(machine-python)
cmake_minimum_required(VERSION 2.6)

set(pysrc
   "src/main.cc"
   "src/machine.cc"
   "src/ProbabilityDistribution.cc"
   "src/MultiVariateNormalDistribution.cc"
   "src/MultiVariateDiagonalGaussianDistribution.cc"
   )
 
include(../../cmake/macros.cmake)

torch_python_bindings(machine "${pysrc}")
torch_python_install(machine)
