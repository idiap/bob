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

torch_python_package_bindings(machine "${pysrc}")
torch_python_install(machine)

# Our test units
torch_python_add_test(python-machine-gmm-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_gmm.py)
