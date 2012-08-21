project(math-python)
cmake_minimum_required(VERSION 2.6)

set(ccsrc
   "src/interiorpointLP.cc"
   "src/linsolve.cc"
   "src/norminv.cc"
   "src/stats.cc"
   "src/histogram.cc"
   "src/main.cc"
   )

bob_python_package_bindings(math "${ccsrc}" "python_integration")

# Our test units
bob_python_add_unittest(math lib/test/interiorpointLP.py)
bob_python_add_unittest(math lib/test/linsolve.py)
bob_python_add_unittest(math lib/test/norminv.py)
bob_python_add_unittest(math lib/test/stats.py)
bob_python_add_unittest(math lib/test/histogram.py)
