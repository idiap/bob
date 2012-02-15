project(math-python)
cmake_minimum_required(VERSION 2.6)

set(ccsrc
   "src/interiorpointLP.cc"
   "src/linsolve.cc"
   "src/norminv.cc"
   "src/stats.cc"
   "src/main.cc"
   )

set(pysrc
   "lib/__init__.py"
   "lib/test/__init__.py"
   "lib/test/interiorpointLP.py"
   "lib/test/linsolve.py"
   "lib/test/norminv.py"
   "lib/test/stats.py"
   )
 
bob_python_package_bindings(math "${ccsrc}" "${pysrc}" "core_array")

# Our test units
bob_python_add_unittest(math lib/test/interiorpointLP.py main)
bob_python_add_unittest(math lib/test/linsolve.py main)
bob_python_add_unittest(math lib/test/norminv.py main)
bob_python_add_unittest(math lib/test/stats.py main)
