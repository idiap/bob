project(measure-python)
cmake_minimum_required(VERSION 2.6)

set(ccsrc
   "src/error.cc"
   "src/main.cc"
   )

bob_python_package_bindings(measure "${ccsrc}" "python_integration")

# Our test units
bob_python_add_unittest(measure lib/test/error.py)
bob_python_add_unittest(measure lib/test/scripts.py)
