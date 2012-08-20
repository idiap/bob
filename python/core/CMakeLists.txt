project(core-python)
cmake_minimum_required(VERSION 2.6)

# Python bindings
set(ccsrc
   #"src/array_wrapper_example.cc"
   "src/exception.cc"
   "src/logging.cc"
   "src/profile.cc"
   "src/convert.cc"
   "src/tinyvector.cc"
   "src/typeinfo.cc"
   "src/version.cc"
   "src/main.cc"
   )

set(ccsrc_random
   "src/random.cc"
   "src/main_random.cc"
   )

bob_python_package_bindings(core "${ccsrc}" "python_integration")
bob_python_subpackage_bindings(core random "${ccsrc_random}"
  "python_integration")

# Our test units
# TTT bob_python_add_unittest(core lib/test/exception.py)
# TTT bob_python_add_unittest(core lib/test/convert.py)
