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

if(GOOGLE_PERFTOOLS_FOUND)
  # Special hack to get libprofiler.so linked in.
  target_link_libraries(pybob_core ${GOOGLE_PERFTOOLS_LIBRARY})
endif()

bob_python_subpackage_bindings(core random "${ccsrc_random}"
  "python_integration")

# Our test units
bob_python_add_unittest(core lib/test/exception.py)
bob_python_add_unittest(core lib/test/convert.py)
