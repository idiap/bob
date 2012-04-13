project(core-python-integration)
cmake_minimum_required(VERSION 2.6)

# Python helper for blitz::Array<> <-> Numpy.ndarray conversion. This base
# helper *must* be used by all other extensions and is versioned, like any
# other public library in Bob.

set(src
   "src/ndarray.cc"
   "src/blitz_numpy.cc"
   "src/ndarray_numpy.cc"
   )

bob_library(python_integration "${src}" "core" "${Boost_PYTHON_LIBRARY_RELEASE};${PYTHON_LIBRARIES}" "core")
set_target_properties(bob_python_integration PROPERTIES COMPILE_FLAGS
  "-Wno-long-long -Wno-unused-function")
