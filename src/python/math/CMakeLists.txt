project(math-python)
cmake_minimum_required(VERSION 2.6)

set(pysrc
   "src/eig.cc"
   "src/exception.cc"
   "src/linsolve.cc"
   "src/lu_det.cc"
   "src/norminv.cc"
   "src/svd.cc"
   "src/stats.cc"
   "src/main.cc"
   )
 
include(../../cmake/macros.cmake)

torch_python_package_bindings(math "${pysrc}")
target_link_libraries(pytorch_math pytorch_core_utils)
torch_python_install(math)

# Our test units
torch_python_add_test(python-math-eigenvalue-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_eig.py)
torch_python_add_test(python-math-stats-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_stats.py)
