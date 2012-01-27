project(math-python)
cmake_minimum_required(VERSION 2.6)

set(pysrc
   "src/eig.cc"
   "src/linsolve.cc"
   "src/norminv.cc"
   "src/sqrtm.cc"
   "src/stats.cc"
   "src/main.cc"
   )
 
include(../../cmake/macros.cmake)

bob_python_package_bindings(math "${pysrc}")
add_dependencies(pybob_math python-core_header_install)
target_link_libraries(pybob_math pybob_core_ndarray)

bob_python_install(math)

# Our test units
bob_python_add_test(python-math-eigenvalue-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_eig.py)
bob_python_add_test(python-math-linsolve-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_linsolve.py)
bob_python_add_test(python-math-norminv-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_norminv.py)
bob_python_add_test(python-math-sqrtm-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_sqrtm.py)
bob_python_add_test(python-math-stats-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_stats.py)
