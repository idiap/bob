project(measure-python)
cmake_minimum_required(VERSION 2.6)

set(pysrc
   "src/error.cc"
   "src/main.cc"
   "src/gabor_jet_similarities.cc"
   )

include(../../cmake/macros.cmake)

bob_python_package_bindings(measure "${pysrc}")
add_dependencies(pybob_measure python-core_header_install)
target_link_libraries(pybob_measure pybob_core_ndarray)

bob_python_install(measure)
  
# Our test units
bob_python_add_test(python-measure-error ${CMAKE_CURRENT_SOURCE_DIR}/test/test_error.py)

# Installs and tests all python scripts
bob_python_add_test(python-measure-compute-perf
  ${CMAKE_CURRENT_SOURCE_DIR}/script/compute_perf.py --self-test)
bob_python_add_test(python-measure-eval-threshold
  ${CMAKE_CURRENT_SOURCE_DIR}/script/eval_threshold.py --self-test)
bob_python_add_test(python-measure-apply-threshold
  ${CMAKE_CURRENT_SOURCE_DIR}/script/apply_threshold.py --self-test)

file(COPY script/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin FILES_MATCHING PATTERN "*.py")
