project(measure-python)
cmake_minimum_required(VERSION 2.6)

set(pysrc
   "src/error.cc"
   "src/main.cc"
   )

include(../../cmake/macros.cmake)

torch_python_package_bindings(measure "${pysrc}")
add_dependencies(pytorch_measure python-core_header_install)
target_link_libraries(pytorch_measure pytorch_core_ndarray)

torch_python_install(measure)
  
# Our test units
torch_python_add_test(python-measure-error ${CMAKE_CURRENT_SOURCE_DIR}/test/test_error.py)

# Installs and tests all python scripts
torch_python_add_test(python-measure-compute-perf
  ${CMAKE_CURRENT_SOURCE_DIR}/script/compute_perf.py --self-test)
torch_python_add_test(python-measure-eval-threshold
  ${CMAKE_CURRENT_SOURCE_DIR}/script/eval_threshold.py --self-test)
torch_python_add_test(python-measure-apply-threshold
  ${CMAKE_CURRENT_SOURCE_DIR}/script/apply_threshold.py --self-test)

file(COPY script/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin FILES_MATCHING PATTERN "*.py")
