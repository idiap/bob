project(measure-python)
cmake_minimum_required(VERSION 2.6)

set(ccsrc
   "src/error.cc"
   "src/main.cc"
   )

set(pysrc
   "lib/__init__.py"
   "lib/load.py"
   "lib/plot.py"
   )

bob_python_package_bindings(measure "${ccsrc}" "${pysrc}" "core_array")

# Our test units
bob_python_add_test(python-measure-error ${CMAKE_CURRENT_SOURCE_DIR}/test/test_error.py)

# Installs and tests all python scripts
bob_python_add_test(python-measure-compute-perf
  ${CMAKE_CURRENT_SOURCE_DIR}/script/compute_perf.py --self-test)
bob_python_add_test(python-measure-eval-threshold
  ${CMAKE_CURRENT_SOURCE_DIR}/script/eval_threshold.py --self-test)
bob_python_add_test(python-measure-apply-threshold
  ${CMAKE_CURRENT_SOURCE_DIR}/script/apply_threshold.py --self-test)

