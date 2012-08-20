project(measure-python)
cmake_minimum_required(VERSION 2.6)

set(ccsrc
   "src/error.cc"
   "src/main.cc"
   )

bob_python_package_bindings(measure "${ccsrc}" "python_integration")

# Our test units
# TTT bob_python_add_unittest(measure lib/test/error.py)

# Installs and test our scripts
# TTT bob_python_add_test(measure bob_compute_perf.py --devel=${CMAKE_CURRENT_SOURCE_DIR}/data/dev-4col.txt --test=${CMAKE_CURRENT_SOURCE_DIR}/data/test-4col.txt  --self-test)

# TTT bob_python_add_test(measure bob_eval_threshold.py --scores=${CMAKE_CURRENT_SOURCE_DIR}/data/dev-4col.txt --self-test)

# TTT bob_python_add_test(measure bob_apply_threshold.py --scores=${CMAKE_CURRENT_SOURCE_DIR}/data/test-4col.txt --self-test)
