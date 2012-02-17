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
   "lib/test/__init__.py"
   "lib/test/error.py"
   "lib/script/__init__.py"
   "lib/script/apply_threshold.py"
   "lib/script/compute_perf.py"
   "lib/script/eval_threshold.py"
   )

bob_python_package_bindings(measure "${ccsrc}" "${pysrc}" "core_array")

# Our test units
bob_python_add_unittest(measure lib/test/error.py main
  ${CMAKE_CURRENT_SOURCE_DIR}/data)

# Installs and test our scripts
bob_python_script(measure "compute_perf.py" "lib/script/compute_perf.py" "main")
bob_python_add_test(measure compute_perf.py --devel=${CMAKE_CURRENT_SOURCE_DIR}/data/dev-4col.txt --test=${CMAKE_CURRENT_SOURCE_DIR}/data/test-4col.txt  --self-test)

bob_python_script(measure "eval_threshold.py" "lib/script/eval_threshold.py" "main")
bob_python_add_test(measure eval_threshold.py --scores=${CMAKE_CURRENT_SOURCE_DIR}/data/dev-4col.txt --self-test)

bob_python_script(measure "apply_threshold.py" "lib/script/apply_threshold.py" "main")
bob_python_add_test(measure apply_threshold.py --scores=${CMAKE_CURRENT_SOURCE_DIR}/data/test-4col.txt --self-test)
