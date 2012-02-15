project(trainer-python)
cmake_minimum_required(VERSION 2.6)

set(ccsrc
   "src/linear.cc"
   "src/kmeans.cc"
   "src/gmm.cc"
   "src/rprop.cc"
   "src/backprop.cc"
   "src/jfa.cc"
   "src/wiener.cc"
   "src/empca.cc"
   "src/plda.cc"
   "src/main.cc"
   )

set(ccsrc_overload
   "src/kmeans_wrappers.cc"
   "src/gmm_wrappers.cc"
   "src/main_overload.cc"
   )

set(pysrc
   "lib/__init__.py"
   "lib/overload/__init__.py"
   "lib/script/__init__.py"
   "lib/script/gmm/__init__.py"
   "lib/script/gmm/adapt.py"
   "lib/script/gmm/test.py"
   "lib/script/gmm/train.py"
   "lib/test/__init__.py"
   "lib/test/backprop.py"
   "lib/test/jfa.py"
   "lib/test/linear.py"
   "lib/test/plda.py"
   "lib/test/rprop.py"
   "lib/test/shuffler.py"
   "lib/test/trainer.py"
   )

bob_python_package_bindings(trainer "${ccsrc}" "${pysrc}" "core_array")
bob_python_submodule_bindings(trainer overload "${ccsrc_overload}" "core_array")

# Installs and test our scripts
bob_python_script(trainer "gmm_train.py" "lib/script/gmm/train.py" "main")
bob_python_add_test(trainer gmm_train.py --self-test)

bob_python_script(trainer "gmm_adapt.py" "lib/script/gmm/adapt.py" "main")
#bob_python_add_test(trainer gmm_adapt.py) ##??

bob_python_script(trainer "gmm_test.py" "lib/script/gmm/test.py" "main")
#bob_python_add_test(trainer gmm_test.py) ##??

# Our test units
bob_python_add_unittest(trainer lib/test/trainer.py main
  ${CMAKE_CURRENT_SOURCE_DIR}/data)
bob_python_add_unittest(trainer lib/test/linear.py main)
bob_python_add_unittest(trainer lib/test/shuffler.py main)
bob_python_add_unittest(trainer lib/test/rprop.py main)
bob_python_add_unittest(trainer lib/test/backprop.py main)
bob_python_add_unittest(trainer lib/test/jfa.py main)
bob_python_add_unittest(trainer lib/test/plda.py main)
