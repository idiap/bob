project(trainer-python)
cmake_minimum_required(VERSION 2.6)

set(pysrc
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

set(pysrc_overload
   "src/kmeans_wrappers.cc"
   "src/gmm_wrappers.cc"
   "src/main_overload.cc"
   )

bob_python_submodule_bindings(trainer overload "${pysrc_overload}" "core_array")
bob_python_package_bindings(trainer "${pysrc}" "core_array")

# Our test units
bob_python_add_test(python-trainer-kmeans-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_trainer.py)

bob_python_add_test(python-trainer-linear-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_linear.py)

bob_python_add_test(python-trainer-mlp-shuffler-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_shuffler.py)

bob_python_add_test(python-trainer-mlp-rprop-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_rprop.py)

bob_python_add_test(python-trainer-mlp-backprop-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_backprop.py)

bob_python_add_test(python-trainer-jfa-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_jfa.py)

bob_python_add_test(python-trainer-plda-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_plda.py)

bob_python_add_test(python-trainer-gmm_train
  ${CMAKE_CURRENT_SOURCE_DIR}/script/gmm_train.py --self-test)
