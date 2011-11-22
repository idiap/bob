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

include(../../cmake/macros.cmake)

torch_python_package_bindings(trainer "${pysrc}") 

add_dependencies(pytorch_trainer python-core_header_install)
target_link_libraries(pytorch_trainer pytorch_core_ndarray)

torch_python_submodule_bindings(trainer overload "${pysrc_overload}")

target_link_libraries(pytorch_trainer_overload pytorch_core_ndarray)
add_dependencies(pytorch_trainer_overload python-core_header_install trainer-cxx_header_install)

torch_python_install(trainer)
  
# Our test units
torch_python_add_test(python-trainer-kmeans-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_trainer.py)

torch_python_add_test(python-trainer-linear-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_linear.py)

torch_python_add_test(python-trainer-mlp-shuffler-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_shuffler.py)

torch_python_add_test(python-trainer-mlp-rprop-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_rprop.py)

torch_python_add_test(python-trainer-mlp-backprop-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_backprop.py)

torch_python_add_test(python-trainer-jfa-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_jfa.py)

torch_python_add_test(python-trainer-plda-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_plda.py)

torch_python_add_test(python-trainer-gmm_train
  ${CMAKE_CURRENT_SOURCE_DIR}/script/gmm_train.py --self-test)

# Installs all python scripts
file(COPY script/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin FILES_MATCHING PATTERN "*.py")

# Our examples (run and install)
#torch_example_install("python" "trainer" example/banca_dct_gmm.py)
