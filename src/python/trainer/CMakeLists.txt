project(trainer-python)
cmake_minimum_required(VERSION 2.6)

set(pysrc
   "src/main.cc"
   "src/exception.cc"
   )

include(../../cmake/macros.cmake)

torch_python_package_bindings(trainer "${pysrc}") 
add_dependencies(pytorch_trainer core-array-python_header_install trainer-cxx_header_install)
torch_python_install(trainer)
  
# Our test units
torch_python_add_test(python-trainer-kmeans-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_trainer.py)

torch_python_add_test(python-trainer-gmm_train
  ${CMAKE_CURRENT_SOURCE_DIR}/script/gmm_train.py --self-test)

# Installs all python scripts
file(COPY script/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin FILES_MATCHING PATTERN "*.py")

# Our examples (run and install)
#torch_example_install("python" "trainer" example/banca_dct_gmm.py)