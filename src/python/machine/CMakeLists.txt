project(machine-python)
cmake_minimum_required(VERSION 2.6)

set(pysrc
   "src/exception.cc"
   "src/gmm.cc"
   "src/activation.cc"
   "src/linear.cc"
   "src/mlp.cc"
   "src/linearscoring.cc"
   "src/ztnorm.cc"
   "src/jfa.cc"
   "src/wiener.cc"
   "src/main.cc"
   )

include(../../cmake/macros.cmake)

torch_python_package_bindings(machine "${pysrc}")
add_dependencies(pytorch_machine core-array-python_header_install machine-cxx_header_install)
torch_python_install(machine)
  
# Our test units
torch_python_add_test(python-machine-kmeans-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_machine.py)
torch_python_add_test(python-machine-linear-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_linear.py)
torch_python_add_test(python-machine-mlp-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_mlp.py)
torch_python_add_test(python-machine-linearscoring-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_linearscoring.py)
torch_python_add_test(python-machine-ztnorm-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_ztnorm.py)

# Installs all python scripts
#file(COPY script/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin FILES_MATCHING PATTERN "*.py")

# Our examples (run and install)
torch_python_add_test(python-machine-xml2hdf5-example
  ${CMAKE_CURRENT_SOURCE_DIR}/example/xml2hdf5.py ignore --self-test)
torch_example_install("python" "machine" example/xml2hdf5.py)
