project(machine-python)
cmake_minimum_required(VERSION 2.6)

set(ccsrc
   "src/gmm.cc"
   "src/activation.cc"
   "src/linear.cc"
   "src/mlp.cc"
   "src/linearscoring.cc"
   "src/ztnorm.cc"
   "src/jfa.cc"
   "src/wiener.cc"
   "src/plda.cc"
   "src/version.cc"
   "src/main.cc"
   )
# If we have libsvm installed, enable the compilation of relevant modules
if(LIBSVM_FOUND)
  include_directories(${LIBSVM_INCLUDE_DIR})
  set(ccsrc "src/svm.cc;${ccsrc}")
endif()

set(pysrc
   "lib/__init__.py"
   )

bob_python_package_bindings(machine "${ccsrc}" "${pysrc}" "core_array")

# Our test units
bob_python_add_test(python-machine-kmeans-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_machine.py)
bob_python_add_test(python-machine-linear-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_linear.py)
bob_python_add_test(python-machine-mlp-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_mlp.py)
bob_python_add_test(python-machine-linearscoring-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_linearscoring.py)
bob_python_add_test(python-machine-plda-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_plda.py)
bob_python_add_test(python-machine-ztnorm-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_ztnorm.py)
bob_python_add_test(python-machine-svm-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_svm.py)

# Installs all python scripts
#file(COPY script/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin FILES_MATCHING PATTERN "*.py")

# Our examples (run and install)
bob_python_add_test(python-machine-xml2hdf5-example
  ${CMAKE_CURRENT_SOURCE_DIR}/example/xml2hdf5.py ignore --self-test)
bob_example_install("python" "machine" example/xml2hdf5.py)
