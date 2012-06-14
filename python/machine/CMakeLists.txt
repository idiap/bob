project(machine-python)
cmake_minimum_required(VERSION 2.6)

set(ccsrc
   "src/machine.cc"
   "src/gabor.cc"
   "src/gaussian.cc"
   "src/gmm.cc"
   "src/kmeans.cc"
   "src/activation.cc"
   "src/linear.cc"
   "src/mlp.cc"
   "src/linearscoring.cc"
   "src/ztnorm.cc"
   "src/jfa.cc"
   "src/wiener.cc"
   "src/plda.cc"
   "src/bic.cc"
   "src/version.cc"
   "src/main.cc"
   )

# If we have libsvm installed, enable the compilation of relevant modules
if(LIBSVM_FOUND)
  include_directories(${LIBSVM_INCLUDE_DIR})
  set(ccsrc "src/svm.cc;${ccsrc}")
endif()

bob_python_package_bindings(machine "${ccsrc}" "python_integration")

# Our test units
bob_python_add_unittest(machine lib/test/gaussian.py)
bob_python_add_unittest(machine lib/test/gmm.py)
bob_python_add_unittest(machine lib/test/kmeans.py)
bob_python_add_unittest(machine lib/test/jfa.py)
bob_python_add_unittest(machine lib/test/linear.py)
bob_python_add_unittest(machine lib/test/mlp.py)
bob_python_add_unittest(machine lib/test/linearscoring.py)
bob_python_add_unittest(machine lib/test/plda.py)
bob_python_add_unittest(machine lib/test/ztnorm.py)

if(LIBSVM_FOUND)
  bob_python_add_unittest(machine lib/test/svm.py)
endif()

# Our examples (run and install)
bob_python_example(machine xml2hdf5.py lib/example/xml2hdf5.py main)
bob_python_add_test(machine-1 xml2hdf5.py --self-test ${CMAKE_CURRENT_SOURCE_DIR}/data/network.xml)
bob_python_add_test(machine-2 xml2hdf5.py --self-test ${CMAKE_CURRENT_SOURCE_DIR}/data/network-without-bias.xml)
