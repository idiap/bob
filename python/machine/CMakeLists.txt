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
   "lib/example/__init__.py"
   "lib/example/xml2hdf5.py"
   "lib/test/__init__.py"
   #"lib/test/gmm_test_benchmark.py"
   "lib/test/linear.py"
   "lib/test/linearscoring.py"
   "lib/test/machine.py"
   "lib/test/mlp.py"
   "lib/test/plda.py"
   "lib/test/svm.py"
   "lib/test/ztnorm.py"
   )

bob_python_package_bindings(machine "${ccsrc}" "${pysrc}" "core_array")

# Our test units
bob_python_add_unittest(machine lib/test/machine.py main
  ${CMAKE_CURRENT_SOURCE_DIR}/data)
bob_python_add_unittest(machine lib/test/linear.py main
  ${CMAKE_CURRENT_SOURCE_DIR}/data)
bob_python_add_unittest(machine lib/test/mlp.py main
  ${CMAKE_CURRENT_SOURCE_DIR}/data)
bob_python_add_unittest(machine lib/test/linearscoring.py main
  ${CMAKE_CURRENT_SOURCE_DIR}/data)
bob_python_add_unittest(machine lib/test/plda.py main
  ${CMAKE_CURRENT_SOURCE_DIR}/data)
bob_python_add_unittest(machine lib/test/ztnorm.py main
  ${CMAKE_CURRENT_SOURCE_DIR}/data)
bob_python_add_unittest(machine lib/test/svm.py main
  ${CMAKE_CURRENT_SOURCE_DIR}/data)
#bob_python_add_unittest(machine lib/test/gmm_test_benchmark.py main
#  ${CMAKE_CURRENT_SOURCE_DIR}/data)

# TODO: Our examples (run and install)
#bob_python_example(machine "video2frame.py" "lib/example/video2frame.py" "main")
#bob_python_add_test(io video2frame.py --self-test)
