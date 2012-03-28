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

# If we have libsvm installed, enable the compilation of relevant modules
if(LIBSVM_FOUND)
  include_directories(${LIBSVM_INCLUDE_DIR})
  set(ccsrc "src/svm.cc;${ccsrc}")
endif()

set(ccsrc_overload
   "src/kmeans_wrappers.cc"
   "src/gmm_wrappers.cc"
   "src/main_overload.cc"
   )

bob_python_package_bindings(trainer "${ccsrc}" "core_array")
bob_python_subpackage_bindings(trainer overload "${ccsrc_overload}" "core_array")

# Our test units
bob_python_add_unittest(trainer lib/test/trainer.py)
bob_python_add_unittest(trainer lib/test/linear.py)
bob_python_add_unittest(trainer lib/test/shuffler.py)
bob_python_add_unittest(trainer lib/test/rprop.py)
bob_python_add_unittest(trainer lib/test/backprop.py)
bob_python_add_unittest(trainer lib/test/jfa.py)
bob_python_add_unittest(trainer lib/test/plda.py)

if(LIBSVM_FOUND)
  bob_python_add_unittest(trainer lib/test/svm.py main 
    "${CMAKE_CURRENT_SOURCE_DIR}/../machine/data")
endif()
