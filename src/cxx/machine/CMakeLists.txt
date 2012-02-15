PROJECT(machine-cxx)
cmake_minimum_required(VERSION 2.6)

# This defines the dependencies of this package
set(deps core io math sp) #other bob subprojects
set(shared "")

# This defines the list of source files inside this package.
set(src
    "src/Exception.cc"
    "src/KMeansMachine.cc"
    "src/Gaussian.cc"
    "src/GMMMachine.cc"
    "src/GMMLLRMachine.cc"
    "src/GMMStats.cc"
    "src/EigenMachineException.cc"
    "src/TwoDPCAMachine.cc"
    "src/LinearMachine.cc"
    "src/MLP.cc"
    "src/MLPException.cc"
    "src/LinearScoring.cc"
    "src/ZTNorm.cc"
    "src/JFAMachine.cc"
    "src/JFAMachineException.cc"
    "src/WienerMachine.cc"
    "src/PLDAMachine.cc"
)

# If we have libsvm installed, enable the compilation of relevant modules
if(LIBSVM_FOUND)
  include_directories(${LIBSVM_INCLUDE_DIR})
  set(shared "${shared};${LIBSVM_LIBRARY}")
  set(src "${src};src/SVM.cc")
endif(LIBSVM_FOUND)

bob_library(machine "${src}" "${deps}" "${shared}")

# Defines tests for this package
bob_test(machine linear test/linear.cc)
set_property(TEST cxx-machine-linear APPEND PROPERTY ENVIRONMENT
  "BOB_MACHINE_TESTDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/../../python/machine/test/data")
