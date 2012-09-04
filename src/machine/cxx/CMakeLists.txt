project(bob_machine)

# This defines the dependencies of this package
set(bob_deps "bob_ip")
set(shared "${bob_deps}")

# This defines the list of source files inside this package.
set(src
  "Exception.cc"
  "KMeansMachine.cc"
  "Gaussian.cc"
  "GMMMachine.cc"
  "GMMLLRMachine.cc"
  "GMMStats.cc"
  "EigenMachineException.cc"
  "TwoDPCAMachine.cc"
  "LinearMachine.cc"
  "MLP.cc"
  "MLPException.cc"
  "LinearScoring.cc"
  "ZTNorm.cc"
  "JFAMachine.cc"
  "JFAMachineException.cc"
  "WienerMachine.cc"
  "PLDAMachine.cc"
  "GaborGraphMachine.cc"
  "GaborJetSimilarities.cc"
  "BICMachine.cc"
  )

# If we have libsvm installed, enable the compilation of relevant modules
if(LIBSVM_FOUND)
  include_directories(SYSTEM ${LIBSVM_INCLUDE_DIR})
  list(APPEND shared "${LIBSVM_LIBRARY}")
  list(APPEND src "SVM.cc")
endif(LIBSVM_FOUND)

bob_add_library(${PROJECT_NAME} "${src}")
target_link_libraries(${PROJECT_NAME} ${shared})

# Defines tests for this package
bob_add_test(${PROJECT_NAME} linear test/linear.cc)
bob_add_test(${PROJECT_NAME} gabor test/gabor.cc)

# Pkg-Config generator
bob_pkgconfig(${PROJECT_NAME} "${bob_deps}")
