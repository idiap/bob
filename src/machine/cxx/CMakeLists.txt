project(bob_machine)

# This defines the dependencies of this package
set(bob_deps "bob_ip")
set(shared "${bob_deps}")
set(incdir ${cxx_incdir})

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
  "IVectorMachine.cc"
  "WienerMachine.cc"
  "PLDAMachine.cc"
  "GaborGraphMachine.cc"
  "GaborJetSimilarities.cc"
  "BICMachine.cc"
  )

# If we have libsvm installed, enable the compilation of relevant modules
if(WITH_LIBSVM)
  list(APPEND incdir ${LIBSVM_INCLUDE_DIR})
  list(APPEND shared "${LIBSVM_LIBRARY}")
  list(APPEND src "SVM.cc")
endif(WITH_LIBSVM)

# Define the library, compilation and linkage options
bob_sort_headers(incdir)
foreach(inc ${incdir})
  include_directories(SYSTEM ${inc})
endforeach()
bob_add_library(${PROJECT_NAME} "${src}")
target_link_libraries(${PROJECT_NAME} ${shared})

# Defines tests for this package
bob_add_test(${PROJECT_NAME} linear test/linear.cc)
bob_add_test(${PROJECT_NAME} gabor test/gabor.cc)

# Pkg-Config generator
bob_pkgconfig(${PROJECT_NAME} "${bob_deps}")
