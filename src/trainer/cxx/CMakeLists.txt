PROJECT(bob_trainer)

# This defines the dependencies of this package
set(bob_deps "bob_io;bob_machine;bob_math")
set(shared "${bob_deps}")
set(incdir ${cxx_incdir})

# This defines the list of source files inside this package.
set(src
  "SVDPCATrainer.cc"
  "FisherLDATrainer.cc"
  "KMeansTrainer.cc"
  "GMMTrainer.cc"
  "MAP_GMMTrainer.cc"
  "ML_GMMTrainer.cc"
  "Exception.cc"
  "TwoDPCATrainer.cc"
  "DataShuffler.cc"
  "MLPRPropTrainer.cc"
  "MLPBackPropTrainer.cc"
  "JFATrainer.cc"
  "WienerTrainer.cc"
  "EMPCATrainer.cc"
  "PLDATrainer.cc"
  "BICTrainer.cc"
  "LLRTrainer.cc"
  )

if(LIBSVM_FOUND)
  list(APPEND incdir ${LIBSVM_INCLUDE_DIR})
  list(APPEND shared "${LIBSVM_LIBRARY}")
  list(APPEND src "SVMTrainer.cc")
endif()

# Define the library, compilation and linkage options
bob_sort_headers(incdir)
foreach(inc ${incdir})
  include_directories(SYSTEM ${inc})
endforeach()
bob_add_library(${PROJECT_NAME} "${src}")
target_link_libraries(${PROJECT_NAME} ${shared})

# Defines tests for this package
bob_add_test(${PROJECT_NAME} jfa test/jfa.cc)
bob_add_test(${PROJECT_NAME} bic test/bic.cc)

# Pkg-Config generator
bob_pkgconfig(${PROJECT_NAME} "${bob_deps}")
