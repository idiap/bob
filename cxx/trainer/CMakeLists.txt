PROJECT(trainer-cxx)
cmake_minimum_required(VERSION 2.6)

# This defines the dependencies of this package
set(deps core io machine math) #other bob subprojects
set(shared "")

# This defines the list of source files inside this package.
set(src
    "src/SVDPCATrainer.cc"
    "src/FisherLDATrainer.cc"
    "src/KMeansTrainer.cc"
    "src/GMMTrainer.cc"
    "src/MAP_GMMTrainer.cc"
    "src/ML_GMMTrainer.cc"
    "src/Exception.cc"
    "src/TwoDPCATrainer.cc"
    "src/DataShuffler.cc"
    "src/MLPRPropTrainer.cc"
    "src/MLPBackPropTrainer.cc"
    "src/JFATrainer.cc"
    "src/WienerTrainer.cc"
    "src/EMPCATrainer.cc"
    "src/PLDATrainer.cc"
    "src/BICTrainer.cc"
)

if(LIBSVM_FOUND)
  include_directories(SYSTEM ${LIBSVM_INCLUDE_DIR})
  set(shared "${shared};${LIBSVM_LIBRARY}")
  set(src "${src};src/SVMTrainer.cc")
endif()

bob_library(trainer "${src}" "${deps}" "${shared}")

# Defines tests for this package
bob_test(trainer jfa test/jfa.cc)
bob_test(trainer bic test/bic.cc)

# Defines environment variable used to localize data for the tests
#set_property(TEST cxx-trainer-gabor APPEND PROPERTY ENVIRONMENT
#  "BOB_IP_TESTDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/test/data")
