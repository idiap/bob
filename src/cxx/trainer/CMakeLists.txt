PROJECT(trainer-cxx)
cmake_minimum_required(VERSION 2.6)

# This defines the dependencies of this package
set(deps core database machine) #other torch subprojects
set(shared "")

# This defines the list of source files inside this package.
set(src
    "src/SimpleFrameSampler.cc"
    "src/SVDPCATrainer.cc"
    "src/KMeansTrainer.cc"
    "src/GMMTrainer.cc"
    "src/MAP_GMMTrainer.cc"
    "src/ML_GMMTrainer.cc"
)

include(../../cmake/macros.cmake)
torch_header_install(trainer-cxx trainer "torch::trainer")
torch_library(trainer "${src}" "${deps}" "${shared}")
add_dependencies(torch_trainer trainer-cxx_header_install)

# Defines tests for this package
#torch_test(trainer block test/block.cc)

# Defines environment variable used to localize data for the tests
#set_property(TEST cxx-trainer-gabor APPEND PROPERTY ENVIRONMENT
#  "TORCH_IP_TESTDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/test/data")

