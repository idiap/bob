project(core)
cmake_minimum_required(VERSION 2.6)

# This defines the dependencies of this package
set(deps th) #other torch subprojects
set(shared "") #shared externals to link against (link)

# This defines the list of source files inside this package.
set(src
   "src/general.cc"
    "src/VariableCollector.cc"
    "src/Object.cc"
    "src/Parameters.cc"
    "src/File.cc"
#    "src/spCore.cc"
#    "src/spCoreChain.cc"
    "src/DataSet.cc"
    "src/MemoryDataSet.cc"
    "src/DiskDataSet.cc"
    "src/Tensor.cc"
    "src/TensorFile.cc"
    "src/TensorList.cc"
    "src/TensorListTarget.cc"
    "src/MTimer.cc"
    "src/CmdOption.cc"
    "src/FileListCmdOption.cc"
    "src/FileList.cc"
    "src/CmdLine.cc"
    "src/CmdFile.cc")

include(../macros.cmake)
torch_library(core "${src}" "${deps}" "${shared}")

# Defines tests for this package
torch_test(core tensor test/tensor.cc)

# Defines benchmarks we make available
torch_test(core tensor benchmark/tensor.cc)
