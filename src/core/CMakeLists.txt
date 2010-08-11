project(core)
cmake_minimum_required(VERSION 2.4)

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
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
torch_library(core "${src}" "${deps}" "${shared}")
