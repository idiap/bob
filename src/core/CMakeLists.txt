project(core)
cmake_minimum_required(VERSION 2.6)

# This defines the dependencies of this package
set(deps th) #other torch subprojects
set(shared "${Boost_THREAD_LIBRARY};${Boost_FILESYSTEM_LIBRARY};${Boost_IOSTREAMS_LIBRARY}") #shared externals to link against

# If we have google-perftools installed, enable the HAS_GOOGLE_PERFTOOLS flag,
# so profiling extensions can be compiled in.
if(GOOGLE_PERFTOOLS_FOUND)
  set(shared "${shared};${GOOGLE_PERFTOOLS_LIBRARIES}")
endif(GOOGLE_PERFTOOLS_FOUND)

# This defines the list of source files inside this package.
set(src
   "src/general.cc"
    "src/VariableCollector.cc"
    "src/Object.cc"
    "src/Parameters.cc"
    "src/File.cc"
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
    "src/CmdFile.cc"
    "src/Random.cc"
    "src/ListDataSet.cc"
    "src/Exception.cc"
    "src/logging.cc"
    )

include(../cmake/macros.cmake)
torch_library(core "${src}" "${deps}" "${shared}")

# Defines tests for this package
torch_test(core tensor test/tensor.cc)
torch_test(core adapter test/blitzAdapter.cc)
torch_test(core logging test/logging.cc)

# Defines benchmarks we make available
torch_benchmark(core tensor benchmark/tensor.cc)

# Python bindings
set(pysrc
   "python/src/exception.cc"
   "python/src/object.cc"
   "python/src/file.cc"
   "python/src/tensor.cc"
   "python/src/tensorfile.cc"
   "python/src/profile.cc"
   "python/src/DataSet.cc"
   "python/src/ListDataSet.cc"
   "python/src/main.cc"
   )
 
torch_python_bindings(core "${pysrc}")
torch_python_install(core)

