project(core)
cmake_minimum_required(VERSION 2.6)

# This defines the dependencies of this package
set(deps "") #other torch subprojects
set(shared "${Boost_SYSTEM_LIBRARY};${Boost_THREAD_LIBRARY};${Boost_FILESYSTEM_LIBRARY};${Boost_IOSTREAMS_LIBRARY};${Boost_DATE_TIME_LIBRARY};${cblas_LIBRARIES}")
add_definitions(-DUSE_CBLAS)

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
   "python/src/logging.cc"
   "python/src/object.cc"
   "python/src/file.cc"
   "python/src/tensor.cc"
   "python/src/tensorfile.cc"
   "python/src/profile.cc"
   "python/src/DataSet.cc"
   "python/src/ListDataSet.cc"
   "python/src/main.cc"
   )
 
set(pysrc_array
   "python/src/tinyvector.cc"
   "python/src/array.cc"
   "python/src/array_bool.cc"
   "python/src/array_int8.cc"
   "python/src/array_int16.cc"
   "python/src/array_int32.cc"
   "python/src/array_int64.cc"
   "python/src/array_uint8.cc"
   "python/src/array_uint16.cc"
   "python/src/array_uint32.cc"
   "python/src/array_uint64.cc"
   "python/src/array_float32.cc"
   "python/src/array_float64.cc"
   "python/src/array_float128.cc"
   "python/src/array_complex64.cc"
   "python/src/array_complex128.cc"
   "python/src/array_complex256.cc"
   "python/src/main_array.cc"
   )

torch_python_bindings(core "${pysrc}")

#some pre-compiled header black magic: read macros.cmake for details. if you
#don't know what pre-compiled headers are good for, google for it.
torch_add_python_pch("python/src/array.h" pysrc_array)
torch_python_submodule(core array "${pysrc_array}")

torch_python_install(core)

