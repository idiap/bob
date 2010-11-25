project(core)
cmake_minimum_required(VERSION 2.6)

# This defines the dependencies of this package
set(deps "") #other torch subprojects
set(shared "${Boost_SYSTEM_LIBRARY};${Boost_THREAD_LIBRARY};${Boost_FILESYSTEM_LIBRARY};${Boost_IOSTREAMS_LIBRARY};${Boost_DATE_TIME_LIBRARY};${cblas_LIBRARIES};${LIBXML2_LIBRARIES}")
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
    "src/DatasetXML.cc"
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
   "python/src/array_bool_1.cc"
   "python/src/array_int8_1.cc"
   "python/src/array_uint8_1.cc"
   "python/src/array_int64_1.cc"
   "python/src/array_uint64_1.cc"
   "python/src/array_float32_1.cc"
   "python/src/array_complex64_1.cc"
   "python/src/array_int16_1.cc"
   "python/src/array_uint16_1.cc"
   "python/src/array_float64_1.cc"
   "python/src/array_int32_1.cc"
   "python/src/array_uint32_1.cc"
   "python/src/array_float128_1.cc"
   "python/src/array_complex128_1.cc"
   "python/src/array_complex256_1.cc"
   "python/src/array_bool_2.cc"
   "python/src/array_int8_2.cc"
   "python/src/array_uint8_2.cc"
   "python/src/array_int64_2.cc"
   "python/src/array_uint64_2.cc"
   "python/src/array_float32_2.cc"
   "python/src/array_complex64_2.cc"
   "python/src/array_int16_2.cc"
   "python/src/array_uint16_2.cc"
   "python/src/array_float64_2.cc"
   "python/src/array_int32_2.cc"
   "python/src/array_uint32_2.cc"
   "python/src/array_float128_2.cc"
   "python/src/array_complex128_2.cc"
   "python/src/array_complex256_2.cc"
   "python/src/array_bool_3.cc"
   "python/src/array_int8_3.cc"
   "python/src/array_uint8_3.cc"
   "python/src/array_int64_3.cc"
   "python/src/array_uint64_3.cc"
   "python/src/array_float32_3.cc"
   "python/src/array_complex64_3.cc"
   "python/src/array_int16_3.cc"
   "python/src/array_uint16_3.cc"
   "python/src/array_float64_3.cc"
   "python/src/array_int32_3.cc"
   "python/src/array_uint32_3.cc"
   "python/src/array_float128_3.cc"
   "python/src/array_complex128_3.cc"
   "python/src/array_complex256_3.cc"
   "python/src/array_bool_4.cc"
   "python/src/array_int8_4.cc"
   "python/src/array_uint8_4.cc"
   "python/src/array_int64_4.cc"
   "python/src/array_uint64_4.cc"
   "python/src/array_float32_4.cc"
   "python/src/array_complex64_4.cc"
   "python/src/array_int16_4.cc"
   "python/src/array_uint16_4.cc"
   "python/src/array_float64_4.cc"
   "python/src/array_int32_4.cc"
   "python/src/array_uint32_4.cc"
   "python/src/array_float128_4.cc"
   "python/src/array_complex128_4.cc"
   "python/src/array_complex256_4.cc"
   "python/src/main_array.cc"
   )

torch_python_bindings(core "${pysrc}")
torch_python_submodule(core array "${pysrc_array}" "core/python/array.h")

torch_python_install(core)
