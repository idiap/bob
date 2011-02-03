project(core-cxx)
cmake_minimum_required(VERSION 2.6)

# This defines the dependencies of this package
set(deps "") #other torch subprojects
set(shared "${Boost_SYSTEM_LIBRARY_RELEASE};${Boost_THREAD_LIBRARY_RELEASE};${Boost_FILESYSTEM_LIBRARY_RELEASE};${Boost_IOSTREAMS_LIBRARY_RELEASE};${Boost_DATE_TIME_LIBRARY_RELEASE};${cblas_LIBRARIES};${LIBXML2_LIBRARIES};${Boost_PROGRAM_OPTIONS_LIBRARY_RELEASE}")

# If we have google-perftools installed, enable the HAS_GOOGLE_PERFTOOLS flag,
# so profiling extensions can be compiled in.
if(GOOGLE_PERFTOOLS_FOUND)
  set(shared "${shared};${GOOGLE_PERFTOOLS_LIBRARIES}")
endif(GOOGLE_PERFTOOLS_FOUND)

# This defines the list of source files inside this package.
set(src
   "src/general.cc"
    "src/blitz_misc.cc"
    "src/dataset_common.cc"
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
    "src/Dataset2.cc"
    "src/Arrayset.cc"
    "src/XMLParser.cc"
    "src/XMLWriter.cc"
    "src/BinFileHeader.cc"
    "src/BinFile.cc"
    )

include(../../cmake/macros.cmake)
torch_library(core "${src}" "${deps}" "${shared}")
torch_header_install(core "torch::core")

# Defines tests for this package
torch_test(core tensor test/tensor.cc)
torch_test(core adapter test/blitzAdapter.cc)
torch_test(core logging test/logging.cc)
torch_test(core binary_file test/binary_file.cc)
torch_test(core blitzarray test/blitzArray.cc)

# Defines benchmarks we make available
torch_benchmark(core tensor benchmark/tensor.cc)

# This is where we install the XML schema definition
set(schemadir share/torch/schema)
install(FILES schema/dataset.xsd DESTINATION ${schemadir})
