project(database-cxx)
cmake_minimum_required(VERSION 2.6)

# This defines the dependencies of this package
set(deps "core") #other torch subprojects
set(shared "${LIBXML2_LIBRARIES}")

# If we have google-perftools installed, enable the HAS_GOOGLE_PERFTOOLS flag,
# so profiling extensions can be compiled in.
if(GOOGLE_PERFTOOLS_FOUND)
  set(shared "${shared};${GOOGLE_PERFTOOLS_LIBRARIES}")
endif(GOOGLE_PERFTOOLS_FOUND)

# This defines the list of source files inside this package.
set(src
    "src/dataset_common.cc"
    "src/InlinedArrayImpl.cc"
    "src/ArrayCodec.cc"
    "src/ArrayCodecRegistry.cc"
    "src/ExternalArrayImpl.cc"
    "src/Array.cc"
    "src/BinFileHeader.cc"
    "src/BinFile.cc"
    "src/BinaryArrayCodec.cc"
    "src/InlinedArraysetImpl.cc"
    "src/ArraysetCodec.cc"
    "src/ArraysetCodecRegistry.cc"
    "src/BinaryArraysetCodec.cc"
    "src/ExternalArraysetImpl.cc"
    "src/Arrayset.cc"
    "src/Dataset.cc"
    "src/Relation.cc"
    "src/Relationset.cc"
    "src/Rule.cc"
    "src/XMLParser.cc"
    #"src/XMLWriter.cc"
    )

include(../../cmake/macros.cmake)
torch_library(database "${src}" "${deps}" "${shared}")
torch_header_install(database "torch::database")

# Defines tests for this package
torch_test(database binary_file test/binary_file.cc)
torch_test(database array test/array.cc)
torch_test(database arrayset test/arrayset.cc)

# This is where we install the XML schema definition
set(schemadir share/torch/schema)
install(FILES schema/dataset.xsd DESTINATION ${schemadir})
