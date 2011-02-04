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
    "src/Array.cc"
    #"src/Dataset.cc"
    #"src/XMLParser.cc"
    #"src/XMLWriter.cc"
    #"src/Relationset.cc"
    #"src/Rule.cc"
    #"src/Member.cc"
    #"src/Relation.cc"
    #"src/Arrayset.cc"
    #"src/BinFileHeader.cc"
    #"src/BinFile.cc"
    )

include(../../cmake/macros.cmake)
torch_library(database "${src}" "${deps}" "${shared}")
torch_header_install(database "torch::database")

# Defines tests for this package
torch_test(database binary_file test/binary_file.cc)

# This is where we install the XML schema definition
set(schemadir share/torch/schema)
install(FILES schema/dataset.xsd DESTINATION ${schemadir})
