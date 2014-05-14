project(bob_io)

# HDF5
set(HDF5_FIND_COMPONENTS C CXX)
include(FindHDF5)

# This defines the dependencies of this package
set(bob_deps "bob_core")
set(shared "${bob_deps};${HDF5_hdf5_LIBRARY_RELEASE}")
set(incdir ${cxx_incdir};${HDF5_CXX_INCLUDE_DIR})
set(libdir "")

# This defines the list of source files inside this package.
set(src
    "reorder.cc"

    "File.cc"
    "CodecRegistry.cc"
    "utils.cc"

    "HDF5Types.cc"
    "HDF5Utils.cc"
    "HDF5Group.cc"
    "HDF5Dataset.cc"
    "HDF5Attribute.cc"
    "HDF5File.cc"

    "TensorFileHeader.cc"
    "TensorFile.cc"

    # File implementations
    "HDF5ArrayFile.cc"
    "CSVFile.cc"
    "TensorArrayFile.cc"
    "T3File.cc"
    )

# Define the library, compilation and linkage options
bob_sort_headers(incdir)
foreach(inc ${incdir})
  include_directories(SYSTEM ${inc})
endforeach()
bob_add_library(${PROJECT_NAME} "${src}")
target_link_libraries(${PROJECT_NAME} ${shared})

# Defines tests for this package
bob_add_test(${PROJECT_NAME} hdf5 test/hdf5.cc)
bob_add_test(${PROJECT_NAME} tensor_codec test/tensor_codec.cc)

# Pkg-Config generator
bob_pkgconfig(${PROJECT_NAME} "${bob_deps}")

set(ENABLED_PACKAGES "${ENABLED_PACKAGES}" PARENT_SCOPE)
