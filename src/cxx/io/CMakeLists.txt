project(io-cxx)
cmake_minimum_required(VERSION 2.6)

# This defines the dependencies of this package
set(deps "core") #other torch subprojects
set(shared "${Boost_REGEX_LIBRARY_RELEASE};${ImageMagick_LIBRARIES};${ffmpeg_LIBRARIES};${HDF5_hdf5_LIBRARY_RELEASE}")

# If we have google-perftools installed, enable the HAS_GOOGLE_PERFTOOLS flag,
# so profiling extensions can be compiled in.
if(GOOGLE_PERFTOOLS_FOUND)
  set(shared "${shared};${GOOGLE_PERFTOOLS_LIBRARIES}")
endif(GOOGLE_PERFTOOLS_FOUND)

# This defines the list of source files inside this package.
set(src
    "src/Exception.cc"

    "src/buffer.cc"
    "src/carray.cc"
    
    "src/Array.cc"
    "src/File.cc"
    "src/CodecRegistry.cc"
    
    "src/Arrayset.cc"

    "src/HDF5Exception.cc"
    "src/HDF5Types.cc"
    "src/HDF5Utils.cc"
    "src/HDF5File.cc"
    "src/HDF5ArrayFile.cc"

    "src/ImageFile.cc"

    "src/BinFileHeader.cc"
    "src/BinFile.cc"
    #"src/BinaryArrayCodec.cc"
    
    "src/VideoReader.cc"
    "src/VideoException.cc"
    "src/VideoWriter.cc"
    #"src/VideoArrayCodec.cc"

    "src/TensorFileHeader.cc"
    "src/TensorFile.cc"
    #"src/TensorArrayCodec.cc"

    #"src/T3BinaryArrayCodec.cc"
    
    )

# If we have matio installed, enable the compilation of relevant modules
if(MATIO_FOUND)
  set(shared "${shared};${matio_LIBRARY}")
  set(src "${src};src/MatUtils.cc")
  #set(src "${src};src/MatArrayCodec.cc")
endif(MATIO_FOUND)

include(../../cmake/macros.cmake)
torch_header_install(io-cxx io "torch::io")
torch_library(io "${src}" "${deps}" "${shared}")
add_dependencies(torch_io io-cxx_header_install)

# Defines tests for this package
torch_test(io array test/array.cc)
torch_test(io arrayset test/arrayset.cc)
torch_test(io hdf5 test/hdf5.cc)
torch_test(io image_codec test/image_codec.cc)
#torch_test(io tensor_codec test/tensor_codec.cc)

# Defines environment variable used to localize data for the tests
#set_property(TEST cxx-io-tensor_codec APPEND PROPERTY ENVIRONMENT
#  "TORCH_TESTDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/test/data")
