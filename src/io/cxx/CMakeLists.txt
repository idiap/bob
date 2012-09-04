project(bob_io)

# This defines the dependencies of this package
set(bob_deps "bob_core")
set(shared "${bob_deps};${HDF5_hdf5_LIBRARY_RELEASE}")
set(incdir "${HDF5_CXX_INCLUDE_DIR}")
set(libdir "")

# This defines the list of source files inside this package.
set(src
    "Exception.cc"
    "reorder.cc"

    "Array.cc"
    "File.cc"
    "CodecRegistry.cc"
    
    "Arrayset.cc"

    "HDF5Exception.cc"
    "HDF5Types.cc"
    "HDF5Utils.cc"
    "HDF5Group.cc"
    "HDF5Dataset.cc"
    "HDF5Attribute.cc"
    "HDF5File.cc"
    "HDF5ArrayFile.cc"

    "CSVFile.cc"

    "BinFileHeader.cc"
    "BinFile.cc"
    "BinaryArrayFile.cc"
    
    "TensorFileHeader.cc"
    "TensorFile.cc"
    "TensorArrayFile.cc"

    "T3File.cc"
    )

# If we have matio installed, enable the compilation of relevant modules
if(matio_FOUND)
  list(APPEND shared "${matio_LIBRARIES}")
  list(APPEND src 
    "MatUtils.cc"
    "MatFile.cc"
    )
  list(APPEND incdir "${matio_INCLUDE_DIRS}")
  list(APPEND libdir "${matio_LIBRARY_DIRS}")
endif(matio_FOUND)

if(ImageMagick_FOUND)
  list(APPEND shared "${ImageMagick_LIBRARIES}")
  list(APPEND src
    "ImageFile.cc"
    )
  list(APPEND incdir "${ImageMagick_INCLUDE_DIRS}")
  list(APPEND libdir "${ImageMagick_LIBRARY_DIRS}")
endif(ImageMagick_FOUND)

if(FFMPEG_FOUND)
  list(APPEND shared "${FFMPEG_LIBRARIES}")
  list(APPEND src 
    "VideoReader.cc"
    "VideoException.cc"
    "VideoWriter.cc"
    "VideoFile.cc"
  )
  list(APPEND incdir "${FFMPEG_INCLUDE_DIRS}")
  list(APPEND libdir "${FFMPEG_LIBRARY_DIRS}")
  add_definitions("-D__STDC_CONSTANT_MACROS")
endif(FFMPEG_FOUND)

# Define the library, compilation and linkage options
list(REMOVE_DUPLICATES incdir)
foreach(inc ${incdir})
  include_directories(SYSTEM ${inc})
endforeach()
bob_add_library(${PROJECT_NAME} "${src}")
list(REMOVE_DUPLICATES libdir)
set_property(TARGET ${PROJECT_NAME} 
  APPEND PROPERTY LINK_DIRECTORIES ${libdir})
target_link_libraries(${PROJECT_NAME} ${shared})

# Defines tests for this package
bob_add_test(${PROJECT_NAME} array test/array.cc)
bob_add_test(${PROJECT_NAME} arrayset test/arrayset.cc)
bob_add_test(${PROJECT_NAME} hdf5 test/hdf5.cc)
bob_add_test(${PROJECT_NAME} tensor_codec test/tensor_codec.cc)

if(ImageMagick_FOUND)
  bob_add_test(${PROJECT_NAME} image_codec test/image_codec.cc)
endif(ImageMagick_FOUND)

# Pkg-Config generator
bob_pkgconfig(${PROJECT_NAME} "${bob_deps}")
