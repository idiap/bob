project(bob_io)

# This defines the dependencies of this package
set(bob_deps "bob_core")
set(shared "${bob_deps};${HDF5_hdf5_LIBRARY_RELEASE}")
set(incdir ${cxx_incdir};${HDF5_CXX_INCLUDE_DIR})
set(libdir "")

# This defines the list of source files inside this package.
set(src
    "Exception.cc"
    "reorder.cc"

    "File.cc"
    "CodecRegistry.cc"
    "utils.cc"
    
    "HDF5Exception.cc"
    "HDF5Types.cc"
    "HDF5Utils.cc"
    "HDF5Group.cc"
    "HDF5Dataset.cc"
    "HDF5Attribute.cc"
    "HDF5File.cc"

    "BinFileHeader.cc"
    "BinFile.cc"

    "TensorFileHeader.cc"
    "TensorFile.cc"

    # File implementations
    "HDF5ArrayFile.cc"
    "CSVFile.cc"
    "BinaryArrayFile.cc"
    "TensorArrayFile.cc"
    "T3File.cc"
    "ImageBmpFile.cc"
    )

# If we have matio installed, enable the compilation of relevant modules
if(matio_FOUND)
  list(APPEND shared "${matio_RESOLVED_LIBRARIES}")
  list(APPEND src 
    "MatUtils.cc"
    "MatFile.cc"
    )
  list(APPEND incdir "${matio_INCLUDE_DIRS}")
endif(matio_FOUND)

if(NETPBM_FOUND)
  list(APPEND shared "${NETPBM_LIBRARIES}")
  list(APPEND src
    "ImageNetpbmFile.cc"
    )
  list(APPEND incdir "${NETPBM_INCLUDE_DIRS}")
endif(NETPBM_FOUND)

if(JPEG_FOUND)
  list(APPEND shared "${JPEG_LIBRARIES}")
  list(APPEND src
    "ImageJpegFile.cc"
    )
  list(APPEND incdir "${JPEG_INCLUDE_DIRS}")
endif(JPEG_FOUND)

if(PNG_FOUND)
  list(APPEND shared "${PNG_LIBRARIES}")
  list(APPEND src
    "ImagePngFile.cc"
    )
  list(APPEND incdir "${PNG_INCLUDE_DIRS}")
endif(PNG_FOUND)

if(TIFF_FOUND)
  list(APPEND shared "${TIFF_LIBRARIES}")
  list(APPEND src
    "ImageTiffFile.cc"
    )
  list(APPEND incdir "${TIFF_INCLUDE_DIR}")
endif(TIFF_FOUND)

if(GIF_FOUND)
  list(APPEND shared "${GIF_LIBRARIES}")
  list(APPEND src
    "ImageGifFile.cc"
    )
  list(APPEND incdir "${GIF_INCLUDE_DIR}")
endif(GIF_FOUND)

if(FFMPEG_FOUND)
  list(APPEND shared "${FFMPEG_RESOLVED_LIBRARIES}")
  list(APPEND src 
    "VideoException.cc"
    "VideoFile.cc"
    "VideoUtilities.cc"
    "VideoWriter.cc"
    "VideoReader.cc"
  )
  list(APPEND incdir "${FFMPEG_INCLUDE_DIRS}")
  add_definitions("-D__STDC_CONSTANT_MACROS")
endif(FFMPEG_FOUND)

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

if(NETPBM_FOUND AND JPEG_FOUND AND PNG_FOUND AND TIFF_FOUND AND GIF_FOUND)
  bob_add_test(${PROJECT_NAME} image_codec test/image_codec.cc)
endif(NETPBM_FOUND AND JPEG_FOUND AND PNG_FOUND AND TIFF_FOUND AND GIF_FOUND)

# Pkg-Config generator
bob_pkgconfig(${PROJECT_NAME} "${bob_deps}")
