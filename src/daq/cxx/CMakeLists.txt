project(bob_daq)

# This defines the dependencies of this package
set(bob_deps "bob_io;bob_ip;bob_visioner")
set(shared "${bob_deps}")
set(incdir ${cxx_incdir};${ImageMagick_INCLUDE_DIRS};${FFMPEG_INCLUDE_DIRS})

# This is required for FFMPEG compilation
add_definitions("-D__STDC_CONSTANT_MACROS")

# This defines the list of source files inside this package.
set(src
    "Controller.cc"
    "Camera.cc"
    "Display.cc"
    "FaceLocalization.cc"
    "OutputWriter.cc"
    "SimpleController.cc"
    "QtDisplay.cc"
    "VideoReaderCamera.cc"
    "ConsoleDisplay.cc"
    "CaptureSystem.cc"
    "VisionerFaceLocalization.cc"
    "BobOutputWriter.cc"
    )

if (V4L2_FOUND)
  list(APPEND src 
    "V4LCamera.cc"
    )
  list(APPEND incdir ${V4L2_INCLUDE_DIRS})
endif()

if (OpenCV_FOUND)
  list(APPEND src 
    "OpenCVFaceLocalization.cc"
    )
  list(APPEND incdir ${OpenCV_INCLUDE_DIRS})
  list(APPEND shared ${OpenCV_LIBRARIES})
endif()

include (${QT_USE_FILE})
list(APPEND incdir ${QT_INCLUDE_DIR})
list(APPEND shared ${QT_LIBRARIES})

qt4_wrap_cpp(MOC_SRC "${CMAKE_SOURCE_DIR}/include/bob/daq/QtDisplay.h")
list(INSERT src 0 "${MOC_SRC}")

# Define the library, compilation and linkage options
bob_sort_headers(incdir)
foreach(inc ${incdir})
  include_directories(SYSTEM ${inc})
endforeach()
bob_add_library(${PROJECT_NAME} "${src}")
target_link_libraries(${PROJECT_NAME} ${shared})

# Pkg-Config generator
bob_pkgconfig(${PROJECT_NAME} "${bob_deps}")
