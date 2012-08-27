project(daq-cxx)
cmake_minimum_required(VERSION 2.6)

# This defines the dependencies of this package
set(deps "ip;io;visioner") #other bob subprojects
set(shared "")

# This defines the list of source files inside this package.
set(src
    "src/Controller.cc"
    "src/Camera.cc"
    "src/Display.cc"
    "src/FaceLocalization.cc"
    "src/OutputWriter.cc"
    "src/SimpleController.cc"
    "src/QtDisplay.cc"
    "src/BobOutputWriter.cc"
    "src/VideoReaderCamera.cc"
    "src/ConsoleDisplay.cc"
    "src/CaptureSystem.cc"
    "src/VisionerFaceLocalization.cc"
    )

if (V4L2_FOUND)
  list(APPEND src 
    "src/V4LCamera.cc"
    )
endif()

if (OpenCV_FOUND)

  list(APPEND src 
    "src/OpenCVFaceLocalization.cc"
    )
  
  link_directories(${OpenCV_LIB_DIR})
  include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})

  list(APPEND shared ${OpenCV_LIBRARIES})
endif()

include (${QT_USE_FILE})
include_directories(SYSTEM ${QT_INCLUDE_DIR})
list(APPEND shared ${QT_LIBRARIES})

qt4_wrap_cpp(MOC_SRC "daq/QtDisplay.h")
list(INSERT src 0 "${MOC_SRC}")

bob_library(daq "${src}" "${deps}" "${shared}")
