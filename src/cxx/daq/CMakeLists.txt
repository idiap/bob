project(daq-cxx)
cmake_minimum_required(VERSION 2.6)

# This defines the dependencies of this package
set(deps "visioner;core;ip;io") #other bob subprojects
set(shared "")

# If we have google-perftools installed, enable the HAS_GOOGLE_PERFTOOLS flag,
# so profiling extensions can be compiled in.
if(GOOGLE_PERFTOOLS_FOUND)
  set(shared "${shared};${GOOGLE_PERFTOOLS_LIBRARIES}")
endif(GOOGLE_PERFTOOLS_FOUND)

# This defines the list of source files inside this package.
set(src
    "src/Controller.cc"
    "src/Camera.cc"
    "src/CaptureSystem.cc"
    "src/Display.cc"
    "src/FaceLocalization.cc"
    "src/OutputWriter.cc"
    "src/SimpleController.cc"
    "src/VisionerFaceLocalization.cc"
    "src/QtDisplay.cc"
    "src/BobOutputWriter.cc"
    "src/VideoReaderCamera.cc"
    "src/ConsoleDisplay.cc"
    )

if (V4L2_FOUND)
  list(APPEND src 
    "src/V4LCamera.cc"
    )
endif()

if (OPENCV_FOUND)
  list(APPEND src 
    "src/OpenCVOutputWriter.cc"
    "src/OpenCVFaceLocalization.cc"
    "src/OpenCVDisplay.cc"
    )

  link_directories(${OpenCV_LIB_DIR})
  include_directories(${OpenCV_INCLUDE_DIRS})

  list(APPEND shared ${OpenCV_LIBS})
endif()


include (${QT_USE_FILE})
include_directories(${VISIONER_INCLUDE_DIRS} ${QT_INCLUDE_DIR})
list(APPEND shared ${QT_LIBRARIES})

qt4_wrap_cpp(MOC_SRC "daq/QtDisplay.h")
list(INSERT src 0 "${MOC_SRC}")


include(../../cmake/macros.cmake)
bob_header_install(daq-cxx daq "bob::daq")
bob_library(daq "${src}" "${deps}" "")
target_link_libraries(bob_daq  ${shared})
add_dependencies(bob_daq visioner)

