project(bob_daq)

# This defines the dependencies of this package
set(bob_deps "bob_ip;bob_visioner")
set(shared "${bob_deps}")
set(incdir "${ImageMagick_INCLUDE_DIRS}")
set(libdir "${ImageMagick_LIBRARY_DIRS}")

# This defines the list of source files inside this package.
set(src
    "Controller.cc"
    "Camera.cc"
    "Display.cc"
    "FaceLocalization.cc"
    "OutputWriter.cc"
    "SimpleController.cc"
    "QtDisplay.cc"
    "BobOutputWriter.cc"
    "VideoReaderCamera.cc"
    "ConsoleDisplay.cc"
    "CaptureSystem.cc"
    "VisionerFaceLocalization.cc"
    )

if (V4L2_FOUND)
  list(APPEND src 
    "V4LCamera.cc"
    )
endif()

if (OpenCV_FOUND)
  list(APPEND src 
    "OpenCVFaceLocalization.cc"
    )
  list(APPEND incdir ${OpenCV_INCLUDE_DIRS})
  list(APPEND libdir ${OpenCV_LIB_DIR})
  list(APPEND shared ${OpenCV_LIBRARIES})
endif()

include (${QT_USE_FILE})
list(APPEND incdir ${QT_INCLUDE_DIR})
list(APPEND shared ${QT_LIBRARIES})

qt4_wrap_cpp(MOC_SRC "${CMAKE_SOURCE_DIR}/include/bob/daq/QtDisplay.h")
list(INSERT src 0 "${MOC_SRC}")

list(REMOVE_DUPLICATES incdir)
foreach(inc ${incdir})
  include_directories(SYSTEM ${inc})
endforeach()
bob_add_library(${PROJECT_NAME} "${src}")
list(REMOVE_DUPLICATES libdir)
set_property(TARGET ${PROJECT_NAME} 
  APPEND PROPERTY LINK_DIRECTORIES ${libdir})
target_link_libraries(${PROJECT_NAME} ${shared})

# Pkg-Config generator
bob_pkgconfig(${PROJECT_NAME} "${bob_deps}")
