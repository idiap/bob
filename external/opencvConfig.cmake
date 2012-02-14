find_package(OpenCV)
  
if(OpenCV_FOUND)
  find_package_message(OpenCV "Found OpenCV ${OpenCV_VERSION}: ${OpenCV_LIBS}" "[${OpenCV_LIBS}][${OpenCV_INCLUDE_DIR}]")
  add_definitions("-D HAVE_OPENCV=1")
endif()
