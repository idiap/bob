find_package(OpenCV)
  
if(OpenCV_FOUND)
  add_definitions("-DHAVE_OPENCV=1")
  find_package_message(OpenCV "Found OpenCV ${OpenCV_VERSION}: ${OpenCV_DIR}" "[${OpenCV_DIR}]")
endif()
