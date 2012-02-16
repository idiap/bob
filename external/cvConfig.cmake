find_package(OpenCV)
  
if(OpenCV_FOUND)
  add_definitions("-DHAVE_OPENCV=1")
  add_definitions("-DOPENCV_VERSION=\"${OpenCV_VERSION}\"")
  find_package_message(OpenCV "Found OpenCV ${OpenCV_VERSION}: ${OpenCV_DIR}" "[${OpenCV_DIR}]")
endif()
