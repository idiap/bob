find_package(OpenCV)
  
if(OpenCV_FOUND)
  message( STATUS "OpenCV ${OpenCV_VERSION} FOUND, compiling add-ons...")
  add_definitions("-D HAVE_OPENCV=1")
else()
  message( STATUS "OpenCV NOT FOUND: Disabling...")
endif()
