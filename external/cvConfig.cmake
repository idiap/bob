include(FindPkgConfig)

if(CMAKE_VERSION VERSION_LESS "2.8.2")
  pkg_check_modules(OpenCV opencv)
else()
  #starting at cmake-2.8.2, the QUIET option can be used
  pkg_check_modules(OpenCV QUIET opencv)
endif()

if(OpenCV_FOUND)

  #checks to see if libcvaux is installed - optional in some systems
  find_file(OpenCV_CVAUX_H_FOUND NAMES cvaux.h PATHS ${OpenCV_INCLUDE_DIRS})
  if (NOT OpenCV_CVAUX_H_FOUND)
    message(WARNING "OpenCV (version ${OpenCV_VERSION}) was found, but cvaux.h was not - Be aware!")
  endif()
  find_library(OpenCV_CVAUX_FOUND NAMES cvaux PATHS ${OpenCV_LIB_DIRS})
  if (NOT OpenCV_CVAUX_FOUND)
    message(WARNING "OpenCV (version ${OpenCV_VERSION}) was found, but libcvaux was not - I'm removing it from the required library set. If you need to compile OpenCV cvaux-enabled applications, please have that installed on your system")
    list(REMOVE_ITEM OpenCV_LIBRARIES cvaux)
  endif()

  #checks to see if libhighgui is installed - optional in some systems
  find_file(OpenCV_HIGHGUI_H_FOUND NAMES highgui.h PATHS ${OpenCV_INCLUDE_DIRS})
  if (NOT OpenCV_HIGHGUI_H_FOUND)
    message(WARNING "OpenCV (version ${OpenCV_VERSION}) was found, but highgui.h was not - Be aware!")
  endif()
  find_library(OpenCV_HIGHGUI_FOUND NAMES highgui PATHS ${OpenCV_LIB_DIRS})
  if (NOT OpenCV_HIGHGUI_FOUND)
    message(WARNING "OpenCV (version ${OpenCV_VERSION}) was found, but libhighgui was not - I'm removing it from the required library set. If you need to compile OpenCV/GUI enabled applications, please have that installed on your system")
    list(REMOVE_ITEM OpenCV_LIBRARIES highgui)
  endif()

  add_definitions("-DHAVE_OPENCV=1")
  add_definitions("-DOPENCV_VERSION=\"${OpenCV_VERSION}\"")
  find_package_message(MyOpenCV "Found OpenCV ${OpenCV_VERSION}: ${OpenCV_INCLUDE_DIRS} - ${OpenCV_LIBRARIES}" "[${OpenCV_LIBRARIES}][${OpenCV_INCLUDE_DIRS}]")
endif()
