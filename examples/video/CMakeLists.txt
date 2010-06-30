#
# Top level project
#
PROJECT(EXAMPLE)

CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

SET(EXAMPLE_SRC
	testio.cc
  #	mobioVideoScan.cc
	checkvideo.cc
	checkvideos.cc
  #	video2tensor.cc
	makeFaceFinder.cc
	fdvideo.cc)
#	fdtvideo.cc)

LINK_DIRECTORIES(${INSTALL_DIR}/lib;/usr/lib/atlas)
INCLUDE_DIRECTORIES(${TORCH5SPRO_INCLUDE})

FOREACH(file ${EXAMPLE_SRC})
	STRING(REGEX REPLACE "([a-z;A-Z;0-9]+).*" "\\1" file_exe "${file}" )
	ADD_EXECUTABLE(${file_exe} ${file})
  ADD_DEPENDENCIES(${file_exe} jpeg) #has to compile before
  ADD_DEPENDENCIES(${file_exe} ffmpeg) #has to compile before
  TARGET_LINK_LIBRARIES(${file_exe} torch5spro)
  TARGET_LINK_LIBRARIES(${file_exe} libjpeg.so)
  TARGET_LINK_LIBRARIES(${file_exe} libcblas.so)
  TARGET_LINK_LIBRARIES(${file_exe} libavformat.so)
  TARGET_LINK_LIBRARIES(${file_exe} libavcodec.so)
  TARGET_LINK_LIBRARIES(${file_exe} libavutil.so)
  TARGET_LINK_LIBRARIES(${file_exe} libswscale.so)
  ADD_CUSTOM_TARGET(install-${file_exe} COMMAND cp -f ${CMAKE_CURRENT_BINARY_DIR}/${file_exe} ${INSTALL_DIR}/bin)

  ADD_DEPENDENCIES(build-examples ${file_exe})
  ADD_DEPENDENCIES(install-examples install-${file_exe})
ENDFOREACH(file)
