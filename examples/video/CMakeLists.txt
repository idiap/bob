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
  ADD_DEPENDENCIES(${file_exe} torch5spro-uniheader) #has to be ready
  if (${TORCH_LINKAGE} STREQUAL dynamic)
    TARGET_LINK_LIBRARIES(${file_exe} torch5spro)
    TARGET_LINK_LIBRARIES(${file_exe} libjpeg.so)
    TARGET_LINK_LIBRARIES(${file_exe} libcblas.so)
    TARGET_LINK_LIBRARIES(${file_exe} libavformat.so)
    TARGET_LINK_LIBRARIES(${file_exe} libavcodec.so)
    TARGET_LINK_LIBRARIES(${file_exe} libavutil.so)
    TARGET_LINK_LIBRARIES(${file_exe} libswscale.so) #and we need to be GPL?
  else (${TORCH_LINKAGE} STREQUAL dynamic)
    TARGET_LINK_LIBRARIES(${file_exe} torch5spro-static)
    TARGET_LINK_LIBRARIES(${file_exe} libjpeg.a)
    TARGET_LINK_LIBRARIES(${file_exe} libcblas.a)
    TARGET_LINK_LIBRARIES(${file_exe} libatlas.a)
    TARGET_LINK_LIBRARIES(${file_exe} libavformat.a)
    TARGET_LINK_LIBRARIES(${file_exe} libavcodec.a)
    TARGET_LINK_LIBRARIES(${file_exe} libavutil.a)
    TARGET_LINK_LIBRARIES(${file_exe} libswscale.a) #and we need to be GPL?
  endif (${TORCH_LINKAGE} STREQUAL dynamic)
  
  ADD_CUSTOM_TARGET(install-${file_exe} COMMAND cp -f ${CMAKE_CURRENT_BINARY_DIR}/${file_exe} ${INSTALL_DIR}/bin DEPENDS ${file_exe})
  ADD_DEPENDENCIES(all-examples ${file_exe})
  ADD_DEPENDENCIES(install-all-examples install-${file_exe})
ENDFOREACH(file)
