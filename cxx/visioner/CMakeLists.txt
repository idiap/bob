project(visioner-cxx)
cmake_minimum_required(VERSION 2.6)

include("ExternalProject")

if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/project/CMakeLists.txt)

  message(WARNING "warning: you should do 'git submodule init; git submodule update' on the root of bob before attempting to run a build that includes the visioner. I'm disabling it for the time being. Re-run cmake to add it back after a proper initialization if you require that functionality.")

  set(BOB_VISIONER_ENABLED OFF PARENT_SCOPE)

else(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/project/CMakeLists.txt)

  ExternalProject_Add(visioner
    DOWNLOAD_COMMAND ""
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/project
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR} -DLBFGS_INCLUDE_DIR=${BOB_LBFGS_HEADER_DIRS} -DLBFGS_LIBRARY_DIR=${CMAKE_BINARY_DIR}/lib
    INSTALL_DIR ${CMAKE_BINARY_DIR}
    )

  set(BOB_VISIONER_ENABLED ON PARENT_SCOPE)

  add_dependencies(visioner bob_lbfgs)

  ExternalProject_Get_Property(visioner BINARY_DIR)
  ExternalProject_Get_Property(visioner SOURCE_DIR)


  if(APPLE)
    set(PATH_TO_SO ${CMAKE_BINARY_DIR}/lib/libbob_visioner.dylib)
  else()
    set(PATH_TO_SO ${CMAKE_BINARY_DIR}/lib/libbob_visioner.so)
  endif()

  set_target_properties(bob_visioner PROPERTIES IMPORTED_LOCATION ${PATH_TO_SO})

  install(FILES ${PATH_TO_SO} DESTINATION lib)

  set(BOB_VISIONER_HEADER_DIRS
    "${SOURCE_DIR}/src/ml"
    "${SOURCE_DIR}/src/proc"
    "${SOURCE_DIR}/src/util"
    "${SOURCE_DIR}/src/vision"
    CACHE INTERNAL "Visioner include directories"
    )

  foreach(HEADER_DIR in ${BOB_VISIONER_HEADER_DIRS})
    get_filename_component(DIR_NAME ${HEADER_DIR} NAME)
    install(DIRECTORY ${HEADER_DIR}
            DESTINATION include/bob/visioner/${DIR_NAME}
            FILES_MATCHING PATTERN "*.h")
  endforeach()

endif(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/project/CMakeLists.txt)
